"""ทดสอบ evaluate.py --engine glm และการหยุดเมื่อ Gemini ตอบ 429 — ไม่เรียก Gemini/Ollama จริง

- server ถูกแทนด้วย stub (ไม่ต้องมี torch/OBB): class ของแต่ละภาพมาจาก meter_type ใน ground truth
- Ollama จำลองด้วย mock httpx.post/get — คำตอบสร้างจากเฉลย แล้วใส่ความผิดที่รู้ผลแน่นอนไว้
ใช้ ground_truth.xlsx + eval_images/ จริงของโปรเจค (ผลเขียนลงโฟลเดอร์ชั่วคราว ไม่แตะ eval_output/)
รัน: python test_evaluate_engine.py
"""
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)

import evaluate  # noqa: E402

OUT = Path(tempfile.mkdtemp(prefix="eval_out_"))
evaluate.OUT_ROOT = OUT

sets, problems, _ = evaluate.load_ground_truth()
assert not problems, problems
TRUTH = {s["set_id"]: {f: v for f, (v, k) in s["truth"].items()} for s in sets}
TYPE_BY_IMAGE = {p.name: s["meter_type"] for s in sets for p in s["paths"]}
SET_BY_IMAGE = {p.name: s["set_id"] for s in sets for p in s["paths"]}

# ── stub ของ server ──
calls = {"gemini": 0, "current_image": None}


def fake_preprocess(img, request_id, idx, source_name=""):
    calls["current_image"] = source_name
    cls = "analog_dial" if TYPE_BY_IMAGE[source_name] == "analog" else "digital_screen"
    return img, cls


class FakeModels:
    def generate_content(self, **kw):
        calls["gemini"] += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED. quota exceeded")


server = types.SimpleNamespace(
    SEND_DESKEWED_TO_GEMINI=True, GEMINI_MODEL="gemini-test", OCR_ENGINE="gemini",
    obb_model=object(), preprocess_meter_image=fake_preprocess,
    check_request_meter_types=lambda names, classes: None,
    client=types.SimpleNamespace(models=FakeModels()),
    types=types.SimpleNamespace(GenerateContentConfig=lambda **kw: kw),
    build_gemini_prompt=lambda classes: "prompt", SYSTEM_INSTRUCTION="sys",
    MeterReading=None, parse_gemini_response=lambda r: None,
)
sys.modules["server"] = server

# ── Ollama จำลอง ──
LABELS = {"pressure_bar": "P", "temperature_c": "T", "correction_factor": "C",
          "qm_m3_h": "Qm", "qb_sm3_h": "Qb", "correct_v_scm": "Vb", "line_v_m3": "Vm"}
# ความผิดที่ตั้งใจใส่ (set_id -> (field, ค่าที่ GLM ตอบ))
ANALOG_OVERRIDE = {"S03": "161567"}       # เฉลย 161567.0 -> ขาดหลักช่องแดง (0) ต้องนับผิด แม้ float เท่ากัน
EVC_OVERRIDE = {"S01": ("pressure_bar", "9.65")}   # ผิด 1 ค่า
SENT = []


def analog_reply(set_id):
    if set_id in ANALOG_OVERRIDE:
        return ANALOG_OVERRIDE[set_id] + "\n"
    v = TRUTH[set_id]["turbine_v_m3"]
    return v.replace(".", "") + "m³\n"           # GLM ไม่ใส่จุดทศนิยม (ตาม prompt)


def evc_reply(set_id):
    lines = ["INDEX VALUE ALARM DATA CONFIG"]
    for f, lab in LABELS.items():
        v = TRUTH[set_id][f]
        if v.lower() in ("na", "?"):
            continue
        if EVC_OVERRIDE.get(set_id, (None,))[0] == f:
            v = EVC_OVERRIDE[set_id][1]
        lines.append(f"{lab} = {v}")
    lines.append("Zb/Z = 1.0171")                # ต้องถูกทิ้ง
    return "\n".join(lines) + "\n"


class Resp:
    def __init__(self, data, status=200):
        self._d, self.status_code, self.text = data, status, json.dumps(data)

    def json(self):
        return self._d


def fake_post(url, json=None, timeout=None):
    prompt = json["messages"][0]["content"][0]["text"]
    image = calls["current_image"]
    set_id = SET_BY_IMAGE[image]
    analog = "counter digits" in prompt
    SENT.append((set_id, "analog" if analog else "evc"))
    content = analog_reply(set_id) if analog else evc_reply(set_id)
    return Resp({"choices": [{"message": {"content": content}, "finish_reason": "stop",
                              "logprobs": {"content": []}}]})


def fake_get(url, timeout=None):
    return Resp({"models": [{"name": "glm-ocr:latest"}]})


def run(argv):
    with mock.patch.object(sys, "argv", ["evaluate.py"] + argv):
        try:
            evaluate.main()
        except SystemExit as e:
            return e.code
    return 0


# preprocess ถูกเรียกทีละภาพก่อน engine.read -> ภาพสุดท้ายของชุดคือภาพที่ใช้ตัดสินคำตอบ (ทุกชุดในเฉลยมีภาพเดียว ยกเว้น S01)
# S01 มี 2 ภาพ: ทั้งสองได้คำตอบชุดเดียวกัน -> ค่าเหมือนกัน -> merge ได้ค่าเดียว
with mock.patch("httpx.post", fake_post), mock.patch("httpx.get", fake_get):
    code = run(["--engine", "glm", "--mode", "deskew", "--run-id", "t1"])
assert code in (0, None), code

rdir = OUT / "glm_deskew" / "t1"
assert rdir.is_dir(), list(OUT.rglob("*"))
fr = {(r["set_id"], r["field"]): r for r in __import__("csv").DictReader(open(rdir / "field_results.csv", encoding="utf-8-sig"))}
sr = {r["set_id"]: r for r in __import__("csv").DictReader(open(rdir / "set_results.csv", encoding="utf-8-sig"))}
summary = (rdir / "summary.md").read_text(encoding="utf-8")

assert all(r["status"] == "ok" for r in sr.values()), sr
# 1. turbine: ตัวเลขตรงนับถูก แม้ไม่มีจุดทศนิยม
analog_ids = [s["set_id"] for s in sets if s["meter_type"] == "analog"]
for sid in analog_ids:
    r = fr[(sid, "turbine_v_m3")]
    assert r["scoring"] == "digits", r
    expect = "misread" if sid in ANALOG_OVERRIDE else "correct"
    assert r["outcome"] == expect, (sid, r)
print(f"✅ turbine ตัวเลขตรง: {len(analog_ids) - 1} ชุดถูก, S03 (161567 vs 161567.0) นับผิดถูกต้อง")
# 2. ฟิลด์อื่นเข้มงวด
assert fr[("S01", "pressure_bar")]["outcome"] == "misread"
assert fr[("S01", "pressure_bar")]["scoring"] == "strict"
assert fr[("S01", "temperature_c")]["outcome"] == "correct"
assert all(r["scoring"] == "strict" for (sid, f), r in fr.items() if f != "turbine_v_m3")
print("✅ ฟิลด์อื่นให้คะแนนแบบเข้มงวด (S01 pressure ผิดถูกจับได้)")
# 3. ค่าดิบเป็นสตริงตามที่ GLM เขียน ไม่ผ่าน float
s01_line = fr[("S01", "line_v_m3")]
assert s01_line["predicted"] == "273155.00", s01_line   # 0 นำหน้าตัดโดย engine, ศูนย์ท้ายคงอยู่ (ไม่ผ่าน float)
assert s01_line["outcome"] == "correct"
print(f"✅ predicted คงสตริงเดิม: {s01_line['predicted']}")
# 4. Zb/Z ไม่ลงฟิลด์ไหน, NA ไม่ hallucinate
assert not any(r["outcome"] == "hallucinated" for r in fr.values())
print("✅ ไม่มี hallucinated (Zb/Z ถูกทิ้ง)")
# 5. analog ใช้ prompt analog, digital ใช้ prompt EVC
for sid, kind in SENT:
    assert kind == ("analog" if sid in analog_ids else "evc"), (sid, kind)
print(f"✅ prompt ตามชนิดภาพ ({len(SENT)} คำขอ, 1 ต่อภาพ)")
# 6. raw เก็บข้อความ GLM + confidence + เวลา
meta = json.loads((rdir / "raw" / "S02.json").read_text(encoding="utf-8"))
assert meta["engine"] == "glm" and meta["model"] == "glm-ocr:latest"
assert meta["glm_raw_text"] and "confidence_scores" in meta and meta["elapsed_s"] is not None
assert "gemini_model" not in meta
# 7. summary มีหัวข้อวิธีให้คะแนน + เวลา + engine
assert "engine `glm`" in summary and "วิธีให้คะแนน turbine_v_m3" in summary and "เวลาเรียก engine" in summary
strict_line = [l for l in summary.splitlines() if "ถ้าใช้เกณฑ์เข้มงวด" in l][0]
print("✅ summary:", strict_line.strip())
print("   " + [l for l in summary.splitlines() if l.startswith("1. ")][0])

# 8. rescore ได้ผลเดิม
with mock.patch("httpx.post", fake_post), mock.patch("httpx.get", fake_get):
    assert run(["--engine", "glm", "--mode", "deskew", "--run-id", "t1", "--rescore"]) in (0, None)
print("✅ --rescore ใช้โฟลเดอร์ glm_deskew ถูกต้อง")

# 9. Ollama ไม่เปิด -> หยุดก่อนเริ่ม ไม่สร้าง raw
import httpx  # noqa: E402


def down(*a, **k):
    raise httpx.ConnectError("refused")


with mock.patch("httpx.get", down), mock.patch("httpx.post", down):
    code = run(["--engine", "glm", "--mode", "deskew", "--run-id", "t_down"])
assert isinstance(code, str) and "Ollama" in code, code
assert not (OUT / "glm_deskew" / "t_down" / "raw").exists()
print(f"✅ Ollama ไม่เปิด -> หยุด: {code.splitlines()[-1][:70]}")

# 10. Gemini 429 -> หยุดหลังคำขอแรก
code = run(["--mode", "deskew", "--run-id", "g1", "--yes", "--delay", "0"])
assert calls["gemini"] == 1, calls
graw = list((OUT / "deskew" / "g1" / "raw").glob("*.json"))
assert len(graw) == 1, graw
gm = json.loads(graw[0].read_text(encoding="utf-8"))
assert gm["status"] == "api_error" and gm["quota_exhausted"] and gm["attempts"] == 1
print(f"✅ Gemini 429 -> เรียก 1 ครั้งแล้วหยุด ไม่ retry ({len(sets) - 1} ชุดไม่ถูกเรียก)")

# 11. Gemini 503 ติดกัน 2 ชุด -> หยุด (ชุดที่ 3 ไม่ถูกเรียก)
def unavailable(**kw):
    calls["gemini"] += 1
    raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand")


calls["gemini"] = 0
with mock.patch.object(FakeModels, "generate_content", lambda self, **kw: unavailable(**kw)):
    run(["--mode", "deskew", "--run-id", "g503", "--yes", "--delay", "0", "--retries", "0"])
assert calls["gemini"] == 2, calls
assert len(list((OUT / "deskew" / "g503" / "raw").glob("*.json"))) == 2
print("✅ Gemini 503 ติดกัน 2 ชุด -> หยุด ไม่เรียกชุดที่เหลือ")

# 12. compare.md รวม run ของทั้งสอง engine
assert (OUT / "compare.md").exists()
print("✅ compare.md:", [l.split("|")[1].strip() for l in (OUT / "compare.md").read_text(encoding="utf-8").splitlines() if l.startswith("| ") and "/" in l.split("|")[1]])

shutil.rmtree(OUT)
print("\nผ่านทั้งหมด")
