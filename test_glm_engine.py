"""
เทสต์ GlmEngine (ocr_engines/glm_engine.py) — ไม่ต้องเปิด Ollama: mock httpx.post ด้วย response จริงที่บันทึกไว้
(test_data/glm_ollama_test5.json = glm-ocr:latest อ่าน bin/test5.jpg ผ่าน pipeline จริง)

ครอบคลุม:
  1. prompt ชุด GLM ล็อกข้อความ + payload: ไม่มี response_format, temperature 0, logprobs, top_logprobs 3, max_tokens 256
  2. confidence = min prob ของ token ที่เป็น "ค่า" (ไม่รวมป้าย/หน่วย), ° แยก byte ข้าม token ยังแมปได้,
     แมปไม่ได้ / ไม่มี logprob -> None (ไม่เดา)
  3. response จริง test5 -> EngineResult: ค่าตรง spike, correct_v/line_v = null, confidence ราย 8 ฟิลด์, MeterReading float
  4. หลายภาพ: ภาพ analog ใช้ prompt analog, ค่าต่างกันระหว่างภาพ -> null, analog อ่านไม่ได้ -> null สุภาพ
  5. Ollama ไม่ได้รัน -> error บอกให้เปิด ollama serve; โมเดลไม่มี -> บอก ollama pull
  6. ผ่าน /api/ocr (OCR_ENGINE=glm): Ollama ปิด -> 500 ข้อความอ่านรู้เรื่อง แล้ว server ยังตอบคำขอถัดไปได้;
     Ollama ตอบ -> 200 + บันทึกได้ + readings.engine = "glm"

รัน: python test_glm_engine.py
"""
import copy
import json
import math
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

TMP = tempfile.mkdtemp(prefix="glm_engine_test_")
os.environ["METER_DB_PATH"] = os.path.join(TMP, "test.db")
os.environ["OCR_REQUEST_DIR"] = os.path.join(TMP, "ocr_requests")
os.environ["OCR_ENGINE"] = "glm"

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

import db  # noqa: E402
import server  # noqa: E402
from ocr_engines import glm_engine, prompts  # noqa: E402

FIXTURE = os.path.join("test_data", "glm_ollama_test5.json")
TEST5 = os.path.join("bin", "test5.jpg")
with open(FIXTURE, encoding="utf-8") as f:
    TEST5_RESPONSE = json.load(f)
CONTENT = TEST5_RESPONSE["choices"][0]["message"]["content"]
TOKENS = TEST5_RESPONSE["choices"][0]["logprobs"]["content"]

BLANK = Image.new("RGB", (64, 32), "white")


def fake_post(*responses):
    """httpx.post ปลอม: คืน response ตามลำดับ + เก็บ payload ที่ส่งไว้ตรวจ"""
    sent = []
    queue = list(responses)

    def _post(url, json=None, timeout=None):
        sent.append(json)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        r = MagicMock()
        r.status_code = item.get("_status", 200)
        r.json.return_value = item
        r.text = str(item)
        return r
    return _post, sent


def chat_response(content, tokens=None):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop",
                         "logprobs": {"content": tokens or []}}]}


def char_level_confidence(content, tokens, value):
    """คำนวณแยกอีกวิธี (ตำแหน่งตัวอักษรของ token string) ไว้เทียบกับ engine (ตำแหน่ง byte)"""
    starts, pos = [], 0
    for t in tokens:
        starts.append((pos, pos + len(t["token"]), math.exp(t["logprob"])))
        pos += len(t["token"])
    probs, i = [], content.find(value)
    while i != -1:
        probs += [p for s, e, p in starts if s < i + len(value) and e > i]
        i = content.find(value, i + 1)
    return min(probs)


def test_prompt_and_payload():
    assert prompts.GLM_EVC_PROMPT == (
        "Read the text on this gas meter display.\n"
        "Copy every label, number, decimal point and unit exactly as shown, one item per line.\n"
        "If something is not visible, leave it out. Do not guess.")
    assert prompts.GLM_ANALOG_PROMPT == (
        "Read the counter digits on this gas meter from left to right, "
        "including red or highlighted digits.\n"
        "Write only the digits, with no spaces and no decimal point.\n"
        "If the digits cannot be read, write nothing. Do not guess.")
    # กัน prompt รั่ว: โมเดลถอดข้อความจึงลอก prompt ออกมาได้ -> prompt ต้องไม่มีอะไรที่กลายเป็นค่าได้
    from ocr_engines import glm_labels
    for prompt in (prompts.GLM_EVC_PROMPT, prompts.GLM_ANALOG_PROMPT):
        assert not any(ch.isdigit() for ch in prompt), "ห้ามมีตัวเลข/ตัวอย่างค่าจริงใน prompt"
        assert "=" not in prompt and ":" not in prompt, "ห้ามมีตัวคั่นแบบป้าย"
        for word in ("LABEL", "VALUE", "DIGITS", "UNIT"):
            assert word not in prompt, f"ห้ามมี placeholder '{word}'"
        for fields, bare in ((None, None), (glm_labels.ANALOG_FIELDS, "turbine_v_m3")):
            leaked = glm_labels.parse_labeled_text(prompt + "\n" + prompt, fields, bare_number_field=bare)
            assert leaked.values == [], "ลอก prompt ทั้งย่อหน้าแล้วต้องไม่เกิดค่าใดๆ"
    assert prompts.GLM_EVC_PROMPT not in prompts.SYSTEM_INSTRUCTION, "ต้องเป็นชุดแยก ไม่ใช่ข้อความของ Gemini"
    assert len(prompts.GLM_EVC_PROMPT) < len(prompts.SYSTEM_INSTRUCTION) / 5, "prompt GLM ต้องสั้น"

    post, sent = fake_post(chat_response(""))
    with patch.object(glm_engine.httpx, "post", post):
        server.engine.read([BLANK], ["digital_screen"])
    p = sent[0]
    assert "response_format" not in p, "ห้ามบังคับ json_schema/json_object"
    assert (p["temperature"], p["logprobs"], p["top_logprobs"], p["max_tokens"]) == (0, True, 3, 256), p
    assert p["model"] == "glm-ocr:latest" and p["messages"][0]["content"][0]["text"] == prompts.GLM_EVC_PROMPT
    assert p["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    print("✅ 1. prompt GLM ล็อกข้อความ (สั้น แยกจาก Gemini) + payload ไม่มี response_format, max_tokens 256")


def test_confidence_rules():
    # P = 9.656 bar — token ของค่า: "9" ".6" "56" (prob 0.9, 0.5, 0.8) ป้าย/หน่วยมี prob ต่ำแต่ต้องไม่ถูกนับ
    content = "P = 9.656 bar"
    toks = [{"token": "P", "logprob": math.log(0.1)}, {"token": " = ", "logprob": math.log(0.1)},
            {"token": "9", "logprob": math.log(0.9)}, {"token": ".6", "logprob": math.log(0.5)},
            {"token": "56", "logprob": math.log(0.8)}, {"token": " bar", "logprob": math.log(0.01)}]
    assert abs(glm_engine.value_confidence(content, toks, [(4, 9)]) - 0.5) < 1e-9, "ต้องเป็น min ของ token ค่า"
    # token คร่อมค่ากับหน่วย ("6 b") ถูกนับ เพราะมีหลักของค่าอยู่ในนั้น
    toks2 = toks[:4] + [{"token": "56 b", "logprob": math.log(0.3)}, {"token": "ar", "logprob": 0.0}]
    assert abs(glm_engine.value_confidence(content, toks2, [(4, 9)]) - 0.3) < 1e-9
    # ° (2 byte) แยกข้าม token: string ต่อกันไม่ตรง แต่ bytes ตรง -> ยังแมปได้
    content = "T = 25.00 °C"
    toks3 = [{"token": "T = ", "logprob": 0.0, "bytes": list(b"T = ")},
             {"token": "25.00", "logprob": math.log(0.7), "bytes": list(b"25.00")},
             {"token": " �", "logprob": 0.0, "bytes": [32, 194]},
             {"token": "�C", "logprob": 0.0, "bytes": [176, 67]}]
    assert abs(glm_engine.value_confidence(content, toks3, [(4, 9)]) - 0.7) < 1e-9
    # แมปไม่ได้ -> None (ไม่เดา)
    assert glm_engine.value_confidence(content, [], [(4, 9)]) is None, "ไม่มี logprobs"
    bad = copy.deepcopy(toks3)
    bad[1]["bytes"] = list(b"25.01")
    assert glm_engine.value_confidence(content, bad, [(4, 9)]) is None, "token ไม่ตรงข้อความ"
    nolp = copy.deepcopy(toks3)
    del nolp[1]["logprob"]
    assert glm_engine.value_confidence(content, nolp, [(4, 9)]) is None, "token ของค่าไม่มี logprob"
    assert glm_engine.value_confidence(content, toks3, [(4, 4)]) is None, "span ว่าง"
    # max_tokens ตัดกลาง "³" (2 byte): content ไม่มี byte ที่ไม่ครบ แต่ token สุดท้ายยังมี -> ยังแมปได้ (เคสจริง S01_p2)
    content = "Qm = 477.5 m"
    cut = [{"token": "Qm = ", "logprob": 0.0, "bytes": list(b"Qm = ")},
           {"token": "477.5", "logprob": math.log(0.6), "bytes": list(b"477.5")},
           {"token": " m", "logprob": 0.0, "bytes": list(b" m")},
           {"token": "�", "logprob": 0.0, "bytes": [194]}]
    assert abs(glm_engine.value_confidence(content, cut, [(5, 10)]) - 0.6) < 1e-9
    # Ollama ตัด whitespace ท้าย content ทิ้ง แต่ token ยังมี -> ยังแมปได้ (เคสจริง S01_p2 / hard_1)
    for ws in ([32], [10], [32, 10, 10]):
        trailing = copy.deepcopy(cut)
        trailing[-1]["bytes"] = ws
        assert abs(glm_engine.value_confidence(content, trailing, [(5, 10)]) - 0.6) < 1e-9, ws
    # ส่วนเกินเป็นตัวอักษรครบ / ยาวเกิน -> ไม่ใช่กรณีถูกตัด -> None
    for extra in ([88], [194, 179, 88, 89]):
        bad = copy.deepcopy(cut)
        bad[-1]["bytes"] = extra
        assert glm_engine.value_confidence(content, bad, [(5, 10)]) is None, extra
    print("✅ 2. confidence = min ของ token ค่า (ไม่นับป้าย/หน่วย), ° แยก byte ได้, แมปไม่ได้ -> None")


def test_real_test5_response():
    post, _ = fake_post(TEST5_RESPONSE)
    with patch.object(glm_engine.httpx, "post", post):
        res = server.engine.read([BLANK], ["digital_screen"])
    assert res.engine == "glm" and res.raw_source == "text" and res.raw_text == CONTENT
    assert res.raw_values == {"correct_v_scm": None, "line_v_m3": None, "pressure_bar": "9.656",
                              "temperature_c": "25.00", "correction_factor": "9.3648", "qm_m3_h": "477.0",
                              "qb_sm3_h": "4467.4", "turbine_v_m3": None}, res.raw_values
    assert set(res.confidence_scores) == set(res.raw_values), "confidence ต้องครบทุกฟิลด์"
    for key, value in res.raw_values.items():
        conf = res.confidence_scores[key]
        if value is None:
            assert conf is None, f"{key} ไม่มีค่า -> confidence ต้อง None"
        else:
            assert conf is not None and 0 < conf <= 1, (key, conf)
    # pressure เทียบกับการคำนวณอีกวิธี (ทุกครั้งที่ "9.656" ปรากฏ)
    assert abs(res.confidence_scores["pressure_bar"] - char_level_confidence(CONTENT, TOKENS, "9.656")) < 1e-12
    r = res.reading
    assert (r.pressure_bar, r.temperature_c, r.correction_factor, r.qm_m3_h, r.qb_sm3_h) == \
        (9.656, 25.0, 9.3648, 477.0, 4467.4)
    assert r.correct_v_scm is None and r.line_v_m3 is None and r.low_confidence_fields == []
    assert res.meter_type == "digital" and res.detected_format == "digital screen x1"
    assert "glm" not in res.detected_format.lower(), "detected_format ขึ้นบนหน้า OCR — ห้ามมีชื่อ engine"
    print("✅ 3. response จริง test5: 5 ค่าตรง spike, correct_v/line_v = null, confidence ครบ 8 ฟิลด์")
    print("   confidence:", {k: (round(v, 4) if v is not None else None) for k, v in res.confidence_scores.items()})


def test_multi_image():
    evc = chat_response("Converted Index = 00123.5 m3\nP = 9.656 bar\n")
    analog = chat_response("Counter = 01234567\n")
    post, sent = fake_post(evc, analog)
    with patch.object(glm_engine.httpx, "post", post):
        res = server.engine.read([BLANK, BLANK], ["digital_screen", "analog_dial"])
    assert [s["messages"][0]["content"][0]["text"] for s in sent] == \
        [prompts.GLM_EVC_PROMPT, prompts.GLM_ANALOG_PROMPT], "ภาพ analog ต้องใช้ prompt analog"
    assert res.raw_values["correct_v_scm"] == "123.5" and res.raw_values["turbine_v_m3"] == "1234567"
    assert res.confidence_scores["turbine_v_m3"] is None, "ไม่มี logprobs -> None"
    assert res.detected_format == "digital screen x1, analog dial x1"

    post, _ = fake_post(chat_response("P = 9.656 bar\n"), chat_response("P = 9.665 bar\n"))
    with patch.object(glm_engine.httpx, "post", post):
        res = server.engine.read([BLANK, BLANK], ["digital_screen", "digital_screen"])
    assert res.raw_values["pressure_bar"] is None, "ค่าต่างกันระหว่างภาพต้องเป็น null ไม่เลือกให้"

    # ถูก max_tokens ตัดกลางบรรทัดสุดท้าย ("P = 9.") -> ทิ้งบรรทัดนั้น ไม่ให้ทำค่าที่ถูกข้างบนกลายเป็น null
    # (เคสจริง eval_images/S01_p2.jpg)
    cut = chat_response("P = 9.655 bar\nT = 25.01 °C\nP = 9.")
    cut["choices"][0]["finish_reason"] = "length"
    post, _ = fake_post(cut)
    with patch.object(glm_engine.httpx, "post", post):
        res = server.engine.read([BLANK], ["digital_screen"])
    assert res.raw_values["pressure_bar"] == "9.655" and res.raw_values["temperature_c"] == "25.01", res.raw_values
    assert glm_engine.complete_lines("P = 9.655\nP = 9.", "stop") == "P = 9.655\nP = 9.", "จบปกติต้องไม่ตัด"
    assert glm_engine.complete_lines("P = 9.", "length") == ""

    # analog: GLM คืนตัวเลขเดี่ยว (ไม่มีป้าย) -> turbine; ตัวเลขหลายชุด -> null (เคสจริง S13 / S12)
    post, _ = fake_post(chat_response("16897967m³\n\n16897967m³\n"))
    with patch.object(glm_engine.httpx, "post", post):
        res = server.engine.read([BLANK], ["analog_dial"])
    assert res.raw_values["turbine_v_m3"] == "16897967", res.raw_values
    post, _ = fake_post(chat_response("6876729\n\n1689796\n"))
    with patch.object(glm_engine.httpx, "post", post):
        res = server.engine.read([BLANK], ["analog_dial"])
    assert res.raw_values["turbine_v_m3"] is None, "ตัวเลขเดี่ยวหลายชุด -> กำกวม -> null"
    # ภาพจอ EVC คืนตัวเลขเดี่ยว -> ไม่ลงฟิลด์ใด (ไม่จับคู่ด้วยตำแหน่ง)
    post, _ = fake_post(chat_response("67203344\n"))
    with patch.object(glm_engine.httpx, "post", post):
        res = server.engine.read([BLANK], ["digital_screen"])
    assert all(v is None for v in res.raw_values.values())

    # analog อ่านไม่ได้ / โมเดลลอก template -> null อย่างสุภาพ ไม่ error
    post, _ = fake_post(chat_response("Counter = DIGITS\nThe image shows a meter.\n"))
    with patch.object(glm_engine.httpx, "post", post):
        res = server.engine.read([BLANK], ["analog_dial"])
    assert all(v is None for v in res.raw_values.values()) and res.meter_type == "analog"
    assert glm_engine.ANALOG_CLASS == server.ANALOG_CLASS and glm_engine.DIGITAL_CLASS == server.DIGITAL_CLASS
    print("✅ 4. หลายภาพ: prompt ตามชนิดภาพ, ค่าต่างกัน -> null, analog อ่านไม่ได้ -> null ไม่ error")


def test_ollama_errors():
    post, _ = fake_post(httpx.ConnectError("refused"))
    with patch.object(glm_engine.httpx, "post", post):
        try:
            server.engine.read([BLANK], ["digital_screen"])
        except glm_engine.OllamaUnavailableError as e:
            assert "ollama serve" in str(e), str(e)
        else:
            raise AssertionError("Ollama ปิดต้อง raise OllamaUnavailableError")
    post, _ = fake_post({"_status": 404, "error": {"message": 'model "glm-ocr:latest" not found'}})
    with patch.object(glm_engine.httpx, "post", post):
        try:
            server.engine.read([BLANK], ["digital_screen"])
        except RuntimeError as e:
            assert "ollama pull glm-ocr:latest" in str(e), str(e)
        else:
            raise AssertionError("ไม่มีโมเดลต้อง raise")
    print("✅ 5. Ollama ปิด -> บอกให้เปิด ollama serve, ไม่มีโมเดล -> บอก ollama pull")


def test_endpoint(client):
    assert server.engine.name == "glm"
    with open(TEST5, "rb") as f:
        files = [("files", ("test5.jpg", f.read(), "image/jpeg"))]

    post, _ = fake_post(httpx.ConnectError("refused"))
    with patch.object(glm_engine.httpx, "post", post):
        r = client.post("/api/ocr", files=files)
    assert r.status_code == 500 and "ollama serve" in r.json()["detail"], r.text
    assert client.get("/api/reading-fields").status_code == 200, "server ต้องไม่ล่ม"

    post, _ = fake_post(TEST5_RESPONSE)
    with patch.object(glm_engine.httpx, "post", post):
        r = client.post("/api/ocr", files=files)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["raw_values"]["pressure_bar"] == "9.656" and data["editor"]["values"]["temperature_c"] == "25.00"
    assert data["raw_values"]["correct_v_scm"] is None and data["raw_source"] == "text"
    assert "glm" not in json.dumps(data).lower(), "response ที่หน้า OCR เห็นต้องไม่มีชื่อ engine"

    run = client.post("/api/meter-runs", json={"company": "glm", "run_label": "a"}).json()
    saved = client.post("/api/readings", json={
        "meter_run_id": run["id"], "request_id": data["request_id"], "recorded_at": "2026-10-08T10:00",
        "entered_values": data["editor"]["values"], "turbine_trim_digits": 0})
    assert saved.status_code == 201, saved.text
    row = db.get_reading(saved.json()["reading"]["id"])
    assert row["engine"] == "glm"
    # confidence: เก็บลง DB อย่างเดียว ไม่ส่งให้หน้าเว็บ (ไม่สัมพันธ์กับความถูกต้อง — ดู PROJECT_CONTEXT)
    conf = row["confidence_scores"]
    assert isinstance(conf, dict) and 0 < conf["pressure_bar"] <= 1 and conf["correct_v_scm"] is None, conf
    assert "confidence_scores" not in json.dumps(data), "response ของ /api/ocr ต้องไม่มี confidence_scores"
    assert row["low_confidence_fields"] == [], "GLM ไม่ติดป้าย 'ตรวจสอบ'"
    print("✅ 6. /api/ocr: Ollama ปิด -> 500 อ่านรู้เรื่อง + server ยังตอบได้; ปกติ -> 200, บันทึก engine = 'glm'")


def main():
    client = TestClient(server.app)
    tests = [test_prompt_and_payload, test_confidence_rules, test_real_test5_response, test_multi_image,
             test_ollama_errors, lambda: test_endpoint(client)]
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"❌ {getattr(fn, '__name__', 'test')}: {e}")
    print("─" * 60)
    if failed:
        print(f"❌ FAIL — {failed} เทสต์ไม่ผ่าน")
        sys.exit(1)
    print("🎉 ผ่านทั้งหมด")


if __name__ == "__main__":
    main()
