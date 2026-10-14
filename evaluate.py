# -*- coding: utf-8 -*-
"""
evaluate.py — วัดความแม่นยำการอ่านค่าของระบบ Gas Meter OCR
=============================================================
เรียก pipeline จริงจาก server.py โดยตรง (ไม่ผ่าน HTTP และไม่แก้ server.py)
ลำดับเดียวกับ perform_ocr():
  exif_transpose + RGB -> preprocess_meter_image() -> check_request_meter_types()
  -> build_gemini_prompt() -> client.models.generate_content() -> parse_gemini_response()

ข้อมูลเข้า
  eval_images/        ภาพทดสอบ
  ground_truth.xlsx   ชีต ground_truth (แถว 1 = ชื่อคอลัมน์, แถว 2 = คำอธิบาย)

คำสั่ง
  python evaluate.py --check                                    ตรวจ ground truth อย่างเดียว
  python evaluate.py --check --mode deskew --run-id r1          ตรวจ + ดูสถานะ raw ของ run
  python evaluate.py --mode deskew --run-id r1                  รัน (ถามยืนยันก่อนเรียก Gemini)
  python evaluate.py --mode crop --run-id r1 --yes              รันโดยไม่ถาม
  python evaluate.py --mode deskew --run-id r1 --retry-errors   รันซ้ำเฉพาะชุด api_error
  python evaluate.py --mode deskew --run-id r1 --force          รันใหม่ทุกชุด (ทับ raw เดิม)
  python evaluate.py --mode deskew --run-id r1 --rescore        คำนวณผลใหม่จาก raw (ไม่เรียก Gemini)
  python evaluate.py --compare                                  สร้าง eval_output/compare.md ใหม่

  --mode deskew = server.SEND_DESKEWED_TO_GEMINI = True  (ส่งภาพที่ดัดตรงแล้ว)
  --mode crop   = server.SEND_DESKEWED_TO_GEMINI = False (ส่ง crop แนวแกน ไม่ใช่ภาพต้นฉบับ)

  --engine glm  = อ่านด้วย GLM-OCR ผ่าน Ollama บนเครื่อง (ไม่มีโควต้า ไม่ถามยืนยัน)
    python evaluate.py --engine glm --mode deskew --run-id glm19
    pipeline ภาพเดียวกัน (OBB -> deskew -> detailEnhance) แล้วเรียก GlmEngine.read ทั้งชุดในครั้งเดียว
    เหมือน /api/ocr; ผลอยู่ที่ eval_output/glm_<mode>/<run_id>/
    ให้คะแนนเหมือน Gemini ทุกฟิลด์ ยกเว้น turbine_v_m3 ใช้ "ตัวเลขตรง" (ตัดจุดและ 0 นำหน้าแล้วเทียบ)
    เพราะ prompt ของ GLM สั่งให้อ่านลูกกลิ้งโดยไม่ใส่จุดทศนิยม (การวางทศนิยมเป็นหน้าที่ checkbox ของผู้ใช้)

  Gemini: เจอ 429 (โควต้าหมด) -> ไม่ retry และหยุดทั้งรอบทันที (retry มีแต่เผาโควต้า — PROJECT_CONTEXT ข้อ 13)
          ชุดที่เหลือรันวันถัดไปด้วย --retry-errors

ผลลัพธ์ eval_output/<mode>/<run_id>/ (Gemini) หรือ eval_output/glm_<mode>/<run_id>/ (GLM)
  raw/<set_id>.json            สถานะการเรียก, OBB class, error และ response.text
  raw/<set_id>.response.txt    response.text ดิบจาก Gemini
  field_results.csv, set_results.csv, summary.md
และ eval_output/compare.md เมื่อมีผลตั้งแต่ 2 run ขึ้นไป
"""
import argparse
import csv
import io
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd

# ทำงานจากโฟลเดอร์ของสคริปต์เสมอ (server.py อ้าง best_obb.pt แบบ relative path)
os.chdir(Path(__file__).resolve().parent)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

# ── CONFIG ────────────────────────────────────────────────────────────
GT_XLSX = Path("ground_truth.xlsx")
GT_SHEET = "ground_truth"
IMG_DIR = Path("eval_images")
OUT_ROOT = Path("eval_output")

FIELDS = [
    "correct_v_scm", "line_v_m3", "pressure_bar", "temperature_c",
    "correction_factor", "turbine_v_m3", "qm_m3_h", "qb_sm3_h", "pulse_weight",
]
MODES = {"deskew": True, "crop": False}   # ค่าที่ตั้งให้ server.SEND_DESKEWED_TO_GEMINI
ENGINES = ["gemini", "glm"]
TURBINE_FIELD = "turbine_v_m3"
SMALL_N = 5                               # จำนวนน้อยกว่านี้ แสดงเป็นจำนวนพร้อมหมายเหตุ
QUOTA_CODE = 429                          # โควต้าหมด -> หยุดทั้งรอบ ไม่ retry
RETRYABLE_CODES = {500, 502, 503, 504}
NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")

# สถานะรายชุด
OK, BLOCKED, API_ERROR, NOT_RUN = "ok", "blocked_mixed_types", "api_error", "not_run"
# outcome รายฟิลด์
CORRECT, MISREAD, HALLUCINATED, MISSED, SKIPPED = (
    "correct", "misread", "hallucinated", "missed", "skipped")
EVALUATED = (CORRECT, MISREAD, HALLUCINATED, MISSED)

FIELD_COLUMNS = ["set_id", "group", "n_images", "field", "truth", "predicted", "outcome",
                 "scoring", "decimal_shift", "swapped_field", "low_confidence"]
SET_COLUMNS = ["set_id", "group", "n_images", "meter_type_truth", "meter_type_pred",
               "detected_format", "decimal_slots_truth", "obb_classes", "n_correct",
               "n_evaluated", "status", "meter_type_correct", "exact_match", "elapsed_s", "error"]


# ── GROUND TRUTH ──────────────────────────────────────────────────────
def _cell(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return str(v).strip()   # strip() ตัดทั้งช่องว่าง tab และขึ้นบรรทัด


def classify_truth(value):
    """ค่าจริง 1 ช่อง -> NUM / NA / ? / BLANK / BAD"""
    if value == "":
        return "BLANK"
    if value.lower() == "na":
        return "NA"
    if value == "?":
        return "?"
    if NUM_RE.match(value):
        return "NUM"
    return "BAD"


def safe_name(set_id):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", set_id)


def image_files():
    if not IMG_DIR.is_dir():
        return []
    return sorted(p for p in IMG_DIR.iterdir() if p.is_file() and not p.name.startswith("."))


def load_ground_truth():
    """อ่านและตรวจ ground_truth.xlsx -> (sets, problems, warnings)
    problems = ต้องหยุด ห้ามรัน, warnings = ข้อสังเกตที่ไม่ต้องหยุด"""
    problems, warnings = [], []
    if not GT_XLSX.exists():
        return [], [f"ไม่พบไฟล์ {GT_XLSX}"], warnings
    if not IMG_DIR.is_dir():
        return [], [f"ไม่พบโฟลเดอร์ {IMG_DIR}/"], warnings
    try:
        raw = pd.read_excel(GT_XLSX, sheet_name=GT_SHEET, header=None,
                            dtype=str, keep_default_na=False)
    except ValueError as e:
        return [], [f"อ่านชีต '{GT_SHEET}' ไม่ได้: {e}"], warnings

    header = [_cell(v) for v in raw.iloc[0]]
    col = {}
    for i, name in enumerate(header):
        if name and name not in col:
            col[name] = i
    required = ["set_id", "images", "group", "meter_type", "decimal_slots"] + FIELDS
    missing = [c for c in required if c not in col]
    if missing:
        return [], [f"ไม่พบคอลัมน์: {', '.join(missing)}"], warnings

    files = image_files()
    by_stem = defaultdict(list)
    for p in files:
        by_stem[p.stem.strip().lower()].append(p)

    sets, used = [], defaultdict(list)
    for i in range(2, len(raw)):          # ข้ามแถว 1 (ชื่อคอลัมน์) และแถว 2 (คำอธิบาย)
        excel_row = i + 1
        r = {name: _cell(raw.iat[i, j]) for name, j in col.items()}
        if all(v == "" for v in r.values()):
            continue
        set_id = r["set_id"]
        if set_id.upper() == "EXAMPLE":
            continue
        where = f"{set_id or '(ไม่มี set_id)'} (แถว {excel_row})"
        if not set_id:
            problems.append(f"{where}: set_id ว่าง")
            continue

        names = [n.strip() for n in r["images"].split(";") if n.strip()]
        if not names:
            problems.append(f"{where}: ช่อง images ว่าง")
        paths = []
        for n in names:
            cands = by_stem.get(Path(n).stem.strip().lower(), [])
            if not cands:
                problems.append(f"{where}: ไม่พบไฟล์ภาพสำหรับ '{n}' ใน {IMG_DIR}/")
            elif len(cands) > 1:
                problems.append(f"{where}: '{n}' ตรงกับหลายไฟล์: "
                                + ", ".join(p.name for p in cands))
            else:
                paths.append(cands[0])
                used[cands[0].name].append(set_id)

        meter_type = r["meter_type"].lower()
        if meter_type == "":
            problems.append(f"{where}: meter_type ว่าง")
        elif meter_type not in ("digital", "analog", "?"):
            problems.append(f"{where}: meter_type ต้องเป็น digital / analog / ? "
                            f"แต่ได้ '{r['meter_type']}'")

        truth = {}
        for f in FIELDS:
            kind = classify_truth(r[f])
            if kind == "BLANK":
                problems.append(f"{where}: ฟิลด์ {f} ว่าง")
            elif kind == "BAD":
                problems.append(f"{where}: ฟิลด์ {f} ต้องเป็นตัวเลข / NA / ? แต่ได้ '{r[f]}'")
            truth[f] = (r[f], kind)

        if not r["group"]:
            warnings.append(f"{where}: group ว่าง (จะแสดงเป็น '(ว่าง)' ในรายงาน)")

        sets.append({
            "set_id": set_id, "excel_row": excel_row, "group": r["group"] or "(ว่าง)",
            "meter_type": meter_type, "decimal_slots": r["decimal_slots"],
            "images": names, "paths": paths, "truth": truth,
        })

    for sid, n in Counter(s["set_id"] for s in sets).items():
        if n > 1:
            problems.append(f"set_id ซ้ำ {n} แถว: {sid}")
    collide = defaultdict(set)
    for s in sets:
        collide[safe_name(s["set_id"])].add(s["set_id"])
    for key, ids in collide.items():
        if len(ids) > 1:
            problems.append(f"set_id {sorted(ids)} ใช้ชื่อไฟล์ raw ชนกัน ({key})")
    for name, ids in used.items():
        if len(ids) > 1:
            warnings.append(f"ไฟล์ {name} ถูกใช้ในหลายชุด: {', '.join(ids)}")
    unused = [p.name for p in files if p.name not in used]
    if unused:
        warnings.append(f"ไฟล์ใน {IMG_DIR}/ ที่ไม่ถูกอ้างถึง: {', '.join(unused)}")
    if not sets and not problems:
        problems.append("ไม่พบชุดภาพใน ground truth")
    return sets, problems, warnings


def print_check_report(sets, problems, warnings):
    files = image_files()
    single = sum(1 for s in sets if len(s["images"]) == 1)
    print("=" * 72)
    print(f"ตรวจ ground truth: {GT_XLSX} (ชีต {GT_SHEET})")
    print("=" * 72)
    print(f"จำนวนชุด {len(sets)} | จำนวนภาพที่อ้างถึง {sum(len(s['images']) for s in sets)}"
          f" | ไฟล์ใน {IMG_DIR}/ {len(files)}")
    print(f"ชุดภาพเดียว {single} | ชุดหลายภาพ {len(sets) - single}")
    print(f"meter_type: {dict(sorted(Counter(s['meter_type'] for s in sets).items()))}")
    print(f"group: {dict(sorted(Counter(s['group'] for s in sets).items()))}")

    print("\nการจับคู่ชื่อไฟล์:")
    print(f"  {'set_id':<9}{'group':<11}{'type':<9}{'slots':<7}ชื่อใน ground truth -> ไฟล์")
    for s in sets:
        if len(s["paths"]) == len(s["images"]):
            pairs = ", ".join(f"{n} -> {p.name}" for n, p in zip(s["images"], s["paths"]))
        else:
            pairs = f"{'; '.join(s['images'])} -> จับคู่ไม่ครบ"
        print(f"  {s['set_id']:<9}{s['group']:<11}{s['meter_type']:<9}"
              f"{s['decimal_slots']:<7}{pairs}")

    print("\nค่าจริงรายฟิลด์:")
    print(f"  {'field':<20}{'ตัวเลข':>8}{'NA':>6}{'?':>5}")
    total = Counter()
    for f in FIELDS:
        c = Counter(s["truth"][f][1] for s in sets)
        total.update(c)
        print(f"  {f:<20}{c['NUM']:>8}{c['NA']:>6}{c['?']:>5}")
    print(f"  {'รวม':<20}{total['NUM']:>8}{total['NA']:>6}{total['?']:>5}")
    q = [f"{s['set_id']}.{f}" for s in sets for f in FIELDS if s["truth"][f][1] == "?"]
    if q:
        print(f"  ช่องที่เป็น ? (ไม่นับ): {', '.join(q)}")
    mq = [s["set_id"] for s in sets if s["meter_type"] == "?"]
    if mq:
        print(f"  meter_type เป็น ? (ไม่นับ): {', '.join(mq)}")

    if warnings:
        print(f"\nข้อสังเกต (ไม่ต้องหยุด): {len(warnings)}")
        for w in warnings:
            print(f"  - {w}")
    print(f"\nปัญหาที่ต้องแก้ก่อนรัน: {len(problems)}")
    for p in problems:
        print(f"  - {p}")


# ── RAW STORAGE ───────────────────────────────────────────────────────
def run_dir(mode, run_id, engine="gemini"):
    # Gemini คงโครงเดิม eval_output/<mode>/<run_id> ; engine อื่นแยกโฟลเดอร์ไม่ให้ raw ปนกัน
    return OUT_ROOT / (mode if engine == "gemini" else f"{engine}_{mode}") / run_id


def meta_path(rdir, set_id):
    return rdir / "raw" / f"{safe_name(set_id)}.json"


def read_meta(rdir, set_id):
    p = meta_path(rdir, set_id)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_meta(rdir, meta):
    (rdir / "raw").mkdir(parents=True, exist_ok=True)
    with open(meta_path(rdir, meta["set_id"]), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    txt = rdir / "raw" / f"{safe_name(meta['set_id'])}.response.txt"
    if meta["response_text"] is not None:
        txt.write_text(meta["response_text"], encoding="utf-8")
    elif txt.exists():
        txt.unlink()   # ไฟล์ที่สคริปต์นี้สร้างเองจากรอบก่อน — ไม่ให้ค้างไม่ตรงกับ meta


def print_run_status(sets, rdir):
    counts = Counter()
    for s in sets:
        meta = read_meta(rdir, s["set_id"])
        counts[meta["status"] if meta else NOT_RUN] += 1
    print(f"\nสถานะ raw ใน {rdir}/: ok {counts[OK]} | blocked_mixed_types {counts[BLOCKED]}"
          f" | api_error {counts[API_ERROR]} | ยังไม่ได้รัน {counts[NOT_RUN]}")


# ── PIPELINE CALL (เหมือน perform_ocr) ─────────────────────────────────
def error_code(exc):
    for attr in ("code", "status_code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    m = re.search(r"\b(429|500|502|503|504)\b", str(exc))
    return int(m.group(1)) if m else None


def process_set(server, s, mode, run_id, retries, delay, engine="gemini", glm=None):
    """เตรียมภาพชุดเดียวด้วย pipeline เดียวกับ /api/ocr แล้วเรียก engine 1 ครั้งต่อชุด
    glm = GlmEngine (เมื่อ engine == "glm") — เก็บผลเป็น JSON รูปเดียวกับของ Gemini ให้ score_run ใช้ร่วม"""
    from PIL import Image, ImageOps

    meta = {
        "set_id": s["set_id"], "mode": mode, "run_id": run_id, "engine": engine,
        "send_deskewed_to_gemini": server.SEND_DESKEWED_TO_GEMINI,
        "model": glm.model if engine == "glm" else server.GEMINI_MODEL,
        "images": [p.name for p in s["paths"]],
        "obb_classes": [], "status": None, "error": None,
        "called_api": False, "attempts": 0, "response_text": None, "elapsed_s": None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if engine == "gemini":
        meta["gemini_model"] = server.GEMINI_MODEL   # ชื่อเดิม (raw รุ่นก่อนใช้ key นี้)
    request_id = f"eval_{engine}_{mode}_{run_id}_{safe_name(s['set_id'])}"

    images, classes = [], []
    try:
        for idx, p in enumerate(s["paths"]):
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(p.read_bytes())))
            if img.mode != "RGB":
                img = img.convert("RGB")
            processed, cls = server.preprocess_meter_image(img, request_id, idx, source_name=p.name)
            images.append(processed)
            classes.append(cls)
    except Exception as e:
        meta.update(status=API_ERROR, error=f"เตรียมภาพไม่สำเร็จ: {type(e).__name__}: {e}")
        return meta
    meta["obb_classes"] = classes

    if server.check_request_meter_types([p.name for p in s["paths"]], classes):
        meta["status"] = BLOCKED
        return meta

    if engine == "glm":
        return _call_glm(glm, images, classes, meta)
    return _call_gemini(server, images, classes, meta, retries, delay)


def _call_glm(glm, images, classes, meta):
    """GlmEngine.read ตัวเดียวกับระบบจริง -> เก็บ raw_values (สตริงตามที่ GLM เขียน) เป็น JSON
    Ollama ติดต่อไม่ได้ -> raise ออกไปให้ main หยุดทั้งรอบ (ทุกชุดต่อจากนี้จะพังเหมือนกัน)"""
    from ocr_engines.glm_engine import OllamaUnavailableError

    meta["called_api"] = True
    meta["attempts"] = 1
    t0 = time.time()
    try:
        result = glm.read(images, classes)
    except OllamaUnavailableError:
        raise
    except Exception as e:
        meta.update(status=API_ERROR, error=f"{type(e).__name__}: {e}",
                    elapsed_s=round(time.time() - t0, 2))
        return meta
    meta["elapsed_s"] = round(time.time() - t0, 2)
    meta["response_text"] = json.dumps({
        "meter_type": result.meter_type,
        "detected_format": result.detected_format,
        **result.raw_values,
        "low_confidence_fields": [],
    }, ensure_ascii=False, indent=2)
    meta["glm_raw_text"] = result.raw_text              # ข้อความที่ GLM ถอดจากจอ (ไว้ตรวจว่าทำไมได้/ไม่ได้ค่า)
    meta["confidence_scores"] = result.confidence_scores  # เก็บไว้วิเคราะห์ ไม่ใช้ให้คะแนน
    meta["status"] = OK
    return meta


def _call_gemini(server, images, classes, meta, retries, delay):
    contents = list(images) + [server.build_gemini_prompt(classes)]
    config = server.types.GenerateContentConfig(
        system_instruction=server.SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=server.MeterReading,
        temperature=0.1,
    )
    meta["called_api"] = True
    for attempt in range(retries + 1):
        meta["attempts"] = attempt + 1
        t0 = time.time()
        try:
            response = server.client.models.generate_content(
                model=server.GEMINI_MODEL, contents=contents, config=config)
        except Exception as e:
            code = error_code(e)
            if code == QUOTA_CODE:
                # โควต้าหมด: retry ไม่ช่วยจนกว่าจะรีเซ็ต และอาจเผาโควต้า -> ให้ main หยุดทั้งรอบ
                meta.update(status=API_ERROR, error=f"{type(e).__name__}: {e}", quota_exhausted=True)
                return meta
            if code in RETRYABLE_CODES and attempt < retries:
                wait = max(delay, 2.0) * (2 ** attempt)
                print(f"    HTTP {code} — รอ {wait:.0f} วินาทีแล้วลองใหม่ "
                      f"({attempt + 1}/{retries})", flush=True)
                time.sleep(wait)
                continue
            meta.update(status=API_ERROR, error=f"{type(e).__name__}: {e}")
            return meta
        meta["elapsed_s"] = round(time.time() - t0, 2)

        try:
            meta["response_text"] = response.text
        except Exception as e:
            meta.update(status=API_ERROR, error=f"อ่าน response.text ไม่ได้: {type(e).__name__}: {e}")
            return meta
        try:
            server.parse_gemini_response(response)   # ผ่านด่านเดียวกับระบบจริง
        except Exception as e:
            meta.update(status=API_ERROR,
                        error=f"parse_gemini_response ไม่ผ่าน: {type(e).__name__}: {e}")
            return meta
        meta["status"] = OK
        return meta

    meta.update(status=API_ERROR, error="ลองครบจำนวนครั้งแล้ว")
    return meta


# ── SCORING ───────────────────────────────────────────────────────────
def plain_number(x):
    """float -> ข้อความตัวเลขแบบไม่ใช้ exponent (12.0 -> 12)"""
    s = format(Decimal(repr(x)), "f")
    return s[:-2] if s.endswith(".0") else s


def coerce_pred(v):
    """ค่าจาก JSON -> float / None / ค่าที่ไม่ใช่ตัวเลข (คงไว้ตามเดิม)"""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        t = v.strip()
        if t == "" or t.lower() == "null":
            return None
        if NUM_RE.match(t):
            return float(t)
    return v


def display_pred(p):
    if p is None:
        return "null"
    if isinstance(p, float):
        return plain_number(p)
    return str(p)


def digit_core(s):
    return s.lstrip("+-").replace(".", "").strip("0")


def is_decimal_shift(truth_str, pred):
    """หลักตัวเลขเหมือนกันเมื่อตัดจุดทศนิยมและเลข 0 หน้า/ท้ายออก (และเครื่องหมายเหมือนกัน)"""
    if truth_str.strip().startswith("-") != (pred < 0):
        return False
    a, b = digit_core(truth_str), digit_core(plain_number(pred))
    return a != "" and a == b


def turbine_digits(s):
    """สตริงตัวเลขของลูกกลิ้ง: ตัดจุดและ 0 นำหน้า ("1529623.69" -> "152962369")
    ไม่ตัด 0 ท้าย — ลูกกลิ้งช่องแดงที่เป็น 0 ก็เป็นหลักหนึ่ง (161567.0 -> "1615670")"""
    return str(s).strip().replace(".", "").lstrip("0") or "0"


def digits_rule_applies(engine, field):
    """GLM อ่านลูกกลิ้งโดยไม่ใส่จุดทศนิยม (prompt สั่งไว้ — การวางทศนิยมเป็นหน้าที่ checkbox ของผู้ใช้)
    -> turbine ของ GLM ให้คะแนนแบบตัวเลขตรง; Gemini และฟิลด์อื่นทั้งหมดให้คะแนนแบบเข้มงวดเหมือนเดิม"""
    return engine == "glm" and field == TURBINE_FIELD


def score_run(sets, rdir):
    """อ่าน raw ของ run แล้วเทียบกับ ground truth ปัจจุบัน -> (field_rows, set_rows, notes)"""
    field_rows, set_rows, notes = [], [], []

    gt_keys = {safe_name(s["set_id"]) for s in sets}
    if (rdir / "raw").is_dir():
        for p in sorted((rdir / "raw").glob("*.json")):
            if p.stem not in gt_keys:
                notes.append(f"มีไฟล์ raw ของชุดที่ไม่อยู่ใน ground truth: {p.name} (ไม่นำมาคิด)")

    for s in sets:
        meta = read_meta(rdir, s["set_id"])
        file_names = [p.name for p in s["paths"]]
        base = {"set_id": s["set_id"], "group": s["group"], "n_images": len(s["images"])}
        hidden = {"_kind": None, "_meter_type": s["meter_type"], "_images": "; ".join(file_names),
                  "_detected_format": "", "_decimal_slots": s["decimal_slots"], "_pred_is_num": False}

        status = meta["status"] if meta else NOT_RUN
        error = (meta or {}).get("error") or ""
        engine = (meta or {}).get("engine") or "gemini"   # raw รุ่นก่อนมีแต่ Gemini
        if meta and meta.get("images") != file_names:
            notes.append(f"{s['set_id']}: ภาพใน ground truth ({', '.join(file_names)}) ไม่ตรงกับตอนรัน "
                         f"({', '.join(meta.get('images') or [])}) — ควรรันชุดนี้ใหม่ด้วย --force")

        pred = None
        if status == OK:
            try:
                pred = json.loads(meta["response_text"])
                if not isinstance(pred, dict):
                    raise ValueError("JSON ไม่ใช่ object")
            except Exception as e:
                status, error = API_ERROR, f"อ่าน response_text ใน raw ไม่ได้: {e}"

        set_row = {
            **base, "meter_type_truth": s["meter_type"], "meter_type_pred": "",
            "detected_format": "", "decimal_slots_truth": s["decimal_slots"],
            "obb_classes": ";".join(c or "none" for c in (meta or {}).get("obb_classes", [])),
            "n_correct": "", "n_evaluated": "", "status": status,
            "meter_type_correct": "", "exact_match": "",
            "elapsed_s": "" if (meta or {}).get("elapsed_s") is None else meta["elapsed_s"],
            "error": error,
        }

        if status != OK:
            for f in FIELDS:
                field_rows.append({**base, **hidden, "_kind": s["truth"][f][1], "field": f,
                                   "truth": s["truth"][f][0], "predicted": "", "outcome": status,
                                   "scoring": "", "decimal_shift": "", "swapped_field": "", "low_confidence": ""})
            set_rows.append(set_row)
            continue

        detected_format = str(pred.get("detected_format") or "")
        hidden["_detected_format"] = detected_format
        low = {str(x).strip() for x in (pred.get("low_confidence_fields") or [])}
        truth_nums = {f: float(v) for f, (v, k) in s["truth"].items() if k == "NUM"}

        n_eval = n_ok = 0
        for f in FIELDS:
            truth_str, kind = s["truth"][f]
            raw = pred.get(f)
            p = coerce_pred(raw)
            is_num = isinstance(p, float)
            decimal_shift, swapped = False, ""
            digits_rule = digits_rule_applies(engine, f)

            if kind == "?":
                outcome = SKIPPED
            elif kind == "NUM":
                if p is None:
                    outcome = MISSED
                elif digits_rule:
                    # ตัดสินด้วยตัวเลขอย่างเดียว ไม่ถอยไปเทียบ float: เฉลย 161567.0 (ช่องแดงเป็น 0)
                    # กับ GLM "161567" มีค่าเท่ากันแบบ float แต่ GLM อ่านขาดไป 1 หลัก -> ต้องนับผิด
                    outcome = (CORRECT if is_num and turbine_digits(raw) == turbine_digits(truth_str)
                               else MISREAD)
                elif is_num and p == float(truth_str):
                    outcome = CORRECT
                else:
                    outcome = MISREAD
                    decimal_shift = is_num and is_decimal_shift(truth_str, p)
            else:   # NA
                outcome = CORRECT if p is None else HALLUCINATED

            if outcome in (MISREAD, HALLUCINATED) and is_num:
                swapped = ";".join(g for g in FIELDS
                                   if g != f and g in truth_nums and truth_nums[g] == p)
            if outcome != SKIPPED:
                n_eval += 1
                n_ok += outcome == CORRECT

            field_rows.append({
                **base, **hidden, "_kind": kind, "_pred_is_num": is_num,
                "field": f, "truth": truth_str,
                # GLM เก็บค่าเป็นสตริงตามที่โมเดลเขียน -> แสดงตามนั้น (ไม่ผ่าน float)
                "predicted": raw.strip() if isinstance(raw, str) and is_num else display_pred(p),
                "outcome": outcome, "scoring": "digits" if digits_rule else "strict",
                "decimal_shift": decimal_shift, "swapped_field": swapped,
                "low_confidence": f in low,
            })

        mt_pred = str(pred.get("meter_type") or "").strip()
        set_row.update(
            meter_type_pred=mt_pred, detected_format=detected_format,
            n_correct=n_ok, n_evaluated=n_eval,
            meter_type_correct="" if s["meter_type"] == "?" else mt_pred.lower() == s["meter_type"],
            exact_match=n_eval > 0 and n_ok == n_eval,
        )
        set_rows.append(set_row)

    return field_rows, set_rows, notes


def compute_metrics(field_rows, set_rows):
    ev = [r for r in field_rows if r["outcome"] in EVALUATED]
    num = [r for r in ev if r["_kind"] == "NUM"]
    na = [r for r in ev if r["_kind"] == "NA"]
    ok_sets = [r for r in set_rows if r["status"] == OK]
    mt = [r for r in ok_sets if r["meter_type_correct"] != ""]

    def count(rows, outcome):
        return sum(1 for r in rows if r["outcome"] == outcome)

    return {
        "num_total": len(num), "num_correct": count(num, CORRECT),
        "num_misread": count(num, MISREAD), "num_missed": count(num, MISSED),
        "na_total": len(na), "na_correct": count(na, CORRECT), "na_hall": count(na, HALLUCINATED),
        "skipped": sum(1 for r in field_rows if r["outcome"] == SKIPPED),
        "sets_ok": len(ok_sets), "exact": sum(1 for r in ok_sets if r["exact_match"] is True),
        "mt_total": len(mt), "mt_correct": sum(1 for r in mt if r["meter_type_correct"] is True),
        "status": Counter(r["status"] for r in set_rows),
    }


# ── REPORTING ─────────────────────────────────────────────────────────
def frac(k, n):
    if n == 0:
        return "0/0 (ไม่มีข้อมูล)"
    if n < SMALL_N:
        return f"{k}/{n} (จำนวนน้อย)"
    return f"{k}/{n} = {k / n * 100:.1f}%"


def md(v):
    return str(v).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def breakdown(lines, title, head, rows, key_fn, order=None):
    buckets = defaultdict(list)
    for r in rows:
        buckets[key_fn(r)].append(r)
    keys = [k for k in (order or sorted(buckets)) if k in buckets]
    lines += ["", f"### {title}", "",
              f"| {head} | ค่าจริงมีตัวเลข: correct | misread | missed | ค่าจริง NA: hallucinated |",
              "|---|---|---:|---:|---|"]
    for k in keys:
        rs = buckets[k]
        num = [r for r in rs if r["_kind"] == "NUM"]
        na = [r for r in rs if r["_kind"] == "NA"]
        lines.append(
            f"| {md(k)} | {frac(sum(r['outcome'] == CORRECT for r in num), len(num))} "
            f"| {sum(r['outcome'] == MISREAD for r in num)} "
            f"| {sum(r['outcome'] == MISSED for r in num)} "
            f"| {frac(sum(r['outcome'] == HALLUCINATED for r in na), len(na))} |")


def engine_flag(engine):
    return "" if engine == "gemini" else f" --engine {engine}"


def build_summary(sets, field_rows, set_rows, notes, rdir, mode, run_id, engine="gemini"):
    m = compute_metrics(field_rows, set_rows)
    st = m["status"]
    metas = [x for x in (read_meta(rdir, s["set_id"]) for s in sets) if x]
    models = sorted({str(x.get("model") or x.get("gemini_model")) for x in metas})
    flags = sorted({str(x.get("send_deskewed_to_gemini")) for x in metas})
    ev = [r for r in field_rows if r["outcome"] in EVALUATED]

    L = [f"# ผลประเมินการอ่านค่า — engine `{engine}` / mode `{mode}` / run `{run_id}`", "",
         f"- สร้างเมื่อ: {datetime.now():%Y-%m-%d %H:%M:%S}",
         f"- ground truth: `{GT_XLSX}` (แก้ไขล่าสุด "
         f"{datetime.fromtimestamp(GT_XLSX.stat().st_mtime):%Y-%m-%d %H:%M:%S})",
         f"- engine: {engine} | model: {', '.join(models) or '-'} | "
         f"SEND_DESKEWED_TO_GEMINI: {', '.join(flags) or '-'}",
         f"- ชุดทั้งหมด {len(set_rows)}: ประเมินได้ {st[OK]} | blocked_mixed_types {st[BLOCKED]}"
         f" | api_error {st[API_ERROR]} | ยังไม่ได้รัน {st[NOT_RUN]}"]

    err_ids = [r["set_id"] for r in set_rows if r["status"] == API_ERROR]
    if err_ids:
        L += ["", f"> ⚠️ **คำเตือน: ยังมี {len(err_ids)} ชุดที่เป็น api_error** "
                  f"({', '.join(err_ids)})",
              "> ตัวเลขทั้งหมดในรายงานนี้ **ไม่รวม** ชุดเหล่านี้ รันซ้ำเฉพาะชุดที่ error ด้วย:",
              f"> `python evaluate.py{engine_flag(engine)} --mode {mode} --run-id {run_id} --retry-errors`"]
    nr_ids = [r["set_id"] for r in set_rows if r["status"] == NOT_RUN]
    if nr_ids:
        L += ["", f"> ⚠️ **ยังไม่ได้รัน {len(nr_ids)} ชุด** ({', '.join(nr_ids)}) — ไม่นับในรายงานนี้"]
    if len(models) > 1 or len(flags) > 1:
        L += ["", "> ⚠️ **raw ใน run นี้ใช้ค่าตั้งไม่เหมือนกันทุกชุด** (model หรือ "
                  "SEND_DESKEWED_TO_GEMINI) — ควรรันใหม่ด้วย --force"]
    for n in notes:
        L += ["", f"> ⚠️ {md(n)}"]

    L += ["", "## ตัวเลขหลัก", "",
          f"1. **ฟิลด์ที่ค่าจริงมีตัวเลข — correct: {frac(m['num_correct'], m['num_total'])}** "
          f"(field accuracy)",
          f"2. **ฟิลด์ที่ค่าจริงเป็น NA — hallucinated: {frac(m['na_hall'], m['na_total'])}**",
          "",
          "สองตัวเลขนี้วัดคนละเรื่องจึงไม่รวมกัน: (1) อ่านค่าที่มีอยู่ได้ถูกไหม "
          "(2) แต่งตัวเลขขึ้นมาในช่องที่ไม่มีค่าไหม",
          "",
          f"- ไม่นับฟิลด์ที่ค่าจริงเป็น `?` ({m['skipped']} ฟิลด์) และชุดที่ไม่ได้ประเมิน",
          f"- ค่าที่มีจำนวนน้อยกว่า {SMALL_N} แสดงเป็นจำนวนเท่านั้น พร้อมหมายเหตุ (จำนวนน้อย)"]

    digits_rows = [r for r in ev if r["scoring"] == "digits" and r["_kind"] == "NUM"]
    if digits_rows:
        strict_ok = [r for r in digits_rows if r["outcome"] == CORRECT
                     and r["_pred_is_num"] and float(r["predicted"]) == float(r["truth"])]
        L += ["", "## วิธีให้คะแนน turbine_v_m3", "",
              f"- engine `{engine}`: **ตัวเลขตรง** — ตัดจุดทศนิยมและ 0 นำหน้าแล้วเทียบ "
              "(prompt ของ GLM สั่งให้อ่านลูกกลิ้งโดยไม่ใส่จุด การวางทศนิยมเป็นหน้าที่ checkbox ของผู้ใช้)",
              "- ฟิลด์อื่นทั้งหมดให้คะแนนแบบเข้มงวด (ค่าต้องเท่ากันรวมตำแหน่งทศนิยม) เหมือน Gemini",
              f"- turbine ที่ค่าจริงมีตัวเลข {len(digits_rows)} ฟิลด์: ตัวเลขตรง "
              f"{sum(r['outcome'] == CORRECT for r in digits_rows)} | "
              f"ถ้าใช้เกณฑ์เข้มงวดจะถูก {len(strict_ok)}"]

    times = [float(r["elapsed_s"]) for r in set_rows if r["status"] == OK and r["elapsed_s"] != ""]
    if times:
        ts = sorted(times)
        mid = len(ts) // 2
        median = ts[mid] if len(ts) % 2 else (ts[mid - 1] + ts[mid]) / 2
        L += ["", "## เวลาเรียก engine ต่อชุด", "",
              f"- {len(ts)} ชุด: เฉลี่ย {sum(ts) / len(ts):.2f} s | มัธยฐาน {median:.2f} s | "
              f"ช่วง {ts[0]:.2f}–{ts[-1]:.2f} s",
              "- วัดเฉพาะการเรียก engine (ไม่รวม OBB/preprocess) — Gemini รวมเครือข่ายและคิวของ server"]

    n1, n2 = m["num_total"], m["na_total"]
    L += ["", "## ผลรวมรายฟิลด์", "",
          "| ค่าจริง | ทั้งหมด | correct | misread | missed | hallucinated |",
          "|---|---:|---|---|---|---|",
          f"| (ก) มีตัวเลข | {n1} | {frac(m['num_correct'], n1)} | {frac(m['num_misread'], n1)} "
          f"| {frac(m['num_missed'], n1)} | — |",
          f"| (ข) NA | {n2} | {frac(m['na_correct'], n2)} | — | — | {frac(m['na_hall'], n2)} |"]

    ok_sets = [r for r in set_rows if r["status"] == OK]
    exact_ids = [r["set_id"] for r in ok_sets if r["exact_match"] is True]
    L += ["", "## Set-level exact match", "",
          f"- ชุดที่ทุกฟิลด์ถูก: **{frac(m['exact'], m['sets_ok'])}**",
          "- นิยาม: ทุกฟิลด์ตัวเลขที่ค่าจริงไม่ใช่ `?` ได้ outcome = correct "
          "(รวมช่อง NA ที่ระบบตอบ null) ไม่รวม meter_type",
          f"- ชุดที่ exact match: {', '.join(exact_ids) or '-'}"]

    L += ["", "## ความแม่นยำของ meter_type", "",
          f"- **{frac(m['mt_correct'], m['mt_total'])}**"]
    mt_wrong = [r for r in ok_sets if r["meter_type_correct"] is False]
    if mt_wrong:
        L += ["", "| set_id | ค่าจริง | ระบบตอบ | obb_classes |", "|---|---|---|---|"]
        L += [f"| {md(r['set_id'])} | {md(r['meter_type_truth'])} | {md(r['meter_type_pred'])} "
              f"| {md(r['obb_classes'])} |" for r in mt_wrong]

    L += ["", "## แยกย่อย"]
    breakdown(L, "ตาม group", "group", ev, lambda r: r["group"])
    breakdown(L, "ตามฟิลด์", "field", ev, lambda r: r["field"], order=FIELDS)
    breakdown(L, "ตาม meter_type (ค่าจริง)", "meter_type", ev, lambda r: r["_meter_type"])
    breakdown(L, "ชุดภาพเดียว vs หลายภาพ", "ชุด", ev,
              lambda r: "ภาพเดียว" if r["n_images"] == 1 else "หลายภาพ",
              order=["ภาพเดียว", "หลายภาพ"])

    answered = [r for r in ev if r["_pred_is_num"]]
    L += ["", "## low_confidence_fields", "",
          "นับเฉพาะฟิลด์ที่ระบบตอบเป็นตัวเลข (Gemini ติดธงได้เฉพาะฟิลด์ที่ตอบค่า)", "",
          "| | ฟิลด์ที่ตอบตัวเลข | correct | misread | hallucinated |",
          "|---|---:|---|---:|---:|"]
    for label, flag in (("ติดธง", True), ("ไม่ติดธง", False)):
        rs = [r for r in answered if r["low_confidence"] is flag]
        L.append(f"| {label} | {len(rs)} | {frac(sum(r['outcome'] == CORRECT for r in rs), len(rs))} "
                 f"| {sum(r['outcome'] == MISREAD for r in rs)} "
                 f"| {sum(r['outcome'] == HALLUCINATED for r in rs)} |")
    flagged_null = [r for r in ev if r["low_confidence"] is True and not r["_pred_is_num"]]
    if flagged_null:
        L.append(f"\nติดธงทั้งที่ระบบตอบ null: {len(flagged_null)} ฟิลด์ (ขัดกับกฎใน prompt)")

    misread = [r for r in ev if r["outcome"] == MISREAD]
    swap_base = [r for r in ev if r["outcome"] in (MISREAD, HALLUCINATED) and r["_pred_is_num"]]
    swapped = [r for r in swap_base if r["swapped_field"]]
    L += ["", "## decimal_shift และ swapped_field", "",
          f"- decimal_shift: {sum(r['decimal_shift'] is True for r in misread)} ฟิลด์ "
          f"(จาก misread {len(misread)} ฟิลด์)",
          f"- swapped_field: {len(swapped)} ฟิลด์ "
          f"(จาก misread + hallucinated ที่ตอบเป็นตัวเลข {len(swap_base)} ฟิลด์)"]

    bad = [r for r in ev if r["outcome"] != CORRECT]
    L += ["", f"## รายการฟิลด์ที่ไม่ correct ({len(bad)} ฟิลด์)", ""]
    if bad:
        L += ["| set_id | ภาพ | field | ค่าจริง | ระบบตอบ | outcome | scoring | decimal_shift "
              "| swapped_field | low_conf | detected_format | decimal_slots_truth |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in bad:
            L.append(f"| {md(r['set_id'])} | {md(r['_images'])} | {r['field']} | {md(r['truth'])} "
                     f"| {md(r['predicted'])} | {r['outcome']} | {r['scoring']} | {r['decimal_shift']} "
                     f"| {md(r['swapped_field'])} | {r['low_confidence']} "
                     f"| {md(r['_detected_format'])} | {md(r['_decimal_slots'])} |")
    else:
        L.append("ไม่มี")

    not_ok = [r for r in set_rows if r["status"] != OK]
    if not_ok:
        L += ["", "## ชุดที่ไม่ได้ประเมิน", "", "| set_id | status | obb_classes | error |",
              "|---|---|---|---|"]
        L += [f"| {md(r['set_id'])} | {r['status']} | {md(r['obb_classes'])} | {md(r['error'])[:300]} |"
              for r in not_ok]

    return "\n".join(L) + "\n"


def write_run_outputs(sets, rdir, mode, run_id, engine="gemini"):
    field_rows, set_rows, notes = score_run(sets, rdir)
    rdir.mkdir(parents=True, exist_ok=True)
    for path, columns, rows in ((rdir / "field_results.csv", FIELD_COLUMNS, field_rows),
                                (rdir / "set_results.csv", SET_COLUMNS, set_rows)):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    (rdir / "summary.md").write_text(
        build_summary(sets, field_rows, set_rows, notes, rdir, mode, run_id, engine), encoding="utf-8")

    m = compute_metrics(field_rows, set_rows)
    st = m["status"]
    print("\n" + "=" * 72)
    print(f"ผล engine {engine} / mode {mode} / run {run_id}")
    print("=" * 72)
    print(f"ชุด: ประเมินได้ {st[OK]} | blocked_mixed_types {st[BLOCKED]} | "
          f"api_error {st[API_ERROR]} | ยังไม่ได้รัน {st[NOT_RUN]}")
    print(f"(1) ฟิลด์ที่ค่าจริงมีตัวเลข — correct:      {frac(m['num_correct'], m['num_total'])}")
    print(f"(2) ฟิลด์ที่ค่าจริงเป็น NA — hallucinated: {frac(m['na_hall'], m['na_total'])}")
    if st[API_ERROR]:
        print(f"คำเตือน: ยังมี {st[API_ERROR]} ชุด api_error — "
              f"รันซ้ำด้วย{engine_flag(engine)} --mode {mode} --run-id {run_id} --retry-errors")
    print(f"ไฟล์ผลลัพธ์: {rdir}/ (field_results.csv, set_results.csv, summary.md)")


def discover_runs():
    runs = []
    if OUT_ROOT.is_dir():
        for mode_dir in sorted(p for p in OUT_ROOT.iterdir() if p.is_dir()):
            for rdir in sorted(p for p in mode_dir.iterdir() if p.is_dir()):
                if any((rdir / "raw").glob("*.json")):
                    runs.append((f"{mode_dir.name}/{rdir.name}", rdir))
    return runs


def write_compare(sets):
    runs = discover_runs()
    if len(runs) < 2:
        print(f"\ncompare.md: มีผล {len(runs)} run — ต้องมีอย่างน้อย 2 run จึงจะสร้าง")
        return
    scored = []
    for label, rdir in runs:
        fr, sr, _ = score_run(sets, rdir)
        scored.append((label, fr, sr, compute_metrics(fr, sr)))

    L = ["# เปรียบเทียบผลระหว่าง run", "",
         f"- สร้างเมื่อ: {datetime.now():%Y-%m-%d %H:%M:%S}",
         "- ทุก run คำนวณใหม่จากไฟล์ raw ด้วย ground_truth.xlsx ปัจจุบัน",
         f"- ค่าที่มีจำนวนน้อยกว่า {SMALL_N} แสดงเป็นจำนวนพร้อมหมายเหตุ (จำนวนน้อย)", "",
         "| run | ประเมินได้ | api_error | blocked | ยังไม่ได้รัน | (1) ค่าจริงมีตัวเลข: correct "
         "| (2) ค่าจริง NA: hallucinated | set exact match | meter_type |",
         "|---|---:|---:|---:|---:|---|---|---|---|"]
    for label, _, _, m in scored:
        st = m["status"]
        L.append(f"| {label} | {st[OK]} | {st[API_ERROR]} | {st[BLOCKED]} | {st[NOT_RUN]} "
                 f"| {frac(m['num_correct'], m['num_total'])} | {frac(m['na_hall'], m['na_total'])} "
                 f"| {frac(m['exact'], m['sets_ok'])} | {frac(m['mt_correct'], m['mt_total'])} |")
    with_err = [label for label, _, _, m in scored if m["status"][API_ERROR]]
    if with_err:
        L += ["", f"> ⚠️ run ที่ยังมีชุด api_error: {', '.join(with_err)} — ผลรวมเทียบกันได้ไม่เต็มที่"]

    status_by = {label: {r["set_id"]: r for r in sr} for label, _, sr, _ in scored}
    field_by = {label: {(r["set_id"], r["field"]): r for r in fr} for label, fr, _, _ in scored}
    labels = [label for label, *_ in scored]

    status_diff = [s["set_id"] for s in sets
                   if len({status_by[l][s["set_id"]]["status"] for l in labels}) > 1]
    if status_diff:
        L += ["", "## ชุดที่สถานะต่างกันระหว่าง run", "",
              "| set_id | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
        for sid in status_diff:
            L.append(f"| {md(sid)} | " + " | ".join(status_by[l][sid]["status"] for l in labels) + " |")

    diffs = []
    for s in sets:
        for f in FIELDS:
            rows = [field_by[l][(s["set_id"], f)] for l in labels]
            usable = [r for r in rows if r["outcome"] in EVALUATED]
            if len(usable) < 2:
                continue
            if len({r["outcome"] for r in usable}) > 1:
                diffs.append((s, f, rows, "outcome"))
            elif len({r["predicted"] for r in usable}) > 1:
                diffs.append((s, f, rows, "ค่าที่ตอบ"))
    L += ["", f"## ฟิลด์ที่ผลต่างกันระหว่าง run ({len(diffs)} ฟิลด์)", "",
          "เทียบเฉพาะ run ที่ชุดนั้นประเมินได้ ช่องแสดง `outcome (ค่าที่ตอบ)`", ""]
    if diffs:
        L += ["| set_id | field | ค่าจริง | " + " | ".join(labels) + " | ต่างที่ |",
              "|---|---|---|" + "---|" * len(labels) + "---|"]
        for s, f, rows, what in diffs:
            cells = [f"{r['outcome']} ({md(r['predicted'])})" if r["outcome"] in EVALUATED
                     else r["outcome"] for r in rows]
            L.append(f"| {md(s['set_id'])} | {f} | {md(s['truth'][f][0])} | "
                     + " | ".join(cells) + f" | {what} |")
    else:
        L.append("ไม่มี")

    mt_diff = []
    for s in sets:
        rows = [status_by[l][s["set_id"]] for l in labels]
        usable = [r for r in rows if r["status"] == OK]
        if len(usable) >= 2 and len({r["meter_type_pred"].lower() for r in usable}) > 1:
            mt_diff.append((s, rows))
    if mt_diff:
        L += ["", "## meter_type ที่ตอบต่างกันระหว่าง run", "",
              "| set_id | ค่าจริง | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
        for s, rows in mt_diff:
            L.append(f"| {md(s['set_id'])} | {s['meter_type']} | "
                     + " | ".join(md(r["meter_type_pred"]) if r["status"] == OK else r["status"]
                                  for r in rows) + " |")

    OUT_ROOT.mkdir(exist_ok=True)
    (OUT_ROOT / "compare.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\ncompare.md: เทียบ {len(runs)} run -> {OUT_ROOT / 'compare.md'}")


def load_glm_engine():
    """สร้าง GlmEngine ตัวเดียวกับที่ server ใช้ แล้วเช็กว่า Ollama เปิดอยู่และมีโมเดล — ไม่พร้อมให้หยุดก่อนเริ่ม"""
    import httpx
    import ocr_engines
    glm = ocr_engines.create_engine("glm")
    try:
        tags = httpx.get(f"{glm.base_url}/api/tags", timeout=5).json()
    except Exception:
        sys.exit(f"หยุด: ติดต่อ Ollama ที่ {glm.base_url} ไม่ได้ — เปิด `ollama serve` แล้วรันใหม่")
    names = {m.get("name") for m in tags.get("models", [])}
    if glm.model not in names:
        sys.exit(f"หยุด: Ollama ยังไม่มีโมเดล '{glm.model}' — ติดตั้งด้วย `ollama pull {glm.model}` "
                 f"(ที่มี: {', '.join(sorted(n for n in names if n)) or '-'})")
    return glm


# ── MAIN ──────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="วัดความแม่นยำการอ่านค่าของระบบ Gas Meter OCR")
    ap.add_argument("--mode", choices=list(MODES), help="deskew หรือ crop")
    ap.add_argument("--engine", choices=ENGINES, default="gemini",
                    help="gemini (default) หรือ glm (GLM-OCR ผ่าน Ollama บนเครื่อง)")
    ap.add_argument("--run-id", default="run1", help="ชื่อรอบการรัน (default: run1)")
    ap.add_argument("--check", action="store_true", help="ตรวจ ground truth อย่างเดียว")
    ap.add_argument("--rescore", action="store_true", help="คำนวณผลใหม่จาก raw ไม่เรียก Gemini")
    ap.add_argument("--force", action="store_true", help="รันใหม่ทุกชุด ทับ raw เดิม")
    ap.add_argument("--retry-errors", action="store_true", help="รันซ้ำเฉพาะชุดที่เป็น api_error")
    ap.add_argument("--compare", action="store_true", help="สร้าง eval_output/compare.md ใหม่")
    ap.add_argument("--yes", action="store_true", help="ไม่ถามยืนยันก่อนเรียก Gemini")
    ap.add_argument("--delay", type=float, default=4.0, help="วินาทีระหว่างการเรียก (default 4)")
    ap.add_argument("--retries", type=int, default=3,
                    help="Gemini: จำนวนครั้งที่ลองซ้ำเมื่อเจอ 5xx (429 ไม่ลองซ้ำ — หยุดทั้งรอบ)")
    args = ap.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id):
        ap.error("--run-id ใช้ได้เฉพาะ A-Z a-z 0-9 _ . -")
    if args.force and args.retry_errors:
        ap.error("ใช้ --force กับ --retry-errors พร้อมกันไม่ได้")
    if args.rescore and (args.force or args.retry_errors):
        ap.error("--rescore ไม่เรียก Gemini จึงใช้กับ --force / --retry-errors ไม่ได้")
    if not (args.check or args.compare) and not args.mode:
        ap.error("ต้องระบุ --mode deskew หรือ --mode crop")
    if args.delay < 0 or args.retries < 0:
        ap.error("--delay และ --retries ต้องไม่ติดลบ")
    return args


def main():
    args = parse_args()
    sets, problems, warnings = load_ground_truth()

    if args.compare:
        if problems:
            print_check_report(sets, problems, warnings)
            sys.exit("หยุด: แก้ปัญหาใน ground truth ก่อน")
        write_compare(sets)
        return

    print_check_report(sets, problems, warnings)
    if args.check:
        if args.mode and not problems:
            print_run_status(sets, run_dir(args.mode, args.run_id, args.engine))
        sys.exit(1 if problems else 0)
    if problems:
        sys.exit("\nหยุด: แก้ปัญหาใน ground truth ก่อน ห้ามรันต่อ")

    rdir = run_dir(args.mode, args.run_id, args.engine)
    if args.rescore:
        if not any((rdir / "raw").glob("*.json")):
            sys.exit(f"\nไม่พบไฟล์ raw ใน {rdir}/raw — ต้องรันก่อนจึงจะ --rescore ได้")
        write_run_outputs(sets, rdir, args.mode, args.run_id, args.engine)
        write_compare(sets)
        return

    metas = {s["set_id"]: read_meta(rdir, s["set_id"]) for s in sets}
    if args.force:
        targets = list(sets)
    elif args.retry_errors:
        targets = [s for s in sets if metas[s["set_id"]] and metas[s["set_id"]]["status"] == API_ERROR]
    else:
        targets = [s for s in sets if metas[s["set_id"]] is None]
    n_have = sum(1 for m in metas.values() if m)

    print("\n" + "=" * 72)
    print(f"แผนการรัน: engine {args.engine} | mode {args.mode} "
          f"(SEND_DESKEWED_TO_GEMINI = {MODES[args.mode]}) | run {args.run_id}")
    print("=" * 72)
    print(f"โฟลเดอร์ผล: {rdir}/ | มี raw แล้ว {n_have}/{len(sets)} ชุด")
    if args.retry_errors:
        print(f"รันซ้ำเฉพาะ api_error: {len(targets)} ชุด")
    elif args.force:
        print(f"--force: รันใหม่ทุกชุด {len(targets)} ชุด (ทับ raw เดิม)")
    else:
        print(f"จะรัน {len(targets)} ชุดที่ยังไม่มี raw | ข้าม {len(sets) - len(targets)} ชุด")

    if targets:
        glm = args.engine == "glm"
        if glm:
            print(f"เรียก GLM (Ollama บนเครื่อง ไม่มีโควต้า) {len(targets)} ครั้ง — ไม่ถามยืนยัน")
        else:
            print(f"เรียก Gemini สูงสุด {len(targets)} ครั้ง (ชุดที่ถูกบล็อกจะไม่เรียก) "
                  f"เว้นระยะ {args.delay:g} วินาที, retry 5xx {args.retries} ครั้ง, เจอ 429 หยุดทันที")
        print("ชุด: " + ", ".join(s["set_id"] for s in targets))
        if not glm and not args.yes:
            if not sys.stdin.isatty():
                sys.exit("\nต้องยืนยันก่อนเรียก Gemini: รันในเทอร์มินัลแล้วตอบ y หรือใส่ --yes")
            if input("\nยืนยันเรียก Gemini? พิมพ์ y แล้วกด Enter: ").strip().lower() != "y":
                print("ยกเลิก — ยังไม่ได้เรียก Gemini")
                return

        print("\nกำลังโหลด server.py ...", flush=True)
        import server   # โหลดหลังยืนยันเท่านั้น (โหลดโมเดล OBB + engine ตาม .env)

        server.SEND_DESKEWED_TO_GEMINI = MODES[args.mode]   # ตั้งค่าในสคริปต์นี้เท่านั้น
        if server.obb_model is None:
            sys.exit("หยุด: server.py โหลดโมเดล OBB ไม่ได้ — ผลประเมินจะไม่ตรงกับระบบที่ใช้งานจริง")

        glm_engine = None
        if glm:
            glm_engine = load_glm_engine()
        elif server.client is None:
            sys.exit(f"หยุด: server.py ไม่ได้สร้าง Gemini client (OCR_ENGINE ใน .env = "
                     f"'{server.OCR_ENGINE}') — ตั้ง OCR_ENGINE=gemini ก่อนรัน --engine gemini")

        from ocr_engines.glm_engine import OllamaUnavailableError
        last_api_call = None
        for i, s in enumerate(targets, 1):
            if not glm and last_api_call is not None and args.delay > 0:
                wait = args.delay - (time.time() - last_api_call)
                if wait > 0:
                    time.sleep(wait)
            print(f"[{i}/{len(targets)}] {s['set_id']} ({len(s['paths'])} ภาพ) ...", flush=True)
            try:
                meta = process_set(server, s, args.mode, args.run_id, args.retries, args.delay,
                                   engine=args.engine, glm=glm_engine)
            except OllamaUnavailableError as e:
                sys.exit(f"\nหยุด: {e}\nชุดที่รันไปแล้วถูกเก็บไว้ รันคำสั่งเดิมซ้ำได้ (จะรันเฉพาะชุดที่ยังไม่มีผล)")
            if meta["called_api"]:
                last_api_call = time.time()
            save_meta(rdir, meta)
            classes = ", ".join(c or "none" for c in meta["obb_classes"]) or "-"
            extra = f" — {meta['error'][:160]}" if meta["error"] else ""
            took = f" | {meta['elapsed_s']:.1f} s" if meta.get("elapsed_s") is not None else ""
            print(f"    {meta['status']} | OBB: {classes}{took}{extra}", flush=True)
            if meta.get("quota_exhausted"):
                rest = [t["set_id"] for t in targets[i:]]
                print(f"\n⛔ Gemini ตอบ 429 (โควต้าหมด) — หยุดทันที ไม่เรียกชุดที่เหลือ "
                      f"({', '.join(rest) or '-'})")
                print(f"   รอโควต้ารีเซ็ตแล้วรัน: python evaluate.py --mode {args.mode} "
                      f"--run-id {args.run_id} --retry-errors  (และคำสั่งเดิมสำหรับชุดที่ยังไม่ได้รัน)")
                break
    else:
        print("ไม่มีชุดที่ต้องเรียก engine — คำนวณผลจาก raw ที่มีอยู่")

    write_run_outputs(sets, rdir, args.mode, args.run_id, args.engine)
    write_compare(sets)


if __name__ == "__main__":
    main()
