# -*- coding: utf-8 -*-
"""
evaluate.py — Evaluation Harness สำหรับระบบ Gas Meter OCR
===========================================================
ยิงภาพทดสอบทุกเคสเข้า API -> เทียบกับเฉลย (ground truth) -> สรุป field-level accuracy

วิธีใช้:
    1) เตรียม eval_dataset/ground_truth.csv และรูปใน eval_dataset/images/
    2) รันเซิร์ฟเวอร์ OCR ให้พร้อม (uvicorn server:app --port 8000)
    3) รัน:  python evaluate.py --label baseline
    4) แก้ระบบ (เช่น ปิด detailEnhance) แล้วรันใหม่:  python evaluate.py --label no_enhance
    5) เทียบ:  python evaluate.py --compare baseline no_enhance

ผลลัพธ์:
    - ตารางสรุป accuracy ต่อฟิลด์บนหน้าจอ
    - results/results_<label>.csv  รายละเอียดรายเคส (เปิดใน Excel ได้)
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests

# ── CONFIG ────────────────────────────────────────────────────────────
API_URL = os.environ.get("OCR_API_URL", "http://127.0.0.1:8000/api/ocr")
DATASET_DIR = Path("eval_dataset")
GROUND_TRUTH_CSV = DATASET_DIR / "ground_truth.csv"
IMAGES_DIR = DATASET_DIR / "images"
RESULTS_DIR = Path("results")

# ฟิลด์ตัวเลขที่ประเมิน (ตรงกับ MeterReading schema ฝั่ง server)
NUMERIC_FIELDS = [
    "correct_v_scm", "line_v_m3", "pressure_bar", "temperature_c",
    "correction_factor", "turbine_v_m3", "qm_m3_h", "qb_sm3_h", "pulse_weight",
]
# ฟิลด์ข้อความที่ประเมินแบบ exact string (ตัวพิมพ์เล็ก)
TEXT_FIELDS = ["meter_type"]

ALL_FIELDS = TEXT_FIELDS + NUMERIC_FIELDS


# ── GROUND TRUTH LOADING ──────────────────────────────────────────────
def load_ground_truth():
    """อ่าน ground_truth.csv -> list ของ dict ต่อเคส
    ช่องว่างใน CSV = เฉลยเป็น null (ค่านั้นไม่ปรากฏบนมิเตอร์)"""
    if not GROUND_TRUTH_CSV.exists():
        sys.exit(f"ไม่พบ {GROUND_TRUTH_CSV} — สร้างจาก ground_truth_template.csv ก่อนครับ")

    cases = []
    with open(GROUND_TRUTH_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = (row.get("case_id") or "").strip()
            images_raw = (row.get("images") or "").strip()
            if not case_id or not images_raw:
                continue

            image_paths = []
            for name in images_raw.split(";"):
                p = IMAGES_DIR / name.strip()
                if not p.exists():
                    sys.exit(f"[{case_id}] ไม่พบไฟล์ภาพ: {p}")
                image_paths.append(p)

            expected = {}
            for field in TEXT_FIELDS:
                v = (row.get(field) or "").strip().lower()
                expected[field] = v if v else None
            for field in NUMERIC_FIELDS:
                v = (row.get(field) or "").strip()
                expected[field] = float(v) if v else None

            cases.append({"case_id": case_id, "images": image_paths, "expected": expected})

    if not cases:
        sys.exit("ground_truth.csv ว่างเปล่า — ใส่เคสทดสอบอย่างน้อย 1 แถวก่อนครับ")
    return cases


# ── API CALL ──────────────────────────────────────────────────────────
def call_ocr_api(image_paths, retries=2):
    """ส่งภาพทั้งชุดของ 1 เคสเข้า /api/ocr แบบ multipart"""
    for attempt in range(retries + 1):
        files = [
            ("files", (p.name, open(p, "rb"), "image/jpeg"))
            for p in image_paths
        ]
        try:
            resp = requests.post(API_URL, files=files, timeout=120)
            resp.raise_for_status()
            return resp.json(), None
        except Exception as e:
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            return None, str(e)
        finally:
            for _, (_, fh, _) in files:
                fh.close()


# ── COMPARISON ────────────────────────────────────────────────────────
def compare_field(expected, predicted, tolerance=0.0):
    """เทียบค่าฟิลด์เดียว: คืน (ถูก/ผิด)
    - ทั้งคู่ null = ถูก (ระบบรู้ว่าไม่มีค่า = ไม่ hallucinate)
    - ฝั่งเดียว null = ผิด
    - ตัวเลข: exact match เป็น default, ใส่ tolerance แบบ relative ได้"""
    if expected is None and predicted is None:
        return True
    if expected is None or predicted is None:
        return False
    if isinstance(expected, str):
        return str(predicted).strip().lower() == expected
    try:
        predicted = float(predicted)
    except (TypeError, ValueError):
        return False
    if tolerance <= 0:
        return predicted == expected
    if expected == 0:
        return abs(predicted) <= tolerance
    return abs(predicted - expected) / abs(expected) <= tolerance


def evaluate_case(case, tolerance):
    """ประเมิน 1 เคส -> dict ผลรายฟิลด์"""
    prediction, error = call_ocr_api(case["images"])
    result = {
        "case_id": case["case_id"],
        "n_images": len(case["images"]),
        "api_error": error or "",
    }
    if error:
        for field in ALL_FIELDS:
            result[f"{field}_ok"] = 0
            result[f"{field}_expected"] = case["expected"][field]
            result[f"{field}_predicted"] = ""
        return result

    for field in ALL_FIELDS:
        expected = case["expected"][field]
        predicted = prediction.get(field)
        ok = compare_field(expected, predicted, tolerance)
        result[f"{field}_ok"] = int(ok)
        result[f"{field}_expected"] = expected if expected is not None else ""
        result[f"{field}_predicted"] = predicted if predicted is not None else ""
    return result


# ── REPORTING ─────────────────────────────────────────────────────────
def summarize(results):
    """สรุป accuracy ต่อฟิลด์ + ภาพรวม (นับเฉพาะเคสที่ API ไม่ error)"""
    ok_cases = [r for r in results if not r["api_error"]]
    n_err = len(results) - len(ok_cases)

    print("\n" + "=" * 62)
    print(f"  สรุปผล: {len(ok_cases)} เคสประเมินได้ | {n_err} เคส API error")
    print("=" * 62)
    print(f"  {'Field':<20} {'Correct':>8} {'Total':>7} {'Accuracy':>10}")
    print("  " + "-" * 47)

    total_ok, total_n = 0, 0
    for field in ALL_FIELDS:
        # นับเฉพาะเคสที่ฟิลด์นี้ "มีความหมาย" (เฉลยมีค่า หรือระบบทายว่ามีค่า)
        relevant = [
            r for r in ok_cases
            if r[f"{field}_expected"] != "" or r[f"{field}_predicted"] != ""
        ]
        n = len(relevant)
        ok = sum(r[f"{field}_ok"] for r in relevant)
        acc = f"{ok / n * 100:.1f}%" if n else "  n/a"
        print(f"  {field:<20} {ok:>8} {n:>7} {acc:>10}")
        total_ok += ok
        total_n += n

    print("  " + "-" * 47)
    overall = f"{total_ok / total_n * 100:.1f}%" if total_n else "n/a"
    print(f"  {'OVERALL':<20} {total_ok:>8} {total_n:>7} {overall:>10}")

    # เคสที่ผิดเยอะสุด — จุดเริ่มไล่ debug
    def case_errors(r):
        return sum(1 - r[f"{f}_ok"] for f in ALL_FIELDS)

    worst = sorted(ok_cases, key=case_errors, reverse=True)[:5]
    worst = [r for r in worst if case_errors(r) > 0]
    if worst:
        print("\n  เคสที่ผิดเยอะสุด (ไล่ debug จากตรงนี้):")
        for r in worst:
            bad_fields = [f for f in ALL_FIELDS if not r[f"{f}_ok"]]
            print(f"    {r['case_id']}: ผิด {len(bad_fields)} ฟิลด์ -> {', '.join(bad_fields)}")
    print()


def save_results(results, label):
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"results_{label}.csv"
    fieldnames = list(results[0].keys())
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"💾 บันทึกผลรายเคส: {out}")


def compare_runs(labels):
    """เทียบ accuracy ต่อฟิลด์ระหว่างหลายรอบการทดลอง (อ่านจาก results_<label>.csv)"""
    runs = {}
    for label in labels:
        path = RESULTS_DIR / f"results_{label}.csv"
        if not path.exists():
            sys.exit(f"ไม่พบ {path} — รัน python evaluate.py --label {label} ก่อนครับ")
        with open(path, newline="", encoding="utf-8-sig") as f:
            runs[label] = [r for r in csv.DictReader(f) if not r["api_error"]]

    print("\n" + "=" * 62)
    print("  เปรียบเทียบ accuracy ต่อฟิลด์ระหว่างรอบการทดลอง")
    print("=" * 62)
    header = f"  {'Field':<20}" + "".join(f"{l[:12]:>14}" for l in labels)
    print(header)
    print("  " + "-" * (20 + 14 * len(labels)))

    for field in ALL_FIELDS:
        row = f"  {field:<20}"
        for label in labels:
            rows = [
                r for r in runs[label]
                if r[f"{field}_expected"] != "" or r[f"{field}_predicted"] != ""
            ]
            if rows:
                acc = sum(int(r[f"{field}_ok"]) for r in rows) / len(rows) * 100
                row += f"{acc:>13.1f}%"
            else:
                row += f"{'n/a':>14}"
        print(row)
    print()


# ── MAIN ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Gas Meter OCR Evaluation Harness")
    parser.add_argument("--label", default=None,
                        help="ชื่อรอบการทดลอง เช่น baseline, no_enhance, obb")
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="relative tolerance ของตัวเลข (default 0 = exact match)")
    parser.add_argument("--compare", nargs="+", default=None,
                        help="เทียบผลหลายรอบ เช่น --compare baseline no_enhance")
    args = parser.parse_args()

    if args.compare:
        compare_runs(args.compare)
        return

    if not args.label:
        sys.exit("ระบุ --label ด้วยครับ เช่น: python evaluate.py --label baseline")

    cases = load_ground_truth()
    print(f"📋 โหลด ground truth: {len(cases)} เคส | tolerance={args.tolerance}")
    print(f"🎯 API: {API_URL}\n")

    results = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case['case_id']} ({len(case['images'])} ภาพ)...", end=" ", flush=True)
        r = evaluate_case(case, args.tolerance)
        n_ok = sum(r[f"{f}_ok"] for f in ALL_FIELDS)
        if r["api_error"]:
            print(f"❌ API error: {r['api_error'][:60]}")
        else:
            print(f"✓ {n_ok}/{len(ALL_FIELDS)} ฟิลด์ถูก")
        results.append(r)

    summarize(results)
    save_results(results, args.label)


if __name__ == "__main__":
    main()
