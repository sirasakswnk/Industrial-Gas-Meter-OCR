"""
การทดลอง: ทำไม classical CV ถึงหากรอบหน้าปัดมิเตอร์ไม่ได้
=========================================================
สคริปต์ทดลองแยกต่างหาก — เก็บหลักฐานเชิงตัวเลข + ภาพ สำหรับใส่รายงาน/สไลด์

! ไม่เกี่ยวกับ pipeline ที่ใช้งานจริง (server.py ใช้ YOLO-OBB แล้ว)
  สคริปต์นี้ไม่ import และไม่แก้ server.py

รัน 3 วิธีคลาสสิกกับทุกภาพใน experiment_images/ แล้ววัดว่า:
  - contour ใหญ่สุดกินพื้นที่กี่ % ของภาพ (ถ้าเจอจอจริงควรกินพื้นที่มาก)
  - approxPolyDP ได้สี่เหลี่ยม 4 จุดไหม (eps 0.02 / 0.04 / 0.06)

วิธีที่ทดสอบ:
  1. canny    — Canny auto-threshold จาก median + CLOSE 7x7
  2. otsu     — Otsu threshold + CLOSE 9x9
     otsu_inv — Otsu แบบ inverted (จอมืดบนพื้นสว่าง) + CLOSE 9x9
                (otsu กับ otsu_inv คือ "วิธีที่ 2" เดียวกัน แค่แยกสองขั้ว จึงรายงานทั้งคู่)
  3. bright   — threshold ที่ percentile 75 ของความสว่าง + OPEN 7x7 + CLOSE 15x15

กลุ่มภาพ (แยกตาม prefix ชื่อไฟล์):
  hard_*   = พลาสติกห่อ / แสงสะท้อนแรง
  tilt_*   = เอียง 45 องศา และตะแคง 90 องศา
  analog_* = มิเตอร์ตัวเลขหมุน
  normal_* = ถ่ายตรง แสงดี

รัน:
    python experiment_classical_cv.py                 # ใช้ experiment_images/
    python experiment_classical_cv.py path/to/folder  # ระบุโฟลเดอร์เอง
"""
import csv
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_DIR = "experiment_images"
OUT_DIR = "experiment_output"
CSV_PATH = "experiment_result.csv"

APPROX_EPS = (0.02, 0.04, 0.06)   # สัดส่วนของ perimeter ที่ใช้ทำ approxPolyDP
SAVE_MAX_W = 1600                 # ย่อภาพก่อนเซฟ (คำนวณจริงบนภาพเต็มความละเอียด)
GROUPS = ("normal", "tilt", "analog", "hard")


# ── helpers ───────────────────────────────────────────────────────────
def group_of(filename):
    """แยกกลุ่มจาก prefix ชื่อไฟล์ เช่น hard_1.jpeg -> hard"""
    stem = os.path.basename(filename).lower()
    for g in GROUPS:
        if stem.startswith(g + "_"):
            return g
    return "other"


def save_view(img, path):
    """เซฟภาพสำหรับดู/ใส่สไลด์ (ย่อถ้ากว้างเกิน SAVE_MAX_W)"""
    if img.shape[1] > SAVE_MAX_W:
        scale = SAVE_MAX_W / img.shape[1]
        img = cv2.resize(img, (SAVE_MAX_W, int(img.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def analyze_mask(mask, img_area):
    """หา contour จาก mask แล้ววัดผล

    คืน: (n_contours, largest_pct, quad หรือ None, eps ที่ได้ quad หรือ None)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0.0, None, None

    largest = max(contours, key=cv2.contourArea)
    largest_pct = cv2.contourArea(largest) / img_area * 100.0

    # ลอง approxPolyDP หลายค่า eps — ได้ 4 จุดเมื่อไหร่ถือว่า "เจอ quad"
    peri = cv2.arcLength(largest, True)
    for eps in APPROX_EPS:
        approx = cv2.approxPolyDP(largest, eps * peri, True)
        if len(approx) == 4:
            return len(contours), largest_pct, approx.reshape(4, 2), eps

    return len(contours), largest_pct, None, None


def quad_area_pct_of(quad, img_area):
    """พื้นที่ของ quad ที่เลือกได้ เทียบกับพื้นที่ภาพ (%)

    ต่างจาก largest_contour_pct: อันนั้นคือพื้นที่ contour ดิบ (ขอบหยักตามจริง)
    ส่วนอันนี้คือพื้นที่สี่เหลี่ยมหลัง approxPolyDP ที่จะถูกเอาไปใช้ deskew จริง
    """
    if quad is None:
        return None
    return cv2.contourArea(quad.astype(np.float32)) / img_area * 100.0


# ── 3 วิธีคลาสสิก — แต่ละตัวคืน mask ไบนารีให้ analyze_mask ──────────
def method_canny(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blur))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(blur, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def method_otsu(gray, inverted=False):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    flags = cv2.THRESH_BINARY_INV if inverted else cv2.THRESH_BINARY
    _, th = cv2.threshold(blur, 0, 255, flags + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    return cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)


def method_bright(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_val = float(np.percentile(blur, 75))
    _, th = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, k_open)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)


METHODS = {
    "canny":    lambda g: method_canny(g),
    "otsu":     lambda g: method_otsu(g, inverted=False),
    "otsu_inv": lambda g: method_otsu(g, inverted=True),
    "bright":   lambda g: method_bright(g),
}


# ── รันหนึ่งภาพ ───────────────────────────────────────────────────────
def run_one(path, rows):
    name = os.path.splitext(os.path.basename(path))[0]
    # โหลดแบบ EXIF-aware ให้ตรงกับ pipeline จริง (server.py ใช้ exif_transpose)
    # ถ้าใช้ cv2.imread ตรงๆ ภาพที่มี EXIF orientation=6 จะถูกประมวลผลผิดแนว
    # และเทียบกับผล YOLO-OBB ไม่ได้ (คนละแนวภาพ)
    try:
        pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception as e:
        print(f"  [!] อ่านภาพไม่ได้: {path} ({e})")
        return
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_area = img.shape[0] * img.shape[1]
    print(f"\n[img] {os.path.basename(path)}  ({img.shape[1]}x{img.shape[0]})  [{group_of(path)}]")

    for method, fn in METHODS.items():
        mask = fn(gray)
        n_contours, largest_pct, quad, eps = analyze_mask(mask, img_area)

        # เซฟ mask หลัง morphology ไว้ดูว่าทำไมถึงพัง
        save_view(mask, os.path.join(OUT_DIR, f"{name}_{method}.jpg"))

        # ถ้าได้ quad วาดทับภาพต้นฉบับ เพื่อให้เห็นว่ามันไปจับผิดตรงไหน
        if quad is not None:
            overlay = img.copy()
            thick = max(2, img.shape[1] // 400)
            cv2.polylines(overlay, [quad.astype(np.int32)], True, (0, 0, 255), thick)
            for pt in quad.astype(np.int32):
                cv2.circle(overlay, (int(pt[0]), int(pt[1])),
                           max(4, img.shape[1] // 200), (0, 255, 255), -1)
            save_view(overlay, os.path.join(OUT_DIR, f"{name}_{method}_result.jpg"))

        q_pct = quad_area_pct_of(quad, img_area)
        flag = "QUAD OK " if quad is not None else "no quad "
        eps_txt = f"(eps={eps}, quad={q_pct:.2f}%)" if eps else ""
        print(f"   {method:<9} contours={n_contours:<6} largest={largest_pct:6.2f}%  {flag}{eps_txt}")

        rows.append({
            "filename": os.path.basename(path),
            "group": group_of(path),
            "method": method,
            "largest_contour_pct": round(largest_pct, 2),
            "quad_area_pct": round(q_pct, 2) if q_pct is not None else "",
            "found_quad": quad is not None,
            "n_contours": n_contours,
            "quad_eps": eps if eps else "",
        })


# ── สรุปผล ────────────────────────────────────────────────────────────
def summarize(rows, n_images):
    print("\n" + "=" * 70)
    print("สรุปภาพรวม — แต่ละวิธีหา quad สำเร็จกี่ภาพ")
    print("=" * 70)
    print(f"{'method':<10}{'สำเร็จ':<14}{'largest_contour_pct':<30}")
    print(f"{'':<10}{'':<14}{'เฉลี่ย':>10}{'ต่ำสุด':>10}{'สูงสุด':>10}")
    print("-" * 70)

    for method in METHODS:
        mrows = [r for r in rows if r["method"] == method]
        ok = sum(1 for r in mrows if r["found_quad"])
        pcts = [r["largest_contour_pct"] for r in mrows]
        succ = f"{ok}/{n_images}"
        print(f"{method:<10}{succ:<14}"
              f"{np.mean(pcts):>10.2f}{np.min(pcts):>10.2f}{np.max(pcts):>10.2f}")

    # Otsu รวมสองขั้ว (สำเร็จถ้าขั้วใดขั้วหนึ่งได้ quad) — มองเป็น "วิธีที่ 2" เดียว
    by_file = defaultdict(bool)
    for r in rows:
        if r["method"] in ("otsu", "otsu_inv") and r["found_quad"]:
            by_file[r["filename"]] = True
    print("-" * 70)
    print(f"{'otsu (รวมสองขั้ว)':<24}{sum(by_file.values())}/{n_images}")

    # ── แยกตามกลุ่ม ──
    groups_present = [g for g in GROUPS if any(r["group"] == g for r in rows)]
    groups_present += [g for g in sorted({r["group"] for r in rows})
                       if g not in groups_present]

    print("\n" + "=" * 70)
    print("อัตราความสำเร็จแยกตามกลุ่มภาพ (หา quad ได้ / ทั้งหมด)")
    print("=" * 70)
    print(f"{'method':<10}" + "".join(f"{g:>14}" for g in groups_present))
    print("-" * 70)
    for method in METHODS:
        line = f"{method:<10}"
        for g in groups_present:
            grows = [r for r in rows if r["method"] == method and r["group"] == g]
            ok = sum(1 for r in grows if r["found_quad"])
            line += f"{f'{ok}/{len(grows)}':>14}"
        print(line)

    print("\n" + "=" * 70)
    print("largest_contour_pct เฉลี่ย แยกกลุ่ม (ยิ่งต่ำ = contour แตกเป็นชิ้น หาจอไม่เจอ)")
    print("=" * 70)
    print(f"{'method':<10}" + "".join(f"{g:>14}" for g in groups_present))
    print("-" * 70)
    for method in METHODS:
        line = f"{method:<10}"
        for g in groups_present:
            pcts = [r["largest_contour_pct"] for r in rows
                    if r["method"] == method and r["group"] == g]
            line += f"{np.mean(pcts):>13.2f}%"
        print(line)


def main():
    img_dir = sys.argv[1] if len(sys.argv) > 1 else IMAGE_DIR
    if not os.path.isdir(img_dir):
        print(f"[!] ไม่พบโฟลเดอร์ '{img_dir}'")
        print(f"    กรุณาสร้างโฟลเดอร์ '{IMAGE_DIR}/' ข้างๆ สคริปต์นี้ แล้ววางภาพทดสอบ")
        print("    ตั้งชื่อตาม prefix: hard_*, tilt_*, analog_*, normal_*")
        sys.exit(1)

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    paths = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir)
                   if f.lower().endswith(exts))
    if not paths:
        print(f"[!] ไม่พบไฟล์ภาพใน '{img_dir}'")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"ทดลอง classical CV กับ {len(paths)} ภาพ จาก '{img_dir}'")

    rows = []
    for p in paths:
        run_one(p, rows)

    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "group", "method", "largest_contour_pct", "quad_area_pct",
            "found_quad", "n_contours", "quad_eps",
        ])
        writer.writeheader()
        writer.writerows(rows)

    summarize(rows, len(paths))
    print(f"\nภาพ intermediate: {OUT_DIR}/")
    print(f"ตารางผลลัพธ์:     {CSV_PATH}")


if __name__ == "__main__":
    main()
