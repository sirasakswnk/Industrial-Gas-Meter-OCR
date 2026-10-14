"""
การทดลอง: รัน YOLO-OBB กับภาพชุดเดียวกับการทดลอง Classical CV
==============================================================
สคริปต์ทดลองแยกต่างหาก — ไม่แก้ server.py (import มาใช้อย่างเดียว)

ใช้ฟังก์ชันของระบบจริงจาก server.py:
  - detect_meter_obb()      รัน best_obb.pt ด้วย OBB_CONF_THRESHOLD เดียวกับระบบ
  - deskew_from_corners()   warp 4 มุมให้ตรง (ภายในเรียก expand_corners ให้แล้ว)
  - ImageOps.exif_transpose + convert("RGB") เหมือน endpoint จริง

หมายเหตุสำคัญ: deskew_from_corners() เรียก expand_corners(OBB_WARP_PADDING) อยู่ข้างในแล้ว
จึงไม่เรียก expand_corners ซ้ำจากข้างนอก (ไม่งั้นจะเผื่อขอบสองรอบ ~12% แทน 6%)

ผลลัพธ์ -> experiment_obb/
  <ชื่อภาพ>_obb_result.jpg   วาดกรอบ OBB ทับภาพต้นฉบับ (สไตล์เดียวกับ Classical CV)
  <ชื่อภาพ>_obb_deskew.jpg   ภาพหลัง deskew (เฉพาะภาพที่ตรวจเจอ)
  compare_<ชื่อภาพ>.jpg      เทียบ 5 ช่อง: Canny | Otsu | Otsu กลับสี | Bright | YOLO-OBB
  obb_result.csv             filename, group, found, class_name, conf, box_area_pct, inference_ms
  review.csv                 14 ภาพ x 5 วิธี = 70 แถว (คอลัมน์ correct เว้นว่างให้กรอกเอง)

รัน:
    python experiment_obb.py
"""
import csv
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

import server  # ระบบจริง — import อย่างเดียว ไม่แก้ไข

IMAGE_DIR = "experiment_images"
CLASSICAL_OUT = "experiment_output"       # ผลจาก experiment_classical_cv.py
CLASSICAL_CSV = "experiment_result.csv"
OUT_DIR = "experiment_obb"
OBB_CSV = os.path.join(OUT_DIR, "obb_result.csv")
REVIEW_CSV = "review.csv"

GROUPS = ("normal", "tilt", "analog", "hard")

# 5 วิธีที่นำมาเทียบกันในภาพ compare (ลำดับซ้าย -> ขวา)
PANELS = [
    ("canny",    "Canny"),
    ("otsu",     "Otsu"),
    ("otsu_inv", "Otsu กลับสี"),
    ("bright",   "Bright"),
    ("obb",      "YOLO-OBB"),
]

CELL_W, CELL_H = 460, 420      # ขนาดช่องภาพในกริด compare
TITLE_H = 44                   # ความสูงแถบชื่อวิธีเหนือแต่ละช่อง
BG = (245, 245, 245)

# ฟอนต์ที่รองรับภาษาไทยบน Windows (ไล่ลองตามลำดับ)
FONT_CANDIDATES = [
    "C:/Windows/Fonts/leelawui.ttf",   # Leelawadee UI
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/cordia.ttc",
    "C:/Windows/Fonts/arial.ttf",      # สำรอง (ไทยจะเป็นกล่อง)
]


def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def group_of(filename):
    stem = os.path.basename(filename).lower()
    for g in GROUPS:
        if stem.startswith(g + "_"):
            return g
    return "other"


def save_view(cv_img, path, max_w=1600):
    """เซฟภาพ BGR (ย่อถ้ากว้างเกิน) — ให้เหมือนสไตล์ของ experiment_classical_cv.py"""
    if cv_img.shape[1] > max_w:
        scale = max_w / cv_img.shape[1]
        cv_img = cv2.resize(cv_img, (max_w, int(cv_img.shape[0] * scale)),
                            interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, cv_img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def draw_obb_overlay(cv_img, corners, label):
    """วาดกรอบ OBB ทับภาพ — สี/ความหนาเดียวกับ _result.jpg ของ Classical CV
    (เส้นแดง หนา max(2, w//400) + จุดมุมเหลือง รัศมี max(4, w//200))
    """
    overlay = cv_img.copy()
    thick = max(2, cv_img.shape[1] // 400)
    cv2.polylines(overlay, [corners.astype(np.int32)], True, (0, 0, 255), thick)
    for pt in corners.astype(np.int32):
        cv2.circle(overlay, (int(pt[0]), int(pt[1])),
                   max(4, cv_img.shape[1] // 200), (0, 255, 255), -1)

    # เขียน class + conf มุมซ้ายบน (ใส่พื้นหลังทึบให้อ่านออกทุกภาพ)
    scale = max(0.8, cv_img.shape[1] / 1600)
    tthick = max(2, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, tthick)
    pad = int(10 * scale)
    cv2.rectangle(overlay, (0, 0), (tw + pad * 2, th + pad * 2), (0, 0, 0), -1)
    cv2.putText(overlay, label, (pad, th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), tthick, cv2.LINE_AA)
    return overlay


def fit_cell(pil_img, w=CELL_W, h=CELL_H):
    """ย่อภาพให้พอดีช่อง (คงสัดส่วน) แล้ววางกลางพื้นหลังสีเทาอ่อน"""
    canvas = Image.new("RGB", (w, h), BG)
    r = min(w / pil_img.width, h / pil_img.height)
    nw, nh = max(1, int(pil_img.width * r)), max(1, int(pil_img.height * r))
    canvas.paste(pil_img.resize((nw, nh), Image.LANCZOS),
                 ((w - nw) // 2, (h - nh) // 2))
    return canvas


def panel_for(method, name, original_pil):
    """เลือกภาพสำหรับช่องหนึ่งของกริด compare

    ใช้ไฟล์ _result.jpg ถ้ามี (แปลว่าวิธีนั้นหาสี่เหลี่ยมเจอ)
    ถ้าไม่มี ใช้ภาพต้นฉบับพร้อมเขียนทับว่า "ไม่พบสี่เหลี่ยม"
    """
    if method == "obb":
        path = os.path.join(OUT_DIR, f"{name}_obb_result.jpg")
    else:
        path = os.path.join(CLASSICAL_OUT, f"{name}_{method}_result.jpg")

    if os.path.exists(path):
        return fit_cell(Image.open(path).convert("RGB")), True

    cell = fit_cell(original_pil)
    d = ImageDraw.Draw(cell)
    font = load_font(26)
    msg = "ไม่พบสี่เหลี่ยม"
    bbox = d.textbbox((0, 0), msg, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = (cell.width - tw) // 2, (cell.height - th) // 2
    d.rectangle([x - 14, y - 12, x + tw + 14, y + th + 14], fill=(0, 0, 0))
    d.text((x, y), msg, font=font, fill=(255, 90, 90))
    return cell, False


def build_compare(name, original_pil):
    """สร้างภาพเทียบ 5 ช่องในแถวเดียว พร้อมชื่อวิธีเหนือทุกช่อง"""
    grid = Image.new("RGB", (CELL_W * len(PANELS), CELL_H + TITLE_H), (255, 255, 255))
    d = ImageDraw.Draw(grid)
    font = load_font(24)

    for i, (method, title) in enumerate(PANELS):
        cell, _ = panel_for(method, name, original_pil)
        x = i * CELL_W
        grid.paste(cell, (x, TITLE_H))

        d.rectangle([x, 0, x + CELL_W, TITLE_H], fill=(30, 30, 30))
        bbox = d.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        d.text((x + (CELL_W - tw) // 2, 10), title, font=font, fill=(255, 255, 255))
        d.line([(x, TITLE_H), (x, TITLE_H + CELL_H)], fill=(200, 200, 200), width=2)

    grid.save(os.path.join(OUT_DIR, f"compare_{name}.jpg"), quality=90)


def main():
    if not os.path.isdir(IMAGE_DIR):
        print(f"[!] ไม่พบโฟลเดอร์ '{IMAGE_DIR}'")
        return
    os.makedirs(OUT_DIR, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    paths = sorted(os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
                   if f.lower().endswith(exts))

    if server.obb_model is None:
        print("[!] ไม่ได้โหลดโมเดล OBB — ตรวจว่ามี best_obb.pt อยู่")
        return

    # warmup ครั้งแรก (CUDA init + สร้าง graph) เพื่อให้เวลาที่วัดไม่บวมผิดปกติ
    print("warmup ...")
    server.detect_meter_obb(Image.new("RGB", (640, 480), "white"))

    print(f"\nรัน YOLO-OBB กับ {len(paths)} ภาพ (conf >= {server.OBB_CONF_THRESHOLD})\n")
    rows = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        pil = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        cv_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        img_area = pil.width * pil.height

        t0 = time.perf_counter()
        corners, cls_name, conf = server.detect_meter_obb(pil)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        if corners is None:
            print(f"  {os.path.basename(p):<16} ไม่พบกรอบ  ({infer_ms:.1f} ms)")
            rows.append({
                "filename": os.path.basename(p), "group": group_of(p),
                "found": False, "class_name": "", "conf": "",
                "box_area_pct": "", "inference_ms": round(infer_ms, 1),
            })
            build_compare(name, pil)
            continue

        box_pct = cv2.contourArea(corners.astype(np.float32)) / img_area * 100.0

        # 1) ภาพวาดกรอบทับต้นฉบับ
        label = f"{cls_name} {conf:.2f}"
        save_view(draw_obb_overlay(cv_img, corners, label),
                  os.path.join(OUT_DIR, f"{name}_obb_result.jpg"))

        # 2) ภาพหลัง deskew (deskew_from_corners เรียก expand_corners ให้อยู่แล้ว)
        deskewed = server.deskew_from_corners(pil, corners)
        if deskewed is not None:
            deskewed.save(os.path.join(OUT_DIR, f"{name}_obb_deskew.jpg"), quality=92)

        print(f"  {os.path.basename(p):<16} {cls_name:<15} conf={conf:.3f}  "
              f"box={box_pct:5.2f}%  ({infer_ms:.1f} ms)")

        rows.append({
            "filename": os.path.basename(p), "group": group_of(p),
            "found": True, "class_name": cls_name, "conf": round(conf, 4),
            "box_area_pct": round(box_pct, 2), "inference_ms": round(infer_ms, 1),
        })

        # 3) ภาพเทียบ 5 ช่อง
        build_compare(name, pil)

    with open(OBB_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "group", "found", "class_name",
                                          "conf", "box_area_pct", "inference_ms"])
        w.writeheader()
        w.writerows(rows)

    # ── review.csv: 14 ภาพ x 5 วิธี, คอลัมน์ correct เว้นว่างให้กรอกเอง ──
    classical = {}
    if os.path.exists(CLASSICAL_CSV):
        for r in csv.DictReader(open(CLASSICAL_CSV, encoding="utf-8-sig")):
            classical[(r["filename"], r["method"])] = r["found_quad"] == "True"
    else:
        print(f"[!] ไม่พบ {CLASSICAL_CSV} — คอลัมน์ found ของวิธี classical จะว่าง")

    obb_found = {r["filename"]: r["found"] for r in rows}
    review = []
    for p in paths:
        fn = os.path.basename(p)
        for method, _ in PANELS:
            found = obb_found.get(fn) if method == "obb" else classical.get((fn, method))
            review.append({
                "filename": fn, "group": group_of(fn), "method": method,
                "found": found if found is not None else "",
                "correct": "",     # <- กรอกเอง: ถูก / บางส่วน / ผิด
            })

    with open(REVIEW_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "group", "method", "found", "correct"])
        w.writeheader()
        w.writerows(review)

    # ── สรุป (เฉพาะสิ่งที่วัดได้ ไม่ตัดสินว่าวิธีไหนถูก) ──
    found_rows = [r for r in rows if r["found"]]
    print("\n" + "=" * 68)
    print("สรุปผล YOLO-OBB")
    print("=" * 68)
    print(f"ตรวจเจอ           : {len(found_rows)}/{len(rows)} ภาพ")
    if found_rows:
        confs = [r["conf"] for r in found_rows]
        times = [r["inference_ms"] for r in rows]
        print(f"conf ต่ำสุด/สูงสุด : {min(confs):.3f} / {max(confs):.3f}")
        print(f"เวลาเฉลี่ยต่อภาพ  : {np.mean(times):.1f} ms "
              f"(ต่ำสุด {min(times):.1f} / สูงสุด {max(times):.1f})")
    missed = [r["filename"] for r in rows if not r["found"]]
    print(f"ภาพที่ตรวจไม่เจอ  : {', '.join(missed) if missed else 'ไม่มี'}")

    print(f"\nไฟล์ผลลัพธ์: {OUT_DIR}/  |  {OBB_CSV}  |  {REVIEW_CSV}")
    print(f"review.csv มี {len(review)} แถว — กรอกคอลัมน์ correct เอง (ถูก / บางส่วน / ผิด)")


if __name__ == "__main__":
    main()
