"""
สนามทดลอง OBB — ดูว่าโมเดล OBB ตัวเดียวทำงานยังไงกับภาพจริง แล้วเซฟผลให้ดูด้วยตา

สถาปัตยกรรมปัจจุบันคือโมเดลเดียว (best_obb.pt) ทำครบ: ตำแหน่ง + class + มุมหมุน
สคริปต์นี้เรียกฟังก์ชันจริงจาก server.py (ไม่ก๊อปโค้ดซ้ำ) ผลที่เห็นจึงตรงกับของจริง
ไม่ยิง Gemini ไม่เสีย API — รันเล่นได้ตามใจ

รัน:
    python try_obb.py                  # ใช้ภาพทดสอบทั้งหมดที่เจอในโฟลเดอร์
    python try_obb.py test5.jpg        # เจาะจงไฟล์
    python try_obb.py a.jpg b.jpg      # หลายไฟล์

ผลลัพธ์ -> โฟลเดอร์ obb_playground/
    <ชื่อ>_1_deskew.jpg        ภาพที่ warp ดัดตรงจาก 4 มุม (แบบที่ส่ง Gemini จริง)
    <ชื่อ>_2_crop_nodeskew.jpg crop แนวแกนจากมุมชุดเดียวกัน (ไว้เทียบว่าดัดตรงช่วยไหม)
"""
import os
import sys
import glob

from PIL import Image, ImageOps

import server  # โหลด best_obb.pt ตามการตั้งค่าจริงของระบบ

OUT_DIR = "obb_playground"


def run_one(path):
    name = os.path.splitext(os.path.basename(path))[0]
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    print(f"\n{'='*62}\n📷 {os.path.basename(path)}  ({img.width}x{img.height})\n{'='*62}")

    corners, cls_name, conf = server.detect_meter_obb(img)
    if corners is None:
        print("   ⚠️ OBB ไม่เจอกรอบ -> ระบบจริงจะ fallback ส่งภาพเต็มเข้า Gemini")
        return

    print(f"   🎯 เจอ [{cls_name}] conf={conf:.3f}")
    print(f"      4 มุม (pixel): {[tuple(round(v) for v in pt) for pt in corners]}")

    # 1) ภาพที่ดัดตรง — แบบที่ส่ง Gemini จริงเมื่อ SEND_DESKEWED_TO_GEMINI=True
    deskewed = server.deskew_from_corners(img, corners)
    if deskewed is not None:
        p = os.path.join(OUT_DIR, f"{name}_1_deskew.jpg")
        deskewed.save(p, quality=92)
        print(f"   ✅ deskew {deskewed.width}x{deskewed.height} -> {p}")
    else:
        print("   ⚠️ deskew ไม่สำเร็จ -> ระบบจริงจะถอยไปใช้ crop แนวแกน")

    # 2) crop แนวแกนจากมุมชุดเดียวกัน — ไว้เทียบว่าการดัดตรงช่วยจริงไหม
    crop = server.crop_from_corners(img, corners)
    p2 = os.path.join(OUT_DIR, f"{name}_2_crop_nodeskew.jpg")
    crop.save(p2, quality=92)
    print(f"   ✅ crop   {crop.width}x{crop.height} -> {p2}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    args = sys.argv[1:]
    if args:
        targets = args
    else:
        targets = sorted(
            p for ext in ("*.jpg", "*.jpeg", "*.png")
            for p in glob.glob(ext)
            if not p.startswith(("proc_", "manual_", "ready_", "debug_"))
        )
        print(f"ไม่ได้ระบุไฟล์ — ใช้ภาพที่เจอทั้งหมด {len(targets)} ไฟล์")

    for t in targets:
        if not os.path.exists(t):
            print(f"❌ ไม่พบไฟล์: {t}")
            continue
        try:
            run_one(t)
        except Exception as e:
            print(f"❌ {t} พัง: {e}")

    print(f"\n🎉 เสร็จ — เปิดดูภาพผลลัพธ์ในโฟลเดอร์ {OUT_DIR}/")


if __name__ == "__main__":
    main()
