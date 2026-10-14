# คู่มือเทรน YOLO-OBB สำหรับหมุนภาพมิเตอร์ให้ตรง

เป้าหมาย: เทรนโมเดล OBB (Oriented Bounding Box) แยกต่างหาก เพื่อหา 4 มุมของจอ/หน้าปัด
แล้วนำมาหมุนภาพให้ตรง — **โดยไม่แตะโมเดล HBB เดิม** (best.pt) ที่ทำงานหลักอยู่

---

## ภาพรวมสถาปัตยกรรม 2 โมเดล

```
                    ┌─ best.pt (HBB เดิม) ──→ crop + class ──→ Gemini   [งานหลัก]
ภาพต้นฉบับ ──┤
                    └─ best_obb.pt (OBB ใหม่) ──→ 4 มุม ──→ หมุนภาพตรง   [งานเสริม]
```

โค้ด `server.py` โหลด 2 โมเดลแยกกัน ถ้าไม่มี `best_obb.pt` ระบบทำงานปกติทุกอย่าง
แค่ข้ามขั้นหมุนภาพ (ขึ้น log ว่า "ℹ️ ไม่พบโมเดล OBB") — ปลอดภัยไว้ก่อนเสมอ

---

## ส่วนที่ 1: เตรียม Dataset OBB บน Roboflow

### 1.1 สร้าง Project ใหม่ (แยกจากตัวเก่า)
- อย่าแก้ project HBB เดิม — สร้าง project ใหม่ Type = **Object Detection**
  แล้วเลือก annotation แบบ OBB / Oriented (Roboflow รองรับ)
- หรือถ้าจะ label ในตัวเดิม ให้สร้างเป็นคนละ version เพื่อไม่ให้ปนกัน

### 1.2 อัปโหลดรูป
- เริ่ม **300-500 รูปที่หลากหลายที่สุด** พอ ไม่ต้องครบ 835
  เน้น: ภาพเอียงหลายองศา, แสงสะท้อน, พลาสติกห่อ, ทั้ง digital/analog
  (ยิ่ง label เคสยากมาก โมเดลยิ่งแกร่งกับหน้างานจริง)

### 1.3 Label เป็นสี่เหลี่ยม 4 มุม (สำคัญ)
มี 2 วิธี เลือกที่เร็วกว่า:
- **Polygon 4 จุด**: คลิกตามมุมจอทั้ง 4 (บนซ้าย→บนขวา→ล่างขวา→ล่างซ้าย)
  Roboflow แปลงเป็น OBB format ให้ตอน export
- **SAM-assisted** (แนะนำ เพราะคุณถนัด SAM อยู่แล้ว): คลิกที่จอ 1 ที
  SAM สร้าง mask ให้ แล้วแปลงเป็น polygon — เร็วกว่านั่งคลิกทีละมุมมาก

**Class**: ใช้ class เดียวพอ เช่น `screen` (เพราะงานนี้แค่หาจอมาหมุน ไม่ต้องแยก digital/analog
— การแยกประเภทเป็นหน้าที่ของโมเดล HBB เดิมกับ Gemini อยู่แล้ว)

Label ตรงไหน: **กรอบของ "พื้นที่ที่อยากให้หมุนตรง"** — คือขอบจอ LCD หรือขอบแถวตัวเลข analog
ไม่ใช่ทั้งเครื่อง เพราะเราอยากได้ภาพจอที่ตรง ไม่ใช่ตัวเครื่องที่ตรง

### 1.4 Generate Version (ปิด augmentation ที่ทำลายมุม)
- **Auto-Orient**: เปิดได้ (แก้ EXIF)
- **Resize**: 640x640 หรือ 1024x1024 ก็ได้
- **หลีกเลี่ยง augmentation ที่หมุน/พลิกภาพแบบสุ่ม** (rotation, flip) — เพราะจะทำให้
  label มุมเพี้ยน สำหรับงาน OBB ระวังเป็นพิเศษ ใช้แค่ brightness/blur/noise พอ

### 1.5 Export
- Format: **YOLOv8 Oriented Bounding Boxes** (หรือ YOLOv11 OBB)
- ดาวน์โหลดเป็น zip หรือ copy โค้ด `roboflow download` snippet

---

## ส่วนที่ 2: เทรนบนเครื่อง (RTX 4060)

### 2.1 ติดตั้ง (ถ้ายังไม่มี)
```bash
pip install ultralytics roboflow
```

### 2.2 สคริปต์เทรน (train_obb.py)
```python
from ultralytics import YOLO

# base model แบบ OBB (ลงท้าย -obb)
model = YOLO("yolo11m-obb.pt")   # medium — สมดุลความแม่น/ความเร็ว บน RTX 4060

model.train(
    data="path/to/data.yaml",   # ไฟล์จาก Roboflow export
    epochs=100,
    imgsz=640,
    batch=8,                     # 8GB VRAM: เริ่ม 8 ถ้า OOM ลดเป็น 4
    device=0,                    # RTX 4060
    patience=20,                 # early stop ถ้าไม่ดีขึ้นใน 20 epochs
    project="obb_runs",
    name="meter_obb_v1",
    # augmentation: ปิดการหมุน/พลิกที่ทำลาย label OBB
    degrees=0.0,                 # ห้ามหมุนสุ่ม
    flipud=0.0, fliplr=0.0,      # ห้ามพลิก
    # เปิดที่ปลอดภัยกับมุมได้:
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,   # สี/แสง
    translate=0.1, scale=0.3,             # เลื่อน/ซูมเล็กน้อย
)
```

### 2.3 หลังเทรนเสร็จ
- ไฟล์โมเดลอยู่ที่ `obb_runs/meter_obb_v1/weights/best.pt`
- **เปลี่ยนชื่อเป็น `best_obb.pt`** แล้ววางในโฟลเดอร์เดียวกับ `server.py`
- restart เซิร์ฟเวอร์ — จะเห็น log "📐 OBB deskew model loaded: best_obb.pt"

---

## ส่วนที่ 3: ทดสอบว่าทำงาน

1. รันเซิร์ฟเวอร์ อัปโหลดภาพเอียงที่เคยทดสอบ
2. เปิดโฟลเดอร์ `debug_images/` หาไฟล์ `..._1.5_deskewed_obb.jpg`
   → ควรเป็นภาพจอที่หมุนตรงแล้ว
3. ดู log: `📐 OBB deskew OK (conf=0.xx, out=WxH)` = สำเร็จ
   ถ้าขึ้น `ไม่พบกล่อง OBB` = โมเดลยังไม่เจอจอ (เทรนเพิ่ม/ลด conf threshold)

---

## ข้อจำกัดที่ควรรู้ (เขียนลงรายงานได้)

1. **180° ambiguity**: OBB บอกมุมเอียงได้ แต่บางครั้งไม่รู้ว่าหัว-ท้ายด้านไหน
   ภาพที่หมุนออกมาอาจกลับหัวได้ — ถ้าเจอบ่อย เติม heuristic เช็คทีหลัง
   (แต่เราพิสูจน์แล้วว่า Gemini อ่านภาพกลับหัว/เอียงได้ถูก ดังนั้นไม่กระทบ pipeline หลัก)
2. **OBB ยังเป็นสี่เหลี่ยมหมุน** ไม่ใช่สี่เหลี่ยมคางหมู — ถ้าถ่ายเฉียงมากๆ (perspective แรง)
   จะเหลือความบิดอยู่บ้าง แต่ดีกว่า classical CV ที่ทำไม่ได้เลยมาก
3. **ต้องเทรนเพิ่มถ้าเจอมิเตอร์รุ่นใหม่** ที่ไม่เคยเห็น — เป็นธรรมชาติของ learned model

---

## สรุป flow ทั้งหมด

```
Roboflow: label 4 มุม (SAM ช่วย) → export YOLOv11-OBB
   ↓
เครื่อง: train yolo11m-obb.pt → ได้ best.pt
   ↓
เปลี่ยนชื่อเป็น best_obb.pt → วางข้าง server.py → restart
   ↓
ระบบหมุนภาพอัตโนมัติ เก็บใน debug_images/ (deliverable ให้พี่เลี้ยงดู)
   ↓
โมเดล HBB เดิมยังทำงานหลักทุกอย่างเหมือนเดิม ไม่กระทบ
```
