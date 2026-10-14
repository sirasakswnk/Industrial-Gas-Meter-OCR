# PROJECT CONTEXT — Industrial Gas Meter OCR

> เอกสารส่งต่อ context สำหรับ Claude Code / ผู้ร่วมพัฒนา / ผู้รับงานต่อ
> อ่านไฟล์นี้ก่อนเริ่มงาน จะเข้าใจว่าระบบเป็นมาอย่างไรและทำไมถึงออกแบบแบบนี้
> อัปเดตล่าสุด: 2026-10-10 (เพิ่ม ocr_engines/ + GLM-OCR engine ตัวที่สอง + ผลเปรียบเทียบ Gemini vs GLM —
> **หยุดพัฒนาแล้ว** เข้าช่วงเขียนรายงาน; ผลทดลองสำหรับบทที่ 4 อยู่ข้อ 12, โควต้า Gemini สำหรับบทที่ 5 อยู่ข้อ 13)

---

## 1. ภาพรวมโปรเจค

ระบบอ่านค่ามิเตอร์ก๊าซธรรมชาติอุตสาหกรรม (OCR) เป็น **Standalone Prototype (Sandbox)**
สำหรับโปรเจคฝึกงาน ไม่ได้ขึ้น production จริง ไม่แตะระบบหลักของโรงงาน

**Pain points ที่ระบบแก้:**
1. ภาพหน้างานไม่ได้มาตรฐาน — เอียง (±45°), ตะแคง (90°), แสงสะท้อน, คราบฝุ่น, พลาสติกห่อ
2. มิเตอร์ดิจิทัล multi-page — ค่ากระจายหลายหน้าจอ (volume/pressure/temperature) ต้องถ่ายหลายรูปแล้วรวมค่า
3. (เพิ่มหลังพรีเซนต์ครั้งแรก — พี่เลี้ยง/อาจารย์ขอ) ต้องให้คนตรวจ/แก้ค่าที่ AI อ่าน, บันทึกประวัติ,
   และออกใบแจ้งปริมาณการใช้ก๊าซ (อ้างอิงฟอร์มจริง F-ปร.บคต.-0007) ที่มีค่าครั้งก่อน/ครั้งหลัง/ผลต่าง

**สถาปัตยกรรม (ภาพรวม):**
```
[หน้าแรก /]           เลือก Meter Run (1 Run = EVC + Turbine Meter)
     ↓
[หน้า OCR /ocr?run=]  อัปโหลดภาพหลายใบ (EVC หลายหน้าจอ + Turbine 1 หน้าปัด)
     → YOLO-OBB detect (best_obb.pt) บนภาพเต็ม → 4 มุมจอ + class + conf
     → ตัวดักชนิดมิเตอร์ (analog > 1 ภาพ → 400 ไม่เรียกโมเดล)
     → warp จาก 4 มุม (เผื่อขอบ 6%) → OpenCV detailEnhance
     → OCR engine ตาม OCR_ENGINE ใน .env (ข้อ 3.13)
          gemini: Gemini (multimodal, 1 API call, batch ทุกภาพ) → Pydantic MeterReading
          glm:    GLM-OCR บนเครื่องผ่าน Ollama (1 ภาพต่อ 1 คำขอ) → ข้อความอิสระ → ชั้นแปลงป้าย (ข้อ 3.16)
     → ค่าดิบเป็นสตริง → ตาข่าย analog → ค่าเริ่มต้นช่องทีละหลัก
     → เก็บค่าดิบ + ภาพต้นฉบับไว้ฝั่ง server (ocr_requests/<request_id>/)
     ↓  ผู้ใช้ตรวจ/แก้ในช่องทีละหลัก + กรอกวันเวลาที่จด + ติ๊กตัดทศนิยม Turbine
[Export to PDF]       server คำนวณ final_values เอง → SQLite (readings) → PDF (สร้างจาก DB)
     ↓
[หน้าประวัติ /history] กรองตาม Run, ดูค่าที่โมเดลอ่าน vs ค่าที่บันทึก, export PDF ซ้ำ
```

---

## 2. Tech Stack

- **Backend**: FastAPI + Uvicorn (`server.py`)
- **Detection**: YOLO-OBB (Ultralytics) — โมเดลเทรนเอง `best_obb.pt` (yolo11s-obb, 100 epochs,
  imgsz 1024, batch 8) — mAP50 = 0.995, mAP50-95 = 0.957, Precision = 1.000, Recall = 1.000
  dataset 646 ภาพ (train 452 / val 129 / test 65), 2 class: `digital_screen`, `analog_dial`
- **OCR/VLM** (เลือกด้วย `OCR_ENGINE` ใน `.env`, ค่าเริ่มต้น `gemini`):
  - Gemini ผ่าน google-genai SDK — ชื่อโมเดลตั้งใน `.env` (`GEMINI_MODEL`, ใช้ `gemini-3.5-flash`), free tier
  - GLM-OCR (`glm-ocr:latest`, Ollama รายงานขนาด 1.1B F16) รันบนเครื่องผ่าน Ollama 0.33.3
    endpoint `/v1/chat/completions` — ไม่ต้องใช้อินเทอร์เน็ต/API key/โควต้า
- **ฐานข้อมูล**: SQLite ไฟล์เดียว `meter_records.db` (sqlite3 ของ standard library — **ห้ามใช้ PostgreSQL**)
- **PDF**: fpdf2 + uharfbuzz (text shaping ภาษาไทย) + ฟอนต์ Sarabun ฝังในไฟล์ (`fonts/`, ไลเซนส์ OFL)
- **Image**: OpenCV, Pillow, NumPy
- **Frontend**: HTML + CSS + Vanilla JS 3 หน้า (เลือก Run / OCR / ประวัติ) — dark mode, ไม่มี framework
- **เครื่องรัน**: local, i9-14900HX + RTX 4060 (8GB VRAM), Python 3.10
- **ติดตั้ง**: `pip install -r requirements.txt` (torch เป็น build CUDA 12.1 — เครื่องไม่มี GPU ดูหมายเหตุในไฟล์)

---

## 3. การตัดสินใจสำคัญ + เหตุผล (อย่าย้อนกลับโดยไม่มีเหตุผลใหม่)

### 3.1 ใช้ VLM (Gemini) อ่านค่า แทน OCR ดั้งเดิม
เพราะจอ LCD 7-segment + แสงสะท้อน ทำ Tesseract/PaddleOCR พังง่าย VLM ทนกว่ามาก
**ผลทดสอบสำคัญ**: Gemini อ่านภาพเอียง 45°/90° ได้ถูกต้องโดยไม่ต้อง deskew
→ deskew จึงไม่ "จำเป็น" ต่อความแม่นยำ แต่ตอนนี้เทรน OBB เสร็จ จึงเปิดส่งภาพ deskew เข้า Gemini
เป็น default (`SEND_DESKEWED_TO_GEMINI=True`) ตั้ง False กลับไปส่ง crop ไม่ดัดตรงได้เพื่อทำ eval A/B

### 3.2 กัน hallucination เป็นเรื่องหลัก
ค่ามิเตอร์ผิดอันตรายกว่าค่าที่หายไป ทุกฟิลด์เป็น Optional + prompt กำชับ "ไม่เห็นค่า = null ห้ามเดา"

### 3.3 ฟีเจอร์ low_confidence_fields
เมื่อ Gemini เห็นค่าแต่ไม่ชัด (แสงสะท้อน) → **ยังอ่านค่ามาให้** แต่ใส่ชื่อฟิลด์ใน `low_confidence_fields`
→ หน้า OCR ขึ้นป้ายเหลือง "ตรวจสอบ" ผู้ใช้ตัดสินใจเอง
**ข้อควรรู้**: เป็น self-reported confidence — "low" เชื่อได้ แต่ "ไม่ low" ไม่การันตีถูก
**GLM**: `low_confidence_fields` เป็นค่าว่างเสมอ (โมเดลไม่รายงานเอง และ confidence จาก logprobs
ไม่ถูกเอามาใช้แทน — ดูข้อ 3.17)

### 3.4 สถาปัตยกรรมโมเดลเดียว (YOLO-OBB) — แทนของเดิมที่ใช้ 2 โมเดล
**เดิม**: HBB (`best.pt`) หาตำแหน่ง+crop+class แล้วส่งต่อให้ OBB deskew บน crop นั้น
**ตอนนี้**: `best_obb.pt` ตัวเดียว รันบนภาพต้นฉบับเต็ม ได้ตำแหน่ง (4 มุม) + class + มุมหมุน แล้ว warp ตรงๆ

**เหตุผลที่ถอด HBB ออก (มีหลักฐานรองรับ):**
- ทดสอบด้วยมือ 10+ ภาพ (ภาพตรงและเอียง 45°) → OBB ตัวเดียว crop ได้ **ดีกว่า** HBB+OBB
- OBB ให้ทั้งตำแหน่ง/class/มุม ได้ในโมเดลเดียว → HBB กลายเป็นขั้นซ้ำซ้อน, inference เร็วขึ้น

**fail-safe**: OBB ไม่เจอกรอบ / warp พัง → ใช้ภาพเดิมต่อ ไม่ throw
**`best.pt` ไม่ถูกลบ** เก็บไว้เผื่อเทียบย้อนหลัง (ไม่ถูกโหลด, อยู่ใน .gitignore)

### 3.5 หลักการที่ยึดตลอดโปรเจค
- **Evidence-driven**: ทดสอบก่อนสรุป ไม่เดา
- **Fail-safe**: ฟีเจอร์เสริมพังต้องไม่ทำ pipeline หลักพัง
- **แก้ทีละอย่าง**: เปลี่ยนหลายอย่างพร้อมกันแล้วจะไม่รู้ว่าอะไรทำให้ผลต่าง
- **งานที่แตะ UI ต้องเปิด browser ตรวจจริงก่อนปิดงาน** — เทสต์ API ผ่านไม่พอ
  (บทเรียน: หน้าประวัติเคยตารางว่างเพราะ CSS เก่า `opacity: 0` ซึ่งเทสต์ API จับไม่ได้)

### 3.6 ตัวดักชนิดมิเตอร์ในคำขอเดียว — **เปลี่ยนแล้ว** (ผ่อนจากกฎเดิม)
**กฎเดิม (ยกเลิก)**: OBB พบ analog + digital ในคำขอเดียว → 400
เหตุผลเดิม: prompt บอกว่าทุกภาพเป็น "มิเตอร์เครื่องเดียวกัน" ค่าคนละเครื่องจะถูกรวมเงียบๆ

**เหตุผลใหม่ที่เปลี่ยน**: เพิ่มแนวคิด Meter Run แล้ว — **1 Run = EVC (digital) + Turbine Meter (analog)**
ต้องส่งมาด้วยกันในการจดครั้งเดียว และ prompt ข้อ 5 (Device-to-Field Mapping) บังคับให้ค่าจาก
ลูกกลิ้ง analog ลงได้แค่ `turbine_v_m3`

**กฎปัจจุบัน** — `check_request_meter_types()` (ใช้ร่วมกันทั้ง `/api/ocr` และ `evaluate.py`):
- analog + digital มาด้วยกัน → ผ่าน
- digital หลายภาพ → ผ่าน (เช่น Corus หลายหน้าจอ)
- **analog มากกว่า 1 ภาพ → 400 ไม่เรียก Gemini** ("พบมิเตอร์ analog มากกว่า 1 ตัวในคำขอเดียว
  หนึ่งรายการจดรองรับ Turbine Meter ได้ 1 ตัว (รูปที่ ...)") — Turbine มี 1 หน้าปัดต่อ Run เสมอ
- ภาพที่ OBB หาไม่เจอ (None) ไม่ถูกนับ
- วางก่อนบล็อก `try` ของ Gemini (ถ้าอยู่ใน `except Exception` จะถูกแปลงเป็น 500)

**ข้อจำกัด**: ดักไม่ได้ถ้าเป็น EVC คนละ Run (digital class เหมือนกัน), ถ้า OBB จัด class ผิดอาจบล็อกผิด/ปล่อยผ่าน

### 3.7 ตาข่าย analog หลัง parse (ชั้นสำรองของ prompt ข้อ 5)
`apply_analog_only_guard()` — คำขอที่**ไม่มีภาพ digital เลย** (ตาม class ของ OBB) และ (OBB เจอ analog
หรือ `meter_type = analog`) แต่โมเดลใส่ค่าฟิลด์ EVC มา → ทิ้งเป็น null + log warning (ไม่ error)
- **ใช้ class จาก OBB ไม่ใช่ `meter_type` อย่างเดียว**: Gemini ตอบก้อนเดียวต่อคำขอ ถ้าคำขอปน
  EVC + Turbine แล้วได้ `meter_type = analog` การดูแค่ meter_type จะทิ้งค่า EVC ที่ถูกต้องทั้งหมด
  OBB รู้ class เป็นรายภาพ จึงตัดสินได้ถูกกว่า
- คำขอที่มี digital ปน ตาข่ายนี้ไม่แตะ (แยกไม่ได้ว่าค่าไหนมาจากภาพไหน) → พึ่ง prompt ข้อ 5
- ค่าดิบเดิมของฟิลด์ที่ถูกทิ้งเก็บใน `guard_dropped` (result.json) ไว้ตรวจย้อนหลัง
- response มี `guard_applied` → หน้า OCR ขึ้นแถบ "ตรวจพบเฉพาะมิเตอร์ analog ในคำขอนี้ ค่าของ EVC
  จึงถูกตัดออก..." และช่องที่ถูกตัดมีป้าย "ตัดออก" (ไม่ใช่ "ไม่พบค่า" — กันผู้ใช้เข้าใจว่าอ่านไม่ออก)

### 3.8 ค่ามิเตอร์เป็น "สตริงตัวเลข" ตลอดทาง ไม่ผ่าน float
- `str(float)` เติม ".0" ให้จำนวนเต็ม → 1123120.0 → ลบจุด "11231200" → ตัด Turbine ผิดหลัก
- `f"{v:.0f}"` ปัดเศษ → 12345.67 → "12346" → ตัด 2 หลักได้ "123" (สเปคต้องได้ "12345")
- **raw_values** = ข้อความที่โมเดลเขียนมาเป๊ะ: `json.loads(response.text, parse_float=str, parse_int=str)`
  ("4.500" ไม่กลายเป็น 4.5) — ลอก markdown fence ก่อน; ถ้าอ่านข้อความไม่ได้ → fallback จาก float
  (`raw_source = "fallback"` → หน้า OCR ขึ้นเตือน เพราะศูนย์ท้ายทศนิยมอาจหาย เช่น 12345.60 → 12345.6)
- **final_values** = ทศนิยมตายตัวตามตารางฟิลด์ (เติม 0 ถ้าขาด, ตัดทิ้งไม่ปัดถ้าเกิน)
- ผลต่างใน PDF คำนวณด้วย `decimal.Decimal`

### 3.9 server คือผู้ตัดสินค่าที่บันทึก (ไม่เชื่อค่าจาก frontend)
- ตอน OCR server เก็บ raw_values + low_confidence + ภาพต้นฉบับไว้ที่ `ocr_requests/<request_id>/`
- ตอนบันทึก frontend ส่งแค่ "ค่าในช่องทีละหลัก" + จำนวนหลักที่ตัด → server คำนวณ
  `final_values` / `edited_fields` เองด้วย `reading_fields.py` → ค่าที่บันทึกกับที่แสดงตรงกันเสมอ
- raw_values/ภาพดึงจากที่ server เก็บ — frontend แก้หลักฐานไม่ได้; บันทึกผลเดียวกันซ้ำ → 409

### 3.10 PDF สร้างจาก DB ทุกครั้ง (ไม่เก็บไฟล์ PDF)
- export ซ้ำจากหน้าประวัติได้ผลตรงกันเสมอ
- "ค่าครั้งก่อน" ใน PDF = การจด**ก่อนหน้ารายการนั้น** ของ Run เดียวกัน (ตัดสินจาก `recorded_at`)
  ไม่ใช่ค่าล่าสุดของ Run — export รายการเก่าซ้ำหลังมีรายการใหม่แล้วก็ยังถูก

### 3.11 การตัดทศนิยมเป็นพฤติกรรมปกติ → ไม่เตือนบนจอ (log อย่างเดียว)
จอ EVC จริงแสดงทศนิยม (เช่น CORRECT V. 47715600.121) แต่ฟอร์มจริงจดเป็นจำนวนเต็ม
ถ้าเตือนทุกครั้งผู้ใช้จะเลิกอ่านคำเตือนทั้งหมด รวมถึงอันสำคัญ (ตาข่าย analog, fallback)
**ยกเว้น** คำใบ้แถว Turbine "โมเดลอ่านได้ทศนิยม N หลัก — ตรวจว่าติ๊กตัดทศนิยมถูกต้อง" คงไว้
เพราะช่วยตัดสินใจติ๊ก checkbox (ติ๊กผิด = เลขผิด)

### 3.12 ผลต่างค่าสะสมติดลบ = มีอะไรผิดเสมอ → เตือน แต่ไม่บล็อก
มาตรวัดปริมาตรสะสม (CORRECT V., LINE V., TURBINE V.) เดินหน้าอย่างเดียว ไม่มีทางลดลง
ผลต่างติดลบจึงเป็นสัญญาณผิดพลาดเสมอ — จดคนละมิเตอร์ / อ่านเลขผิด / เลือก Run ผิด
(ที่มา: PDF ทดสอบรายการ #3 ผลต่าง CORRECT V. = -7218314 เพราะใช้รูปจากมิเตอร์คนละตัว — ไม่ใช่บั๊ก
แต่ระบบควรบอกผู้ใช้ก่อนบันทึก)
- **server ตัดสิน**: `reading_fields.decreased_fields(final, prev_final)` + endpoint
  `POST /api/meter-runs/{id}/decrease-check` คำนวณ final ด้วย `compute_final_values` ตัวเดียวกับตอนบันทึก
  → Turbine เทียบ**หลังตัดทศนิยม** ตรงกับค่าที่จะลง PDF; เทียบด้วย Decimal (ไม่ใช่เทียบสตริง)
- หน้า OCR เรียกแบบ debounce (200 ms) ทุกครั้งที่แก้ช่อง/ติ๊ก checkbox → แถบเตือนเหนือปุ่ม Export
  1 บรรทัดต่อฟิลด์: "CORRECT V. มีค่าน้อยกว่าครั้งก่อน ตรวจสอบว่าเป็นมิเตอร์ตัวเดียวกันหรืออ่านค่าถูกต้องหรือไม่"
  + ครั้งนี้ / ครั้งก่อน / ผลต่าง
- **ไม่บล็อกการบันทึก** (ปุ่ม Export กดได้ตามปกติ); PDF แสดงผลต่างตามจริง ไม่มีเครื่องหมายพิเศษ
- ไม่เตือน: ผลต่างเป็นศูนย์ (ไม่ได้ใช้ก๊าซ) / ไม่มีค่าครั้งก่อน (Run ใหม่ — หน้า OCR ไม่เรียก API เลย) /
  ครั้งนี้ไม่มีค่า / ค่าไม่สะสม (Pressure ฯลฯ)
- เทียบกับ "ค่าจดครั้งก่อน" ที่หน้า OCR แสดง (รายการล่าสุดของ Run) — ถ้าวันเวลาที่จดเก่ากว่ารายการล่าสุด
  มีคำเตือนวันเวลาแยกอยู่แล้ว

### 3.13 สถาปัตยกรรม OCR engine (`ocr_engines/`) — สลับโมเดลได้ด้วยค่าเดียวใน .env
**เหตุผล**: เดิมโค้ด Gemini กระจายอยู่ใน server.py ทำให้เปลี่ยน/เทียบโมเดลไม่ได้ จึงแยกการเรียกโมเดลออกมา
เป็นชั้นเดียว ส่วนอื่นทั้งหมด (preprocess/OBB, ตัวดัก 3.6, ตาข่าย analog 3.7, ค่าดิบสตริง 3.8,
reading_fields, DB, PDF, UI) ใช้ร่วมกันไม่ว่าจะเลือก engine ไหน

**โครงสร้าง:**
| ไฟล์ | หน้าที่ |
|------|--------|
| `ocr_engines/__init__.py` | registry `ENGINES = {"gemini": ..., "glm": ...}`, `create_engine(name)`, `DEFAULT_ENGINE = "gemini"`, `UnknownEngineError` |
| `ocr_engines/base.py` | `MeterReading` (Pydantic, ย้ายมาจาก server.py ตรงตัว), `EngineResult`, `OcrEngine` (ABC), ตัวช่วยค่าดิบ (`strip_json_fence`, `load_json_as_strings`, `extract_raw_values`) |
| `ocr_engines/prompts.py` | prompt ของ Gemini (`SYSTEM_INSTRUCTION`, `build_user_prompt` — ย้ายมาตรงตัว) + prompt สั้นของ GLM (`GLM_EVC_PROMPT`, `GLM_ANALOG_PROMPT`) |
| `ocr_engines/gemini_engine.py` | `GeminiEngine` — 1 API call ต่อคำขอ (ทุกภาพ), response_schema = MeterReading, fail-fast ถ้าไม่มี `GEMINI_API_KEY` |
| `ocr_engines/glm_engine.py` | `GlmEngine` — เรียก Ollama 1 ภาพต่อ 1 คำขอ, คำนวณ confidence จาก logprobs (ข้อ 3.14, 3.17) |
| `ocr_engines/glm_labels.py` | ชั้นแปลง "ป้าย → ฟิลด์" ของ GLM ไม่มี I/O เทสต์ได้ตรงๆ (ข้อ 3.16) |

**Interface:**
```python
class OcrEngine(ABC):
    name: str                                   # ลงคอลัมน์ readings.engine
    def read(self, images: List[PIL.Image], detected_classes: List[Optional[str]]) -> EngineResult

@dataclass
class EngineResult:
    raw_values: dict             # 8 ฟิลด์ เป็นสตริงตามที่โมเดลเขียน (None = ไม่พบ)
    meter_type: str
    detected_format: str
    confidence_scores: Optional[dict]   # gemini = None, glm = {ฟิลด์: 0..1 หรือ None}
    raw_text: Optional[str]      # ข้อความดิบจากโมเดล ไว้ตรวจย้อนหลัง
    engine: str
    raw_source: str              # "text" | "fallback"
    reading: MeterReading        # ค่าแบบ float สำหรับ response /api/ocr เดิม
```
- `images` = ภาพหลัง preprocess เรียงตามที่อัปโหลด, `detected_classes` = class จาก OBB รายภาพ
- server.py เรียกแค่ `engine.read(...)` ใน `/api/ocr` กับ `/api/ocr/sample` แล้วส่งต่อ `build_ocr_result(engine_result, classes)`
- **เลือก engine**: `OCR_ENGINE` ใน `.env` (ไม่ตั้ง = `gemini`, ไม่สนตัวพิมพ์เล็กใหญ่) — ชื่อที่ไม่รู้จัก →
  server **ไม่ยอมสตาร์ท** พร้อมรายชื่อที่เลือกได้ (fail-fast ไม่ fallback เงียบๆ)
- import แบบ lazy: เลือก glm แล้วไม่ต้องมี `GEMINI_API_KEY`, เลือก gemini แล้วไม่ต้องมี Ollama
- เพิ่ม engine ใหม่ = เขียนคลาสสืบจาก `OcrEngine` + ลงทะเบียนใน `ENGINES` บรรทัดเดียว
- ชื่อ engine ถูกเก็บใน result.json ตอน OCR แล้วลง `readings.engine` ตอนบันทึก (`NOT NULL` ไม่มีค่าเริ่มต้น —
  `insert_reading` ไม่ส่ง engine = error ทันที กันข้อมูลเงียบๆ ว่าอ่านด้วยโมเดลไหน)
- **เฟส A ย้ายโค้ดโดยไม่เปลี่ยนพฤติกรรม**: prompt/parse/ตัวดัก/ตาข่ายไม่ถูกแก้, เทสต์ 7 ชุดเดิมผ่านโดยไม่แก้เทสต์
  (server.py ยัง export ชื่อเดิม เช่น `SYSTEM_INSTRUCTION`, `extract_raw_values` ให้เทสต์/evaluate.py ใช้ได้)
- UI/ข้อความที่ผู้ใช้เห็นไม่มีคำว่า "Gemini" (meta description, FastAPI title เป็นคำกลาง)

### 3.14 GLM-OCR: ผล spike 3 กรณี → ทำไมไม่ใช้ json_schema
spike ยิง GLM-OCR ด้วยภาพ `bin/test5.jpg` (Corus หน้า VALUE: P = 9.656 bar, T = 25.00 °C, Zb/Z = 1.0171,
C = 9.3648, Qm = 477.0 m³/h, Qb = 4467.4 Sm³/h) โดยเปิด logprobs ทั้ง 3 กรณี:

| กรณี | วิธีบังคับรูปแบบ | ผล |
|------|-----------------|-----|
| A | ไม่บังคับเลย | โมเดล**วนซ้ำไม่หยุด 7,397 token** |
| B | `response_format = json_object` | **ค่าถูกครบ** แต่โมเดลตั้งชื่อ key/schema เอง |
| C | `response_format = json_schema` (สิ่งที่ตั้งใจใช้ตอนแรก) | JSON ถูกรูปแบบแต่**ค่าผิด**: pressure_bar = 0, temperature_c = 0, เอาอุณหภูมิไปใส่ line_v_m3; logprob ของ token ที่ถูกบังคับต่ำระดับ 0.0003 |

**ข้อสรุป**: ตอนปล่อยอิสระ โมเดลอ่านจอถูกจริง (ตรวจไขว้: Qb/Qm = 4467.4/477.0 = 9.365 ตรงกับ C = 9.3648)
ปัญหาอยู่ที่**รูปแบบ output ไม่ใช่ความสามารถอ่าน** — grammar ของ json_schema ฝืนโมเดลจนค่าเพี้ยน และที่อันตรายคือ
ผลลัพธ์ "ดูสำเร็จ" (JSON ถูก schema) ทั้งที่ค่าผิดและลงผิดฟิลด์ → **ห้ามใช้ json_schema กับ GLM เด็ดขาด**
ให้โมเดลถอดข้อความอิสระ แล้วให้โค้ดของเรา (ข้อ 3.16) แปลงป้ายเป็นฟิลด์เอง

**พารามิเตอร์ที่ใช้** (`glm_engine._chat`):
- `POST {OLLAMA_BASE_URL}/v1/chat/completions`, `temperature 0`, `logprobs true`, `top_logprobs 3`
- `max_tokens 256` — Ollama แปลงเป็น num_predict กันอาการวนซ้ำแบบกรณี A
  (endpoint `/v1` **ไม่สน `options`** จึงตั้ง num_predict/num_ctx ผ่าน options ไม่ได้ — ตรวจแล้ว)
- num_ctx ใช้ค่าของโมเดล 8192 — เลข 2048 ในแผนเดิมมาจากสคริปต์โมเดล 7B (กัน VRAM) ไม่จำเป็นกับโมเดลเล็กบน 4060;
  ไม่ทำ Modelfile, ไม่ย้ายไป `/api/chat` (ยังไม่ยืนยันว่ารองรับ logprobs)
- **1 ภาพต่อ 1 คำขอ** (ภาพ JPEG q95) แล้วรวมผลหลายภาพเอง — ฟิลด์เดียวกันจากหลายภาพค่าต่างกัน → null
- ถูกตัดที่ max_tokens (`finish_reason = "length"`) → ทิ้งบรรทัดสุดท้ายที่ขาดกลาง (`complete_lines`)
  (เคยเจอ `P = 9.` ครึ่งบรรทัดไปชนกับค่าเต็มแล้วกลายเป็น conflict)
- Ollama ไม่ได้เปิด → error อ่านรู้เรื่อง "ติดต่อ Ollama ที่ ... ไม่ได้ — กรุณาเปิด `ollama serve` แล้วลองใหม่"
  (server ไม่ล่ม แค่คำขอนั้น error), ไม่มีโมเดล → บอกให้ `ollama pull glm-ocr:latest`

### 3.15 ข้อค้นพบ: GLM-OCR "ถอดข้อความตามจอ" ไม่ทำตามคำสั่งรูปแบบ
GLM-OCR เป็นโมเดลถอดข้อความ (OCR) ไม่ใช่โมเดลทำตามคำสั่ง หลักฐานที่พบจริง:
- **ลอก prompt ออกมาเป็นคำตอบ (prompt leakage)**: prompt รุ่นแรกมีตัวอย่าง `LABEL = VALUE UNIT` และ
  `Counter = DIGITS` → ใน 66 รูป โมเดลเขียน "LABEL" ออกมา 8 ครั้ง และ "Counter = DIGITS" 7 ครั้ง;
  ยังพบประโยคที่ไม่มีใน prompt เช่น "Answer the question using only one word or a phrase." โผล่ในคำตอบ
- **ไม่ทำตามรูปแบบที่สั่ง**: สั่ง "ตัวเลขอย่างเดียว ไม่มีช่องว่าง ไม่มีจุด" กับ analog แต่ได้ `0068224;4`,
  `0 0 0 3 3 8 9 7 6`, `48818849=`, `05659261m³`; จอ EVC ถอดตามที่เห็นบนจอ (ป้าย `Vb`/`Vm`/`p1`/`VkT`/`Q`,
  หน่วย `Sm²`) และมักพิมพ์ทั้งจอซ้ำหลายรอบ/ใส่ markdown fence
- **ใส่คำสั่งเพิ่มแก้ไม่ได้**: เติม "Use a dot for decimals" กลับเข้าไป S05 ก็ยังจุดทศนิยมหายเหมือนเดิม

**ผลต่อการออกแบบ:**
- prompt ของ GLM สั้น และ**ไม่มีตัวเลข ไม่มี "=" หรือ ":" ไม่มี placeholder** — ถ้าโมเดลลอก prompt ออกมาทั้งย่อหน้า
  ก็ต้องไม่เกิดค่าใดๆ (เทสต์ล็อกไว้ใน `test_glm_engine.py`) จึง**ไม่ใช้ตัวอย่างค่าจริง** อย่าง "P = 9.656 bar"
  เพราะจะถูกลอกออกมาเป็นค่าที่ดูเหมือนอ่านสำเร็จ
- กฎเนื้อหาตรงกับ Gemini (ไม่เห็นให้เว้น ห้ามเดา) แต่**ความปลอดภัยของค่าอยู่ที่ชั้นแปลงป้าย ไม่ใช่ที่ prompt**
```
GLM_EVC_PROMPT    = "Read the text on this gas meter display.
                     Copy every label, number, decimal point and unit exactly as shown, one item per line.
                     If something is not visible, leave it out. Do not guess."
GLM_ANALOG_PROMPT = "Read the counter digits on this gas meter from left to right, including red or highlighted digits.
                     Write only the digits, with no spaces and no decimal point.
                     If the digits cannot be read, write nothing. Do not guess."
```
- analog: เฟสนี้อ่านเป็นตัวเลขอย่างเดียว ไม่มีกฎช่องแดง (ตั้งใจวัดว่าโมเดลขนาดนี้ทำได้แค่ไหน)

### 3.16 กฎเหล็กการจับคู่ป้าย → ฟิลด์ (`glm_labels.py`)
> **จับคู่ค่ากับฟิลด์ได้ก็ต่อเมื่อ "รู้จักป้ายนั้นในตารางแล้ว" เท่านั้น ห้ามจับคู่ด้วยตำแหน่งเพียงอย่างเดียวในทุกกรณี**

**เหตุผล**: ค่าที่จับคู่ผิดอันตรายกว่า null เพราะมัน "ดูเหมือนสำเร็จ" แล้วถูกบันทึกลง DB/PDF
(เหมือน spike กรณี C ที่เอาอุณหภูมิไปใส่ line_v_m3) ส่วน null ผู้ใช้เห็นทันทีว่าต้องกรอกเอง

**ตารางป้าย** (ตรงตัวหลัง normalize ตัวพิมพ์/ช่องว่าง/จุดท้าย เท่านั้น):
| ฟิลด์ | ป้าย |
|-------|------|
| pressure_bar | P, Pressure |
| temperature_c | T, Temp, Temperature |
| correction_factor | C, Factor, Correction, Correction Factor |
| qm_m3_h / qb_sm3_h | Qm / Qb |
| correct_v_scm | Converted Index, Vb, Converted Vol(ume), CV Idx, *Convened Index* (alias) |
| line_v_m3 | Unconverted Index, Vm, Unconverted Vol(ume), UV Idx, *Unconvened Index* (alias) |
| turbine_v_m3 | Counter |
- **ทิ้งแน่นอน**: Zb/Z, Z/Zb, Zb, Z — ไม่ใช่ correction_factor (spike พบว่าโมเดลสับสนสองค่านี้) เราไม่มีฟิลด์นี้
- **ป้ายที่ไม่รู้จัก → ทิ้ง + log** ห้ามเดา (ที่พบจริง: Bat, K, K1, Q, p1, VkT, Year, IP ฯลฯ — Bat/K/Q/Pulse weight
  ตัดสินใจไม่เพิ่ม เพราะไม่มีฟิลด์รองรับ/กำกวม)
- **ห้าม fuzzy matching / edit distance เด็ดขาด**: "Converted" กับ "Unconverted" ต่างกันแค่ "Un" ถ้าเปิด fuzzy
  ระบบจะสลับยอดสะสมสองตัวที่ใช้คิดเงินได้ → คำที่ OCR สะกดผิด (Convened) เพิ่มเป็น alias ทีละคำอย่างชัดเจนเท่านั้น
  (มีเทสต์ยืนยันว่าคำใกล้เคียงที่ไม่อยู่ในตารางไม่ถูกจับคู่ และ Converted/Unconverted ไม่สลับกัน)
- ค่าที่ป้ายบอกว่าเป็นฟิลด์ที่อุปกรณ์นี้ให้ไม่ได้ (เช่น Counter บนภาพ digital) → ทิ้ง (`not_allowed`)
- ฟิลด์เดียวกันได้หลายครั้ง: ค่าเหมือนกัน (โมเดลพิมพ์ซ้ำ) → ใช้ค่านั้น, ค่าต่างกัน → null (ไม่เลือกให้)
- แปลงเป็นตัวเลขไม่ได้ → null ไม่ throw

**เลย์เอาต์ที่รองรับ** (ทุกแบบต้องรู้จักป้ายก่อน):
1. `ป้าย = ค่า หน่วย` / `ป้าย: ค่า` — แบบหลัก
2. `ป้าย:` อยู่บรรทัดเดียว แล้วค่าอยู่**บรรทัดถัดไปทันที** (ต้องขึ้นต้นด้วยตัวเลข; ถ้าเป็นป้ายอีกตัว = ป้ายแรกไม่มีค่า)
   — จอ Corus หน้า INDEX ถอดออกมาแบบนี้
3. `ป้าย ค่า หน่วย` บรรทัดเดียวไม่มี "=" — ป้ายยาวเกิน 30 ตัวอักษรไม่นับ (กันประโยคที่โมเดลแต่งเอง);
   เก็บคู่แรกของบรรทัดเท่านั้น
4. **ข้อยกเว้นเดียวที่ไม่มีป้าย — ตัวเลขเดี่ยวบนภาพ analog**: OBB บอกว่าเป็น analog และบรรทัดเป็นตัวเลขล้วน
   (ยอมให้มีหน่วย m³/m3 ต่อท้าย) → ลง turbine_v_m3 ได้ เพราะ**กฎอุปกรณ์กำหนดฟิลด์ไว้แล้ว** (analog มีได้ฟิลด์เดียว)
   ไม่ใช่การเดาจากตำแหน่ง; ตัวเลขเดี่ยวหลายชุดที่ค่าต่างกัน → null + log (กำกวม), เลขเดิมซ้ำหลายบรรทัดนับเป็นชุดเดียว

**ราคาของการกรองเข้มงวด (ยอมรับแล้ว) — 3 รูป analog ที่ไม่ได้ค่า:**
| รูป | GLM เขียนมา | ทำไมถูกทิ้ง |
|-----|------------|------------|
| img7 | `0068224;4` | มี `;` แทรก — ไม่ใช่ตัวเลขล้วน |
| analog001 | `0 0 0 3 3 8 9 7 6` | มีช่องว่างคั่นทุกหลัก |
| test7 | `48818849=` (+ บรรทัด `Answer: 48818849`) | "=" ห้อยท้าย; "Answer" ไม่ใช่ป้ายในตาราง |

โมเดล**อ่านเลขถูกทั้ง 3 รูป** แต่รูปแบบมีอักขระแปลกปน จึงไม่ผ่านกฎตัวเลขเดี่ยว — เป็นผลของการเลือกกรองเข้มงวด
ซึ่ง**ยอมเสียค่าที่อ่านถูกบ้าง ดีกว่ารับค่าที่อ่านผิดเข้ามา** (ตัดสินใจไม่แก้: ถ้าผ่อนให้ลบ `;`/ช่องว่าง/`=`
จะเริ่มเดาว่าอักขระไหน "ไม่สำคัญ" — `;4` อาจเป็นหลักทศนิยมหรือขยะก็ได้) แบบเดียวกันกับ EK280 (img1) ที่โมเดลถอด
ยอด Vb/Vm ถูกแต่ไม่ถอดป้าย → ค่าไม่ถูกจับคู่ (ข้อ 12.2)

### 3.17 confidence จาก logprobs: คำนวณและเก็บ แต่ไม่แสดง
**วิธีคำนวณ** (`glm_engine.value_confidence`): หา token ที่ทับช่วงตัวอักษรของ "ค่า" (ไม่รวม token ป้าย/หน่วย)
โดยเทียบตำแหน่งระดับ byte → confidence = **ค่าต่ำสุด** ของ exp(logprob) ในกลุ่มนั้น (ใช้ min ไม่ใช่ค่าเฉลี่ย
เพราะเลขผิดตัวเดียวก็ผิดทั้งค่า); แมป token กลับไม่ได้ → None (ห้ามเดา)
- บั๊กที่เจอและแก้แล้ว: Ollama ตัดช่องว่างท้ายข้อความ (และ byte UTF-8 ที่ไม่ครบ) ออกจาก content แต่ token ยังมี
  → เดิม confidence เป็น None 25/65 ค่า; ตอนนี้ยอมเฉพาะส่วนเกินแบบนี้ที่ท้ายสุด (มีเทสต์ทั้งกรณียอมรับ/ปฏิเสธ)

**ผลวัด (GLM 66 รูป, 65 ค่า ตรวจกับภาพทุกค่า: ผิด 10, ไม่แน่ใจ 2, ที่เหลือ 53 ถูก):**
| | จำนวน | ต่ำสุด | มัธยฐาน | สูงสุด | ≥ 0.9 |
|---|---|---|---|---|---|
| ค่าที่ผิด | 10 | 0.570 | 0.823 | 1.000 (S12 = 0.9997) | 5/10 |
| ค่าที่ถูก | 53 | 0.508 | 0.999 | 1.000 | 46/53 |

| เกณฑ์ (ต่ำกว่านี้ = เตือน) | จับค่าผิดได้ | เตือนค่าถูกโดยไม่จำเป็น |
|---|---|---|
| 0.6 | 2/10 | 1/53 |
| 0.9 | 5/10 | 7/53 |
| 0.99 | 9/10 | 14/53 (26%) |
- ค่าผิดที่ confidence สูงมาก: S12 turbine (หลักที่กำลังหมุนหาย) 0.9997, S19 0.985, S10 Vm 0.951 (หลักแรก 0→8)
- ไม่แน่ใจ 2 ค่า: img4 0.209 (ภาพกลับหัว), S01_p1 0.929 — ไม่นับในตาราง
- โดยรวมค่าผิดมีแนวโน้มต่ำกว่า (มัธยฐาน 0.82 vs 0.999) แต่**ไม่มีเกณฑ์ไหนแยกได้**: เกณฑ์ต่ำจับค่าผิดได้ครึ่งเดียว
  เกณฑ์สูงเตือนค่าถูกถึง 1 ใน 4 และค่าผิดบางตัวได้ ~1.0 อยู่ดี

**การตัดสินใจ: ยกเลิกแผนแสดง/ไฮไลต์ confidence บนหน้า OCR** — เหตุผล: ค่าความมั่นใจสูงบนคำตอบที่ผิด
อันตรายกว่าการไม่มีค่าเลย (ผู้ใช้จะเชื่อแล้วไม่ตรวจ) ตรงข้ามกับเป้าหมายข้อ 3.2
- **เก็บอย่างเดียว**: confidence_scores อยู่ใน result.json ตอน OCR → ลง `readings.confidence_scores` ตอนบันทึก
  (เผื่อวิเคราะห์ย้อนหลัง); **ไม่อยู่ใน response `/api/ocr`**, ไม่แตะ UI; Gemini = NULL
- `low_confidence_fields` ของ GLM = `[]` เสมอ (ไม่แปลง confidence เป็นป้าย "ตรวจสอบ")

### 3.18 สถาปัตยกรรม 2 engine พิสูจน์ตัวเองแล้ว — API ภายนอกล่ม ระบบยังเดินต่อได้
วันที่ 2026-10-08 โควต้า Gemini free tier หมดกลางการทดลอง (ข้อ 13) → Gemini ใช้ไม่ได้ ~16.5 ชั่วโมง
และวันที่ 2026-10-10 Gemini ตอบ 503 "high demand" 2 ใน 4 คำขอ — ทั้งสองเรื่องอยู่นอกการควบคุมของเรา
- เพราะการเรียกโมเดลถูกแยกเป็น `ocr_engines/` (ข้อ 3.13) **ระบบเปลี่ยนไปใช้ GLM-OCR บนเครื่องได้ด้วยการแก้ค่าเดียว**
  ใน `.env`: `OCR_ENGINE=glm` แล้วรีสตาร์ท server — ไม่แก้โค้ด, ไม่ต้องใช้อินเทอร์เน็ต/API key/โควต้า
- ส่วนอื่นเหมือนเดิมทั้งหมด: OBB, ตัวดัก, ตาข่าย analog, ช่องทีละหลัก, การบันทึก, PDF, ประวัติ
  (readings.engine บอกว่าแต่ละรายการอ่านด้วย engine ไหน)
- **ข้อแลกเปลี่ยน**: GLM แม่นน้อยกว่าและได้ค่าน้อยกว่า (ข้อ 12) → ใช้เป็นทางสำรองที่ "ระบบไม่หยุด" โดยคนต้องตรวจ/กรอก
  ค่าที่ขาดในช่องทีละหลัก (ขั้นตอนนี้มีอยู่แล้วในทุกการจด)
- ที่ยืนยันแล้ว: GlmEngine ผ่าน pipeline จริง (OBB → preprocess → engine.read → build_ocr_result) กับ 66 รูป
  และเทสต์ endpoint `/api/ocr` → บันทึก (Ollama จำลอง) ใน `test_glm_engine.py`
  **ยังไม่ได้ทดสอบ**: เปิดหน้าเว็บจริงด้วย `OCR_ENGINE=glm` ครบ flow ใน browser (`.env` ยังเป็น gemini ตามที่สั่ง)

---

## 4. บทเรียน Deskew (สำคัญมาก — อย่าเสียเวลาซ้ำ)

**ลอง classical CV มา 3 วิธี — พังหมดกับภาพหน้างานจริง:**
1. Canny + contour หา quad → edge แตกเป็นชิ้นเพราะพลาสติกห่อ/แสงสะท้อน
2. Otsu threshold → หมุนมั่ว
3. Bright-region segmentation → ยังไม่เสถียร

**ต้นเหตุ:** deskew บนภาพที่ crop แล้ว = จอเต็มเฟรม ไม่มีขอบให้ contour หา (contour ได้แค่ 0.1-1.8%)
แม้ crop กว้างขึ้น edge detection ก็ยังแพ้แสงสะท้อน/พลาสติกโดยธรรมชาติ

**ข้อสรุป → YOLO-OBB** (learned approach ไม่พึ่ง edge detection) — smoke test 8 ภาพจริงเจอครบ 8/8
(conf 0.93–0.97); โค้ด: `detect_meter_obb()` → `expand_corners()` → `order_corners()` → `deskew_from_corners()`
**ข้อจำกัด OBB**: (ก) 180° ambiguity อาจหมุนกลับหัว (Gemini อ่านได้) (ข) แก้ perspective/keystone ไม่ได้

---

## 5. Flow การใช้งาน (หลังเพิ่มระบบบันทึกค่า)

1. **หน้าแรก `/`** — รายการ Meter Run แบบจับคู่เป็นชุด (EVC: Tag/Serial FY, Turbine: Tag/Serial FE)
   - เลือกได้ทีละ 1 Run → "ไปหน้าอ่านค่า" (จำ Run ล่าสุดไว้ preselect)
   - "เพิ่ม Meter Run": company/run_label/tag แปลงเป็นตัวพิมพ์ใหญ่; ซ้ำ → "Meter Run นี้มีอยู่แล้วในระบบ"
   - ปุ่มลบ (ยืนยันในการ์ด): Run ที่มีประวัติการจดแล้วลบไม่ได้ (ปุ่มปิด + server 409)
2. **หน้า OCR `/ocr?run=<id>`** — ไม่มี/ไม่พบ run → เด้งกลับหน้าแรก
   - ซ้าย: อัปโหลดภาพ (EVC หลายหน้าจอ + Turbine 1 รูปในคำขอเดียวได้)
   - ขวา: แต่ละฟิลด์ = 10 ช่องทีละหลัก เติมจากขวา จุดทศนิยมตำแหน่งตายตัว, แก้ได้ทุกช่อง
     - "ค่าจดครั้งก่อน" สีเทาเหนือแต่ละแถว (จาก reading ล่าสุดของ Run; ไม่มีประวัติ = ว่าง)
     - ป้าย: ตรวจสอบ (low confidence) / แก้ไขแล้ว / ไม่พบค่า / ตัดออก (ตาข่าย analog)
     - แถบเตือนผลต่างค่าสะสมติดลบเหนือปุ่ม Export (ข้อ 3.12) — เตือนอย่างเดียว ยังบันทึกได้
   - วัน/เวลาที่จด (default = ตอนนี้, แก้ได้); เตือนถ้าเก่ากว่าการจดครั้งก่อน
   - checkbox ตัดทศนิยม Turbine 2 ช่องใต้ 2 ช่องขวาสุด: ขวา = ตัด 1 หลัก, ซ้าย = ตัด 2 หลัก
     (ติ๊กซ้ายบังคับติ๊กขวา) — ตัดบน "สตริงตัวเลข" ไม่ใช่จากตำแหน่งจุด; แสดง "ค่าที่จะบันทึก"
   - **Export to PDF** = บันทึกลง DB + ดาวน์โหลด PDF ในครั้งเดียว → ล็อกช่อง (ไม่มีการแก้ข้อมูลที่บันทึกแล้ว)
3. **หน้าประวัติ `/history?run=<id>`** — ตารางใหม่สุดก่อน กรองตาม Run
   - คอลัมน์ "แก้ด้วยมือ" = จำนวนฟิลด์ที่ถูกแก้ / ฟิลด์ที่มีค่า (hover ดูชื่อฟิลด์)
   - หัวตาราง: "อัตราค่าที่ไม่ถูกแก้ด้วยมือ X/Y (Z%)" — **ไม่ใช่ความแม่นยำจริง** (ถ้ากดบันทึกโดยไม่ตรวจ
     จะได้ 100% ทั้งที่อาจผิด) นับเฉพาะฟิลด์ที่มีค่า, การติ๊กตัดทศนิยมไม่นับเป็นการแก้
   - รายละเอียด: ค่าที่โมเดลอ่าน vs ค่าที่บันทึก ทุกฟิลด์ + ภาพต้นฉบับ + export PDF ซ้ำ

**PDF (ใบแจ้งปริมาณการใช้ก๊าซ)** — จัดตามฟอร์มจริงแต่ไม่เลียนแบบเป๊ะ, A4 หน้าเดียว:
- หัว: บริษัท / Meter Run / วันเวลาที่จด / วันที่จดครั้งก่อน
- ตาราง 1 EVC (Tag/Serial FY): คอลัมน์ CORRECT V. (SCM), LINE V. (M³) × แถว จดครั้งหลัง (บน) →
  จดครั้งก่อน (กลาง) → ค่าแตกต่าง (ล่าง); ตาราง PRESSURE (Bara) / TEMPERATURE (°C)
- ตาราง 2 Turbine (Tag/Serial FE): TURBINE V. (M³) แถวเดียวกัน
- REMARK: `Run A: CF EVC = 0.9987, CF คำนวณ = -, Qm = 120.5 m3/hr, Qb = -` (จบที่ Qb)
  - ค่า null → "-" **ไม่มีหน่วยต่อท้าย** (แบบเดียวกับ "CF คำนวณ = -" บนฟอร์มจริง)
  - CF คำนวณ = "-" เสมอ (ระบบไม่คำนวณ); CF/Qm/Qb อยู่ใน REMARK เท่านั้น ไม่ซ้ำในตาราง
- ผลต่างเฉพาะค่าสะสม 3 ตัว แสดงตามจริง (ติดลบก็แสดงเลขติดลบ ไม่มีเครื่องหมายพิเศษ);
  ไม่มีครั้งก่อน → ช่องครั้งก่อนและผลต่างว่าง
- **ไม่มี**ป้าย "Remark =" ท้ายบรรทัด REMARK, บรรทัดค่าจากเกจบนท่อ (Pressure/Temperature Gauge) และช่องติ๊ก 5 ช่อง (Calibrate /
  Corrector ขัดข้อง / Turbine ขัดข้อง / เปลี่ยน Corrector / เปลี่ยน Turbine) — เคยใส่แล้วเอาออก เพราะ
  ไม่มีข้อมูลในระบบ เติมไม่ได้สักช่อง; PDF นี้สร้างจาก DB อัตโนมัติ ไม่ใช่ฟอร์มกระดาษให้คนกรอก
  พื้นที่ว่างถาวรทำให้รายงานดูไม่เสร็จ

---

## 6. ฐานข้อมูล (SQLite: `meter_records.db`)

### ตาราง `meter_runs`
| คอลัมน์ | ชนิด | หมายเหตุ |
|---------|------|---------|
| id | INTEGER PK AUTOINCREMENT | |
| company | TEXT NOT NULL | ตัวพิมพ์ใหญ่ เช่น "ASM" |
| run_label | TEXT NOT NULL | ตัวพิมพ์ใหญ่ เช่น "A" |
| tag_fe | TEXT | Tag ของ Turbine Meter เช่น ASM-0330-FE-057A |
| serial_fe | TEXT | คงตามที่กรอก |
| tag_fy | TEXT | Tag ของ Electronic Volume Corrector เช่น ASM-0330-FY-057A |
| serial_fy | TEXT | คงตามที่กรอก |
| created_at | TEXT NOT NULL | ระบบใส่เอง (ISO) |

- `UNIQUE INDEX uq_meter_runs_company_run (company COLLATE NOCASE, run_label COLLATE NOCASE)` —
  กัน Run ซ้ำ (ถ้าซ้ำได้ ประวัติแตกสองก้อนและ "ค่าครั้งก่อน" หายเงียบๆ)
- seed (`SEED_METER_RUNS` ใน db.py) ใส่เฉพาะตอนตารางว่าง — แก้ seed หลังสร้าง DB แล้วไม่มีผล

### ตาราง `readings`
| คอลัมน์ | ชนิด | หมายเหตุ |
|---------|------|---------|
| id | INTEGER PK AUTOINCREMENT | |
| meter_run_id | INTEGER NOT NULL FK → meter_runs(id) | PRAGMA foreign_keys = ON |
| recorded_at | TEXT NOT NULL | วัน/เวลาที่ผู้ใช้กรอก (เวลาที่จด) normalize เป็น YYYY-MM-DDTHH:MM:SS |
| created_at | TEXT NOT NULL | เวลาที่กดบันทึกจริง (ระบบใส่เอง) |
| raw_values | TEXT (JSON) | ค่าดิบตามที่โมเดลเขียน (สตริง) ก่อนคนแก้ |
| final_values | TEXT (JSON) | ค่าหลังคนแก้ + หลังตัดทศนิยม Turbine (สตริง ทศนิยมตายตัว) |
| edited_fields | TEXT (JSON) | ฟิลด์ที่คนแก้ (server เทียบหลัง normalize) |
| low_confidence_fields | TEXT (JSON) | จากโมเดล |
| turbine_trim_digits | INTEGER CHECK IN (0,1,2) | จาก checkbox |
| image_paths | TEXT (JSON) | ภาพต้นฉบับใน ocr_requests/ (relative กับโฟลเดอร์โปรเจค) |
| engine | TEXT NOT NULL (ไม่มีค่าเริ่มต้น) | `gemini` / `glm` — มาจาก result.json, ไม่ส่งมา = error |
| confidence_scores | TEXT (JSON) nullable | GLM: {ฟิลด์: 0..1 หรือ null} เก็บอย่างเดียวไม่แสดง (ข้อ 3.17); Gemini: NULL |

- **หมายเหตุ**: ไฟล์ `meter_records.db` ที่สร้างก่อนเปลี่ยน schema ยังมี `DEFAULT 'gemini'` ค้างในนิยามตาราง
  (SQLite แก้ default ของคอลัมน์เดิมไม่ได้โดยไม่สร้างตารางใหม่ — ไม่ได้ migrate เพราะไม่กระทบ: โค้ดส่ง engine เสมอ)

- `INDEX idx_readings_run_time (meter_run_id, recorded_at)`
- "ค่าครั้งก่อน" = `get_previous_reading()` เรียงตาม recorded_at (เท่ากันใช้ id)

### ตารางฟิลด์ (แหล่งเดียว: `reading_fields.FIELDS`, frontend ดึงผ่าน `/api/reading-fields`)
| ฟิลด์ | label | หน่วย | ทศนิยม | อุปกรณ์ | ค่าสะสม |
|-------|-------|------|:---:|:---:|:---:|
| correct_v_scm | CORRECT V. | SCM | 0 | FY (EVC) | ✓ |
| line_v_m3 | LINE V. | M³ | 0 | FY | ✓ |
| pressure_bar | PRESSURE | Bara | 3 | FY | |
| temperature_c | TEMPERATURE | °C | 2 | FY | |
| correction_factor | CORRECTION FACTOR | — | 4 | FY | |
| qm_m3_h | Qm | m3/h | 1 | FY | |
| qb_sm3_h | Qb | Sm3/h | 1 | FY | |
| turbine_v_m3 | TURBINE V. | M³ | 0 | FE (Turbine) | ✓ |

ทศนิยมตรวจกับฟอร์มจริงแล้ว: Volume 3 ตัวจดเป็นจำนวนเต็ม; Qm/Qb ฟอร์มจริงเขียน 2 ตำแหน่ง แต่
**ตัดสินใจใช้ 1 ตำแหน่ง** (เป็นการตัดสินใจ ไม่ใช่ความพลาด) — ห้ามแก้ตารางนี้โดยไม่ได้ยืนยันกับเจ้าของโปรเจค

---

## 7. ไฟล์ / ฟังก์ชัน / Endpoint (สำหรับรายงานบทที่ 3)

### ไฟล์ที่เพิ่มใหม่
| ไฟล์ | หน้าที่ |
|------|--------|
| `db.py` | ชั้นฐานข้อมูล SQLite: schema, seed, CRUD ของ meter_runs/readings |
| `reading_fields.py` | ตารางฟิลด์ + logic ค่าสตริง: จัดรูปแบบ, ตัด Turbine, คำนวณ final_values/ผลต่าง |
| `report_pdf.py` | สร้าง PDF ใบแจ้งปริมาณการใช้ก๊าซ (fpdf2 + Sarabun + text shaping) |
| `fonts/` | Sarabun-Regular.ttf, Sarabun-Bold.ttf, OFL.txt (ต้อง commit) |
| `static/runs.html`, `runs.js` | หน้าแรก เลือก/เพิ่ม/ลบ Meter Run |
| `static/history.html`, `history.js` | หน้าประวัติ |
| `static/theme.js` | สลับโหมดมืด/สว่าง (ใช้ร่วมทุกหน้า — ย้ายออกมาจาก index.js) |
| `requirements.txt`, `.gitignore` | ส่งมอบงาน |
| `test_db.py`, `test_reading_fields.py`, `test_meter_runs_api.py`, `test_ocr_result.py`, `test_readings_api.py` | เทสต์ |
| `ocr_engines/` (6 ไฟล์ ดูข้อ 3.13) | ชั้น OCR engine: Gemini + GLM-OCR |
| `test_glm_labels.py`, `test_glm_engine.py` | เทสต์ชั้นแปลงป้าย / GlmEngine (mock httpx.post ด้วย response จริง) |
| `test_data/glm_ollama_test5.json` | response จริงของ Ollama กับ bin/test5.jpg (prompt ปัจจุบัน) ใช้ในเทสต์ |

**ไฟล์ที่ลบ**: `gemini_ocr.py`, `test_pipeline.py` — สคริปต์ทดลองเก่าที่ฝัง API key (key ถูก revoke แล้ว)
ตรวจแล้วไม่มีไฟล์ไหน import/อ้างถึงก่อนลบ

### ไฟล์ที่แก้
- `server.py` — เพิ่มชั้นค่าดิบ/ตาข่าย/เก็บผล OCR + endpoint ใหม่ทั้งหมด, prompt (ข้อ 2 + กฎข้อ 5),
  ตัวดัก 3.6 แบบใหม่; **pipeline ภาพเดิมไม่ถูกแตะ** (preprocess_meter_image, detect_meter_obb,
  expand_corners, order_corners, deskew_from_corners, crop_from_corners)
- `static/index.html|index.js|index.css` — หน้า OCR: ช่องทีละหลักแทนตารางเดิม, ค่าครั้งก่อน, วันเวลา,
  checkbox Turbine, Export to PDF, แถบเตือน fallback/ตาข่าย/ผลต่างติดลบ
- `evaluate.py` — ใช้ `check_request_meter_types()` แทน `distinct_meter_types()` (ถูกลบ)
- `test_mixed_meter.py` — เขียนใหม่ตามกฎ 3.6 ใหม่ + ตาข่าย analog

### server.py — ฟังก์ชันที่เพิ่ม/เปลี่ยน
| ฟังก์ชัน | หน้าที่ |
|---------|--------|
| `check_request_meter_types(filenames, classes)` | ตัวดัก 3.6 แบบใหม่ คืนข้อความ error หรือ None |
| `apply_analog_only_guard(result, classes)` | ตาข่าย 3.7 คืน {ฟิลด์: ค่าดิบเดิม} ที่ถูกทิ้ง |
| `extract_raw_values(response, reading)` | (ย้ายไป `ocr_engines/gemini_engine.py`, server.py ยัง export ชื่อเดิม) ค่าดิบสตริงจากข้อความ JSON → (values, "text"/"fallback") |
| `build_ocr_result(engine_result, classes)` | EngineResult → ตาข่าย → ค่าเริ่มต้นช่องทีละหลัก (ไม่ขึ้นกับ engine) |
| `persist_ocr_request(request_id, result, uploads, confidence_scores=None)` | เก็บ result.json (+ ชื่อ engine, confidence_scores) + ภาพต้นฉบับใน ocr_requests/ |
| `load_ocr_request(request_id)` | อ่านผล OCR ที่เก็บไว้ (ตรวจรูปแบบ id กัน path traversal) |
| `create_reading(payload)` | บันทึก: คำนวณ final_values/edited_fields ฝั่ง server → insert |
| `check_decrease(meter_run_id, payload)` | เตือนผลต่างค่าสะสมติดลบเทียบค่าจดครั้งก่อน (ข้อ 3.12) |
| `get_reading_pdf(reading_id)` | สร้าง PDF จาก DB + การจดก่อนหน้า |
| `OcrResult` (Pydantic) | response ของ /api/ocr = MeterReading + raw_values, raw_source, editor, request_id, guard_dropped, guard_applied |

### db.py
`init_db`, `get_conn`, `create_meter_run` (DuplicateMeterRunError), `list_meter_runs` (+reading_count),
`get_meter_run`, `delete_meter_run` (MeterRunInUseError), `insert_reading`, `get_reading`,
`list_readings`, `get_previous_reading(meter_run_id, before_reading_id=None)`

### reading_fields.py
`to_plain_str`, `format_fixed`, `turbine_digits`, `trim_turbine`, `initial_editor_values`,
`normalize_entered`, `compute_final_values`, `find_edited_fields`, `difference`,
`decreased_fields` (ฟิลด์ค่าสะสมที่ผลต่างติดลบ + ข้อความเตือน)

### report_pdf.py
`build_reading_pdf(run, reading, previous)`, `remark_line`, `pdf_filename`, `format_datetime`

### Endpoint
| Method | Path | หน้าที่ |
|--------|------|--------|
| GET | `/` | หน้าเลือก Meter Run |
| GET | `/ocr?run=<id>` | หน้า OCR |
| GET | `/history?run=<id>` | หน้าประวัติ |
| POST | `/api/ocr` | อ่านค่า (multipart files) → OcrResult |
| POST | `/api/ocr/sample` | อ่านภาพตัวอย่าง |
| GET | `/api/reading-fields` | ตารางฟิลด์ |
| GET / POST | `/api/meter-runs` | รายการ Run / เพิ่ม Run (ซ้ำ 409) |
| GET / DELETE | `/api/meter-runs/{id}` | รายละเอียด / ลบ (มีประวัติ 409) |
| GET | `/api/meter-runs/{id}/latest-reading` | ค่าจดครั้งล่าสุด (หน้า OCR) |
| POST | `/api/meter-runs/{id}/decrease-check` | ฟิลด์ค่าสะสมที่ผลต่างติดลบ (เตือน ไม่บล็อก) |
| POST | `/api/readings` | บันทึกการจด (ซ้ำ 409) |
| GET | `/api/readings?meter_run_id=` | ประวัติ |
| GET | `/api/readings/{id}` | รายละเอียด + การจดก่อนหน้า |
| GET | `/api/readings/{id}/images/{n}` | ภาพต้นฉบับ |
| GET | `/api/readings/{id}/pdf` | PDF |

### เทสต์ 9 ชุด (รันทีละไฟล์ `python test_xxx.py` — ไม่เรียก Gemini/Ollama จริง, ใช้ DB/โฟลเดอร์ชั่วคราว)
`test_reading_fields.py` (logic ค่า/ตัด Turbine/กับดัก float), `test_db.py` (+ engine บังคับส่ง),
`test_meter_runs_api.py`, `test_ocr_result.py` (ค่าดิบ/fallback/fence/prompt), `test_readings_api.py`
(บันทึก+PDF+ประวัติ ครบสเปคข้อ 9 + 9b รูปแบบ PDF + 12 เตือนผลต่างติดลบ: ติดลบ/ศูนย์/ไม่มีครั้งก่อน/Turbine หลังตัด/ยังบันทึกได้),
`test_mixed_meter.py` (ตัวดัก 3.6 + ตาข่าย, ใช้ OBB จริง), `test_obb.py`,
`test_glm_labels.py` (ตารางป้าย, 4 เลย์เอาต์, ป้ายไม่รู้จัก, ไม่มี fuzzy, Converted/Unconverted ไม่สลับ, analog หลายชุด),
`test_glm_engine.py` (ล็อก prompt + prompt ลอกออกมาต้องไม่เกิดค่า, กฎ confidence, response จริง test5,
หลายภาพ/analog, Ollama ไม่เปิด/ไม่มีโมเดล, endpoint: confidence ลง DB แต่ไม่อยู่ใน response)
— ผลรันล่าสุด 2026-10-10 ผ่านครบ 9/9 (เก็บ log เต็มทุกครั้งที่รัน)

---

## 8. Roadmap — สถานะปัจจุบัน

**เสร็จแล้ว (ก่อนระบบบันทึกค่า):**
- [x] Step 1-6: .env + fail-fast, EXIF/device/RGB, refactor preprocess, deskew (→ OBB), class hint +
      response.parsed, frontend (dedupe/ลบรายรูป/ค่าดิบ/DOM ปลอดภัย), evaluation harness
- [x] low_confidence_fields + ป้าย "ตรวจสอบ"
- [x] เทรน YOLO-OBB → ถอด HBB เหลือ OBB โมเดลเดียว (ข้อ 3.4)

**เสร็จแล้ว (ระบบบันทึกค่า + Export PDF — ตาม claude_code_brief.md):**
- [x] เฟส 1 SQLite + seed (+ UNIQUE company/run_label, ตัวพิมพ์ใหญ่)
- [x] เฟส 2 หน้าแรก เลือก/เพิ่ม Meter Run (+ ปุ่มลบ)
- [x] เฟส 3 หน้า OCR ช่องทีละหลัก + ค่าครั้งก่อน + วันเวลา + checkbox Turbine
- [x] เฟส 4 บันทึก + PDF (server คำนวณ final_values, PDF จาก DB)
- [x] เฟส 5 หน้าประวัติ (+ อัตราค่าที่ไม่ถูกแก้ด้วยมือ)
- [x] เฟส 6 แก้ prompt (ย่อหน้า 2 + กฎข้อ 5) + ผ่อนตัวดัก 3.6 + ตาข่าย analog
- [x] PDF ตามฟอร์มจริง (แถวบน→ล่าง, REMARK, Bara) — เกจบนท่อ/ช่องติ๊กเคยใส่แล้วเอาออก (ข้อ 5)
- [x] ทดสอบ end-to-end จริง (รูปจริง → Gemini จริง → บันทึก → PDF → ประวัติ) โดยเจ้าของโปรเจค — ผ่าน
      (ระหว่างทางพบบั๊กตารางประวัติว่าง (CSS) แก้แล้ว)
- [x] เตือนผลต่างค่าสะสมติดลบบนหน้า OCR (ข้อ 3.12)

**ยกเลิก:**
- ~~นับจำนวนช่องทีละหลักเทียบกับฟอร์มจริง~~ — `DIGIT_BOXES = 10` ใช้ต่อ (ค่าจริงยาวสุดที่เจอ 8 หลัก
  ยังเหลือที่), ไม่เอาภาพฟอร์มเข้าโปรเจค

**เสร็จแล้ว (รองรับหลาย OCR engine):**
- [x] เฟส A — แยก `ocr_engines/` โดยไม่เปลี่ยนพฤติกรรม (เทสต์เดิมผ่านโดยไม่แก้), OCR_ENGINE + fail-fast,
      readings.engine บังคับ, ลบ gemini_ocr.py/test_pipeline.py, สแกน secret + .gitignore, เอาคำว่า Gemini ออกจาก UI
- [x] spike GLM-OCR 3 กรณี (ข้อ 3.14)
- [x] เฟส B — GlmEngine + ชั้นแปลงป้าย + confidence จาก logprobs + แก้ prompt รั่ว (ข้อ 3.14–3.16)
- [x] confidence เก็บอย่างเดียว ไม่แสดง (ข้อ 3.17)
- [x] วัดผล GLM 66 รูป + เปรียบเทียบ Gemini vs GLM 10 รูป ตรวจด้วยตาทุกค่า (ข้อ 12)

**หยุดพัฒนาแล้ว (2026-10-10)** — เข้าช่วงเขียนรายงาน

**ค้างอยู่ (ถ้าจะพัฒนาต่อ):**
- [ ] รัน evaluation harness (`evaluate.py` + `ground_truth.xlsx` + `eval_images/`) — งานคอขวดคือกรอกเฉลย;
      evaluate.py ยังเรียก Gemini ผ่านชื่อเดิมใน server.py ยังไม่รองรับเลือก engine (ตั้งใจยังไม่แก้)
- [ ] ทดสอบหน้าเว็บจริงครบ flow ด้วย `OCR_ENGINE=glm` ใน browser
- [ ] (ทางเลือก) GLM: แยก 2 คู่ในบรรทัดเดียว (`T 31.40 °C C 24.95448`) — ยังอยู่ในกฎเหล็กเพราะทั้งสองมีป้าย

**คำถามที่ยังไม่ตอบด้วยข้อมูล (ตอบได้ด้วย eval harness):**
- detailEnhance ช่วยหรือทำร้าย Gemini? / detected_format ควรลบไหม? / ภาพ deskewed แม่นกว่าภาพดิบไหม?

### รายการทดลองที่ควรรันบน eval harness
แก้ทีละอย่าง รันด้วย label ต่างกัน แล้วเทียบผล:

| Label            | แก้อะไรก่อนรัน                         | ตอบคำถาม                   |
|------------------|----------------------------------------|-----------------------------|
| baseline         | ไม่แก้อะไร                             | ตัวตั้งต้น                  |
| no_enhance       | คอมเมนต์ `cv2.detailEnhance(...)` ออก  | detailEnhance ช่วยจริงไหม  |
| no_format_field  | ลบ detected_format จาก schema ชั่วคราว | ฟิลด์นี้กระทบ accuracy ไหม |
| crop_input       | `SEND_DESKEWED_TO_GEMINI = False`      | ภาพดัดตรงแม่นขึ้นไหม       |

---

## 9. Config สำคัญ

- `GEMINI_MODEL` (.env) — ชื่อโมเดล Gemini (`gemini-3.5-flash`)
- `GEMINI_API_KEY` (.env) — **ห้าม hardcode**
- `OBB_MODEL_PATH` (.env, optional) — default `best_obb.pt`
- `METER_DB_PATH` (env, optional) — default `meter_records.db` (เทสต์ใช้ไฟล์ชั่วคราว)
- `OCR_REQUEST_DIR` (env, optional) — default `ocr_requests/` (เทสต์ใช้โฟลเดอร์ชั่วคราว)
- `SEND_DESKEWED_TO_GEMINI = True` — False = ส่ง crop แนวแกนไม่ดัดตรง (eval A/B)
- `OBB_WARP_PADDING = 0.06`, `OBB_CONF_THRESHOLD = 0.4`
- `OCR_ENGINE` (.env, optional) — `gemini` (ค่าเริ่มต้น) หรือ `glm`; ชื่ออื่น → server ไม่สตาร์ท (ข้อ 3.13)
- `OLLAMA_BASE_URL` (env, optional) — default `http://localhost:11434`
- `GLM_MODEL` (env, optional) — default `glm-ocr:latest`
- `GLM_MAX_TOKENS` (env, optional) — default 256 (= num_predict กันวนซ้ำ)
- `GLM_TIMEOUT` (env, optional) — default 180 วินาทีต่อภาพ (โหลดโมเดลครั้งแรกช้า)
- `reading_fields.DIGIT_BOXES = 10` — จำนวนช่องทีละหลักต่อฟิลด์ (ค่าจริงยาวสุดที่เจอ 8 หลัก)

---

## 10. ข้อควรระวังตอนพัฒนาต่อ

1. **ห้าม hardcode API key** — .env อยู่ใน .gitignore
2. **ค่ามิเตอร์เป็นสตริงตลอดทาง** (ข้อ 3.8) — ห้าม `str(float)` / `Number()` / ปัดเศษ / counter animation
3. **ห้ามแตะ pipeline ภาพ** ถ้าไม่จำเป็น — ทำงานดีแล้ว
4. **แก้ .css/.js แล้วต้องบวกเลข `?v=`** ในทุก html ที่ใช้ไฟล์นั้น (หน้า HTML ส่ง `Cache-Control: no-cache` แล้ว)
5. **`.params-table` ใช้ร่วมหลายตาราง** — ระวังกฎ CSS เดิม (เคยมี `opacity: 0` ทำตารางประวัติว่าง,
   `td:last-child { text-align: center }` สืบทอดเข้าแถวรายละเอียด)
6. **แตะ UI ต้องเปิด browser ตรวจจริง** ก่อนปิดงาน ถ้าเปิดไม่ได้ต้องแจ้งว่ายังไม่ได้ตรวจส่วนไหน
7. **debug_images/ และ ocr_requests/ โตเรื่อยๆ** ทุก request — เคลียร์ ocr_requests เฉพาะโฟลเดอร์ที่
   result.json มี `reading_id = null` (รายการที่บันทึกแล้วใช้ภาพในนั้นเป็นหลักฐาน)
8. **ห้ามลบ meter_records.db หลังเริ่มบันทึกค่าจริง** — แก้รายชื่อ Run ผ่านหน้าเว็บ (เพิ่ม/ลบ) หรือ SQLite tool
9. **ฟอนต์ใน fonts/ ต้อง commit** — ห้ามพึ่งฟอนต์ที่ติดตั้งในเครื่อง; PDF ต้องเรียก `set_text_shaping(True)`
10. **ภาพทดสอบจาก Canva** ต่างจากหน้างานจริง — พิสูจน์ได้แค่ "โค้ดไม่พัง" ต้องมีภาพถ่ายจริงด้วย
11. **.gitignore**: ignore `*.pt` ยกเว้น `best_obb.pt` (~20MB < 100MB ของ GitHub), ignore `runs/`, `experiment_output.zip`

---

## 11. ข้อจำกัดที่รู้อยู่ (ยอมรับแล้ว)

- โมเดล**เขียน** `1123120.0` มาเองในข้อความ → Turbine ได้ 8 หลัก (แยกไม่ได้ว่าเป็น artifact หรือช่องแดงเลข 0)
  — ช่องทีละหลักโชว์หลักเกิน + คำใบ้ "ทศนิยม 1 หลัก" ให้คนจับได้
- เส้นทาง fallback (อ่านข้อความดิบไม่ได้) อาจเสียศูนย์ท้ายทศนิยม — มีแถบเตือนบนหน้า OCR
- ค่าติดลบ: แสดงเครื่องหมายตามที่โมเดลอ่าน แก้ได้เฉพาะตัวเลข (โรงแยกก๊าซในไทย 18-34°C ไม่มีโอกาสติดลบ)
- ตาข่าย analog ใช้ไม่ได้กับคำขอที่มี digital ปน (ผลรวมเป็นก้อนเดียว) — พึ่ง prompt ข้อ 5
- ตัวดักแยกไม่ได้ EVC คนละ Run (class เดียวกัน)
- "อัตราค่าที่ไม่ถูกแก้ด้วยมือ" ไม่ใช่ความแม่นยำจริง
- **Gemini free tier 20 คำขอ/วัน/โมเดล** (ข้อ 13) — การใช้งาน 1 คำขอต่อการจด 1 ครั้ง แต่การทดลอง/สาธิตกินโควต้าเร็ว
- **GLM-OCR**: ได้ค่าน้อยกว่าและผิดได้ (ข้อ 12) — จุดทศนิยมหาย, หลักแรกผิด, หลักที่กำลังหมุนหาย, ภาพกลับหัว;
  confidence ใช้คัดค่าผิดไม่ได้ (ข้อ 3.17); ป้ายที่ไม่อยู่ในตาราง/ไม่ถูกถอดออกมา → null (ข้อ 3.16)
- GLM อ่านภาพที่ OBB หมุนกลับหัว (180° ambiguity) ผิด (S15: เลข 6 กลับหัวอ่านเป็น 9) — Gemini อ่านได้ตามที่ทดสอบไว้ในข้อ 4

---

## 12. ผลการทดลอง GLM-OCR และการเปรียบเทียบกับ Gemini (สำหรับรายงานบทที่ 4)

ไฟล์ผลดิบทั้งหมดอยู่ใน scratchpad ของ session (ไม่อยู่ในโปรเจค): `glm_batch_v1.jsonl`, `glm_batch_v4.jsonl`,
`gemini_batch.jsonl`, `gemini_10.jsonl`, `compare10_rows.csv` / `compare10_rows.jsonl` (ผลรายค่า 80 ช่อง),
`compare10_summary.txt` และสคริปต์ที่ใช้รัน (`run_glm_batch.py`, `run_gemini_10.py`, `run_gemini_fill.py`, `compare_10.py`)

**เงื่อนไขร่วม**: ทุกภาพผ่าน pipeline จริง (EXIF → OBB → warp → detailEnhance) แล้วเรียก `engine.read([ภาพ], [class])`
**1 ภาพต่อ 1 คำขอทั้งสอง engine** เพื่อเทียบรูปต่อรูป (ต่างจากการใช้งานจริงของ Gemini ที่ส่งทุกภาพในคำขอเดียว);
เวลาที่วัด = เฉพาะ `engine.read` (ไม่รวม OBB/preprocess); GLM รันบน RTX 4060 Laptop (โมเดลโหลดค้างแล้ว),
Gemini = `gemini-3.5-flash` free tier ผ่านอินเทอร์เน็ต; การตรวจค่าด้วยตาทำบนภาพต้นฉบับ (ขยายภาพเมื่อจำเป็น)

### 12.1 GLM-OCR ชุด 66 รูป (ข้อมูล coverage — ไม่ได้เทียบกับ Gemini)
ชุดทดสอบ: `testpic/`, `experiment_images/`, `eval_images/`, `bin/` รวม 66 รูป (OBB: digital 40, analog 26)

| | รอบแรก (prompt มีตัวอย่าง LABEL/DIGITS, รับแค่ `ป้าย = ค่า`) | รอบสุดท้าย (v4) |
|---|---|---|
| รูปที่ได้ค่าอย่างน้อย 1 ค่า | 6 / 66 | **41 / 66** |
| จำนวนค่าที่ได้ | 21 | **65** |
| analog ได้ turbine_v_m3 | 0 / 26 | **23 / 26** (ไม่ได้: img7, analog001, test7 — ข้อ 3.16) |
| digital ได้ค่าอย่างน้อย 1 ค่า | 6 / 40 | 18 / 40 |
| ค่าที่มี confidence | 21 / 21 | 65 / 65 |
| prompt รั่ว ("LABEL" / "Counter = DIGITS") | 8 / 7 ครั้ง | 0 |
| เวลาเฉลี่ยต่อรูป | — | 1.60 s (มัธยฐาน 1.70 s) |

**ความถูกต้อง (ตรวจกับภาพทุกค่า)**: 65 ค่า → ถูก 53, **ผิด 10**, ไม่แน่ใจ 2 — **ไม่มีค่าไหนลงผิดฟิลด์** (กฎเหล็กทำงาน)

| ประเภทความผิดพลาด | จำนวน | ตัวอย่าง |
|---|---|---|
| จุดทศนิยมหาย | 1 | S05 correct_v_scm `47959582516` (จอ `47959582.516`) → ใหญ่ไป 1000 เท่า — อันตรายที่สุด |
| หลักแรกผิด | 1 | S10 (MacBAT) Vm `8299772.90` (จอ `0299772.90`) |
| หลักกลางผิด | 1 | S10 C `0.902139` (จอ `0.908139`) |
| หลักที่กำลังหมุน (ช่องแดง) หาย | 4 | S12, S17, S19, test3 — ขาด 1 หลัก เล็กไป 10 เท่า |
| หลักที่กำลังหมุนอ่านผิด | 1 | 00129-RMG `5659261` (หน้าปัด `0565926,7`) |
| ภาพกลับหัว (OBB หมุน 180°) | 1 | S15 `1209795` — เลข 6 กลับหัวอ่านเป็น 9 |
| อักษรหน่วยถูกอ่านเป็นตัวเลข | 1 | ItronPage3 CV Idx `7546132.6885` (จอ `07546132.688Sm³` ตัว S → 5) — ค่าที่บันทึก (0 ตำแหน่ง) ไม่กระทบ |
| **รวม** | **10** | |
- ไม่แน่ใจ: img4 (ภาพกลับหัว), S01_p1 correct_v_scm (หลักทศนิยมสุดท้ายถูกหน่วยบัง — ค่าที่บันทึกไม่กระทบ)
- confidence ของ 10 ค่าที่ผิด: ข้อ 3.17

### 12.2 เปรียบเทียบ Gemini vs GLM-OCR — 10 รูป
**ทำไม 10 รูป ไม่ใช่ 66**: Gemini free tier ได้ 20 คำขอ/วัน และต้องเหลือโควต้าไว้สาธิต (ข้อ 13)

**กฎเลือกรูป** (กันการเลือกรูปตามผลของ engine ใด): ไล่ตามลำดับไฟล์ `testpic → experiment_images → eval_images → bin`
แล้วเอา N รูปแรกของแต่ละชนิด ข้ามภาพที่เป็นภาพเดียวกันซ้ำ; EK280/MacBAT เอารุ่นละ 1 รูป — ทุกชนิดมีรูปครบ ไม่ต้องหาทดแทน

| # | รูป | ชนิด | เหตุผลที่เลือก |
|---|---|---|---|
| 1 | eval_images/S01_p2.jpg | Corus หน้า VALUE | รูปแรกของชนิดนี้ |
| 2 | bin/test5.jpg | Corus หน้า VALUE | รูปที่ 2 (ข้าม pictureForDemo/digital002.jpg = ภาพเดียวกับ S01_p2); เป็นรูปเดียวกับ spike |
| 3 | testpic/img5.jpg | Corus หน้า INDEX | รูปแรกของชนิดนี้ |
| 4 | experiment_images/normal_2.jpg | Corus หน้า INDEX | รูปที่ 2 |
| 5 | testpic/img1.jpeg | EK280 | EK280 รูปแรก (หน้าหลัก Vb/Vm/P/T/C) |
| 6 | experiment_images/hard_1.jpeg | MacBAT 5 | MacBAT รูปแรก (มีพลาสติกห่อ แสงสะท้อน) |
| 7 | testpic/IMG_0332.jpeg | Itron Corus Evo+ | รูปแรก (หน้า C/K/Q/Qb, จอเล็กในภาพ) |
| 8 | testpic/00129-RMG.jpeg | analog ตรง | analog หน้าตรงรูปแรก (ช่องแดงกำลังหมุน) |
| 9 | testpic/00136-RMG.jpg | analog ตรง | รูปที่ 2 (คนละเครื่องกับรูปแรก) |
| 10 | testpic/IMG_2231.jpeg | analog เอียง | analog เอียงรูปแรก (ลูกกลิ้งแนวตั้ง หมุน 90°) |

**วิธีตรวจ — แบ่งทุกช่อง (10 รูป × 8 ฟิลด์ = 80 ช่อง) เป็น 4 กลุ่ม** แล้วตรวจด้วยตา**ทุกค่า** (รวมกลุ่ม 1):
| กลุ่ม | ความหมาย | จำนวนช่อง | ผลตรวจด้วยตา |
|---|---|---|---|
| 1 | ทั้งคู่ได้ค่าและตรงกัน | 23 | ถูกทั้งคู่ 23/23 |
| 2 | ทั้งคู่ได้ค่าแต่ต่างกัน | 1 | Gemini ถูก, GLM ผิด (00129) |
| 3 | ได้ค่าฝ่ายเดียว | 6 | Gemini ได้ฝ่ายเดียวทั้ง 6: ถูก 5, ไม่แน่ใจ 1; GLM ควรได้ค่าแต่ไม่ได้ 5 |
| 4 | ไม่ได้ค่าทั้งคู่ | 50 | ภาพไม่มีค่านั้นให้อ่านจริงทั้ง 50 ช่อง |
- ตัวสอบเทียบ: ใน 10 ค่าที่ GLM ผิด (ข้อ 12.1) มีอยู่ในชุดนี้ 1 ค่า (00129) ซึ่งตกกลุ่ม 2 ไม่ใช่กลุ่ม 1 → วิธีแบ่งกลุ่มไม่ซ่อนค่าผิด
- การเทียบ: turbine เทียบสายตัวเลข (ตัดจุด/ศูนย์นำหน้า เพราะ Gemini ใส่จุดหน้าช่องแดง GLM ไม่ใส่);
  ฟิลด์อื่นเทียบค่าด้วย Decimal (`25.0` = `25.00`)
- ไม่แน่ใจ 1 ค่า: hard_1 บรรทัดแรก `VkT 4717 m3` — ภาพขยายอ่านได้ `4717` (ไม่เห็นจุด) และ Gemini ใส่ลง correct_v_scm
  (พร้อมติด low_confidence เอง) แต่ป้ายไม่ใช่ Vb, หน่วยเป็น m3 ไม่ใช่ Sm3 และขนาดไม่เข้ากับ Vm × C → ยืนยันไม่ได้ว่า
  เป็น CORRECT V. จึงไม่นับในความแม่นยำ

**ตารางสรุป (บทที่ 4):**
| ตัวชี้วัด (10 รูป, 29 ค่าที่ยืนยันเฉลยได้) | Gemini (gemini-3.5-flash, cloud) | GLM-OCR (local, Ollama) |
|---|---|---|
| ค่าที่อ่านถูก | **29 / 29 (100%)** | 23 / 29 (79.3%) |
| ค่าที่อ่านผิด (ตอบมาแต่ผิด) | 0 | 1 (3.4%) |
| ค่าที่ไม่ได้ (null ทั้งที่ภาพมีค่า) | 0 | 5 (17.2%) |
| ความถูกต้องของค่าที่ตอบมา | 29 / 29 (100%) | 23 / 24 (95.8%) |
| ตอบค่าที่ภาพไม่มี / ลงผิดฟิลด์ | 0 / 0 | 0 / 0 |
| รูปที่ไม่ได้ค่าเลย | 0 / 10 | 0 / 10 |
| รูปที่ได้ครบทุกค่าและถูกทั้งหมด | 10 / 10 | 6 / 10 |
| เวลาเฉลี่ยต่อรูป | 14.40 s (มัธยฐาน 7.13, ช่วง 4.55–38.18) | **1.64 s** (มัธยฐาน 1.80, ช่วง 0.7–2.0) |
| ต้องใช้อินเทอร์เน็ต / โควต้า | ใช่ / 20 คำขอต่อวัน | ไม่ / ไม่มี |

**รายรูป:**
| # | รูป | ค่าในภาพ | Gemini ถูก | GLM ถูก / ผิด / ไม่ได้ | เวลา Gemini (s) | เวลา GLM (s) |
|---|---|---|---|---|---|---|
| 1 | S01_p2 (Corus VALUE) | 5 | 5 | 5 / 0 / 0 | 7.25 | 1.8 |
| 2 | test5 (Corus VALUE) | 5 | 5 | 5 / 0 / 0 | 4.87 | 2.0 |
| 3 | img5 (Corus INDEX) | 2 | 2 | 2 / 0 / 0 | 4.55 | 1.7 |
| 4 | normal_2 (Corus INDEX) | 2 | 2 | 2 / 0 / 0 | 26.05 | 1.8 |
| 5 | img1 (EK280) | 5 | 5 | 2 / 0 / 3 | 33.34 | 1.8 |
| 6 | hard_1 (MacBAT) | 4 (+1 ไม่แน่ใจ) | 4 | 3 / 0 / 1 | 38.18 | 1.0 |
| 7 | IMG_0332 (Itron) | 3 | 3 | 2 / 0 / 1 | 6.88 | 1.8 |
| 8 | 00129-RMG (analog ตรง) | 1 | 1 | 0 / 1 / 0 | 7.01 | 2.0 |
| 9 | 00136-RMG (analog ตรง) | 1 | 1 | 1 / 0 / 0 | 6.70 | 1.8 |
| 10 | IMG_2231 (analog เอียง 90°) | 1 | 1 | 1 / 0 / 0 | 9.14 | 0.7 |
| | **รวม** | **29** | **29** | **23 / 1 / 5** | เฉลี่ย 14.40 | เฉลี่ย 1.64 |

**ประเภทความผิดพลาด (10 รูป):**
| ประเภท | Gemini | GLM |
|---|---|---|
| จุดทศนิยมหาย | 0 | 0 |
| หลักแรกผิด | 0 | 0 |
| หลักที่กำลังหมุน (ช่องแดง) หาย/อ่านผิด | 0 | 1 (00129: `7` → `1`) |
| ภาพกลับหัว | 0 | 0 (ไม่มีภาพกลับหัวในชุด 10 รูปนี้) |
| ไม่ได้ค่า: ป้ายบนจอไม่อยู่ในตาราง | 0 | 2 (MacBAT `p1` → P, Itron `Q` → Qm) |
| ไม่ได้ค่า: โมเดลถอดตัวเลขแต่ไม่ถอดป้าย | 0 | 2 (EK280 Vb, Vm — ตัวเลขถูกแต่กฎเหล็กไม่จับคู่ด้วยตำแหน่ง) |
| ไม่ได้ค่า: 2 คู่ในบรรทัดเดียว | 0 | 1 (EK280 `T 31.40 °C C 24.95448` → ได้แค่ T) |

**ข้อสังเกต:**
- จอ Corus (4 รูป, 14 ค่า) ทั้งสอง engine ถูกครบเท่ากัน — ความต่างอยู่ที่จอรุ่นอื่น (EK280/MacBAT/Itron) และ analog
- ความผิดของ GLM ส่วนใหญ่เป็น "ไม่ได้ค่า" (ปลอดภัย ผู้ใช้เห็นช่องว่าง) มี "ค่าผิด" 1 ค่า แต่เป็นยอด turbine ที่ใช้คิดเงิน
- Gemini ช้ากว่าและแปรปรวนมาก (4.6–38 s ใน 10 รูปนี้; รอบ 20 รูปวันที่ 2026-10-08 เฉลี่ย 21.9 s, มัธยฐาน 9.0 s,
  ช้าสุด 183.7 s ที่ IMG_1513) — ขึ้นกับภาระของ server Google; GLM คงที่ ~1–2 s
- Gemini เขียนตัวเลขใน JSON เป็น number จึงเสียศูนย์ท้าย (`25.00` → `25.0`, `529778.000` → `529778.0`) — ค่าเท่ากัน
  และ final_values เติมศูนย์ตามตารางทศนิยมอยู่แล้ว ไม่กระทบค่าที่บันทึก

**ข้อจำกัดของการทดลอง (ต้องเขียนในรายงาน):**
- ตัวอย่างเล็ก (10 รูป, 29 ค่า) — ตัวเลข % เป็นข้อบ่งชี้ ไม่ใช่ค่าทางสถิติ
- 1 ภาพต่อ 1 คำขอ ไม่ตรงกับการใช้งานจริงของ Gemini (หลายภาพในคำขอเดียว)
- ผล Gemini มาจาก 2 วัน: 6 รูปจากรอบวันที่ 2026-10-08 (ใช้ผลเดิม โค้ด/วิธีเดียวกัน ไม่เรียกซ้ำเพื่อประหยัดโควต้า)
  และ 4 รูปวันที่ 2026-10-10 (S01_p2, test5, normal_2, hard_1) — test5 และ hard_1 ได้ 503 ครั้งแรก ลองใหม่ 1 ครั้งผ่าน
- เวลาของ Gemini รวมเครือข่ายและคิวของ server Google; เวลาของ GLM ไม่รวมการโหลดโมเดลครั้งแรก
- เฉลยได้จากการตรวจด้วยตาบนภาพต้นฉบับโดยผู้ช่วย AI (ขยายภาพเมื่อจำเป็น) — ควรให้คนตรวจทานก่อนใช้ในรายงาน
- GLM ใช้ผลรอบ v4 (2026-10-08) prompt และกฎปัจจุบัน

---

## 13. ข้อจำกัดโควต้า Gemini free tier + เหตุการณ์จริง (สำหรับรายงานบทที่ 5)

**ข้อจำกัด**: Gemini API free tier จำกัด **20 คำขอต่อวันต่อโปรเจคต่อโมเดล**
(quota id `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, metric `generate_content_free_tier_requests`)
- ระบบเรียก 1 คำขอต่อการกด "อ่านค่า" 1 ครั้ง (ทุกภาพในคำขอเดียว) → ใช้งานจริงวันละ ≤ 20 ครั้ง
- งานทดลอง/ประเมินผล (1 รูป = 1 คำขอ) และการสาธิตแย่งโควต้าเดียวกัน
- เกินโควต้า → HTTP 429 `RESOURCE_EXHAUSTED` ทันที พร้อม `retryDelay` จนถึงรอบรีเซ็ต — retry ไม่ช่วย มีแต่ทำให้แย่ลง

**เหตุการณ์ที่เกิดขึ้นจริง:**
1. **2026-10-08** — เริ่มรัน Gemini กับชุด 66 รูปเพื่อเปรียบเทียบกับ GLM (1 รูปต่อ 1 คำขอ, สคริปต์มี retry
   สูงสุด 4 ครั้งเมื่อเจอ 429/503) → เจ้าของโปรเจคสั่งหยุดเพราะจะกินโควต้าที่ต้องใช้สาธิต แต่ตอนหยุด**ใช้ไปแล้ว 20 คำขอ**
   (รูปที่ 0–19 ไม่มี error) — แผนเดิมต้องใช้ 66 คำขอ ซึ่งเกินโควต้า 1 วันไปกว่า 3 เท่า
2. ลดแผนเหลือ 10 รูป (ใช้ผลเดิม 6 รูปที่อ่านไปแล้ว) เขียนสคริปต์ใหม่: ปิด retry ของ SDK
   (`HttpRetryOptions(attempts=1)`), หน่วง ≥ 5 วินาทีระหว่างคำขอ, เจอ error โควต้า → หยุดทันที ห้าม retry
3. คำขอแรกของรอบใหม่ (S01_p2) ได้ **429 ทันที**: `quotaValue 20`, `retryDelay ≈ 59,522 s (~16.5 ชั่วโมง)` → หยุดตามกฎ
   (ใช้ไป 1 คำขอที่ล้มเหลว) — **Gemini ใช้ไม่ได้ทั้งระบบจนโควต้ารีเซ็ต**; ถ้าต้องสาธิตในช่วงนั้น ทางเดียวคือ
   `OCR_ENGINE=glm` (ข้อ 3.18)
4. **2026-10-10** — โควต้ารีเซ็ตแล้ว รัน 4 รูปที่ขาดเป็นงานแรกก่อนใช้ Gemini ทำอย่างอื่น: S01_p2 ✓, test5 ✗ 503,
   normal_2 ✓, hard_1 ✗ 503 (`UNAVAILABLE: This model is currently experiencing high demand`) → รอ 90 วินาทีแล้วลองใหม่
   รูปละ 1 ครั้งเท่านั้น → ผ่านทั้ง 2 รวมใช้ 6 คำขอในวันนั้น (ไม่ทราบว่าคำขอที่ได้ 503 ถูกนับโควต้าหรือไม่)

**บทเรียน:**
- ต้อง**วางงบโควต้าก่อนรันทดลอง** (จำนวนรูป × จำนวนรอบ) และกันโควต้าไว้สาธิตเสมอ
- retry อัตโนมัติกับ 429 = เผาโควต้าที่เหลือ → ปิด retry และหยุดทันทีเมื่อเจอ error โควต้า
- 503 (server Google ภาระสูง) เป็นอีกความเสี่ยงที่ควบคุมไม่ได้ แม้โควต้ายังเหลือ
- ระบบที่พึ่ง API ภายนอกตัวเดียวมีจุดล้มเดียว — สถาปัตยกรรม 2 engine (ข้อ 3.13, 3.18) ทำให้ระบบเดินต่อได้
  ด้วย GLM บนเครื่อง (แลกกับความแม่นยำ ข้อ 12); ทางเลือกอื่นคือ Gemini แบบเสียเงิน (โควต้าสูงกว่ามาก)
