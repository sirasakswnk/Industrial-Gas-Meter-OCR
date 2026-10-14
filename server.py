import io
import os
import json
import re
import time
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps
import cv2
import numpy as np
import torch
from pydantic import BaseModel, Field
from ultralytics import YOLO

# ── 1. ENVIRONMENT & API CONFIGURATION ────────────────────────────────
from dotenv import load_dotenv
load_dotenv()  # โหลดค่าจากไฟล์ .env เข้าสู่ environment

# OCR engine: เลือกจาก .env (OCR_ENGINE=..., ค่าเริ่มต้น "gemini") — โค้ดเรียกโมเดลอยู่ใน ocr_engines/
# ชื่อที่ไม่รู้จัก -> หยุดตั้งแต่เปิด server พร้อมบอกตัวเลือก
# engine แต่ละตัว fail-fast เรื่อง key/config ของตัวเองตอนสร้าง (เช่น Gemini ไม่มี GEMINI_API_KEY)
import ocr_engines
from ocr_engines import MeterReading

OCR_ENGINE = (os.environ.get("OCR_ENGINE") or ocr_engines.DEFAULT_ENGINE).strip().lower()
try:
    engine = ocr_engines.create_engine(OCR_ENGINE)
except ocr_engines.UnknownEngineError as e:
    raise RuntimeError(str(e)) from None
print(f"🔤 OCR engine: {engine.name}")

# ── ชื่อเดิมที่ evaluate.py / เทสต์อ้างผ่าน server.* — pipeline ไม่ได้ใช้แล้ว (เรียกผ่าน engine.read) ──
# คงไว้เพื่อไม่ต้องแก้เทสต์/evaluate.py: เทสต์ mock โมเดลด้วย patch.object(server.client.models, ...)
# ซึ่งเป็น client ตัวเดียวกับที่ GeminiEngine ใช้ (engine อื่นไม่มี client -> None)
from ocr_engines.gemini_engine import GEMINI_MODEL, types, parse_gemini_response, extract_raw_values  # noqa: E402,F401
from ocr_engines.prompts import SYSTEM_INSTRUCTION, build_user_prompt as build_gemini_prompt  # noqa: E402,F401
client = getattr(engine, "client", None)

# เลือก device อัตโนมัติ: มี CUDA ใช้ GPU (RTX 4060) ไม่มีก็ถอยไป CPU
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"🖥️ YOLO inference device: {'CUDA GPU' if DEVICE == 0 else 'CPU'}")

# ── โมเดลเดียวของระบบ: YOLO-OBB (ให้ครบในตัวเดียว: ตำแหน่ง + class + มุมหมุน) ──
# แทนสถาปัตยกรรม 2 โมเดลเดิม (HBB best.pt + OBB) — เหตุผลดู PROJECT_CONTEXT ข้อ 3.4
# best.pt ยังเก็บไว้ในโฟลเดอร์เผื่อเทียบย้อนหลัง แต่ไม่ถูกโหลดแล้ว
# override path ได้ผ่าน .env: OBB_MODEL_PATH=...
OBB_MODEL_PATH = os.environ.get("OBB_MODEL_PATH", "best_obb.pt")
obb_model = None
if os.path.exists(OBB_MODEL_PATH):
    try:
        obb_model = YOLO(OBB_MODEL_PATH)
        print(f"📐 OBB model loaded: {OBB_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ โหลดโมเดล OBB ไม่สำเร็จ ({e}) — fail-safe: จะส่งภาพต้นฉบับเต็มเข้า Gemini")
else:
    print(f"⚠️ ไม่พบโมเดล OBB '{OBB_MODEL_PATH}' — ระบบยังทำงานได้แต่ไม่มี detection/deskew "
          f"(fail-safe: ส่งภาพต้นฉบับเต็มเข้า Gemini)")

# โฟลเดอร์เก็บภาพ intermediate สำหรับ debug ดูย้อนหลัง
# (แยกจาก static/ เพื่อไม่ให้เข้าถึงได้ผ่านเว็บ)
DEBUG_DIR = os.path.join(os.path.dirname(__file__), "debug_images")
os.makedirs(DEBUG_DIR, exist_ok=True)

# เก็บค่าดิบ + ภาพต้นฉบับของทุกคำขอ OCR ไว้ฝั่ง server (ocr_requests/<request_id>/)
# ตอนบันทึก (/api/readings) ดึง raw_values/ภาพจากที่นี่ ไม่เชื่อค่าที่ frontend ส่งกลับมา
# โตเรื่อยๆ เหมือน debug_images (คำขอที่ไม่ได้บันทึกก็ถูกเก็บ) — เคลียร์เฉพาะโฟลเดอร์ที่ไม่มี reading_id
OCR_REQUEST_DIR = os.environ.get("OCR_REQUEST_DIR",
                                 os.path.join(os.path.dirname(__file__), "ocr_requests"))
os.makedirs(OCR_REQUEST_DIR, exist_ok=True)
REQUEST_ID_PATTERN = re.compile(r"^\d+_[0-9a-f]{6}$")

# ฐานข้อมูลบันทึกค่า (SQLite ไฟล์เดียว) — สร้างตาราง + seed Meter Run ตัวอย่างถ้ายังไม่มี
import db
import reading_fields
import report_pdf
db.init_db()

# ── PIPELINE CONFIGURATION ──
# OBB ทำงานบนภาพต้นฉบับเต็ม -> ได้ 4 มุมจอ -> ขยายมุมออกจาก centroid ตามสัดส่วนนี้ก่อน warp
# กันเลขหลักแรก/จุดทศนิยม/แถบแดงของ analog โดนตัด
OBB_WARP_PADDING = 0.06

# ภาพ deskew ถูกเซฟลง debug_images เสมอเพื่อการตรวจสอบย้อนหลัง (deliverable ให้พี่เลี้ยงดู)
# flag นี้เลือกว่าจะส่งภาพแบบไหนเข้า Gemini:
#   True (default)  = ส่งภาพที่ warp จาก 4 มุม OBB แล้ว (ดัดตรง)
#   False           = ส่ง crop แนวแกนจากมุมเดียวกัน (ไม่ดัดตรง) — ไว้ทำ eval A/B
SEND_DESKEWED_TO_GEMINI = True

# เกณฑ์ confidence ขั้นต่ำของ OBB ที่จะยอมใช้ผลมาหมุนภาพ
OBB_CONF_THRESHOLD = 0.4


def save_debug_image(pil_img: Image.Image, request_id: str, idx: int, stage: str):
    """เซฟภาพระหว่างทางของ pipeline ลงเครื่อง: debug_images/<request_id>_imgN_<stage>.jpg"""
    filename = f"{request_id}_img{idx}_{stage}.jpg"
    pil_img.convert("RGB").save(os.path.join(DEBUG_DIR, filename), quality=90)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """เรียงมุม 4 จุดให้เป็นลำดับ: บนซ้าย, บนขวา, ล่างขวา, ล่างซ้าย (ตามเข็มนาฬิกา)

    ใช้การเรียงตามมุมรอบ centroid (atan2) แทนวิธี x+y/y-x แบบเดิม
    เหตุผล: วิธี sum/diff จะ assign มุมซ้ำเมื่อสี่เหลี่ยมหมุนมาก (~45°)
    ซึ่งเป็นสเปคหลักของโปรเจค (ภาพเอียง ±45°) — atan2 ทนทุกองศาโดยไม่มีจุดซ้ำ
    """
    pts = np.asarray(pts, dtype=np.float32)
    center = pts.mean(axis=0)

    # มุมรอบ centroid (แกนภาพ y ชี้ลง): เรียงจากน้อยไปมากได้ลำดับวนรอบจอ
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]

    # จัดให้จุดแรกเป็น "บนซ้าย" (ผลรวม x+y น้อยสุด) แล้ววน list ให้เริ่มจากจุดนั้น
    start = int(np.argmin(ordered.sum(axis=1)))
    rect = np.roll(ordered, -start, axis=0)

    # กัน winding กลับด้าน: ถ้าจุดที่ 2 (ควรเป็นบนขวา) อยู่ซ้ายของจุดแรก แสดงว่าวนทวนเข็ม
    # ให้พลิกลำดับ 3 จุดหลังเป็นตามเข็ม → TL,TR,BR,BL เสมอทุกมุมเอียง
    if rect[1][0] < rect[0][0]:
        rect = rect[[0, 3, 2, 1]]
    return rect.astype(np.float32)


def expand_corners(pts: np.ndarray, ratio: float) -> np.ndarray:
    """ขยาย 4 มุมออกจาก centroid ตามสัดส่วน (0.06 = เผื่อขอบ 6%)

    ใช้ก่อน warp เพื่อกันเลขหลักแรก/จุดทศนิยม/แถบแดงของ analog โดนตัดขาด
    การขยายรอบ centroid เป็น uniform scale จึงไม่ทำให้ลำดับมุมเปลี่ยน
    """
    pts = np.asarray(pts, dtype=np.float32)
    center = pts.mean(axis=0)
    return (center + (pts - center) * (1.0 + ratio)).astype(np.float32)


def detect_meter_obb(pil_img: Image.Image):
    """
    รัน YOLO-OBB บนภาพต้นฉบับเต็ม — โมเดลเดียวให้ครบ: ตำแหน่ง + class + มุมหมุน

    คืน: (corners, class_name, conf) ของกรอบที่ confidence สูงสุด
         corners = ndarray (4,2) พิกัด pixel ของภาพที่ป้อนเข้า (ไม่ต้อง normalize)

    Fail-safe: คืน (None, None, 0.0) ถ้าไม่มีโมเดล / ไม่เจอกรอบ / inference พัง
    ผู้เรียกต้อง fallback ไปใช้ภาพเดิมเอง
    """
    if obb_model is None:
        return None, None, 0.0

    try:
        results = obb_model(pil_img, device=DEVICE, conf=OBB_CONF_THRESHOLD, verbose=False)
        obb = results[0].obb

        if obb is None or len(obb) == 0:
            return None, None, 0.0

        best_idx = int(obb.conf.argmax())
        conf_val = float(obb.conf[best_idx])
        cls_name = obb_model.names[int(obb.cls[best_idx])]
        corners = obb.xyxyxyxy[best_idx].cpu().numpy().reshape(4, 2).astype(np.float32)
        return corners, cls_name, conf_val

    except Exception as e:
        print(f"⚠️ OBB detect failed: {e}")
        return None, None, 0.0


def deskew_from_corners(pil_img: Image.Image, corners: np.ndarray) -> Optional[Image.Image]:
    """
    warp perspective จาก 4 มุมของ OBB -> ภาพจอที่ดัดตรงแล้ว

    ต่างจาก classical CV เดิม: 4 มุมมาจากการเรียนรู้ข้อมูลจริง ไม่พึ่ง edge detection
    จึงทนแสงสะท้อน/พลาสติกห่อ/ฝุ่นได้ดีกว่ามาก
    ขั้นตอน: เผื่อขอบ (OBB_WARP_PADDING) -> order_corners -> warpPerspective
             -> ถ้าสูงกว่ากว้าง (ตัวเลขเรียงตั้ง) หมุน 90° ให้เป็นแนวนอน

    Fail-safe: คืน None ถ้าขนาดผิดปกติ / warp พัง
    """
    try:
        q = order_corners(expand_corners(corners, OBB_WARP_PADDING))

        # ขนาดปลายทางจากค่าเฉลี่ยของด้านคู่ตรงข้าม
        out_w = int((np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2)
        out_h = int((np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2)
        out_w, out_h = max(out_w, 1), max(out_h, 1)
        if out_w < 20 or out_h < 20:
            print(f"📐 OBB deskew: output too small ({out_w}x{out_h})")
            return None

        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(q, dst)
        warped = cv2.warpPerspective(cv_img, M, (out_w, out_h))

        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

    except Exception as e:
        print(f"⚠️ OBB deskew failed (using original): {e}")
        return None


def crop_from_corners(pil_img: Image.Image, corners: np.ndarray) -> Image.Image:
    """crop กรอบสี่เหลี่ยมแนวแกน (ไม่ดัดตรง) ที่ครอบ 4 มุม OBB

    ใช้เมื่อ SEND_DESKEWED_TO_GEMINI=False หรือ warp พัง — ยังได้ crop ไม่เสียของ
    และทำให้เทียบ A/B ได้ว่า "crop เฉยๆ" กับ "crop+deskew" อันไหน Gemini อ่านแม่นกว่า
    """
    q = expand_corners(corners, OBB_WARP_PADDING)
    x1, y1 = int(q[:, 0].min()), int(q[:, 1].min())
    x2, y2 = int(q[:, 0].max()), int(q[:, 1].max())
    return pil_img.crop((
        max(0, x1), max(0, y1),
        min(pil_img.width, x2), min(pil_img.height, y2),
    ))


def preprocess_meter_image(
    original_img: Image.Image,
    request_id: str,
    img_idx: int,
    source_name: str = "",
) -> tuple[Image.Image, Optional[str]]:
    """
    Pipeline กลางสำหรับเตรียมภาพก่อนส่ง Gemini (ใช้ร่วมกันทุก endpoint)

    สถาปัตยกรรมโมเดลเดียว (YOLO-OBB) — แทนของเดิมที่ใช้ HBB + OBB:
      - รัน OBB บน "ภาพต้นฉบับเต็ม" -> ได้ 4 มุมจอ + class + conf ในครั้งเดียว
      - warp จาก 4 มุมนั้นโดยตรง (เผื่อขอบ OBB_WARP_PADDING) -> ภาพจอที่ตรง -> ส่ง Gemini
      - class ที่ OBB ให้มา ใช้เป็น soft hint ส่งต่อให้ build_gemini_prompt()

    รับ: ภาพ PIL ที่ผ่าน exif_transpose + convert("RGB") แล้ว
    คืน: (ภาพพร้อมส่ง Gemini, ชื่อ class ที่ OBB เจอ)
    """
    img_to_process = original_img

    # ── STAGE 1: OBB detect บนภาพต้นฉบับเต็ม (ตำแหน่ง + class + มุม ในครั้งเดียว) ──
    corners, detected_class, conf_val = detect_meter_obb(original_img)

    if corners is None:
        # fail-safe: ไม่เจอกรอบ / ไม่มีโมเดล -> ส่งภาพเดิมเข้า Gemini ต่อ ไม่ throw
        print(f"⚠️ OBB missed. Using image as-is for: {source_name}")
    else:
        print(f"🎯 OBB detected [{detected_class}] conf={conf_val:.2f} from: {source_name}")

        # ── STAGE 1.5: deskew จาก 4 มุม — เซฟเป็น deliverable เสมอเมื่อ warp สำเร็จ ──
        deskewed = deskew_from_corners(original_img, corners)
        if deskewed is not None:
            save_debug_image(deskewed, request_id, img_idx, "1_obb_deskew")

        if SEND_DESKEWED_TO_GEMINI and deskewed is not None:
            img_to_process = deskewed
            print(f"📐 Deskew (OBB) applied → Gemini "
                  f"({deskewed.width}x{deskewed.height}) for: {source_name}")
        else:
            # flag ปิด หรือ warp พัง -> ใช้ crop แนวแกนจากมุมชุดเดียวกัน (ยังได้ crop ไม่เสียของ)
            img_to_process = crop_from_corners(original_img, corners)
            save_debug_image(img_to_process, request_id, img_idx, "1_obb_crop_nodeskew")
            reason = "flag off" if deskewed is not None else "warp failed"
            print(f"📐 ใช้ crop แนวแกนแทน deskew ({reason}) "
                  f"({img_to_process.width}x{img_to_process.height}) for: {source_name}")

    # ── STAGE 2: OpenCV preprocessing ──
    cv_img = cv2.cvtColor(np.array(img_to_process), cv2.COLOR_RGB2BGR)
    cv_img = cv2.detailEnhance(cv_img, sigma_s=10, sigma_r=0.15)
    final_pil_image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    save_debug_image(final_pil_image, request_id, img_idx, "2_sent_to_gemini")

    return final_pil_image, detected_class


ANALOG_CLASS = "analog_dial"
DIGITAL_CLASS = "digital_screen"


def check_request_meter_types(filenames: List[str], detected_classes: List[Optional[str]]) -> Optional[str]:
    """ตัวดักชนิดมิเตอร์ในคำขอเดียว — คืนข้อความ error (ต้องบล็อก) หรือ None (ผ่าน)

    กฎปัจจุบัน (ผ่อนจากข้อ 3.6 เดิมที่บล็อก analog + digital):
      1 Meter Run = EVC (digital, อาจหลายหน้าจอ) + Turbine (analog, 1 หน้าปัดเสมอ)
      - analog + digital มาด้วยกันได้  - digital หลายภาพได้ (เช่น Corus หลายหน้าจอ)
      - analog > 1 ภาพ -> บล็อก (แปลว่าเอา Turbine คนละ Run มาปน)
    ภาพที่ OBB หาไม่เจอ (None) ไม่ถูกนับ — ใช้ร่วมกันทั้ง /api/ocr และ evaluate.py
    """
    analog = [i for i, c in enumerate(detected_classes) if c == ANALOG_CLASS]
    if len(analog) > 1:
        details = ", ".join(f"รูปที่ {i + 1} ({filenames[i]})" for i in analog)
        return ("พบมิเตอร์ analog มากกว่า 1 ตัวในคำขอเดียว "
                f"หนึ่งรายการจดรองรับ Turbine Meter ได้ 1 ตัว ({details})")
    return None


# ฟิลด์ที่มาจาก EVC เท่านั้น (ทุกฟิลด์ตัวเลขยกเว้น turbine_v_m3) — ใช้ในตาข่ายหลัง parse
EVC_ONLY_FIELDS = [k for k in reading_fields.FIELD_KEYS if k != reading_fields.TURBINE_FIELD] + ["pulse_weight"]


def apply_analog_only_guard(result: "OcrResult", detected_classes: List[Optional[str]]) -> dict:
    """ตาข่ายหลัง parse (ชั้นสำรองของกฎข้อ 5 ใน prompt): คำขอที่เป็น analog ล้วน
    แต่โมเดลใส่ค่าในฟิลด์ EVC มา -> ทิ้งเป็น null + log warning (ไม่ error)

    ทำงานเฉพาะเมื่อไม่มีภาพที่ OBB จัดเป็น digital เลย และ (OBB เจอ analog หรือ meter_type = analog)
    คำขอที่มี digital ปนอยู่ไม่แตะ: ผลลัพธ์รวมเป็นก้อนเดียว แยกไม่ได้ว่าค่าไหนมาจากภาพไหน
    (ถ้าดูแค่ meter_type อาจทิ้งค่า EVC ที่ถูกต้องทั้งหมดแบบเงียบๆ)

    คืน {field: ค่าดิบเดิม} ของฟิลด์ที่ถูกทิ้ง (เก็บไว้ตรวจย้อนหลังใน ocr_requests/…/result.json)
    """
    if DIGITAL_CLASS in detected_classes:
        return {}
    if ANALOG_CLASS not in detected_classes and (result.meter_type or "").strip().lower() != "analog":
        return {}

    dropped = {}
    for key in EVC_ONLY_FIELDS:
        raw = result.raw_values.get(key)
        parsed = getattr(result, key, None)
        if raw is None and parsed is None:
            continue
        dropped[key] = raw if raw is not None else reading_fields.to_plain_str(parsed)
        setattr(result, key, None)
        if key in result.raw_values:
            result.raw_values[key] = None
    if dropped:
        result.low_confidence_fields = [f for f in result.low_confidence_fields if f not in dropped]
        print(f"⚠️ analog guard: โมเดลใส่ค่าฟิลด์ EVC มาในคำขอที่เป็น analog ล้วน -> ทิ้ง {dropped}")
    return dropped


def build_ocr_result(engine_result: ocr_engines.EngineResult,
                     detected_classes: List[Optional[str]]) -> "OcrResult":
    """ผลลัพธ์ส่งหน้า OCR: ค่าที่ parse แล้ว + ค่าดิบ (สตริง) + ค่าเริ่มต้นสำหรับช่องทีละหลัก
    ผ่านตาข่าย analog ก่อนคำนวณค่าเริ่มต้นของช่อง (ค่าที่ถูกทิ้งไม่ขึ้นในช่อง)
    ไม่รู้ว่าผลมาจาก engine ไหน — ใช้แค่ EngineResult"""
    result = OcrResult(**engine_result.reading.model_dump(), raw_values=dict(engine_result.raw_values),
                       raw_source=engine_result.raw_source)
    result.guard_dropped = apply_analog_only_guard(result, detected_classes)
    result.guard_applied = bool(result.guard_dropped)
    result.editor = reading_fields.initial_editor_values(result.raw_values)
    if result.editor["truncated_fields"]:
        # ตัดทศนิยมเกินเป็นเรื่องปกติตามสเปค -> log อย่างเดียว ไม่เตือนบนหน้าจอ
        print("ℹ️ ตัดทศนิยมเกินตามสเปค: " + ", ".join(
            f"{k}={result.raw_values[k]}" for k in result.editor["truncated_fields"]))
    return result


def _project_relative(path: str) -> str:
    """path ที่เก็บลง DB: relative กับโฟลเดอร์โปรเจค (ย้ายโฟลเดอร์ได้) ถ้าอยู่นอกโปรเจคใช้ absolute"""
    project = os.path.dirname(os.path.abspath(__file__))
    full = os.path.abspath(path)
    try:
        if os.path.commonpath([project.lower(), full.lower()]) == project.lower():
            return os.path.relpath(full, project).replace(os.sep, "/")
    except ValueError:  # คนละไดรฟ์ (Windows)
        pass
    return full


def _ocr_request_meta_path(request_id: str) -> str:
    return os.path.join(OCR_REQUEST_DIR, request_id, "result.json")


def persist_ocr_request(request_id: str, result: "OcrResult", uploads: List[tuple],
                        confidence_scores: Optional[dict] = None) -> Optional[str]:
    """เก็บค่าดิบ + low_confidence + ภาพต้นฉบับของคำขอ OCR ลง ocr_requests/<request_id>/

    uploads = [(ชื่อไฟล์, bytes), ...] ภาพที่ผู้ใช้ส่งมา (ก่อน preprocess — เป็นหลักฐานตรวจย้อนหลัง)
    Fail-safe: เก็บไม่สำเร็จคืน None — ผล OCR ยังแสดงได้ แค่บันทึกไม่ได้
    """
    try:
        folder = os.path.join(OCR_REQUEST_DIR, request_id)
        os.makedirs(folder, exist_ok=True)
        image_paths = []
        for idx, (name, data) in enumerate(uploads):
            safe = re.sub(r"[^\w.-]", "_", os.path.basename(name or "")) or "image"
            fname = f"img{idx}_{safe}"
            full_path = os.path.join(folder, fname)
            with open(full_path, "wb") as f:
                f.write(data)
            image_paths.append(_project_relative(full_path))

        meta = {
            "request_id": request_id,
            "engine": engine.name,
            "raw_values": result.raw_values,
            "raw_source": result.raw_source,
            "guard_dropped": result.guard_dropped,
            "low_confidence_fields": result.low_confidence_fields,
            # confidence รายฟิลด์จาก engine (GLM: logprobs) — เก็บไว้ลง DB อย่างเดียว ไม่ส่งให้หน้าเว็บ
            # (วัดแล้วไม่สัมพันธ์กับความถูกต้อง: ค่าผิดได้ 0.57-1.0 — แสดงแล้วจะทำให้เชื่อค่าที่ผิด)
            "confidence_scores": confidence_scores,
            "image_paths": image_paths,
            "reading_id": None,  # ใส่เมื่อบันทึกแล้ว — กันบันทึกผลเดียวกันซ้ำ
        }
        with open(_ocr_request_meta_path(request_id), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return request_id
    except Exception as e:
        print(f"⚠️ เก็บผล OCR ไม่สำเร็จ ({request_id}): {e}")
        return None


app = FastAPI(title="Industrial Gas Meter OCR API (YOLO-OBB + OpenCV + VLM)")

# ── 2. PYDANTIC DATA SCHEMA ───────────────────────────────────────────
# MeterReading (schema ผลการอ่าน) อยู่ที่ ocr_engines/base.py

class OcrResult(MeterReading):
    """response ของ /api/ocr — แยกจาก MeterReading เพราะ MeterReading เป็น schema ที่ส่งให้โมเดล"""
    raw_values: Dict[str, Optional[str]] = Field(default_factory=dict)
    raw_source: str = "text"  # "fallback" = ค่าดิบมาจาก float ไม่ใช่ข้อความ -> หน้าจอขึ้นเตือน
    guard_dropped: Dict[str, Optional[str]] = Field(default_factory=dict)  # ค่าที่ตาข่าย analog ทิ้ง
    guard_applied: bool = False  # True = ตาข่ายทิ้งค่า EVC -> หน้า OCR ขึ้นข้อความบอก (ไม่ให้เข้าใจว่าอ่านไม่ออก)
    editor: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = None  # ใช้อ้างอิงตอนบันทึก — None = เก็บผลไม่สำเร็จ (บันทึกไม่ได้)

# ── 4. CORE API ENDPOINTS ─────────────────────────────────────────────

@app.post("/api/ocr", response_model=OcrResult)
async def perform_ocr(files: List[UploadFile] = File(...)):
    """Endpoint หลัก Multi-Page: OBB Detect + Deskew -> ตรวจชนิดมิเตอร์ไม่ปนกัน -> OCR engine -> Unified JSON"""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    request_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    img_idx = 0

    images_for_engine = []
    detected_classes = []
    filenames = []
    uploads = []  # (ชื่อไฟล์, bytes ต้นฉบับ) เก็บเป็นหลักฐานคู่กับผลการอ่าน

    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"Invalid file type for {file.filename}")

        try:
            contents = await file.read()
            original_img = ImageOps.exif_transpose(Image.open(io.BytesIO(contents)))
            if original_img.mode != "RGB":
                original_img = original_img.convert("RGB")

            processed_img, detected_class = preprocess_meter_image(
                original_img, request_id, img_idx, source_name=file.filename
            )
            images_for_engine.append(processed_img)
            detected_classes.append(detected_class)
            filenames.append(file.filename)
            uploads.append((file.filename, contents))
            img_idx += 1

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process image {file.filename}: {str(e)}")

    # ── ดักชนิดมิเตอร์ในคำขอเดียว (analog > 1 ภาพ = Turbine คนละ Run ปนกัน) ──
    # ต้องอยู่ก่อน try ที่เรียกโมเดล: except Exception ข้างล่างจะแปลงทุก error เป็น 500
    # ค่าผิดอันตรายกว่าค่าที่หายไป -> บล็อกเลย ไม่เสียค่า API
    blocked = check_request_meter_types(filenames, detected_classes)
    if blocked:
        raise HTTPException(status_code=400, detail=blocked)

    try:
        engine_result = engine.read(images_for_engine, detected_classes)
        result = build_ocr_result(engine_result, detected_classes)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เรียกโมเดลอ่านค่าไม่สำเร็จ: {str(e)}")

    result.request_id = persist_ocr_request(request_id, result, uploads, engine_result.confidence_scores)
    return result


@app.post("/api/ocr/sample", response_model=OcrResult)
async def perform_ocr_sample(filename: str):
    """ Endpoint ทดสอบภาพเดี่ยวตัวอย่างระบบชิปหลังบ้าน """
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(os.path.dirname(__file__), "static", "samples", safe_filename)

    if not os.path.exists(file_path):
        file_path = os.path.join(os.path.dirname(__file__), safe_filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Sample file {safe_filename} not found.")

    request_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"

    try:
        img = ImageOps.exif_transpose(Image.open(file_path))
        if img.mode != "RGB":
            img = img.convert("RGB")

        img, detected_class = preprocess_meter_image(
            img, request_id, 0, source_name=safe_filename
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to open sample image: {str(e)}")

    try:
        engine_result = engine.read([img], [detected_class])
        result = build_ocr_result(engine_result, [detected_class])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เรียกโมเดลอ่านค่าไม่สำเร็จ: {str(e)}")

    with open(file_path, "rb") as f:
        result.request_id = persist_ocr_request(request_id, result, [(safe_filename, f.read())],
                                                engine_result.confidence_scores)
    return result

# ── 4.5 METER RUN ENDPOINTS (หน้าแรก: เลือก/เพิ่ม Meter Run) ────────────
# ใช้ def ธรรมดา (ไม่ใช่ async) ให้ FastAPI รันใน threadpool เพราะ sqlite3 เป็น blocking I/O

class MeterRunCreate(BaseModel):
    company: str
    run_label: str
    tag_fe: str = ""
    serial_fe: str = ""
    tag_fy: str = ""
    serial_fy: str = ""


@app.get("/api/meter-runs")
def list_meter_runs():
    return db.list_meter_runs()


@app.get("/api/meter-runs/{meter_run_id}")
def get_meter_run(meter_run_id: int):
    run = db.get_meter_run(meter_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="ไม่พบ Meter Run นี้ในระบบ")
    return run


@app.delete("/api/meter-runs/{meter_run_id}", status_code=204)
def delete_meter_run(meter_run_id: int):
    """ลบ Meter Run ที่ยังไม่มีประวัติการจด (มีแล้ว -> 409 เพื่อไม่ให้ประวัติหลุดจาก Run)"""
    try:
        deleted = db.delete_meter_run(meter_run_id)
    except db.MeterRunInUseError as e:
        raise HTTPException(status_code=409,
                            detail=f"ลบไม่ได้ — Meter Run นี้มีประวัติการจด {e.reading_count} รายการ")
    if not deleted:
        raise HTTPException(status_code=404, detail="ไม่พบ Meter Run นี้ในระบบ")
    return Response(status_code=204)


@app.get("/api/meter-runs/{meter_run_id}/latest-reading")
def get_latest_reading(meter_run_id: int):
    """ค่าจดครั้งล่าสุดของ Run (แสดงเป็น "ค่าจดครั้งก่อน" บนหน้า OCR) — ไม่มีประวัติคืน null"""
    if db.get_meter_run(meter_run_id) is None:
        raise HTTPException(status_code=404, detail="ไม่พบ Meter Run นี้ในระบบ")
    return db.get_previous_reading(meter_run_id)


class DecreaseCheck(BaseModel):
    entered_values: Dict[str, Optional[str]]       # ค่าในช่องทีละหลัก (เหมือนตอนบันทึก)
    turbine_trim_digits: int = 0


@app.post("/api/meter-runs/{meter_run_id}/decrease-check")
def check_decrease(meter_run_id: int, payload: DecreaseCheck):
    """เทียบค่าที่กำลังจะบันทึกกับค่าจดครั้งก่อน (ค่าเดียวกับที่หน้า OCR แสดง) -> ฟิลด์ค่าสะสมที่ผลต่างติดลบ
    ใช้ compute_final_values ตัวเดียวกับตอนบันทึก -> Turbine เทียบหลังตัดทศนิยม ตรงกับค่าที่จะลง PDF
    เป็นคำเตือนเท่านั้น ไม่บล็อกการบันทึก"""
    if db.get_meter_run(meter_run_id) is None:
        raise HTTPException(status_code=404, detail="ไม่พบ Meter Run นี้ในระบบ")
    if payload.turbine_trim_digits not in (0, 1, 2):
        raise HTTPException(status_code=400, detail="จำนวนหลักที่ตัดต้องเป็น 0, 1 หรือ 2")
    try:
        final_values = reading_fields.compute_final_values(
            payload.entered_values, payload.turbine_trim_digits)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    previous = db.get_previous_reading(meter_run_id)
    return {
        "previous_reading_id": previous["id"] if previous else None,
        "decreased": reading_fields.decreased_fields(
            final_values, previous["final_values"] if previous else None),
    }


@app.get("/api/reading-fields")
def get_reading_fields():
    """ตารางฟิลด์ (label/หน่วย/ทศนิยมตายตัว) — frontend ใช้วาดช่องทีละหลัก ไม่ต้องเขียนซ้ำใน JS"""
    return {"fields": reading_fields.FIELDS, "digit_boxes": reading_fields.DIGIT_BOXES,
            "turbine_field": reading_fields.TURBINE_FIELD}


@app.post("/api/meter-runs", status_code=201)
def create_meter_run(payload: MeterRunCreate):
    try:
        return db.create_meter_run(**payload.model_dump())
    except db.DuplicateMeterRunError as e:
        raise HTTPException(status_code=409, detail=f"Meter Run นี้มีอยู่แล้วในระบบ ({e})")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# ── 4.6 READING ENDPOINTS (บันทึก + PDF) ───────────────────────────────

class ReadingCreate(BaseModel):
    meter_run_id: int
    request_id: str                                # จาก /api/ocr — ใช้ดึงค่าดิบ/ภาพที่ server เก็บไว้
    recorded_at: str                               # วัน/เวลาที่จด (ผู้ใช้กรอก)
    entered_values: Dict[str, Optional[str]]       # ค่าในช่องทีละหลัก (Turbine = ตัวเลขล้วนก่อนตัด)
    turbine_trim_digits: int = 0


def load_ocr_request(request_id: str) -> dict:
    if not REQUEST_ID_PATTERN.match(request_id or ""):  # กัน path traversal
        raise HTTPException(status_code=400, detail="request_id ไม่ถูกต้อง")
    path = _ocr_request_meta_path(request_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="ไม่พบผลการอ่านนี้ในระบบ กรุณาอ่านค่าใหม่")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/readings", status_code=201)
def create_reading(payload: ReadingCreate):
    """บันทึกการจดค่า 1 ครั้ง — final_values/edited_fields คำนวณที่นี่ (reading_fields.py)
    ไม่เชื่อค่าที่ frontend คำนวณมา, raw_values/ภาพ/low_confidence ดึงจากที่ server เก็บตอน OCR"""
    if db.get_meter_run(payload.meter_run_id) is None:
        raise HTTPException(status_code=404, detail="ไม่พบ Meter Run นี้ในระบบ")
    meta = load_ocr_request(payload.request_id)
    if meta.get("reading_id"):
        raise HTTPException(status_code=409,
                            detail=f"ผลการอ่านนี้ถูกบันทึกไปแล้ว (รายการที่ #{meta['reading_id']})")
    try:
        datetime.fromisoformat(payload.recorded_at)
    except ValueError:
        raise HTTPException(status_code=400, detail="วัน/เวลาที่จดไม่ถูกต้อง")
    if payload.turbine_trim_digits not in (0, 1, 2):
        raise HTTPException(status_code=400, detail="จำนวนหลักที่ตัดต้องเป็น 0, 1 หรือ 2")

    try:
        final_values = reading_fields.compute_final_values(
            payload.entered_values, payload.turbine_trim_digits)
        edited_fields = reading_fields.find_edited_fields(payload.entered_values, meta["raw_values"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if all(v is None for v in final_values.values()):
        raise HTTPException(status_code=400, detail="ไม่มีค่าให้บันทึก")

    reading = db.insert_reading(
        meter_run_id=payload.meter_run_id,
        recorded_at=payload.recorded_at,
        raw_values=meta["raw_values"],
        final_values=final_values,
        edited_fields=edited_fields,
        low_confidence_fields=meta["low_confidence_fields"],
        turbine_trim_digits=payload.turbine_trim_digits,
        image_paths=meta["image_paths"],
        engine=meta["engine"],  # ชื่อ engine ที่อ่านจริงตอน OCR (ไม่เดาจาก engine ปัจจุบัน)
        confidence_scores=meta.get("confidence_scores"),  # Gemini = None, GLM = dict (เก็บอย่างเดียว ไม่แสดง)
    )

    meta["reading_id"] = reading["id"]
    with open(_ocr_request_meta_path(payload.request_id), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {"reading": reading, "pdf_url": f"/api/readings/{reading['id']}/pdf"}


def _run_summary(run: Optional[dict]) -> Optional[dict]:
    if run is None:
        return None
    return {k: run[k] for k in ("id", "company", "run_label", "tag_fe", "serial_fe", "tag_fy", "serial_fy")}


@app.get("/api/readings")
def list_readings(meter_run_id: Optional[int] = None):
    """ประวัติการจด ใหม่สุดก่อน (เรียงตามเวลาที่จด) — กรองตาม Run ได้ (หน้าประวัติ)"""
    runs = {r["id"]: r for r in db.list_meter_runs()}
    return [{**r, "run": _run_summary(runs.get(r["meter_run_id"]))}
            for r in db.list_readings(meter_run_id)]


@app.get("/api/readings/{reading_id}")
def get_reading_detail(reading_id: int):
    """รายละเอียด 1 รายการ + การจดครั้งก่อนหน้า (ตัวเดียวกับที่ใช้ใน PDF)"""
    reading = db.get_reading(reading_id)
    if reading is None:
        raise HTTPException(status_code=404, detail="ไม่พบรายการนี้ในระบบ")
    previous = db.get_previous_reading(reading["meter_run_id"], before_reading_id=reading_id)
    return {
        "reading": reading,
        "run": _run_summary(db.get_meter_run(reading["meter_run_id"])),
        "previous": ({k: previous[k] for k in ("id", "recorded_at", "final_values")}
                     if previous else None),
    }


@app.get("/api/readings/{reading_id}/images/{index}")
def get_reading_image(reading_id: int, index: int):
    """ภาพต้นฉบับที่ใช้อ่านค่ารายการนี้ — path มาจาก DB (server เป็นคนเขียนเอง) ไม่รับ path จากผู้ใช้"""
    reading = db.get_reading(reading_id)
    if reading is None or not (0 <= index < len(reading["image_paths"] or [])):
        raise HTTPException(status_code=404, detail="ไม่พบภาพนี้")
    path = reading["image_paths"][index]
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="ไฟล์ภาพถูกลบไปแล้ว")
    return FileResponse(path)


@app.get("/api/readings/{reading_id}/pdf")
def get_reading_pdf(reading_id: int):
    """PDF ของการจดค่า 1 ครั้ง — สร้างจาก DB ทุกครั้ง (ใช้ทั้งตอนบันทึกและ export ซ้ำจากหน้าประวัติ)
    ค่าครั้งก่อน = การจดที่ "ก่อนหน้า" ครั้งนี้ของ Run เดียวกัน (ไม่ใช่ค่าล่าสุดของ Run)"""
    reading = db.get_reading(reading_id)
    if reading is None:
        raise HTTPException(status_code=404, detail="ไม่พบรายการนี้ในระบบ")
    run = db.get_meter_run(reading["meter_run_id"])
    previous = db.get_previous_reading(reading["meter_run_id"], before_reading_id=reading_id)
    pdf_bytes = report_pdf.build_reading_pdf(run, reading, previous)
    filename = report_pdf.pdf_filename(run, reading)
    return Response(content=pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})

# ── 5. STATIC FILES & ROUTING ─────────────────────────────────────────

@app.get("/static/samples/{filename}")
def get_sample_image(filename: str):
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(os.path.dirname(__file__), "static", "samples", safe_filename)
    if not os.path.exists(file_path):
        file_path = os.path.join(os.path.dirname(__file__), safe_filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Sample file not found")


# หน้า HTML ให้ browser ถามใหม่ทุกครั้ง -> เลข ?v= ของ .css/.js ที่บวกไว้มีผลทันที
NO_CACHE_HEADERS = {"Cache-Control": "no-cache"}


@app.get("/")
def read_runs_page():
    """หน้าแรก: เลือก Meter Run ก่อนเข้าหน้า OCR"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "runs.html"),
                        headers=NO_CACHE_HEADERS)


@app.get("/history")
def read_history_page():
    """หน้าประวัติการจด (กรองตาม Run ได้ด้วย ?run=<id>)"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "history.html"),
                        headers=NO_CACHE_HEADERS)


@app.get("/ocr")
def read_index():
    """หน้า OCR (ต้องมี ?run=<id> — index.js เด้งกลับหน้าแรกถ้าไม่มี/ไม่พบ)"""
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, headers=NO_CACHE_HEADERS)
    return {"message": "Index.html not found. Please place it in static/ folder."}


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# ── 6. AUTOMATED SERVER RUNNER ────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("🚀 PTT Industrial Meter Reader Portal Engine Ready!")
    print("👉 เปิดใช้งานระบบเว็บพอร์ทัลได้ที่: http://127.0.0.1:8000")
    print("=" * 60 + "\n")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)


# uvicorn server:app --reload --port 8000