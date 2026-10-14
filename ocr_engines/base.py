"""
ส่วนกลางของ OCR engine: สัญญา (interface) ที่ทุก engine ต้องทำตาม + การ parse ที่ใช้ร่วมกัน

- MeterReading  = schema ผลการอ่าน (ย้ายมาจาก server.py ไม่เปลี่ยนแม้แต่ description
                  เพราะ Gemini ใช้เป็น response_schema)
- EngineResult  = สิ่งที่ engine คืนให้ server — pipeline ที่เหลือไม่ต้องรู้ว่าใช้ engine ไหน
- extract_raw_values = ลอก ```json fence -> json.loads แบบคงค่าเป็นสตริง (ใช้ได้ทุก engine ที่ตอบเป็น JSON)
"""
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image
from pydantic import BaseModel, Field

import reading_fields


class MeterReading(BaseModel):
    meter_type: str = Field(description="ประเภทมิเตอร์ 'digital' หรือ 'analog'")
    detected_format: str = Field(description="ลักษณะที่วิเคราะห์ได้ เช่น digital page 1-3 aggregated หรือ analog 2 red slots")
    correct_v_scm: Optional[float] = Field(None, description="ค่า CORRECT V.(SCM) หรือ Vb")
    line_v_m3: Optional[float] = Field(None, description="ค่า LINE V.(M3) - *สำหรับ Analog ควรปล่อยให้เป็น null*")
    pressure_bar: Optional[float] = Field(None, description="ค่า PRESSURE (Bar / Baro)")
    temperature_c: Optional[float] = Field(None, description="ค่า TEMPERATURE (°C)")
    correction_factor: Optional[float] = Field(None, description="ค่า CORRECTION FACTOR (C)")
    turbine_v_m3: Optional[float] = Field(None, description="ค่าสะสมปริมาตรก๊าซในหน่วย M3 หากเป็น Analog ให้มองว่าเป็นค่านี้")
    qm_m3_h: Optional[float] = Field(None, description="ค่า Qm (m3/h)")
    qb_sm3_h: Optional[float] = Field(None, description="ค่า Qb (Sm3/h)")
    pulse_weight: Optional[float] = Field(None, description="ค่า Pulse weight")
    low_confidence_fields: List[str] = Field(
        default_factory=list,
        description=(
            "List of field names that were read but with LOW confidence due to glare, "
            "reflection, dust, or partially obscured digits. Still provide your best-guess "
            "value for those fields in their normal field — this list is ONLY a flag for "
            "human verification. Example: ['pressure_bar', 'temperature_c']"
        )
    )


@dataclass
class EngineResult:
    """ผลการอ่านของ engine 1 ครั้ง (ทุกภาพในคำขอรวมเป็นก้อนเดียว)"""
    raw_values: dict                   # 8 ฟิลด์ตาม reading_fields.FIELD_KEYS — สตริงตามที่โมเดลเขียน (None = ไม่พบ)
    meter_type: str
    detected_format: str
    confidence_scores: Optional[dict]  # gemini = None, glm = {ฟิลด์: 0..1 หรือ None} — เก็บลง DB อย่างเดียว ไม่แสดง
    raw_text: Optional[str]            # ข้อความดิบจากโมเดล ไว้ตรวจย้อนหลัง (None = อ่านข้อความไม่ได้)
    engine: str                        # ชื่อ engine (ลงคอลัมน์ readings.engine)
    # สองฟิลด์ด้านล่างเกินจากสเปค — ต้องมีเพื่อไม่ให้พฤติกรรมเปลี่ยน
    raw_source: str                    # "text" | "fallback" (raw_values มาจาก float -> หน้า OCR ขึ้นเตือน)
    reading: MeterReading              # ค่าที่ parse ตาม schema: ฟิลด์ float เดิมใน response /api/ocr,
                                       # low_confidence_fields, pulse_weight (ตาข่าย analog ใช้)


class OcrEngine(ABC):
    """engine อ่านค่ามิเตอร์ — รับภาพที่ผ่าน preprocess แล้ว คืน EngineResult

    ข้อตกลง: error ทุกชนิด (เรียกโมเดลไม่ได้ / parse ไม่ได้) ให้ raise ออกไป
    server แปลงเป็น 500 "เรียกโมเดลอ่านค่าไม่สำเร็จ" ให้เอง
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """ชื่อ engine ตรงกับค่า OCR_ENGINE ใน .env และค่าที่บันทึกลง readings.engine"""

    @abstractmethod
    def read(self, images: List[Image.Image], detected_classes: List[Optional[str]]) -> EngineResult:
        """images = ภาพหลัง preprocess (เรียงตามที่อัปโหลด), detected_classes = class จาก OBB รายภาพ
        (None = OBB หาไม่เจอ) ใช้เป็น hint ให้โมเดล"""


def strip_json_fence(text: str) -> str:
    """ลอก markdown fence (```json ... ```) ออก — SDK parse ผ่านได้แต่ json.loads ดิบพัง"""
    fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", text, re.DOTALL)
    return fence.group(1) if fence else text


def load_json_as_strings(text: str) -> dict:
    """json.loads โดยคงตัวเลขเป็นสตริงตามที่โมเดลเขียน ("4.500" ไม่กลายเป็น 4.5) — ต้องเป็น JSON object"""
    data = json.loads(strip_json_fence(text.strip()), parse_float=str, parse_int=str)
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    return data


def extract_raw_values(text: Optional[str], reading: MeterReading) -> tuple:
    """ค่าดิบ "ตามที่โมเดลเขียนมาเป๊ะๆ" เป็นสตริง (เก็บลง raw_values เพื่อตรวจย้อนหลัง)

    อ่านจากข้อความ JSON ด้วย parse_float/parse_int=str -> "4.500" ไม่กลายเป็น 4.5
    ทางสำรอง (ไม่มีข้อความ / JSON พัง): แปลงจาก float ที่ parse แล้ว ด้วย to_plain_str
      — ไม่เติม ".0" และเก็บทุกหลักที่ float มี (12345.67 คงเป็น "12345.67")
      แต่ float เสียศูนย์ท้ายไปแล้ว (12345.60 -> "12345.6") จึงพยายามใช้ข้อความดิบให้ได้มากที่สุด:
      ลอก markdown fence (```json ... ```) ออกก่อน เพราะ SDK parse ผ่านได้แต่ json.loads ดิบพัง

    text = ข้อความดิบที่โมเดลตอบ (None ได้ -> ใช้ทางสำรอง)
    คืน (raw_values, source) — source = "text" (ข้อความดิบ) | "fallback" (จาก float)
    หน้าจอขึ้นคำเตือนเมื่อเป็น fallback เพราะศูนย์ท้ายทศนิยมอาจหายแบบเงียบๆ
    """
    source = "text"
    try:
        data = load_json_as_strings(text)
    except Exception:
        data = reading.model_dump()
        source = "fallback"
        print("⚠️ อ่านข้อความ JSON ดิบของโมเดลไม่ได้ — ใช้ค่าสำรองจาก float (ศูนย์ท้ายอาจหาย)")
    values = {key: reading_fields.to_plain_str(data.get(key)) for key in reading_fields.FIELD_KEYS}
    return values, source
