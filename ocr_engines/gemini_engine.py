"""
OCR engine: Gemini (google-genai) — ย้ายมาจาก server.py แบบคัดลอกตรง ไม่เปลี่ยนพฤติกรรม

ทุกอย่างที่ผูกกับ Gemini SDK อยู่ในไฟล์นี้: API key, ชื่อโมเดล, client, GenerateContentConfig,
response.parsed / response.text
"""
import json
import os
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .base import EngineResult, MeterReading, OcrEngine
from .base import extract_raw_values as extract_raw_values_from_text
from .prompts import SYSTEM_INSTRUCTION, build_user_prompt

load_dotenv()  # import ไฟล์นี้ตรงๆ (ไม่ผ่าน server) ก็ยังอ่าน .env ได้ — ไม่ทับค่าที่ตั้งไว้แล้ว

# ชื่อโมเดล Gemini: override ได้ผ่าน .env (GEMINI_MODEL=...) โดยไม่ต้องแก้โค้ด
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")


def parse_gemini_response(response) -> "MeterReading":
    """ดึงผลลัพธ์จาก Gemini อย่างปลอดภัย: response.parsed เป็นหลัก, json.loads เป็นสำรอง"""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, MeterReading):
        return parsed
    return MeterReading(**json.loads(response.text))


def response_text(response) -> Optional[str]:
    """response.text อย่างปลอดภัย — SDK อาจ raise ถ้าไม่มีข้อความ (เดิมอ่านใน try ของ extract_raw_values)"""
    try:
        return response.text
    except Exception:
        return None


def extract_raw_values(response, reading: MeterReading) -> tuple:
    """ค่าดิบสตริงจาก response ของ Gemini -> (raw_values, "text"/"fallback") — ดู base.extract_raw_values"""
    return extract_raw_values_from_text(response_text(response), reading)


class GeminiEngine(OcrEngine):
    def __init__(self, model: str = GEMINI_MODEL):
        # Fail-fast: ถ้าไม่มี key ให้หยุดทันทีพร้อมข้อความชัดเจน ดีกว่าไปพังตอน call API
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError(
                "ไม่พบ GEMINI_API_KEY — กรุณาสร้างไฟล์ .env ในโฟลเดอร์โปรเจค "
                "แล้วใส่บรรทัด: GEMINI_API_KEY=your_key_here"
            )
        self.model = model
        self.client = genai.Client()
        print(f"🤖 Gemini model: {self.model}")

    @property
    def name(self) -> str:
        return "gemini"

    def read(self, images: List, detected_classes: List[Optional[str]]) -> EngineResult:
        """1 API call ต่อคำขอ: ทุกภาพ + prompt ท้าย (มี hint จาก OBB) -> MeterReading"""
        contents = list(images)
        contents.append(build_user_prompt(detected_classes))

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=MeterReading,
                temperature=0.1
            ),
        )
        reading = parse_gemini_response(response)
        raw_text = response_text(response)
        raw_values, raw_source = extract_raw_values_from_text(raw_text, reading)
        return EngineResult(
            raw_values=raw_values,
            meter_type=reading.meter_type,
            detected_format=reading.detected_format,
            confidence_scores=None,
            raw_text=raw_text,
            engine=self.name,
            raw_source=raw_source,
            reading=reading,
        )
