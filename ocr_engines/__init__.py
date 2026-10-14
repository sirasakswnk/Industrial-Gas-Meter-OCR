"""
OCR engine ที่ระบบเลือกใช้ได้ — ตั้งผ่าน OCR_ENGINE ใน .env (ค่าเริ่มต้น "gemini")

เพิ่ม engine ใหม่: เขียนคลาสสืบจาก base.OcrEngine แล้วลงทะเบียนใน ENGINES
import แบบ lazy -> engine ที่ไม่ได้เลือกไม่ถูก import (ไม่ต้องมี dependency/API key ของมัน)
"""
import importlib

from .base import EngineResult, MeterReading, OcrEngine

__all__ = ["EngineResult", "MeterReading", "OcrEngine", "DEFAULT_ENGINE", "ENGINES",
           "UnknownEngineError", "available_engines", "create_engine"]

DEFAULT_ENGINE = "gemini"

# ชื่อ -> (module, คลาส)
ENGINES = {
    "gemini": ("ocr_engines.gemini_engine", "GeminiEngine"),
    "glm": ("ocr_engines.glm_engine", "GlmEngine"),   # GLM-OCR ผ่าน Ollama บนเครื่อง
}


class UnknownEngineError(ValueError):
    pass


def available_engines() -> list:
    return sorted(ENGINES)


def create_engine(name: str) -> OcrEngine:
    """สร้าง engine ตามชื่อ (ไม่สนตัวพิมพ์เล็กใหญ่) — ชื่อที่ไม่รู้จัก -> UnknownEngineError พร้อมรายชื่อที่เลือกได้"""
    key = (name or "").strip().lower()
    if key not in ENGINES:
        raise UnknownEngineError(
            f"ไม่รู้จัก OCR_ENGINE = '{name}' — เลือกได้: {', '.join(available_engines())}")
    module_name, class_name = ENGINES[key]
    return getattr(importlib.import_module(module_name), class_name)()
