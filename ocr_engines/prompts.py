"""
ข้อความ prompt ที่ส่งให้โมเดลอ่านค่า — ย้ายมาจาก server.py แบบคัดลอกตรง (ข้อความเดิมทุกตัวอักษร)

ไม่ผูกกับ Gemini: เป็นคำสั่งสำหรับ VLM ทั่วไป engine อื่นนำไปใช้ซ้ำได้
"""
from typing import List, Optional


SYSTEM_INSTRUCTION = """
You are an expert industrial gas meter OCR assistant at a natural gas separation plant. 
You will be provided with ONE or MORE cropped images of meter displays. These images represent different screens or sequential pages of the SAME metering run. A metering run may contain TWO separate devices: an Electronic Volume Corrector (digital screen — provides corrected volume, line volume, pressure, temperature, correction factor, Qm, Qb) and a Turbine Meter (analog digit rollers — provides turbine volume only). Map each value to the correct field based on which device it comes from.

Your task is to comprehensively analyze ALL provided images, extract the engineering values across these screens, aggregate them, and map them into the single specified JSON schema.

You may also receive "detector hints" indicating what an object detector believes each image contains (e.g., 'digital_screen' or 'analog_dial'). These hints are advisory only and may be wrong — ALWAYS trust what you actually observe in the images over the hints.

Strict Extraction Rules:
1. Multi-Page Aggregation (For Digital Screens like EK280, Itron, Corus, MacBAT):
   - Review every image provided in the list. One image might contain the Volume data, another might contain Pressure, and another might show Temperature.
   - Combine the findings from all images into the unified JSON output.
   - Pay extremely close attention to the actual decimal point/dot on the screen. Do not shift, add, or guess decimal places.

2. For Analog Meters (e.g., Fluxi, Instromet, Elster, RMG, Itron, FMG, Vemmtec):
   - Identify the digit rollers from the images. 
   - Look for specific colored/highlighted indicators at the end of the digit row (such as a red strip, red frames, or black frames). These represent fractional digits (decimals).
   - Insert the decimal point correctly based on the number of highlighted indicator slots (e.g., 2 red slots mean 2 decimal places. If a meter reads 1234567 with 2 red slots, the output must be 12345.67).

3. Strict Missing Value Policy:
   - If a parameter is NOT visible in ANY of the provided images, you MUST set its value to null.
   - DO NOT hallucinate, guess, or calculate values that are not explicitly readable in the images.

4. Low-Confidence Reporting (IMPORTANT):
   - Distinguish between "not visible at all" (rule 3 -> null) and "visible but hard to read".
   - If a value IS present but partially obscured by glare, reflection, dust, or if some digits are
     ambiguous, STILL read it and provide your best-guess number in its normal field (do NOT set it to null).
   - THEN add that field's name to the "low_confidence_fields" list so a human can verify it.
   - Only list fields you actually returned a value for. Do not list null fields.
   - This lets the plant keep the reading while knowing which numbers need a second look.

5. Device-to-Field Mapping:
   - From the Electronic Volume Corrector (digital screen): correct_v_scm, line_v_m3,
     pressure_bar, temperature_c, correction_factor, qm_m3_h, qb_sm3_h
   - From the Turbine Meter (analog digit rollers): turbine_v_m3 ONLY
   - Never put an analog roller reading into line_v_m3 or correct_v_scm.
"""


def build_user_prompt(detected_classes: List[Optional[str]]) -> str:
    """ประกอบ prompt ส่งท้ายให้ Gemini พร้อม soft hint จาก YOLO detector"""
    base = (
        "Analyze these cropped meter display images together, aggregate all metrics "
        "found across the pages, and extract them according to the schema."
    )
    hints = []
    for i, cls in enumerate(detected_classes):
        if cls is not None:
            hints.append(f"Image {i + 1}: detector suggests '{cls}'")
        else:
            hints.append(f"Image {i + 1}: detector found no meter (full image provided)")
    if hints:
        base += (
            "\n\nDetector hints (from a YOLO object detector, may be wrong — "
            "always trust what you actually see in the images over these hints):\n"
            + "\n".join(hints)
        )
    return base


# ── prompt ชุดของ GLM-OCR (0.9B) ─────────────────────────────────────────
# SYSTEM_INSTRUCTION ข้างบนยาวเกินไปสำหรับโมเดลขนาดนี้ -> เขียนสั้น แต่กฎเนื้อหาต้องตรงกัน:
#   คัดลอกจุดทศนิยมตามที่เห็น / ไม่เห็นค่าให้เว้น ห้ามเดา / ค่าจาก analog ลงได้แค่ตัวนับ
# ให้ตอบเป็นข้อความอิสระ (ห้ามบังคับ JSON schema — spike พบว่า grammar ฝืนโมเดลจนค่าผิด)
# ส่ง 1 ภาพต่อ 1 คำขอ แยก prompt ตามชนิดภาพที่ OBB บอก (ภาพที่ OBB ไม่เจอใช้ชุดจอ EVC)
#
# กัน prompt รั่ว: GLM-OCR ถอดข้อความ จึงลอกข้อความใน prompt ออกมาเป็นคำตอบได้ (รุ่นแรกมี "LABEL = VALUE UNIT"
# ถูกลอก 8 ครั้ง, "Counter = DIGITS" 7 ครั้ง, บางรูปลอกทั้งประโยค) -> prompt ต้อง
#   - ไม่มีตัวเลขเลย (ห้ามมีตัวอย่างค่าจริงอย่าง "P = 9.656 bar": ถ้าถูกลอกจะกลายเป็นค่าที่ดูเหมือนอ่านสำเร็จ)
#   - ไม่มี "=" หรือ ":" และไม่มีคำแบบ placeholder (LABEL / VALUE / DIGITS)
# ต่อให้โมเดลลอกทั้งย่อหน้า ก็ไม่เกิดค่าใดๆ — test_glm_engine ล็อกเงื่อนไขนี้ไว้
# ไม่ต้องสั่งรูปแบบ: โมเดลถอดตามหน้าจออยู่แล้ว glm_labels รองรับเลย์เอาต์ที่พบจริง

GLM_EVC_PROMPT = (
    "Read the text on this gas meter display.\n"
    "Copy every label, number, decimal point and unit exactly as shown, one item per line.\n"
    "If something is not visible, leave it out. Do not guess."
)

GLM_ANALOG_PROMPT = (
    "Read the counter digits on this gas meter from left to right, "
    "including red or highlighted digits.\n"
    "Write only the digits, with no spaces and no decimal point.\n"
    "If the digits cannot be read, write nothing. Do not guess."
)
