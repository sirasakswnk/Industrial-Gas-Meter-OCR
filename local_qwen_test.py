import os
import ollama
from PIL import Image
from pydantic import BaseModel, Field
from typing import Optional

# 1. นิยามโครงสร้าง JSON เหมือนเดิม
class MeterReading(BaseModel):
    meter_type: str = Field(description="Type of meter: 'digital' or 'analog'")
    detected_format: str = Field(description="Description of format, e.g., 'digital page 1' or 'analog 2 red slots'")
    correct_v_scm: Optional[float] = Field(None, description="CORRECT V.(SCM) or Vb value")
    line_v_m3: Optional[float] = Field(None, description="LINE V.(M3) or Vm value")
    pressure_bar: Optional[float] = Field(None, description="PRESSURE (Bar / Baro)")
    temperature_c: Optional[float] = Field(None, description="TEMPERATURE (°C)")
    correction_factor: Optional[float] = Field(None, description="CONNECTION FACTOR (C)")
    turbine_v_m3: Optional[float] = Field(None, description="TURBINE V.(M3)")
    qm_m3_h: Optional[float] = Field(None, description="Qm (m3/h)")
    qb_sm3_h: Optional[float] = Field(None, description="Qb (Sm3/h)")
    pulse_weight: Optional[float] = Field(None, description="Pulse weight")

# 2. ระบุไฟล์ภาพมิเตอร์
image_path = "ready_screen.png"  # มั่นใจว่าชื่อไฟล์ตรงกับรูปภาพ Corus ในโฟลเดอร์นะครับ

# 3. ยุบคำสั่งทั้งหมดมารวมกันเป็นข้อความเดียวให้โมเดลอ่านทีเดียว
combined_prompt = """
Task: Analyze the provided industrial gas meter image and extract values strictly into the requested JSON schema format.

Strict Extraction Rules:
1. For Digital Screens (e.g., Corus, EK280, Itron, MacBAT):
   - Analyze the digital segments carefully (even if they are low contrast or faint).
   - SPECIFIC FOR CORUS: 
     - Find "Converted Index" (unit: Sm3) and map its numeric value to 'correct_v_scm'.
     - Find "Unconverted Index" (unit: m3) and map its numeric value to 'line_v_m3'.
   - CAUTION: This is a 7-segment display with glare. Pay extreme attention to the difference between '6' and '5', and '9' and '1'. Do not drop leading zeros if they are printed.

2. For Analog Meters (e.g., Fluxi, Instromet, Elster):
   - Identify the digit rollers and check for red strips, red frames, or black frames at the end which represent decimal places. Insert decimal points accordingly.

3. Missing Value Policy:
   - If a parameter is NOT visible in the image, you MUST set its value to null. Do not hallucinate.

Return ONLY a valid JSON string that matches this structure:
{
  "meter_type": "string",
  "detected_format": "string",
  "correct_v_scm": float or null,
  "line_v_m3": float or null,
  "pressure_bar": float or null,
  "temperature_c": float or null,
  "correction_factor": float or null,
  "turbine_v_m3": float or null,
  "qm_m3_h": float or null,
  "qb_sm3_h": float or null,
  "pulse_weight": float or null
}
"""

print("กำลังประมวลผลด้วย Qwen2.5vl:7b บนเครื่องของคุณ (Local)...")

try:
    # 4. เรียกใช้โมเดลโดยส่งข้อมูลเข้าในบทบาท user ทั้งหมด
    response = ollama.chat(
        model='glm-ocr:latest',
        messages=[
            {
                'role': 'user',
                'content': combined_prompt,
                'images': [image_path] 
            }
        ],
        options={
            'temperature': 0.5 # เพิ่มความยืดหยุ่นในการอ่านค่าจางๆ
        },
        # เปลี่ยนจากการส่ง Schema เต็มๆ เป็นการบังคับรับฟอร์แมต JSON ทั่วไปเพื่อลดอาการค้างของ Ollama
        format='json' 
    )

    # 5. แสดงผลลัพธ์
    print("\n--- ผลลัพธ์จาก Qwen2.5vl:7b (Local) ---")
    print(response['message']['content'])

except Exception as e:
    print(f"เกิดข้อผิดพลาดในการรันระบบ Local: {e}")