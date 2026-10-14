import cv2
import ollama

# 1. ระบุชื่อไฟล์ภาพที่คุณ Crop
image_path = "test2.jpg" 
processed_image_path = "ready_screen.png"

try:
    # 2. ปรับแสงด้วย OpenCV (Grayscale + CLAHE)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ ไม่พบไฟล์ภาพ {image_path}")
        exit()
        
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    
    cv2.imwrite(processed_image_path, img_clahe)
    print("✅ ทำ Preprocessing ภาพเรียบร้อยแล้ว...")
except Exception as e:
    print(f"❌ Error OpenCV: {e}")
    exit()

# 3. เตรียมคำสั่งสำหรับดึงค่าพารามิเตอร์ดิบ
combined_prompt = """
Task: Read the displayed values on this industrial gas meter screen.

Instructions:
1. Scan the image from top to bottom.
2. For each label (e.g., P, T, Zb/Z, C, Qm, Qb, Bat), extract the exact value and unit visible next to it.
3. List them as simple text: "Label: Value Unit".
4. If a number is faint or partially obscured, look closely at the shape of the digits to interpret it.
5. Do not summarize or explain. Just output the list of values found.
"""

print(f"กำลังส่งภาพให้ glm-ocr:latest ประมวลผลบนเครื่อง Local...")

try:
    # 4. เรียกใช้งานโมเดลโดยปรับ Temperature เป็น 0.0 เพื่อความเสถียรสูงสุด
    response = ollama.chat(
        model='glm-ocr:latest',
        messages=[
            {
                'role': 'user',
                'content': combined_prompt,
                'images': [processed_image_path] 
            }
        ],
        options={
            'temperature': 0.0,
            'num_predict': 128,   # 💡 จำกัดให้โมเดลพ่นคำตอบออกมากว้างพอสำหรับลิสต์พารามิเตอร์ แต่ไม่ให้ปล่อย VRAM บวม
            'num_ctx': 2048     # 💡 ล็อคขนาด Context Window ไว้ที่ 2048 ไม่ให้ Ollama แอบขยายจน VRAM ล้นการ์ดจอ
            }
    )

    
    # บรรทัดพิมพ์ผลลัพธ์เดิมของคุณในโค้ด
    raw_content = response['message']['content']
    
    # 💡 ใช้ Python ตัดเนื้อหาขยะฝั่ง Markdown ออก
    clean_content = raw_content.split("```markdown")[0].strip()

    print("\n--- 🏆 ผลลัพธ์สุดท้ายจากโมเดล glm-ocr (Cleaned) ---")
    print(clean_content)

except Exception as e:
    print(f"เกิดข้อผิดพลาดในการรันระบบ Local: {e}")