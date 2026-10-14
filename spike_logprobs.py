"""
spike_logprobs.py — ทดสอบว่า Ollama คืน logprobs พร้อมกับการบังคับ JSON ได้หรือไม่

วิธีใช้:
    python spike_logprobs.py <path รูปมิเตอร์>
    เช่น  python spike_logprobs.py test2.jpg

ต้องมี: pip install requests
ต้องรัน Ollama อยู่ (ollama serve) และมีโมเดล glm-ocr:latest

สคริปต์นี้ไม่แตะโค้ดโปรเจคเลย รันแยกต่างหากเพื่อตอบคำถามเดียว:
    logprobs + บังคับ JSON ใช้ด้วยกันได้ไหม
"""

import base64
import json
import sys
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL = "glm-ocr:latest"
TIMEOUT = 300

PROMPT = (
    "Read this industrial gas meter screen. "
    "Return JSON with keys: correct_v_scm, line_v_m3, pressure_bar, "
    "temperature_c, correction_factor. Use null if a value is not visible. "
    "Do not explain."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "correct_v_scm": {"type": ["number", "null"]},
        "line_v_m3": {"type": ["number", "null"]},
        "pressure_bar": {"type": ["number", "null"]},
        "temperature_c": {"type": ["number", "null"]},
        "correction_factor": {"type": ["number", "null"]},
    },
    "required": [
        "correct_v_scm",
        "line_v_m3",
        "pressure_bar",
        "temperature_c",
        "correction_factor",
    ],
    "additionalProperties": False,
}


def build_messages(image_b64: str) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        }
    ]


def run_case(name: str, image_b64: str, response_format) -> dict:
    payload = {
        "model": MODEL,
        "messages": build_messages(image_b64),
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 3,
        "stream": False,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    print(f"\n{'=' * 70}\nกรณี {name}\n{'=' * 70}")
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    except Exception as exc:
        print(f"  เรียก API ไม่สำเร็จ: {exc}")
        return {"case": name, "ok": False, "reason": str(exc)}

    if r.status_code != 200:
        print(f"  HTTP {r.status_code}: {r.text[:400]}")
        return {"case": name, "ok": False, "reason": f"HTTP {r.status_code}"}

    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content", "")
    logprobs = choice.get("logprobs")
    tokens = (logprobs or {}).get("content") or []

    print(f"  ข้อความที่ได้ : {content[:300]}")
    print(f"  มีคีย์ logprobs ในคำตอบ : {logprobs is not None}")
    print(f"  จำนวน token ที่มี logprob : {len(tokens)}")

    if tokens:
        print("  ตัวอย่าง 8 token แรก (token -> ความน่าจะเป็น):")
        import math

        for t in tokens[:8]:
            tok = t.get("token")
            lp = t.get("logprob")
            prob = math.exp(lp) if isinstance(lp, (int, float)) else None
            alts = t.get("top_logprobs") or []
            alt_s = ", ".join(
                f"{a.get('token')!r}:{math.exp(a['logprob']):.3f}"
                for a in alts[:3]
                if isinstance(a.get("logprob"), (int, float))
            )
            print(f"    {tok!r:>12} -> {prob:.4f}   ตัวเลือกอื่น: {alt_s}")

    is_json = False
    try:
        json.loads(content)
        is_json = True
    except Exception:
        pass
    print(f"  ข้อความเป็น JSON ที่ parse ได้ : {is_json}")

    return {
        "case": name,
        "ok": True,
        "has_logprobs": bool(tokens),
        "token_count": len(tokens),
        "valid_json": is_json,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"ไม่พบไฟล์ {image_path}")
        sys.exit(1)

    image_b64 = base64.b64encode(image_path.read_bytes()).decode()

    results = [
        run_case("A: logprobs อย่างเดียว ไม่บังคับ JSON", image_b64, None),
        run_case("B: logprobs + json_object", image_b64, {"type": "json_object"}),
        run_case(
            "C: logprobs + json_schema (แบบที่ระบบจริงจะใช้)",
            image_b64,
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "meter_reading",
                    "schema": SCHEMA,
                    "strict": True,
                },
            },
        ),
    ]

    print(f"\n\n{'=' * 70}\nสรุป\n{'=' * 70}")
    for res in results:
        if not res.get("ok"):
            print(f"  {res['case']}  ->  ล้มเหลว ({res.get('reason')})")
            continue
        print(
            f"  {res['case']}\n"
            f"      logprobs: {'มี' if res['has_logprobs'] else 'ไม่มี'}"
            f" ({res['token_count']} token)   JSON ถูกต้อง: {res['valid_json']}"
        )

    print(
        "\nสิ่งที่ต้องดู: กรณี C ต้องได้ทั้ง logprobs และ JSON\n"
        "ถ้า C ไม่มี logprobs แต่ B มี  -> ใช้ json_object แทน json_schema\n"
        "ถ้าทั้ง B และ C ไม่มี แต่ A มี -> บังคับ JSON แล้ว logprobs หาย\n"
        "   ต้องเปลี่ยนไปใช้ /api/generate หรือเลิกบังคับ schema\n"
        "ถ้า A ก็ไม่มี -> Ollama เวอร์ชันนี้ยังไม่รองรับ ต้องอัปเกรด\n"
    )


if __name__ == "__main__":
    main()
