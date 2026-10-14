"""
OCR engine: GLM-OCR ผ่าน Ollama (POST /v1/chat/completions, รันบนเครื่อง)

ต่างจาก Gemini ตามผล spike logprobs:
- ห้ามใช้ response_format json_schema: grammar ฝืนโมเดล 0.9B จนค่าผิด (prob ของ token ที่ถูกบังคับ ~0.0003)
  -> ให้ตอบข้อความอิสระ แล้วแปลงป้ายเป็นฟิลด์ด้วย glm_labels
  (GLM-OCR ถอดข้อความตามหน้าจอ ไม่ทำตามรูปแบบที่สั่ง — glm_labels รองรับเลย์เอาต์ที่พบจริง)
- ไม่จำกัดความยาว = วนซ้ำไม่หยุด (7397 token) -> max_tokens (= num_predict ของ Ollama)
- confidence รายฟิลด์จาก logprobs: min ของความน่าจะเป็นของ token ที่ประกอบเป็น "ค่า" (ไม่รวมป้าย/หน่วย)
  ใช้ min ไม่ใช่ค่าเฉลี่ย เพราะเลขผิดตัวเดียวก็ผิดทั้งค่า; แมป token กลับหาค่าไม่ได้ -> None (ไม่เดา)

ส่ง 1 ภาพต่อ 1 คำขอ (รูปเดียวกิน prompt ~614 token) แยก prompt ตามชนิดภาพจาก OBB แล้วรวมผล:
ฟิลด์เดียวกันจากหลายภาพ ค่าเหมือนกัน -> ใช้, ค่าต่างกัน -> null (ไม่เลือกให้)

ข้อจำกัด: /v1 ของ Ollama ตั้ง num_ctx ไม่ได้ (เอกสาร Ollama: OpenAI API ไม่มีช่องตั้ง context size;
ทดลองส่ง options.num_ctx แล้วถูกเมิน) -> ใช้ context ตามค่าตั้งของโมเดลใน Ollama
"""
import base64
import io
import math
import os
from typing import Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

import reading_fields

from . import glm_labels
from .base import EngineResult, MeterReading, OcrEngine
from .prompts import GLM_ANALOG_PROMPT, GLM_EVC_PROMPT

load_dotenv()

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
GLM_MODEL = os.environ.get("GLM_MODEL", "glm-ocr:latest")
GLM_MAX_TOKENS = int(os.environ.get("GLM_MAX_TOKENS", "256"))   # = num_predict (กันวนซ้ำ)
GLM_TIMEOUT = float(os.environ.get("GLM_TIMEOUT", "180"))         # วินาทีต่อภาพ (โหลดโมเดลครั้งแรกช้า)

# ต้องตรงกับ server.ANALOG_CLASS / DIGITAL_CLASS (import server ไม่ได้ — วนกัน) — test_glm_engine ตรวจให้
ANALOG_CLASS = "analog_dial"
DIGITAL_CLASS = "digital_screen"


class OllamaUnavailableError(RuntimeError):
    """ติดต่อ Ollama ไม่ได้ — server แปลงเป็น 500 พร้อมข้อความนี้ (server ไม่ล่ม)"""


def encode_image(img) -> str:
    """ภาพ PIL (หลัง preprocess) -> base64 JPEG"""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


# ── confidence จาก logprobs ───────────────────────────────────────────────

def token_byte_spans(content: str, tokens: list) -> Optional[List[Tuple[int, int, Optional[float]]]]:
    """[(byte_start, byte_end, prob)] ของแต่ละ token ในข้อความ (นับเป็น byte ของ UTF-8)

    ใช้ field "bytes" ของ Ollama ถ้ามี (ตัวอักษรหลาย byte เช่น ° อาจถูกแบ่งข้าม token)
    token ต่อกันแล้วไม่ตรงกับข้อความ -> None (แมปไม่ได้ ไม่เดา)
    """
    spans, pos, joined = [], 0, b""
    for t in tokens:
        b = t.get("bytes")
        piece = bytes(b) if isinstance(b, list) else str(t.get("token", "")).encode("utf-8")
        lp = t.get("logprob")
        prob = math.exp(lp) if isinstance(lp, (int, float)) and not isinstance(lp, bool) else None
        spans.append((pos, pos + len(piece), prob))
        pos += len(piece)
        joined += piece
    content_bytes = content.encode("utf-8")
    if joined == content_bytes:
        return spans
    # token ยาวกว่า content เฉพาะ "ท้ายสุด" ใน 2 กรณีที่พบจริงเท่านั้น (อื่นๆ -> None):
    #   - Ollama ตัด whitespace ท้ายข้อความออกจาก content แต่ token ยังมี (" " / "\n" — S01_p2, hard_1)
    #   - ถูก max_tokens ตัดกลางตัวอักษรหลาย byte (เช่น "³") -> byte ที่ไม่ครบไม่อยู่ใน content
    extra = joined[len(content_bytes):]
    if joined.startswith(content_bytes) and extra and (
            not extra.strip() or (len(extra) <= 3 and _is_incomplete_utf8(extra))):
        return spans
    return None


def _is_incomplete_utf8(data: bytes) -> bool:
    try:
        data.decode("utf-8")
        return False
    except UnicodeDecodeError as e:
        return e.reason == "unexpected end of data"


def value_confidence(content: str, tokens: list, char_spans: List[Tuple[int, int]]) -> Optional[float]:
    """confidence ของค่า 1 ฟิลด์ = min ของ prob ของทุก token ที่ทับตำแหน่งค่า (ทุกครั้งที่ค่านี้ปรากฏ)

    token ที่คร่อมค่ากับป้าย/หน่วย (เช่น "0m") ถูกนับด้วย เพราะมีหลักของค่าอยู่ในนั้น
    คืน None ถ้า: ไม่มี logprobs / token ไม่ตรงข้อความ / หา token ของค่าไม่เจอ / token นั้นไม่มี logprob
    """
    if not tokens or not char_spans:
        return None
    tspans = token_byte_spans(content, tokens)
    if tspans is None:
        return None
    probs = []
    for cs, ce in char_spans:
        if ce <= cs:
            return None
        bs, be = len(content[:cs].encode("utf-8")), len(content[:ce].encode("utf-8"))
        hit = [p for s, e, p in tspans if s < be and e > bs]
        if not hit or any(p is None for p in hit):
            return None
        probs.extend(hit)
    return min(probs)


def complete_lines(content: str, finish_reason: Optional[str]) -> str:
    """ถูกตัดด้วย max_tokens (finish_reason = "length") -> ทิ้งบรรทัดสุดท้ายที่ยังเขียนไม่จบ
    (เช่น "P = 9." จะขัดกับ "P = 9.655" ข้างบนจนค่าที่ถูกกลายเป็น null) — คืน prefix ของข้อความ
    ตำแหน่งตัวอักษรจึงยังตรงกับข้อความเต็ม (ใช้หา confidence ได้)"""
    if finish_reason != "length":
        return content
    return content[:content.rfind("\n") + 1]


# ── รวมผลหลายภาพ ──────────────────────────────────────────────────────────

def merge_images(per_image: List[dict]) -> Tuple[Dict[str, Optional[str]], Dict[str, Optional[float]], List[str]]:
    """per_image = [{"fields": {field: FieldValue}, "content": str, "tokens": list}, ...]
    คืน (ค่าราย field, confidence ราย field, ข้อความ log) — ครบ 8 ฟิลด์ (ไม่มีค่า = None)"""
    candidates: Dict[str, list] = {}
    for idx, img in enumerate(per_image):
        for fld, fv in img["fields"].items():
            conf = (value_confidence(img["content"], img["tokens"], fv.spans)
                    if fv.value is not None else None)
            candidates.setdefault(fld, []).append((idx, fv, conf))

    values = {k: None for k in reading_fields.FIELD_KEYS}
    confs = {k: None for k in reading_fields.FIELD_KEYS}
    notes = []
    for fld, cands in candidates.items():
        if any(fv.conflict for _, fv, _ in cands):
            notes.append(f"{fld}: โมเดลเขียนค่าไม่ตรงกันในภาพเดียวกัน -> null")
            continue
        distinct = {fv.value for _, fv, _ in cands if fv.value is not None}
        if len(distinct) > 1:
            notes.append(f"{fld}: ค่าต่างกันระหว่างภาพ {sorted(distinct)} -> null")
            continue
        if not distinct:
            raws = [fv for _, fv, _ in cands]
            notes.append(f"{fld}: มีป้ายแต่ค่าแปลงเป็นตัวเลขไม่ได้ -> null ({len(raws)} ครั้ง)")
            continue
        value = distinct.pop()
        cs = [c for _, fv, c in cands if fv.value == value]
        values[fld] = value
        confs[fld] = None if any(c is None for c in cs) else min(cs)
    return values, confs, notes


def describe_classes(detected_classes: List[Optional[str]]) -> str:
    """ข้อความกลางๆ สำหรับ detected_format (ขึ้นบนหน้า OCR — ห้ามมีชื่อ engine/ผู้ให้บริการ)"""
    names = {DIGITAL_CLASS: "digital screen", ANALOG_CLASS: "analog dial", None: "undetected image"}
    counts: Dict[str, int] = {}
    for c in detected_classes:
        name = names.get(c, str(c))
        counts[name] = counts.get(name, 0) + 1
    return ", ".join(f"{name} x{n}" for name, n in counts.items())


class GlmEngine(OcrEngine):
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = GLM_MODEL,
                 max_tokens: int = GLM_MAX_TOKENS, timeout: float = GLM_TIMEOUT):
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        print(f"🤖 GLM model: {self.model} @ {self.base_url} (max_tokens={self.max_tokens})")
        self._check_ollama()

    @property
    def name(self) -> str:
        return "glm"

    def _check_ollama(self):
        """เช็กตอนเปิด server แบบไม่บล็อก — Ollama ยังไม่เปิดก็แค่เตือน (เปิดทีหลังได้ ไม่ต้อง restart)"""
        try:
            tags = httpx.get(f"{self.base_url}/api/tags", timeout=3).json()
        except Exception:
            print(f"⚠️ ยังติดต่อ Ollama ที่ {self.base_url} ไม่ได้ — ก่อนอ่านค่าให้เปิด `ollama serve`")
            return
        names = {m.get("name") for m in tags.get("models", [])}
        if self.model not in names:
            print(f"⚠️ Ollama ยังไม่มีโมเดล '{self.model}' — ติดตั้งด้วย `ollama pull {self.model}`")

    def _chat(self, prompt: str, image_b64: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]}],
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 3,
            "max_tokens": self.max_tokens,   # Ollama แปลงเป็น num_predict
            "stream": False,
        }
        url = f"{self.base_url}/v1/chat/completions"
        try:
            r = httpx.post(url, json=payload, timeout=self.timeout)
        except httpx.ConnectError:
            raise OllamaUnavailableError(
                f"ติดต่อ Ollama ที่ {self.base_url} ไม่ได้ — กรุณาเปิด `ollama serve` แล้วลองใหม่") from None
        except httpx.TimeoutException:
            raise RuntimeError(f"Ollama ตอบไม่ทันใน {self.timeout:.0f} วินาที") from None
        if r.status_code != 200:
            try:
                msg = r.json().get("error", {})
                msg = msg.get("message") if isinstance(msg, dict) else msg
            except Exception:
                msg = r.text[:200]
            if r.status_code == 404:
                raise RuntimeError(f"Ollama ไม่พบโมเดล '{self.model}' — ติดตั้งด้วย `ollama pull {self.model}` ({msg})")
            raise RuntimeError(f"Ollama ตอบ HTTP {r.status_code}: {msg}")
        return r.json()

    def read(self, images: List, detected_classes: List[Optional[str]]) -> EngineResult:
        per_image, texts = [], []
        for idx, (img, cls) in enumerate(zip(images, detected_classes)):
            analog = cls == ANALOG_CLASS
            data = self._chat(GLM_ANALOG_PROMPT if analog else GLM_EVC_PROMPT, encode_image(img))
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content") or ""
            tokens = (choice.get("logprobs") or {}).get("content") or []

            # analog: ตัวเลขเดี่ยวลง turbine ได้โดยไม่ต้องมีป้าย (กฎอุปกรณ์กำหนดฟิลด์เดียวไว้แล้ว)
            parsed = glm_labels.parse_labeled_text(
                complete_lines(content, choice.get("finish_reason")),
                glm_labels.ANALOG_FIELDS if analog else glm_labels.EVC_FIELDS,
                bare_number_field=reading_fields.TURBINE_FIELD if analog else None)
            tag = f"🔤 GLM ภาพ {idx + 1} ({cls or 'OBB ไม่เจอ'})"
            print(f"{tag}: แปลงป้ายได้ {len(parsed.values)}/{parsed.label_count}"
                  + (f" | finish={choice.get('finish_reason')}" if choice.get("finish_reason") != "stop" else ""))
            if parsed.discarded:
                print(f"{tag}: ทิ้งป้ายที่ไม่ใช่ฟิลด์ {parsed.discarded}")
            if parsed.unknown:
                print(f"{tag}: ทิ้งป้ายที่ไม่รู้จัก {parsed.unknown}")
            if parsed.not_allowed:
                print(f"{tag}: ทิ้งค่าที่อุปกรณ์นี้ให้ไม่ได้ {parsed.not_allowed}")
            if parsed.no_value:
                print(f"{tag}: ป้ายที่ไม่มีค่าตามมา {parsed.no_value}")
            bare = sorted({v.value or v.raw_value for v in parsed.values if v.layout == "bare"})
            if len(bare) > 1:
                print(f"{tag}: พบตัวเลขเดี่ยวหลายชุด {bare} -> กำกวม turbine = null")

            per_image.append({"fields": glm_labels.collapse_fields(parsed), "content": content, "tokens": tokens})
            texts.append(content if len(images) == 1 else f"[ภาพ {idx + 1}: {cls}]\n{content}")

        values, confs, notes = merge_images(per_image)
        for note in notes:
            print(f"⚠️ GLM {note}")

        meter_type = "analog" if detected_classes and all(c == ANALOG_CLASS for c in detected_classes) else "digital"
        reading = MeterReading(
            meter_type=meter_type,
            detected_format=describe_classes(detected_classes),
            **{k: (float(v) if v is not None else None) for k, v in values.items()},
            low_confidence_fields=[],   # GLM ไม่รายงานเอง — confidence อยู่ใน confidence_scores
        )
        return EngineResult(
            raw_values=values,
            meter_type=meter_type,
            detected_format=reading.detected_format,
            confidence_scores=confs,
            raw_text="\n\n".join(texts),
            engine=self.name,
            raw_source="text",
            reading=reading,
        )
