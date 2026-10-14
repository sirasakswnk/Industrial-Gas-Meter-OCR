"""
ชั้นแปลง "ป้าย" ในข้อความอิสระของโมเดล -> ฟิลด์ของระบบ

ใช้กับ engine ที่ตอบเป็นข้อความอิสระ (GLM-OCR) — ไม่บังคับ JSON schema เพราะ spike พบว่า grammar ฝืนโมเดล
จนค่าผิด (pressure/temperature เป็น 0, อุณหภูมิไปลง line_v_m3) แต่ตอนปล่อยอิสระโมเดลอ่านถูก
GLM-OCR "ถอดข้อความตามหน้าจอ" ไม่ทำตามรูปแบบที่สั่ง จึงรองรับเลย์เอาต์ที่พบจริงจากรูปทดสอบ 66 รูป:

  ก. "ป้าย = ค่า หน่วย" / "ป้าย: ค่า"     เช่น  P = 9.656 bar          (Corus หน้า VALUE)
  ข. "ป้าย:" แล้วค่าอยู่บรรทัดถัดไป      เช่น  Unconverted Index:\\n000273155.00m³   (Corus หน้า INDEX)
  ค. "ป้าย ค่า หน่วย" ไม่มีตัวคั่น         เช่น  Vm 000299666.40 m³    (EK280 / MacBAT)
  ง. ตัวเลขเดี่ยว — เฉพาะภาพ analog       เช่น  16897967m³            (ลงได้ฟิลด์เดียวตามกฎอุปกรณ์)

กฎเหล็ก: จับคู่ค่ากับฟิลด์ได้ก็ต่อเมื่อ "รู้จักป้ายนั้นในตาราง" เท่านั้น ห้ามจับคู่ด้วยตำแหน่งอย่างเดียว
(ยกเว้น ง. ที่กฎอุปกรณ์กำหนดฟิลด์ไว้แล้ว) — ค่าที่จับคู่ผิดอันตรายกว่า null เพราะดูเหมือนสำเร็จแล้วถูกบันทึก
- ป้ายที่ไม่รู้จัก -> ทิ้ง + log ไม่เดา; ห้าม fuzzy / edit distance กับป้ายเด็ดขาด
  (Converted กับ Unconverted ต่างกันแค่ "Un" — fuzzy จะสลับยอดสะสมที่ใช้คิดเงินได้)
  คำที่ OCR สะกดผิดต้องเพิ่มเป็น alias ทีละตัวในตาราง
- Zb/Z (compressibility ratio) ไม่ใช่ correction_factor -> ทิ้ง (spike พบโมเดลสับสนสองค่านี้)
- ค่าที่ไม่ใช่ตัวเลขรูปแบบ 123 / 123.45 (ทศนิยมเป็นจุด) -> null ไม่ throw (เช่น "9,656" กำกวม -> null)
- ฟิลด์เดียวกันออกมาหลายครั้ง: ค่าเหมือนกัน -> ใช้ค่านั้น, ค่าต่างกัน -> null (ไม่เลือกให้)
ไม่มี I/O ไม่เรียกโมเดล — เทสต์ได้ตรงๆ (test_glm_labels.py)
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import reading_fields

# ป้าย (หลัง normalize_label) -> ฟิลด์ — ตรงตัวเท่านั้น ไม่มี fuzzy
LABEL_TO_FIELD: Dict[str, str] = {
    "p": "pressure_bar",
    "pressure": "pressure_bar",
    "t": "temperature_c",
    "temp": "temperature_c",
    "temperature": "temperature_c",
    "c": "correction_factor",
    "factor": "correction_factor",
    "correction": "correction_factor",
    "correction factor": "correction_factor",
    "qm": "qm_m3_h",
    "qb": "qb_sm3_h",
    "converted index": "correct_v_scm",
    "vb": "correct_v_scm",
    "converted vol": "correct_v_scm",       # Itron Corus Evo: "Converted vol."
    "converted volume": "correct_v_scm",
    "cv idx": "correct_v_scm",              # Itron Corus: "CV Idx"
    "convened index": "correct_v_scm",      # alias: OCR อ่าน "Converted Index" ผิดเป็น "Convened" (พบจริง)
    "unconverted index": "line_v_m3",
    "vm": "line_v_m3",
    "unconverted vol": "line_v_m3",         # Itron Corus Evo: "Unconverted vol."
    "unconverted volume": "line_v_m3",
    "uv idx": "line_v_m3",                  # Itron Corus: "UV Idx"
    "unconvened index": "line_v_m3",        # alias: OCR อ่าน "Unconverted Index" ผิด (พบจริงคู่กับข้างบน)
    "counter": "turbine_v_m3",
}

# ป้ายที่รู้จักแต่ "ไม่ใช่ฟิลด์ของเรา" — ทิ้งแน่นอน (log แยกจากป้ายที่ไม่รู้จัก)
DISCARD_LABELS = {"zb/z", "z/zb", "zb", "z"}

# ฟิลด์ที่แต่ละอุปกรณ์ให้ได้ (กฎ Device-to-Field Mapping เดียวกับ prompt ข้อ 5 ของ Gemini)
EVC_FIELDS = [k for k in reading_fields.FIELD_KEYS if k != reading_fields.TURBINE_FIELD]
ANALOG_FIELDS = [reading_fields.TURBINE_FIELD]

_BULLET = r"[ \t]*(?:[-*•][ \t]+)?"
# ก. / ข.  ป้าย [=:] ส่วนที่เหลือ
_SEP_LINE = re.compile(_BULLET + r"(?P<label>[^=:]*?[A-Za-z][^=:]*?)[ \t]*(?P<sep>[=:])[ \t]*(?P<rest>.*)$")
# ค่าต้นข้อความ: ส่วนตัวเลขก่อน (แยกหน่วยที่ติดกันได้ เช่น "477.0m3/h") ไม่ขึ้นต้นด้วยตัวเลข -> ทั้งก้อน (กลายเป็น null)
_VALUE_HEAD = re.compile(r"[-+]?\d[\d.,]*|[^ \t]*")
# ข. บรรทัดที่เป็นค่าของป้ายบรรทัดก่อน: ขึ้นต้นด้วยตัวเลข ตามด้วยหน่วย (ถ้ามี) ที่ไม่ขึ้นต้นด้วยตัวเลข
_VALUE_LINE = re.compile(r"[ \t]*(?P<value>[-+]?\d[\d.,]*)(?:[ \t]*\D.*)?$")
# ค. ป้าย (มีตัวอักษร) เว้นวรรค ค่า หน่วย — ป้าย = ข้อความสั้นที่สุดก่อนตัวเลขตัวแรกที่มีช่องว่างนำหน้า
_SPACE_LINE = re.compile(_BULLET + r"(?P<label>[^=:]*?[A-Za-z][^=:]*?)[ \t]+(?P<value>[-+]?\d[\d.,]*)(?:[ \t]*\D.*)?$")
_MAX_LABEL_LEN = 30   # ค. ข้อความยาวกว่านี้คือประโยค ไม่ใช่ป้าย -> ไม่นับเป็นป้ายเลย
# ง. ตัวเลขเดี่ยวทั้งบรรทัด (+ หน่วยลูกบาศก์เมตรที่เขียนได้หลายแบบ)
_BARE_NUMBER = re.compile(r"[ \t]*(?P<value>\d+(?:\.\d+)?)[ \t]*(?:m³|m3|m\^\{3\}|m\^3)?[ \t]*$")
_NUMBER = re.compile(r"^[-+]?\d+(?:\.\d+)?$")

BARE_LABEL = "(ตัวเลขเดี่ยว)"


def normalize_label(label: str) -> str:
    """"**Converted Index:**" -> "converted index", "Pressure (bar)" -> "pressure", "Zb / Z" -> "zb/z" """
    s = label.strip().strip("*_`").strip()
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)          # หน่วยในวงเล็บท้ายป้าย
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s+", " ", s).strip().rstrip(".:").strip()
    return s.lower()


def field_for_label(label: str) -> Optional[str]:
    """ป้าย -> ชื่อฟิลด์ หรือ None (ไม่รู้จัก / ป้ายที่ต้องทิ้ง) — ตรงตัวเท่านั้น"""
    key = normalize_label(label)
    if key in DISCARD_LABELS:
        return None
    return LABEL_TO_FIELD.get(key)


def to_number_str(value: str) -> Optional[str]:
    """ค่าดิบ -> สตริงตัวเลขตามที่โมเดลเขียน (คงศูนย์ท้าย "25.00") หรือ None ถ้าไม่ใช่ตัวเลขชัดเจน"""
    v = (value or "").strip()
    if not _NUMBER.match(v):
        return None
    return reading_fields.to_plain_str(v)


@dataclass
class LabeledValue:
    """ค่าที่แปลงเป็นฟิลด์ได้ 1 รายการ — span = ตำแหน่ง (ตัวอักษร) ของ "ค่า" ในข้อความ (ไม่รวมป้าย/หน่วย)"""
    field: str
    label: str
    value: Optional[str]          # None = มีค่าแต่แปลงเป็นตัวเลขไม่ได้
    raw_value: str
    span: Tuple[int, int]
    layout: str                   # "sep" (ก.) | "next_line" (ข.) | "space" (ค.) | "bare" (ง.)


@dataclass
class ParseResult:
    values: List[LabeledValue] = field(default_factory=list)
    discarded: List[str] = field(default_factory=list)    # ป้ายที่รู้จักแต่ไม่ใช่ฟิลด์ (Zb/Z ...)
    unknown: List[str] = field(default_factory=list)      # ป้ายที่ไม่รู้จัก
    not_allowed: List[str] = field(default_factory=list)  # ฟิลด์ที่อุปกรณ์นี้ให้ไม่ได้ (ผิดกฎ device)
    no_value: List[str] = field(default_factory=list)     # ป้ายที่รู้จักแต่ไม่มีค่าตามมา

    @property
    def label_count(self) -> int:
        """จำนวนป้ายที่พบ (ไม่นับตัวเลขเดี่ยว)"""
        labeled = sum(1 for v in self.values if v.layout != "bare")
        return labeled + len(self.discarded) + len(self.unknown) + len(self.not_allowed) + len(self.no_value)


def _lines(text: str) -> List[Tuple[int, str]]:
    """[(offset ต้นบรรทัด, ข้อความบรรทัดไม่รวม \\n)]"""
    out, pos = [], 0
    for line in text.split("\n"):
        out.append((pos, line))
        pos += len(line) + 1
    return out


def _classify(label: str, allowed_fields, result: ParseResult) -> Optional[str]:
    """ป้าย -> ฟิลด์ที่ใช้ได้ หรือ None (บันทึกเหตุผลลง result แล้ว)"""
    key = normalize_label(label)
    if key in DISCARD_LABELS:
        result.discarded.append(label)
        return None
    fld = LABEL_TO_FIELD.get(key)
    if fld is None:
        result.unknown.append(label)
        return None
    if allowed_fields is not None and fld not in allowed_fields:
        result.not_allowed.append(label)
        return None
    return fld


def parse_labeled_text(text: str, allowed_fields: Optional[List[str]] = None,
                       bare_number_field: Optional[str] = None) -> ParseResult:
    """ข้อความอิสระ -> รายการค่าที่แปลงได้ (เรียงตามตำแหน่งในข้อความ)

    allowed_fields    = ฟิลด์ที่อุปกรณ์ในภาพนี้ให้ได้ (None = ทุกฟิลด์) — ฟิลด์อื่นถูกทิ้ง ไม่ย้ายไปฟิลด์ใกล้เคียง
    bare_number_field = ฟิลด์ที่ "ตัวเลขเดี่ยว" ลงได้ (ใช้เฉพาะภาพ analog = turbine_v_m3; None = ไม่รับตัวเลขเดี่ยว)
    """
    result = ParseResult()
    lines = _lines(text or "")
    i = 0
    while i < len(lines):
        start, line = lines[i]
        i += 1

        m = _SEP_LINE.match(line)
        if m:                                                   # ก. / ข.
            label = m.group("label").strip()
            if not normalize_label(label):
                continue
            fld = _classify(label, allowed_fields, result)
            if fld is None:
                continue
            rest_start = start + m.start("rest")
            if m.group("rest").strip():                         # ก. ค่าอยู่บรรทัดเดียวกัน
                v = _VALUE_HEAD.match(m.group("rest"))
                raw = v.group(0)
                result.values.append(LabeledValue(fld, label, to_number_str(raw), raw,
                                                  (rest_start, rest_start + len(raw)), "sep"))
                continue
            # ข. "ป้าย:" บรรทัดเดียว -> ค่าต้องอยู่บรรทัดถัดไปทันที และขึ้นต้นด้วยตัวเลข
            nxt = _VALUE_LINE.match(lines[i][1]) if m.group("sep") == ":" and i < len(lines) else None
            if nxt is None:                                     # บรรทัดถัดไปเป็นป้ายอื่น/ข้อความ/ไม่มี
                result.no_value.append(label)
                continue
            n_start = lines[i][0]
            raw = nxt.group("value")
            result.values.append(LabeledValue(fld, label, to_number_str(raw), raw,
                                              (n_start + nxt.start("value"), n_start + nxt.end("value")),
                                              "next_line"))
            i += 1                                              # บรรทัดค่าถูกใช้แล้ว
            continue

        m = _SPACE_LINE.match(line)
        if m and len(m.group("label").strip()) <= _MAX_LABEL_LEN:   # ค.
            label = m.group("label").strip()
            fld = _classify(label, allowed_fields, result)
            if fld is not None:
                raw = m.group("value")
                result.values.append(LabeledValue(fld, label, to_number_str(raw), raw,
                                                  (start + m.start("value"), start + m.end("value")), "space"))
            continue

        if bare_number_field is not None:                       # ง.
            m = _BARE_NUMBER.match(line)
            if m:
                raw = m.group("value")
                result.values.append(LabeledValue(bare_number_field, BARE_LABEL, to_number_str(raw), raw,
                                                  (start + m.start("value"), start + m.end("value")), "bare"))
    return result


@dataclass
class FieldValue:
    value: Optional[str]
    spans: List[Tuple[int, int]]   # ทุกตำแหน่งที่ค่านี้ปรากฏ (ใช้หา confidence)
    conflict: bool = False


def collapse_fields(parsed: ParseResult) -> Dict[str, FieldValue]:
    """รวมรายการที่ฟิลด์ซ้ำกัน: ค่าเหมือนกันทุกครั้ง -> ใช้ค่านั้น (เก็บทุก span),
    ต่างกัน/มีอันที่แปลงไม่ได้ -> None (เช่น analog มีตัวเลขเดี่ยวหลายชุด -> กำกวม ไม่เลือกให้)"""
    out: Dict[str, FieldValue] = {}
    for lv in parsed.values:
        cur = out.get(lv.field)
        if cur is None:
            out[lv.field] = FieldValue(value=lv.value, spans=[lv.span])
            continue
        cur.spans.append(lv.span)
        if cur.conflict or cur.value != lv.value:
            cur.value, cur.conflict = None, True
    return out
