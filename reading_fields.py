"""
นิยามฟิลด์ค่ามิเตอร์ + logic จัดรูปแบบค่า/ตัดทศนิยม Turbine — แหล่งเดียวของระบบ

frontend ดึง FIELDS ผ่าน /api/reading-fields (ไม่ต้องเขียนตาราง label/ทศนิยมซ้ำใน JS)
และหน้าบันทึก (เฟส 4) ใช้ฟังก์ชันชุดเดียวกันคำนวณ final_values ฝั่ง server

หลักสำคัญ: ทำงานบน "สตริงตัวเลข" ทั้งหมด ไม่ผ่าน float
  - str(float) เติม ".0" ให้จำนวนเต็ม -> 1123120.0 -> "11231200" -> logic ตัด Turbine ผิดหลัก
  - f"{v:.0f}" ปัดเศษ -> 12345.67 -> "12346" -> ผิดตัวอย่างในสเปค (ต้องได้ 12345)
"""
import re
from decimal import Decimal, InvalidOperation
from typing import Optional, Tuple

# จำนวนช่องตัวเลขต่อฟิลด์บนหน้า OCR (เติมจากขวาไปซ้าย)
DIGIT_BOXES = 10

# ตำแหน่งจุดทศนิยมตายตัวตามสเปค — device: fy = Electronic Volume Corrector, fe = Turbine Meter
# cumulative = ค่าสะสม (คำนวณผลต่างใน PDF + เตือนค่าลดลง)
FIELDS = [
    {"key": "correct_v_scm",     "label": "CORRECT V.",        "unit": "SCM",   "decimals": 0, "device": "fy", "cumulative": True},
    {"key": "line_v_m3",         "label": "LINE V.",           "unit": "M³",    "decimals": 0, "device": "fy", "cumulative": True},
    {"key": "pressure_bar",      "label": "PRESSURE",          "unit": "Bara",  "decimals": 3, "device": "fy", "cumulative": False},
    {"key": "temperature_c",     "label": "TEMPERATURE",       "unit": "°C",    "decimals": 2, "device": "fy", "cumulative": False},
    {"key": "correction_factor", "label": "CORRECTION FACTOR", "unit": "—",     "decimals": 4, "device": "fy", "cumulative": False},
    {"key": "qm_m3_h",           "label": "Qm",                "unit": "m3/h",  "decimals": 1, "device": "fy", "cumulative": False},
    {"key": "qb_sm3_h",          "label": "Qb",                "unit": "Sm3/h", "decimals": 1, "device": "fy", "cumulative": False},
    {"key": "turbine_v_m3",      "label": "TURBINE V.",        "unit": "M³",    "decimals": 0, "device": "fe", "cumulative": True},
]
FIELD_KEYS = [f["key"] for f in FIELDS]
FIELD_DECIMALS = {f["key"]: f["decimals"] for f in FIELDS}
CUMULATIVE_FIELDS = [f["key"] for f in FIELDS if f["cumulative"]]
TURBINE_FIELD = "turbine_v_m3"

_PLAIN_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")


def to_plain_str(value) -> Optional[str]:
    """ค่าตัวเลขจากโมเดล -> สตริงทศนิยมธรรมดา (ไม่มี exponent ไม่มี artifact ของ float)

    - str (จาก json.loads(..., parse_float=str)) -> คงหลักตามที่โมเดลเขียนมา ("4.500" ยังเป็น "4.500")
    - int -> str(int)
    - float (ทางสำรอง เมื่อไม่มีข้อความดิบ) -> จำนวนเต็มใช้ .0f (ไม่เติม ".0"), อื่นๆ ใช้ repr ที่สั้นสุด
    คืน None ถ้าไม่ใช่ตัวเลข
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, float):
            if value != value or value in (float("inf"), float("-inf")):
                return None
            if value.is_integer():
                return f"{value:.0f}"
            d = Decimal(repr(value))
        elif isinstance(value, int):
            return str(value)
        else:
            d = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    if not d.is_finite():
        return None
    s = format(d, "f")  # ขยาย exponent (1.2e5 -> 120000) แต่คงศูนย์ท้ายตามที่เขียน
    return s if _PLAIN_NUMBER.match(s) else None


def _split(s: str) -> Tuple[bool, str, str]:
    neg = s.startswith("-")
    body = s[1:] if neg else s
    int_part, _, frac_part = body.partition(".")
    return neg, int_part, frac_part


def format_fixed(value, decimals: int) -> Tuple[Optional[str], bool]:
    """จัดรูปแบบตามจำนวนทศนิยมตายตัว -> (สตริง, ตัดทศนิยมเกินทิ้งหรือไม่)

    ทศนิยมขาด -> เติม 0 ("4.5" -> "4.500"), เกิน -> ตัดทิ้งไม่ปัด + flag ให้ UI เตือน
    ตัด 0 นำหน้าส่วนจำนวนเต็ม (เหลืออย่างน้อย "0")
    """
    s = to_plain_str(value)
    if s is None:
        return None, False
    neg, int_part, frac_part = _split(s)
    truncated = len(frac_part.rstrip("0")) > decimals  # ศูนย์ท้ายที่เกินไม่นับว่าเสียข้อมูล
    frac_part = frac_part[:decimals].ljust(decimals, "0")
    int_part = int_part.lstrip("0") or "0"
    out = int_part + ("." + frac_part if decimals > 0 else "")
    if neg and out.strip("0.") != "":
        out = "-" + out
    return out, truncated


def turbine_digits(value) -> Tuple[Optional[str], int]:
    """ค่า Turbine -> (สตริงตัวเลขล้วนที่ลบจุดแล้ว, จำนวนหลักทศนิยมที่โมเดลเขียนมา)

    ตามสเปค: "ลบจุดทศนิยมออก -> ได้สตริงตัวเลขล้วน" — ใช้เติมช่องทีละหลักและเป็น input ของ trim_turbine
    "12345.67" -> ("1234567", 2), "1234567" -> ("1234567", 0), float 1123120.0 -> ("1123120", 0)
    ค่าติดลบไม่มีความหมายสำหรับมิเตอร์สะสม -> None
    """
    s = to_plain_str(value)
    if s is None or s.startswith("-"):
        return None, 0
    _, int_part, frac_part = _split(s)
    return (int_part + frac_part).lstrip("0") or "0", len(frac_part)


def trim_turbine(digits: Optional[str], trim: int) -> Optional[str]:
    """ตัดหลักขวาสุดทิ้งตาม checkbox (0/1/2) — ทำบนสตริงตัวเลข ไม่ใช่ตำแหน่งจุดทศนิยม

    "1234567" ตัด 2 -> "12345" | ตัดจนไม่เหลือหลัก -> None
    """
    if trim not in (0, 1, 2):
        raise ValueError(f"trim ต้องเป็น 0, 1 หรือ 2 (ได้ {trim})")
    if digits is None or not digits.isdigit():
        return None
    if trim == 0:
        return digits.lstrip("0") or "0"
    if len(digits) <= trim:
        return None
    return digits[:-trim].lstrip("0") or "0"


def initial_editor_values(raw_values: dict) -> dict:
    """ค่าเริ่มต้นสำหรับเติมช่องทีละหลัก จาก raw_values ของโมเดล

    คืน {"values": {field: สตริงหรือ None}, "truncated_fields": [...], "turbine_raw_decimals": int}
    - ฟิลด์ทั่วไป: format_fixed ตามทศนิยมตายตัว
    - Turbine: สตริงตัวเลขล้วน (ยังไม่ตัด — การตัดเป็นหน้าที่ checkbox ที่ผู้ใช้ติ๊กทีหลัง)
    """
    values, truncated_fields = {}, []
    turbine_raw_decimals = 0
    for key in FIELD_KEYS:
        raw = raw_values.get(key)
        if key == TURBINE_FIELD:
            values[key], turbine_raw_decimals = turbine_digits(raw)
        else:
            values[key], truncated = format_fixed(raw, FIELD_DECIMALS[key])
            if truncated:
                truncated_fields.append(key)
    return {"values": values, "truncated_fields": truncated_fields,
            "turbine_raw_decimals": turbine_raw_decimals}


# ── บันทึกค่า (เฟส 4): คำนวณ final_values ฝั่ง server จากค่าในช่องทีละหลัก ──

def normalize_entered(key: str, value) -> Optional[str]:
    """ตรวจ + normalize ค่าที่ผู้ใช้กรอกในช่องทีละหลัก (สตริง) — ผิดรูปแบบ raise ValueError

    ฟิลด์ทั่วไป: ต้องมีทศนิยมครบตามจำนวนตายตัวพอดี (ช่องทีละหลักบังคับอยู่แล้ว)
    Turbine: สตริงตัวเลขล้วน (ก่อนตัด)
    """
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key}: ค่าต้องเป็นสตริงตัวเลข")
    s = value.strip()
    label = next(f["label"] for f in FIELDS if f["key"] == key)
    if key == TURBINE_FIELD:
        if not s.isdigit():
            raise ValueError(f"{label}: ต้องเป็นตัวเลขล้วน")
        return s.lstrip("0") or "0"
    d = FIELD_DECIMALS[key]
    pattern = rf"^-?\d+\.\d{{{d}}}$" if d > 0 else r"^-?\d+$"
    if not re.match(pattern, s):
        raise ValueError(f"{label}: ต้องมีทศนิยม {d} หลักพอดี")
    out, _ = format_fixed(s, d)
    return out


def compute_final_values(entered: dict, trim: int) -> dict:
    """final_values = ค่าที่คนตรวจแล้ว + ตัดทศนิยม Turbine ตาม checkbox (ทุกค่าเป็นสตริง)"""
    final = {}
    for key in FIELD_KEYS:
        value = normalize_entered(key, entered.get(key))
        if key == TURBINE_FIELD and value is not None:
            trimmed = trim_turbine(value, trim)
            if trimmed is None:
                raise ValueError(f"TURBINE V.: ตัด {trim} หลักแล้วไม่เหลือตัวเลข")
            value = trimmed
        final[key] = value
    return final


def find_edited_fields(entered: dict, raw_values: dict) -> list:
    """ฟิลด์ที่คนแก้ = ค่าในช่อง (normalize) ต่างจากค่าเริ่มต้นที่ได้จากโมเดล (normalize แบบเดียวกัน)

    Turbine เทียบสตริงตัวเลข "ก่อนตัด" — การติ๊กตัดทศนิยมไม่นับว่าเป็นการแก้ค่า
    """
    initial = initial_editor_values(raw_values)["values"]
    return [key for key in FIELD_KEYS if normalize_entered(key, entered.get(key)) != initial[key]]


def difference(current: Optional[str], previous: Optional[str]) -> Optional[str]:
    """ผลต่างค่าสะสม (ครั้งหลัง − ครั้งก่อน) ด้วย Decimal — ไม่มีค่าใดค่าหนึ่ง = None"""
    if current is None or previous is None:
        return None
    try:
        return format(Decimal(current) - Decimal(previous), "f")
    except InvalidOperation:
        return None


def decreased_fields(final: dict, prev_final: Optional[dict]) -> list:
    """ฟิลด์ค่าสะสมที่ผลต่างติดลบ (ครั้งหลัง < ครั้งก่อน) — มาตรวัดปริมาตรสะสมเดินหน้าอย่างเดียว
    ติดลบ = มีอะไรผิดเสมอ (จดคนละมิเตอร์ / อ่านเลขผิด / เลือก Run ผิด) -> เตือน แต่ไม่บล็อกการบันทึก

    ผลต่างเป็นศูนย์ / ไม่มีค่าครั้งก่อน / ไม่มีค่าครั้งนี้ -> ไม่เตือน
    """
    prev_final = prev_final or {}
    out = []
    for f in FIELDS:
        if not f["cumulative"]:
            continue
        diff = difference(final.get(f["key"]), prev_final.get(f["key"]))
        if diff is None or Decimal(diff) >= 0:
            continue
        out.append({
            "key": f["key"], "label": f["label"],
            "current": final[f["key"]], "previous": prev_final[f["key"]], "difference": diff,
            "message": f"{f['label']} มีค่าน้อยกว่าครั้งก่อน "
                       "ตรวจสอบว่าเป็นมิเตอร์ตัวเดียวกันหรืออ่านค่าถูกต้องหรือไม่",
        })
    return out
