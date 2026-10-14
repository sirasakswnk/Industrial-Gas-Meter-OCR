"""
เทสต์ตัวดักชนิดมิเตอร์ในคำขอเดียว (/api/ocr) + ตาข่าย analog หลัง parse — ไม่ยิง Gemini จริง

กฎปัจจุบัน (ผ่อนจากข้อ 3.6 เดิมที่บล็อก analog + digital):
  1 Meter Run = EVC (digital, อาจหลายหน้าจอ) + Turbine (analog, 1 หน้าปัด)
  -> analog + digital ผ่าน, digital หลายภาพผ่าน, analog > 1 ภาพ -> 400

ครอบคลุม:
  1-4. check_request_meter_types()  — ตรรกะ (ไม่นับภาพที่ OBB หาไม่เจอ)
  5-8. apply_analog_only_guard()    — ตาข่ายหลัง parse: analog ล้วน -> ทิ้งค่าฟิลด์ EVC
  9.   analog + digital ผ่าน /api/ocr จริง  -> 200, Gemini ถูกเรียก 1 ครั้ง, ค่า EVC ไม่ถูกทิ้ง
  10.  digital + digital                  -> 200
  11.  analog + analog                    -> 400 + ข้อความใหม่, Gemini ไม่ถูกเรียก
  12.  analog ภาพเดียว แต่โมเดลใส่ค่า EVC มา -> ค่า EVC ถูกทิ้ง, turbine คงอยู่

เคส 9-12 ใช้ OBB จริงกับภาพใน experiment_images/ (class ยืนยันแล้วจาก experiment_obb/obb_result.csv)
  analog_1.jpg / analog_2.jpg = analog_dial, normal_2.jpg / normal_3.png = digital_screen

รัน: python test_mixed_meter.py
"""
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

# ไม่ให้เทสต์เขียนลง DB / ocr_requests ของจริง
_TMP = tempfile.mkdtemp(prefix="mixed_test_")
os.environ["METER_DB_PATH"] = os.path.join(_TMP, "test.db")
os.environ["OCR_REQUEST_DIR"] = os.path.join(_TMP, "ocr_requests")

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402

IMG_DIR = "experiment_images"
ANALOG = "analog_1.jpg"
ANALOG_B = "analog_2.jpg"
DIGITAL_A = "normal_2.jpg"
DIGITAL_B = "normal_3.png"

A, D = server.ANALOG_CLASS, server.DIGITAL_CLASS

# โมเดลตอบค่าครบทั้ง EVC และ Turbine (เหมือนคำขอ EVC + Turbine ที่ถูกต้อง)
FULL_TEXT = ('{"meter_type": "analog", "detected_format": "mock", "correct_v_scm": 1234567, '
             '"line_v_m3": 2345678, "pressure_bar": 4.500, "temperature_c": 25.67, '
             '"turbine_v_m3": 12345.67, "low_confidence_fields": ["pressure_bar", "turbine_v_m3"]}')


def upload(*names):
    """เตรียม multipart files สำหรับ /api/ocr"""
    files = []
    for n in names:
        mime = "image/png" if n.lower().endswith(".png") else "image/jpeg"
        with open(os.path.join(IMG_DIR, n), "rb") as f:
            files.append(("files", (n, f.read(), mime)))
    return files


def fake_gemini_response(text=FULL_TEXT):
    """response ปลอมที่ parse_gemini_response() อ่านได้ (ใช้ .parsed) + ข้อความดิบ"""
    resp = MagicMock()
    resp.text = text
    resp.parsed = server.MeterReading.model_validate_json(text)
    return resp


def fake_result(text=FULL_TEXT):
    """OcrResult ก่อนผ่านตาข่าย (สำหรับเทสต์ apply_analog_only_guard ตรงๆ)"""
    resp = fake_gemini_response(text)
    reading = server.parse_gemini_response(resp)
    raw, source = server.extract_raw_values(resp, reading)
    return server.OcrResult(**reading.model_dump(), raw_values=raw, raw_source=source)


# ── 1-4: ตรรกะ check_request_meter_types ──────────────────────────────
def test_check_mixed_allowed():
    assert server.check_request_meter_types(["a.jpg", "d.jpg"], [A, D]) is None
    assert server.check_request_meter_types(["d1", "a", "d2"], [D, A, D]) is None
    print("✅ test_check_mixed_allowed: analog + digital -> ผ่าน (1 Run = EVC + Turbine)")


def test_check_many_digital_allowed():
    assert server.check_request_meter_types(["1", "2", "3"], [D, D, D]) is None
    print("✅ test_check_many_digital_allowed: digital หลายหน้าจอ -> ผ่าน")


def test_check_two_analog_blocked():
    msg = server.check_request_meter_types(["d.jpg", "a1.jpg", "a2.jpg"], [D, A, A])
    assert msg and msg.startswith("พบมิเตอร์ analog มากกว่า 1 ตัวในคำขอเดียว "
                                  "หนึ่งรายการจดรองรับ Turbine Meter ได้ 1 ตัว"), msg
    assert "รูปที่ 2 (a1.jpg)" in msg and "รูปที่ 3 (a2.jpg)" in msg, msg
    print("✅ test_check_two_analog_blocked: analog 2 ภาพ -> บล็อก + ระบุรูป")


def test_check_ignores_none():
    assert server.check_request_meter_types(["a", "x"], [A, None]) is None
    assert server.check_request_meter_types(["x", "y"], [None, None]) is None
    assert server.check_request_meter_types([], []) is None
    print("✅ test_check_ignores_none: ภาพที่ OBB หาไม่เจอไม่ถูกนับ")


# ── 5-8: ตาข่าย analog หลัง parse ───────────────────────────────────────
def test_guard_analog_only_drops_evc():
    r = fake_result()
    dropped = server.apply_analog_only_guard(r, [A])
    assert set(dropped) == {"correct_v_scm", "line_v_m3", "pressure_bar", "temperature_c"}, dropped
    assert dropped["pressure_bar"] == "4.500", "ต้องเก็บค่าดิบเดิมไว้ตรวจย้อนหลัง"
    assert r.correct_v_scm is None and r.raw_values["correct_v_scm"] is None
    assert r.turbine_v_m3 == 12345.67 and r.raw_values["turbine_v_m3"] == "12345.67", "turbine ต้องคงอยู่"
    assert r.low_confidence_fields == ["turbine_v_m3"], "ฟิลด์ที่ถูกทิ้งต้องหลุดจาก low_confidence"
    print("✅ test_guard_analog_only_drops_evc: analog ล้วน -> ทิ้งฟิลด์ EVC, turbine คงอยู่")


def test_guard_mixed_untouched():
    # meter_type = "analog" แต่มีภาพ digital ปน -> ห้ามแตะ (ค่า EVC ถูกต้องจากจอ digital)
    r = fake_result()
    assert server.apply_analog_only_guard(r, [D, A]) == {}
    assert r.correct_v_scm == 1234567 and r.raw_values["pressure_bar"] == "4.500"
    print("✅ test_guard_mixed_untouched: มี digital ปน -> ไม่ทิ้งอะไร แม้ meter_type = analog")


def test_guard_digital_only_untouched():
    r = fake_result(FULL_TEXT.replace('"analog"', '"digital"'))
    assert server.apply_analog_only_guard(r, [D, D]) == {}
    print("✅ test_guard_digital_only_untouched: digital ล้วน -> ไม่ทิ้งอะไร")


def test_guard_undetected_uses_meter_type():
    # OBB หาไม่เจอทุกภาพ: ใช้ meter_type ตัดสิน
    r = fake_result()
    assert set(server.apply_analog_only_guard(r, [None])) >= {"correct_v_scm"}
    r = fake_result(FULL_TEXT.replace('"analog"', '"digital"'))
    assert server.apply_analog_only_guard(r, [None]) == {}
    print("✅ test_guard_undetected_uses_meter_type: OBB ไม่เจอ -> ใช้ meter_type")


# ── 9-12: ผ่าน /api/ocr จริง (OBB จริง + Gemini mock) ────────────────────
def test_endpoint_allows_mixed(client):
    with patch.object(server.client.models, "generate_content",
                      return_value=fake_gemini_response()) as gen:
        r = client.post("/api/ocr", files=upload(ANALOG, DIGITAL_A))
    assert r.status_code == 200, f"คาดหวัง 200 แต่ได้ {r.status_code}: {r.text}"
    assert gen.call_count == 1, f"Gemini ควรถูกเรียก 1 ครั้ง แต่ได้ {gen.call_count}"
    data = r.json()
    assert data["guard_dropped"] == {} and data["editor"]["values"]["correct_v_scm"] == "1234567", \
        "คำขอ EVC + Turbine ต้องไม่ถูกทิ้งค่า EVC"
    assert data["guard_applied"] is False
    print("✅ test_endpoint_allows_mixed: analog + digital -> 200 + ค่า EVC ครบ")


def test_endpoint_allows_same_type(client):
    with patch.object(server.client.models, "generate_content",
                      return_value=fake_gemini_response()) as gen:
        r = client.post("/api/ocr", files=upload(DIGITAL_A, DIGITAL_B))
    assert r.status_code == 200, f"คาดหวัง 200 แต่ได้ {r.status_code}: {r.text}"
    assert gen.call_count == 1, f"Gemini ควรถูกเรียก 1 ครั้ง แต่ได้ {gen.call_count}"
    print("✅ test_endpoint_allows_same_type: digital + digital -> 200")


def test_endpoint_blocks_two_analog(client):
    with patch.object(server.client.models, "generate_content") as gen:
        r = client.post("/api/ocr", files=upload(DIGITAL_A, ANALOG, ANALOG_B))
    assert r.status_code == 400, f"คาดหวัง 400 แต่ได้ {r.status_code}: {r.text}"
    detail = r.json()["detail"]
    assert detail.startswith("พบมิเตอร์ analog มากกว่า 1 ตัวในคำขอเดียว "
                             "หนึ่งรายการจดรองรับ Turbine Meter ได้ 1 ตัว"), detail
    assert ANALOG in detail and ANALOG_B in detail, f"ไม่ระบุชื่อไฟล์: {detail}"
    assert not gen.called, "Gemini ถูกเรียก ทั้งที่ควรบล็อกก่อน"
    print("✅ test_endpoint_blocks_two_analog: analog 2 ภาพ -> 400 + ไม่เรียก Gemini")
    print(f"   detail: {detail}")


def test_endpoint_analog_guard(client):
    with patch.object(server.client.models, "generate_content",
                      return_value=fake_gemini_response()):
        r = client.post("/api/ocr", files=upload(ANALOG))
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["correct_v_scm"] is None and data["raw_values"]["correct_v_scm"] is None
    assert data["editor"]["values"]["correct_v_scm"] is None, "ค่าที่ถูกทิ้งต้องไม่ขึ้นในช่อง"
    assert data["editor"]["values"]["turbine_v_m3"] == "1234567"
    assert data["guard_dropped"]["correct_v_scm"] == "1234567"
    assert data["guard_applied"] is True, "ต้องส่งสัญญาณให้หน้า OCR ขึ้นข้อความ"
    print("✅ test_endpoint_analog_guard: analog ภาพเดียว + ค่า EVC หลุดมา -> ถูกทิ้ง, turbine คงอยู่")


def main():
    missing = [n for n in (ANALOG, ANALOG_B, DIGITAL_A, DIGITAL_B)
               if not os.path.exists(os.path.join(IMG_DIR, n))]
    if missing:
        print(f"❌ ไม่พบภาพทดสอบใน {IMG_DIR}/: {missing}")
        sys.exit(1)

    client = TestClient(server.app)
    tests = [
        (test_check_mixed_allowed, False),
        (test_check_many_digital_allowed, False),
        (test_check_two_analog_blocked, False),
        (test_check_ignores_none, False),
        (test_guard_analog_only_drops_evc, False),
        (test_guard_mixed_untouched, False),
        (test_guard_digital_only_untouched, False),
        (test_guard_undetected_uses_meter_type, False),
        (test_endpoint_allows_mixed, True),
        (test_endpoint_allows_same_type, True),
        (test_endpoint_blocks_two_analog, True),
        (test_endpoint_analog_guard, True),
    ]
    failed = 0
    for fn, needs_client in tests:
        try:
            fn(client) if needs_client else fn()
        except AssertionError as e:
            failed += 1
            print(f"❌ {fn.__name__}: {e}")
    print("─" * 60)
    if failed:
        print(f"❌ FAIL — {failed} เทสต์ไม่ผ่าน")
        sys.exit(1)
    print("🎉 ผ่านทั้งหมด")


if __name__ == "__main__":
    main()
