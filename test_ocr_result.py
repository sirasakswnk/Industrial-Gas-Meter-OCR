"""
เทสต์ response ของ /api/ocr หลังเพิ่ม raw_values + editor (เฟส 3) — mock โมเดล ไม่เสียค่า API

ครอบคลุม:
  1. raw_values มาจากข้อความ JSON ตามที่โมเดลเขียน ("4.500" ไม่กลายเป็น 4.5)
  2. โมเดลเขียน 1123120 (จำนวนเต็ม) -> raw "1123120" ไม่มี ".0"
  3. ข้อความดิบใช้ไม่ได้ -> fallback จาก float ที่ parse แล้ว โดยไม่เติม ".0"
  3b. fallback + Turbine 12345.67 -> เก็บทุกหลัก, ตัด 2 -> "12345"
  3c. ข้อความมี markdown fence -> ลอกแล้วใช้ข้อความดิบ (12345.60 คงศูนย์ท้าย)
  4. editor.values / truncated_fields / turbine_raw_decimals ถูกต้อง
  5. /api/reading-fields คืน 8 ฟิลด์ตามสเปค
  6. /api/meter-runs/{id}/latest-reading: ไม่มีประวัติ -> null, มี -> ล่าสุด, Run ไม่มี -> 404
  7. prompt (เฟส 6): ย่อหน้า 2 แบบใหม่ + กฎข้อ 5 Device-to-Field Mapping + กฎข้อ 2 (ช่องแดง) ยังอยู่

ใช้ DB ชั่วคราวผ่าน METER_DB_PATH, ภาพทดสอบ experiment_images/normal_2.jpg
รัน: python test_ocr_result.py
"""
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

TMP = tempfile.mkdtemp(prefix="ocrresult_test_")
os.environ["METER_DB_PATH"] = os.path.join(TMP, "test.db")
os.environ["OCR_REQUEST_DIR"] = os.path.join(TMP, "ocr_requests")

from fastapi.testclient import TestClient  # noqa: E402

import db  # noqa: E402
import server  # noqa: E402

IMG = os.path.join("experiment_images", "normal_2.jpg")

MODEL_TEXT = """{"meter_type": "digital", "detected_format": "mock",
  "correct_v_scm": 123456, "pressure_bar": 4.500, "temperature_c": 25.678,
  "turbine_v_m3": 1123120, "low_confidence_fields": ["pressure_bar"]}"""


def fake_response(text, turbine=1123120.0):
    resp = MagicMock()
    resp.text = text
    resp.parsed = server.MeterReading(
        meter_type="digital", detected_format="mock", correct_v_scm=123456.0,
        pressure_bar=4.5, temperature_c=25.678, turbine_v_m3=turbine,
        low_confidence_fields=["pressure_bar"],
    )
    return resp


def post_ocr(client, text, turbine=1123120.0):
    with open(IMG, "rb") as f:
        files = [("files", ("normal_2.jpg", f.read(), "image/jpeg"))]
    with patch.object(server.client.models, "generate_content",
                      return_value=fake_response(text, turbine)):
        r = client.post("/api/ocr", files=files)
    assert r.status_code == 200, r.text
    return r.json()


def test_raw_from_text(client):
    data = post_ocr(client, MODEL_TEXT)
    raw = data["raw_values"]
    assert raw["pressure_bar"] == "4.500", f"ต้องคงศูนย์ท้ายตามที่โมเดลเขียน: {raw['pressure_bar']}"
    assert raw["turbine_v_m3"] == "1123120", f"ห้ามมี .0: {raw['turbine_v_m3']}"
    assert raw["correct_v_scm"] == "123456" and raw["line_v_m3"] is None
    assert data["raw_source"] == "text"
    assert data["pressure_bar"] == 4.5, "ฟิลด์เดิม (parsed) ต้องยังอยู่ครบ"
    assert data["low_confidence_fields"] == ["pressure_bar"]

    ed = data["editor"]
    assert ed["values"]["pressure_bar"] == "4.500"
    assert ed["values"]["temperature_c"] == "25.67"
    assert ed["values"]["turbine_v_m3"] == "1123120"
    assert ed["truncated_fields"] == ["temperature_c"]
    assert ed["turbine_raw_decimals"] == 0
    print("✅ test_raw_from_text: raw ตามข้อความดิบ + editor ถูกต้อง")


def test_raw_fallback(client):
    data = post_ocr(client, "not json at all")
    raw = data["raw_values"]
    assert raw["turbine_v_m3"] == "1123120", f"fallback จาก float ห้ามเติม .0: {raw['turbine_v_m3']}"
    assert raw["pressure_bar"] == "4.5", raw["pressure_bar"]
    assert data["raw_source"] == "fallback", "ต้องบอก frontend ให้ขึ้นเตือน"
    assert data["editor"]["values"]["pressure_bar"] == "4.500"
    print("✅ test_raw_fallback: ข้อความดิบพัง -> fallback ไม่เติม .0")


def test_fallback_turbine_keeps_decimals(client):
    # fallback + float 12345.67: ต้องเก็บทุกหลัก (ห้าม .0f -> "12346" -> ตัด 2 -> "123")
    data = post_ocr(client, "not json at all", turbine=12345.67)
    assert data["raw_values"]["turbine_v_m3"] == "12345.67", data["raw_values"]["turbine_v_m3"]
    digits = data["editor"]["values"]["turbine_v_m3"]
    assert digits == "1234567", digits
    assert data["editor"]["turbine_raw_decimals"] == 2
    assert server.reading_fields.trim_turbine(digits, 2) == "12345"
    assert data["raw_source"] == "fallback"
    print("✅ test_fallback_turbine_keeps_decimals: fallback 12345.67 ตัด 2 -> 12345")


def test_fenced_text_uses_raw(client):
    # SDK parse ข้อความที่มี markdown fence ได้ -> ต้องลอก fence แล้วใช้ข้อความดิบ (ไม่ตกไป fallback)
    fenced = '```json\n{"meter_type": "analog", "detected_format": "m", "turbine_v_m3": 12345.60}\n```'
    data = post_ocr(client, fenced, turbine=12345.6)
    assert data["raw_values"]["turbine_v_m3"] == "12345.60", "ต้องได้ศูนย์ท้ายจากข้อความดิบ"
    assert data["raw_source"] == "text", "ลอก fence แล้วต้องไม่ตกไป fallback"
    digits = data["editor"]["values"]["turbine_v_m3"]
    assert server.reading_fields.trim_turbine(digits, 2) == "12345", digits
    print("✅ test_fenced_text_uses_raw: ```json``` -> ใช้ข้อความดิบ 12345.60 ตัด 2 -> 12345")


def test_reading_fields(client):
    data = client.get("/api/reading-fields").json()
    got = [(f["key"], f["decimals"]) for f in data["fields"]]
    assert got == [("correct_v_scm", 0), ("line_v_m3", 0), ("pressure_bar", 3), ("temperature_c", 2),
                   ("correction_factor", 4), ("qm_m3_h", 1), ("qb_sm3_h", 1),
                   ("turbine_v_m3", 0)], got
    assert data["digit_boxes"] == 10 and data["turbine_field"] == "turbine_v_m3"
    print("✅ test_reading_fields: 8 ฟิลด์ + ทศนิยมตามสเปค")


def test_latest_reading(client):
    r = client.get("/api/meter-runs/1/latest-reading")
    assert r.status_code == 200 and r.json() is None, r.text
    for when in ("2026-10-01T09:00", "2026-10-05T09:00"):
        db.insert_reading(1, when, {}, {"correct_v_scm": when[8:10]}, [], [], 0, [], engine="gemini")
    latest = client.get("/api/meter-runs/1/latest-reading").json()
    assert latest["final_values"]["correct_v_scm"] == "05", latest
    assert client.get("/api/meter-runs/999/latest-reading").status_code == 404
    print("✅ test_latest_reading: null / ล่าสุด / 404")


def test_prompt_phase6(client):
    p = server.SYSTEM_INSTRUCTION
    assert "SAME metering run" in p and "TWO separate devices" in p, "ย่อหน้าที่ 2 ต้องเป็นแบบใหม่"
    assert "SAME physical gas meter unit" not in p, "ข้อความเดิมต้องถูกแทนที่"
    assert "5. Device-to-Field Mapping:" in p
    assert "turbine_v_m3 ONLY" in p
    assert "Never put an analog roller reading into line_v_m3 or correct_v_scm." in p
    assert "2 red slots mean 2 decimal places" in p, "กฎข้อ 2 (นับช่องแดง) ต้องคงไว้"
    assert p.index("4. Low-Confidence") < p.index("5. Device-to-Field"), "กฎข้อ 5 ต้องต่อท้ายข้อ 4"
    print("✅ test_prompt_phase6: prompt แก้ครบ 2 จุด, กฎช่องแดงยังอยู่")


def main():
    if not os.path.exists(IMG):
        print(f"❌ ไม่พบภาพทดสอบ {IMG}")
        sys.exit(1)
    client = TestClient(server.app)
    tests = [test_raw_from_text, test_raw_fallback, test_fallback_turbine_keeps_decimals,
             test_fenced_text_uses_raw, test_reading_fields, test_latest_reading, test_prompt_phase6]
    failed = 0
    for fn in tests:
        try:
            fn(client)
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
