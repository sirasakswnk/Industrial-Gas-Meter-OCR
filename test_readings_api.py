"""
เทสต์บันทึกค่า + PDF แบบครบวงจร (เฟส 4) ตามสเปคข้อ 9 — mock โมเดล ไม่เสียค่า API

ใช้ DB และโฟลเดอร์ ocr_requests ชั่วคราว (METER_DB_PATH / OCR_REQUEST_DIR) ไม่แตะของจริง

ครอบคลุม:
  1. บันทึกครั้งแรกของ Run ใหม่ -> ไม่มีค่าครั้งก่อน, PDF ช่องครั้งก่อน/ผลต่างว่าง
  2. บันทึกครั้งที่สอง -> PDF ค่าครั้งก่อน + ผลต่างถูก (Decimal)
  3. ค่าลดลง -> ยังบันทึกได้ (201) และ PDF แสดงผลต่างติดลบ
  4. checkbox ตัด Turbine 1 หลัก / 2 หลัก -> ตัดถูกหลัก
  5. แก้ค่าด้วยมือ -> raw_values ≠ final_values และ edited_fields ถูกบันทึก
  6. engine = "gemini" ทุกแถว, confidence_scores = null
  7. server ไม่เชื่อค่าจาก frontend: raw/ภาพมาจากที่ server เก็บ, final คำนวณใหม่
  8. กันบันทึกซ้ำ (409), request_id ปลอม/ไม่มี (400/404), ค่าผิดรูปแบบ (400)
  9. PDF ฝังฟอนต์ Sarabun + ข้อความไทย "ใบแจ้งปริมาณการใช้ก๊าซ" / "ค่าจดครั้งก่อน" อ่านกลับได้ตรง
  10. หน้าประวัติ: รายการ (กรองตาม Run, ใหม่สุดก่อน, มีชื่อ Run), รายละเอียด + ครั้งก่อน, ภาพ, /history
  11. ค่าครั้งก่อน: Run ใหม่บันทึก 2 รายการ -> รายการแรกไม่มีครั้งก่อน, รายการที่ 2 ดึงค่ารายการแรก
      ได้ครบทุกทาง (หน้า OCR / รายละเอียดหน้าประวัติ / PDF)
  12. เตือนผลต่างค่าสะสมติดลบ (/decrease-check): ติดลบ / ศูนย์ / ไม่มีครั้งก่อน — ไม่บล็อกการบันทึก

รัน: python test_readings_api.py
"""
import io
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

TMP = tempfile.mkdtemp(prefix="readings_test_")
os.environ["METER_DB_PATH"] = os.path.join(TMP, "test.db")
os.environ["OCR_REQUEST_DIR"] = os.path.join(TMP, "ocr_requests")

from fastapi.testclient import TestClient  # noqa: E402
from pypdf import PdfReader  # noqa: E402

import db  # noqa: E402
import server  # noqa: E402

IMG = os.path.join("experiment_images", "normal_2.jpg")


def model_text(correct, line, turbine, pressure="4.5"):
    return (f'{{"meter_type": "digital", "detected_format": "mock", "correct_v_scm": {correct}, '
            f'"line_v_m3": {line}, "pressure_bar": {pressure}, "temperature_c": 25.678, '
            f'"correction_factor": 0.9987, "qm_m3_h": 120.5, "turbine_v_m3": {turbine}, '
            f'"low_confidence_fields": ["pressure_bar"]}}')


def ocr(client, text):
    resp = MagicMock()
    resp.text = text
    resp.parsed = server.MeterReading.model_validate_json(text)
    with open(IMG, "rb") as f:
        files = [("files", ("normal_2.jpg", f.read(), "image/jpeg"))]
    with patch.object(server.client.models, "generate_content", return_value=resp):
        r = client.post("/api/ocr", files=files)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["request_id"], "ต้องได้ request_id สำหรับบันทึก"
    return data


def save(client, run_id, data, recorded_at, trim=0, overrides=None):
    entered = dict(data["editor"]["values"])
    entered.update(overrides or {})
    return client.post("/api/readings", json={
        "meter_run_id": run_id, "request_id": data["request_id"], "recorded_at": recorded_at,
        "entered_values": entered, "turbine_trim_digits": trim,
    })


def pdf_text(client, reading_id):
    r = client.get(f"/api/readings/{reading_id}/pdf")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf", r.text
    assert "attachment" in r.headers["content-disposition"]
    return PdfReader(io.BytesIO(r.content)), PdfReader(io.BytesIO(r.content)).pages[0].extract_text()


def cum(text, header):
    """ตารางค่าสะสมใน PDF (แถว จดครั้งหลัง/จดครั้งก่อน/ค่าแตกต่าง) ใต้หัวตารางที่ขึ้นต้นด้วย header
    -> {"จดครั้งหลัง": [...], "จดครั้งก่อน": [...], "ค่าแตกต่าง": [...]} (ช่องว่างไม่ถูกนับ)"""
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith(header))
    out = {}
    for ln in lines[i + 1:i + 4]:
        label, *values = ln.split()
        out[label] = values
    assert list(out) == ["จดครั้งหลัง", "จดครั้งก่อน", "ค่าแตกต่าง"], f"ลำดับแถวผิด: {list(out)}"
    return out


EVC = "CORRECT V. (SCM) LINE V. (M³)"
TURBINE = "TURBINE V. (M³)"


class Ctx:
    pass


def test_first_reading(client, ctx):
    ctx.run = client.post("/api/meter-runs", json={"company": "test", "run_label": "a",
                                                   "tag_fe": "t-fe-1", "tag_fy": "t-fy-1"}).json()
    data = ocr(client, model_text(1234000, 2400000, "12300.45"))
    r = save(client, ctx.run["id"], data, "2026-10-01T09:00", trim=2)
    assert r.status_code == 201, r.text
    ctx.first = r.json()["reading"]
    assert r.json()["pdf_url"] == f"/api/readings/{ctx.first['id']}/pdf"

    _, text = pdf_text(client, ctx.first["id"])
    assert cum(text, EVC) == {"จดครั้งหลัง": ["1234000", "2400000"], "จดครั้งก่อน": [], "ค่าแตกต่าง": []}
    assert cum(text, TURBINE) == {"จดครั้งหลัง": ["12300"], "จดครั้งก่อน": [], "ค่าแตกต่าง": []}
    assert "ยังไม่มีประวัติ" in text
    print("✅ 1. ครั้งแรก: ไม่มีค่าครั้งก่อน, PDF ช่องครั้งก่อน/ผลต่างว่าง")


def test_second_reading(client, ctx):
    data = ocr(client, model_text(1234567, 2345678, "12345.67"))
    r = save(client, ctx.run["id"], data, "2026-10-07T13:30", trim=2)
    assert r.status_code == 201, "ค่า LINE V. ลดลงต้องยังบันทึกได้: " + r.text
    ctx.second = r.json()["reading"]

    latest = client.get(f"/api/meter-runs/{ctx.run['id']}/latest-reading").json()
    assert latest["id"] == ctx.second["id"]

    reader, text = pdf_text(client, ctx.second["id"])
    evc = cum(text, EVC)
    assert evc["จดครั้งหลัง"] == ["1234567", "2345678"] and evc["จดครั้งก่อน"] == ["1234000", "2400000"], evc
    assert evc["ค่าแตกต่าง"][0] == "567", evc
    assert cum(text, TURBINE) == {"จดครั้งหลัง": ["12345"], "จดครั้งก่อน": ["12300"], "ค่าแตกต่าง": ["45"]}
    print("✅ 2. ครั้งที่สอง: ค่าครั้งก่อน + ผลต่างใน PDF ถูก")

    assert evc["ค่าแตกต่าง"][1] == "-54322", evc
    print("✅ 3. ค่าลดลง: บันทึกได้ (201) + PDF ผลต่างติดลบ")

    # PDF ครั้งแรก export ซ้ำหลังมีครั้งที่สองแล้ว -> ยังไม่มีค่าครั้งก่อน (ใช้ "ก่อนหน้าครั้งนั้น")
    _, first_text = pdf_text(client, ctx.first["id"])
    assert cum(first_text, EVC)["จดครั้งก่อน"] == [], "export ซ้ำครั้งแรกต้องไม่มีค่าครั้งก่อน"

    ctx.reader, ctx.text = reader, text


def test_turbine_trim(client, ctx):
    run = client.post("/api/meter-runs", json={"company": "test", "run_label": "trim"}).json()
    for trim, expected in ((0, "1123120"), (1, "112312"), (2, "11231")):
        data = ocr(client, model_text(1, 1, 1123120))
        r = save(client, run["id"], data, f"2026-10-0{trim + 1}T10:00", trim=trim)
        assert r.status_code == 201, r.text
        got = r.json()["reading"]
        assert got["final_values"]["turbine_v_m3"] == expected, (trim, got["final_values"])
        assert got["turbine_trim_digits"] == trim
        assert got["raw_values"]["turbine_v_m3"] == "1123120", "raw ต้องตามที่โมเดลเขียน (ไม่มี .0)"
    print("✅ 4. ตัด Turbine 0/1/2 หลัก -> 1123120 / 112312 / 11231")

    # ข้อจำกัดที่รู้อยู่ (รายงานไว้ตั้งแต่เฟส 3): ถ้าโมเดล "เขียน" 1123120.0 มาเองในข้อความ
    # แยกไม่ได้ว่าเป็น artifact หรือช่องแดง 1 ช่องที่เป็นเลข 0 -> ได้ 8 หลัก
    # UI จับได้: ช่องทีละหลักโชว์หลักเกิน + เตือน "โมเดลอ่านได้ทศนิยม 1 หลัก — ตรวจ checkbox"
    data = ocr(client, model_text(1, 1, "1123120.0"))
    assert data["editor"]["values"]["turbine_v_m3"] == "11231200"
    assert data["editor"]["turbine_raw_decimals"] == 1, "ต้องมีสัญญาณให้ UI เตือน"
    print("✅ 4b. ข้อจำกัดที่รู้อยู่: โมเดลเขียน 1123120.0 เอง -> 8 หลัก + turbine_raw_decimals=1 (UI เตือน)")


def test_manual_edit(client, ctx):
    data = ocr(client, model_text(1300000, 2500000, "12400.00"))
    r = save(client, ctx.run["id"], data, "2026-10-08T09:00", trim=2,
             overrides={"pressure_bar": "4.505", "qb_sm3_h": "540.2"})
    assert r.status_code == 201, r.text
    got = r.json()["reading"]
    assert got["raw_values"]["pressure_bar"] == "4.5" and got["final_values"]["pressure_bar"] == "4.505"
    assert got["raw_values"]["qb_sm3_h"] is None and got["final_values"]["qb_sm3_h"] == "540.2"
    assert got["edited_fields"] == ["pressure_bar", "qb_sm3_h"], got["edited_fields"]
    assert got["low_confidence_fields"] == ["pressure_bar"]
    print("✅ 5. แก้มือ: raw ≠ final + edited_fields = ['pressure_bar', 'qb_sm3_h']")


def test_engine_column(client, ctx):
    with db.get_conn() as conn:
        rows = conn.execute("SELECT engine, confidence_scores FROM readings").fetchall()
    assert rows and all(r["engine"] == "gemini" and r["confidence_scores"] is None for r in rows), \
        [tuple(r) for r in rows]
    print(f"✅ 6. engine = 'gemini' ครบทั้ง {len(rows)} แถว, confidence_scores = NULL")


def test_server_authority(client, ctx):
    got = ctx.second
    assert got["raw_values"]["turbine_v_m3"] == "12345.67", "raw ต้องมาจากที่ server เก็บตอน OCR"
    assert len(got["image_paths"]) == 1
    img = got["image_paths"][0]
    full = img if os.path.isabs(img) else os.path.join(os.path.dirname(server.__file__), img)
    assert os.path.exists(full), f"ภาพต้นฉบับต้องถูกเก็บไว้: {img}"
    # ส่ง final ที่ frontend "คำนวณผิด" มาไม่ได้ เพราะ server รับเฉพาะค่าในช่อง แล้วคำนวณเอง
    data = ocr(client, model_text(1, 1, "999.99"))
    r = save(client, ctx.run["id"], data, "2026-10-09T09:00", trim=2)
    assert r.json()["reading"]["final_values"]["turbine_v_m3"] == "999"
    print("✅ 7. raw/ภาพมาจาก server, final คำนวณฝั่ง server")


def test_rejections(client, ctx):
    data = ocr(client, model_text(1, 1, 1))
    assert save(client, ctx.run["id"], data, "2026-10-10T09:00").status_code == 201
    r = save(client, ctx.run["id"], data, "2026-10-10T09:00")
    assert r.status_code == 409 and "บันทึกไปแล้ว" in r.json()["detail"], r.text

    data = ocr(client, model_text(1, 1, 1))
    bad = [
        (dict(request_id="../../etc"), 400),
        (dict(request_id="1_abcdef"), 404),
        (dict(meter_run_id=999), 404),
        (dict(recorded_at="not a date"), 400),
        (dict(turbine_trim_digits=3), 400),
        (dict(entered_values={"pressure_bar": "4.5"}), 400),
        (dict(entered_values={}), 400),
    ]
    base = {"meter_run_id": ctx.run["id"], "request_id": data["request_id"],
            "recorded_at": "2026-10-11T09:00", "entered_values": data["editor"]["values"],
            "turbine_trim_digits": 0}
    for override, status in bad:
        r = client.post("/api/readings", json={**base, **override})
        assert r.status_code == status, (override, r.status_code, r.text)
    assert client.get("/api/readings/9999/pdf").status_code == 404
    print("✅ 8. ปฏิเสธ: บันทึกซ้ำ 409, request_id ปลอม 400/404, ค่าผิดรูปแบบ 400")


def test_pdf_thai_font(client, ctx):
    fonts = set()
    for f in ctx.reader.pages[0]["/Resources"]["/Font"].values():
        fo = f.get_object()
        desc = fo["/DescendantFonts"][0].get_object()["/FontDescriptor"].get_object()
        assert "/FontFile2" in desc, "ฟอนต์ต้องถูกฝังในไฟล์"
        fonts.add(str(fo["/BaseFont"]).split("+")[-1])
    assert fonts == {"Sarabun", "SarabunBold"}, fonts
    for phrase in ("ใบแจ้งปริมาณการใช้ก๊าซ", "ค่าจดครั้งก่อน", "จดครั้งหลัง", "ค่าแตกต่าง"):
        assert phrase in ctx.text, f"ไม่พบข้อความไทย '{phrase}' ใน PDF"
    print("✅ 9. PDF ฝัง Sarabun + ข้อความไทยอ่านกลับได้ตรง")

    # รูปแบบตามฟอร์มจริง: Bara, REMARK (CF/Qm/Qb, null -> "-" ไม่มีหน่วย)
    # ไม่มีส่วนที่ระบบเติมไม่ได้ (ป้าย "Remark =" / เกจบนท่อ / ช่องติ๊กสถานะ)
    lines = ctx.text.splitlines()
    assert "PRESSURE Bara 4.500" in lines, "หน่วย Pressure ต้องเป็น Bara"
    assert "Run A: CF EVC = 0.9987, CF คำนวณ = -, Qm = 120.5 m3/hr, Qb = -" in lines, \
        [ln for ln in lines if ln.startswith("Run ")]
    assert not any(ln.startswith(("CORRECTION FACTOR", "Qm ", "Qb ")) for ln in lines), \
        "CF/Qm/Qb ต้องอยู่ใน REMARK เท่านั้น ไม่ซ้ำในตาราง"
    assert "Remark =" not in ctx.text, "ระบบไม่มีช่องกรอกหมายเหตุ -> ต้องไม่มีป้าย 'Remark ='"
    for gone in ("Pressure Gauge", "Temperature Gauge", "Psig", "Calibrate", "ขัดข้อง", "เปลี่ยน Corrector"):
        assert gone not in ctx.text, f"PDF ต้องไม่มี '{gone}' (เอาเกจ/ช่องติ๊กออกแล้ว)"
    assert len(ctx.reader.pages) == 1
    print("✅ 9b. PDF: Bara, บรรทัด REMARK (null = \"-\" ไม่มีหน่วย), ไม่มี Remark =/เกจ/ช่องติ๊ก, 1 หน้า")


def test_history_api(client, ctx):
    all_rows = client.get("/api/readings").json()
    mine = client.get(f"/api/readings?meter_run_id={ctx.run['id']}").json()
    assert len(all_rows) > len(mine) > 0, (len(all_rows), len(mine))
    assert all(r["meter_run_id"] == ctx.run["id"] for r in mine)
    times = [r["recorded_at"] for r in mine]
    assert times == sorted(times, reverse=True), "ต้องใหม่สุดก่อน (ตามเวลาที่จด)"
    assert mine[0]["run"]["company"] == "TEST" and mine[0]["run"]["run_label"] == "A"
    edited_row = next(r for r in mine if r["edited_fields"])
    assert edited_row["edited_fields"] == ["pressure_bar", "qb_sm3_h"]

    detail = client.get(f"/api/readings/{ctx.second['id']}").json()
    assert detail["reading"]["id"] == ctx.second["id"]
    assert detail["previous"]["id"] == ctx.first["id"], "ครั้งก่อนต้องตรงกับที่ใช้ใน PDF"
    assert detail["run"]["tag_fy"] == "T-FY-1"
    assert client.get(f"/api/readings/{ctx.first['id']}").json()["previous"] is None
    assert client.get("/api/readings/9999").status_code == 404

    img = client.get(f"/api/readings/{ctx.second['id']}/images/0")
    assert img.status_code == 200 and img.content[:2] == bytes([0xFF, 0xD8]), "ต้องได้ภาพ JPEG ต้นฉบับ"
    assert client.get(f"/api/readings/{ctx.second['id']}/images/5").status_code == 404
    assert client.get(f"/api/readings/{ctx.second['id']}/images/-1").status_code == 404

    page = client.get("/history?run=1")
    assert page.status_code == 200 and 'id="historyTable"' in page.text
    print("✅ 10. ประวัติ: กรอง/เรียง/ชื่อ Run, รายละเอียด+ครั้งก่อน, ภาพ, หน้า /history")


def test_previous_reading_two_saves(client, ctx):
    run = client.post("/api/meter-runs", json={"company": "prev", "run_label": "a"}).json()
    latest = f"/api/meter-runs/{run['id']}/latest-reading"

    # ก่อนบันทึกครั้งแรก: หน้า OCR ไม่มีค่าครั้งก่อน
    assert client.get(latest).json() is None

    first = save(client, run["id"], ocr(client, model_text(47715600, 43084, "100.00")),
                 "2026-10-07T14:24", trim=2).json()["reading"]
    detail1 = client.get(f"/api/readings/{first['id']}").json()
    assert detail1["previous"] is None, "รายการแรกของ Run ต้องไม่มีครั้งก่อน"
    _, pdf1 = pdf_text(client, first["id"])
    assert cum(pdf1, EVC) == {"จดครั้งหลัง": ["47715600", "43084"], "จดครั้งก่อน": [], "ค่าแตกต่าง": []},         "PDF รายการแรก: ครั้งก่อน/ผลต่างต้องว่าง"

    # ก่อนบันทึกครั้งที่ 2: หน้า OCR ต้องเห็นค่าของรายการแรก
    shown = client.get(latest).json()
    assert shown["id"] == first["id"]
    assert shown["final_values"]["correct_v_scm"] == "47715600"
    assert shown["final_values"]["line_v_m3"] == "43084"

    second = save(client, run["id"], ocr(client, model_text(47716000, 43100, "105.00")),
                  "2026-10-07T14:31", trim=2).json()["reading"]
    detail2 = client.get(f"/api/readings/{second['id']}").json()
    assert detail2["previous"]["id"] == first["id"]
    assert detail2["previous"]["final_values"]["correct_v_scm"] == "47715600"
    _, pdf2 = pdf_text(client, second["id"])
    assert cum(pdf2, EVC) == {"จดครั้งหลัง": ["47716000", "43100"], "จดครั้งก่อน": ["47715600", "43084"],
                              "ค่าแตกต่าง": ["400", "16"]}, cum(pdf2, EVC)
    assert cum(pdf2, TURBINE) == {"จดครั้งหลัง": ["105"], "จดครั้งก่อน": ["100"], "ค่าแตกต่าง": ["5"]}
    print("✅ 11. ค่าครั้งก่อน: รายการแรกไม่มี, รายการที่ 2 ดึงรายการแรกได้ (หน้า OCR / ประวัติ / PDF)")


def test_decrease_check(client, ctx):
    run = client.post("/api/meter-runs", json={"company": "dec", "run_label": "a"}).json()
    url = f"/api/meter-runs/{run['id']}/decrease-check"

    def check(correct, line, turbine, trim=2):
        r = client.post(url, json={"entered_values": {"correct_v_scm": correct, "line_v_m3": line,
                                                      "turbine_v_m3": turbine},
                                   "turbine_trim_digits": trim})
        assert r.status_code == 200, r.text
        return r.json()

    # ไม่มีค่าครั้งก่อน -> ไม่เตือน
    got = check("1", "1", "1", trim=0)
    assert got == {"previous_reading_id": None, "decreased": []}, got

    first = save(client, run["id"], ocr(client, model_text(47715600, 43084, "12345.67")),
                 "2026-10-07T14:24", trim=2).json()["reading"]
    assert first["final_values"]["turbine_v_m3"] == "12345"

    # ติดลบ -> เตือนพร้อมชื่อฟิลด์ (เทียบกับรายการล่าสุดที่หน้า OCR แสดง)
    got = check("40497286", "2086449", "1234499", trim=2)   # Turbine ตัด 2 หลัก -> 12344 < 12345
    assert got["previous_reading_id"] == first["id"]
    assert [(d["label"], d["difference"]) for d in got["decreased"]] ==         [("CORRECT V.", "-7218314"), ("TURBINE V.", "-1")], got
    assert got["decreased"][0]["message"].startswith("CORRECT V. มีค่าน้อยกว่าครั้งก่อน")

    # Turbine เทียบ "หลังตัด" เหมือนค่าที่จะบันทึก: ไม่ติ๊ก -> 1234499 > 12345 -> ไม่เตือน
    assert [d["key"] for d in check("47715600", "43084", "1234499", trim=0)["decreased"]] == []

    # ศูนย์ -> ไม่เตือน
    assert check("47715600", "43084", "1234567")["decreased"] == []

    # ค่าในช่องผิดรูปแบบ / Run ไม่มี -> 400 / 404 (หน้า OCR ซ่อนคำเตือน)
    bad = client.post(url, json={"entered_values": {"correct_v_scm": "12a"}})
    assert bad.status_code == 400, bad.text
    assert client.post("/api/meter-runs/9999/decrease-check",
                       json={"entered_values": {}}).status_code == 404

    # เตือนอย่างเดียว ไม่บล็อก: บันทึกค่าที่ติดลบได้ และ PDF แสดงผลต่างตามจริง (ไม่มีเครื่องหมายพิเศษ)
    r = save(client, run["id"], ocr(client, model_text(40497286, 2086449, "12344.99")),
             "2026-10-07T14:31", trim=2)
    assert r.status_code == 201, r.text
    _, text = pdf_text(client, r.json()["reading"]["id"])
    assert cum(text, EVC)["ค่าแตกต่าง"] == ["-7218314", "2043365"], cum(text, EVC)
    assert cum(text, TURBINE)["ค่าแตกต่าง"] == ["-1"]
    print("✅ 12. ผลต่างติดลบ: เตือน+ชื่อฟิลด์ / ศูนย์ไม่เตือน / ไม่มีครั้งก่อนไม่เตือน / "
          "Turbine เทียบหลังตัด / บันทึกได้ + PDF ตามจริง")


def main():
    if not os.path.exists(IMG):
        print(f"❌ ไม่พบภาพทดสอบ {IMG}")
        sys.exit(1)
    client = TestClient(server.app)
    ctx = Ctx()
    tests = [test_first_reading, test_second_reading, test_turbine_trim, test_manual_edit,
             test_engine_column, test_server_authority, test_rejections, test_pdf_thai_font,
             test_history_api, test_previous_reading_two_saves, test_decrease_check]
    try:
        for fn in tests:
            fn(client, ctx)
    except AssertionError as e:
        print(f"❌ {fn.__name__}: {e}")
        print("─" * 60)
        print("❌ FAIL")
        sys.exit(1)
    print("─" * 60)
    print("🎉 ผ่านทั้งหมด")


if __name__ == "__main__":
    main()
