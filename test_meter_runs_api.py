"""
เทสต์ API หน้าแรก (เลือก/เพิ่ม Meter Run) + routing หน้าเว็บ — ไม่เรียก Gemini

ใช้ DB ชั่วคราวผ่าน METER_DB_PATH (ต้องตั้งก่อน import server) ไม่แตะ meter_records.db จริง

ครอบคลุม:
  1. GET /        -> หน้าเลือก Meter Run (runs.html)
  2. GET /ocr     -> หน้า OCR (index.html)
  3. GET /api/meter-runs           -> seed 2 Run
  4. GET /api/meter-runs/{id}      -> 200 / ไม่พบ 404
  5. POST /api/meter-runs          -> 201 + ตัดช่องว่าง
  6. POST Run ซ้ำ (ต่างตัวพิมพ์)    -> 409 + "Meter Run นี้มีอยู่แล้วในระบบ"
  7. POST company ว่าง -> 400, ขาดฟิลด์บังคับ -> 422
  8. DELETE: ไม่มีประวัติ -> 204, มีประวัติ -> 409, ไม่พบ -> 404, list มี reading_count

รัน: python test_meter_runs_api.py
"""
import os
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix="meterapi_test_")
os.environ["METER_DB_PATH"] = os.path.join(_TMP, "test.db")
os.environ["OCR_REQUEST_DIR"] = os.path.join(_TMP, "ocr_requests")

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402


def test_pages(client):
    r = client.get("/")
    assert r.status_code == 200 and 'id="runList"' in r.text, "/ ควรเป็นหน้าเลือก Meter Run"
    r = client.get("/ocr?run=1")
    assert r.status_code == 200 and 'id="dropzone"' in r.text, "/ocr ควรเป็นหน้า OCR"
    print("✅ test_pages: / = เลือก Run, /ocr = หน้า OCR")


def test_list_seed(client):
    r = client.get("/api/meter-runs")
    assert r.status_code == 200, r.text
    runs = r.json()
    assert [(x["company"], x["run_label"]) for x in runs] == [("ASM", "A"), ("ASM", "B")], runs
    print("✅ test_list_seed: seed ASM A/B")


def test_get_one(client):
    assert client.get("/api/meter-runs/1").json()["tag_fy"] == "ASM-0330-FY-057A"
    r = client.get("/api/meter-runs/999")
    assert r.status_code == 404 and "ไม่พบ" in r.json()["detail"], r.text
    print("✅ test_get_one: 200 / 404")


def test_create(client):
    r = client.post("/api/meter-runs", json={
        "company": " COFCO ", "run_label": "A",
        "tag_fe": "COFCO-FE-01", "serial_fe": "SFE", "tag_fy": "COFCO-FY-01", "serial_fy": "SFY",
    })
    assert r.status_code == 201, r.text
    run = r.json()
    assert run["company"] == "COFCO" and run["id"] == 3 and run["tag_fy"] == "COFCO-FY-01", run
    assert len(client.get("/api/meter-runs").json()) == 3
    print("✅ test_create: 201 + ตัดช่องว่าง")


def test_duplicate(client):
    r = client.post("/api/meter-runs", json={"company": "asm", "run_label": "a"})
    assert r.status_code == 409, r.text
    assert "Meter Run นี้มีอยู่แล้วในระบบ" in r.json()["detail"], r.text
    assert "ASM Run A" in r.json()["detail"], "ควรบอกชื่อ Run ตามที่มีในระบบ ไม่ใช่ที่ผู้ใช้พิมพ์"
    assert len(client.get("/api/meter-runs").json()) == 3, "Run ซ้ำต้องไม่ถูกเพิ่ม"
    print(f"✅ test_duplicate: 409 — {r.json()['detail']}")


def test_invalid(client):
    r = client.post("/api/meter-runs", json={"company": "  ", "run_label": "A"})
    assert r.status_code == 400, r.text
    r = client.post("/api/meter-runs", json={"company": "X"})
    assert r.status_code == 422, r.text
    print("✅ test_invalid: company ว่าง 400, ขาด run_label 422")


def test_delete(client):
    import db
    run = client.post("/api/meter-runs", json={"company": "del", "run_label": "x"}).json()
    assert run["id"] in [r["id"] for r in client.get("/api/meter-runs").json()]
    r = client.delete(f"/api/meter-runs/{run['id']}")
    assert r.status_code == 204, r.text
    assert run["id"] not in [r["id"] for r in client.get("/api/meter-runs").json()]
    assert client.delete(f"/api/meter-runs/{run['id']}").status_code == 404

    db.insert_reading(1, "2026-10-01T09:00", {}, {"correct_v_scm": "1"}, [], [], 0, [], engine="gemini")
    listed = {r["id"]: r for r in client.get("/api/meter-runs").json()}
    assert listed[1]["reading_count"] == 1 and listed[2]["reading_count"] == 0
    r = client.delete("/api/meter-runs/1")
    assert r.status_code == 409 and "มีประวัติการจด 1 รายการ" in r.json()["detail"], r.text
    assert client.get("/api/meter-runs/1").status_code == 200, "Run ที่มีประวัติต้องยังอยู่"
    print(f"✅ test_delete: 204 / 404 / 409 — {r.json()['detail']}")


def main():
    client = TestClient(server.app)
    tests = [test_pages, test_list_seed, test_get_one, test_create, test_duplicate, test_invalid,
             test_delete]
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
