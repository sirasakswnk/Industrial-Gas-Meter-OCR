"""
เทสต์ชั้นฐานข้อมูล (db.py) — ใช้ไฟล์ SQLite ชั่วคราว ไม่แตะ meter_records.db จริง

ครอบคลุม:
  1. init_db สร้างตาราง + คอลัมน์ครบตามสเปค
  2. seed ใส่ 2 Run ครั้งแรก และรัน init_db ซ้ำไม่ seed ซ้ำ
  3. create_meter_run / get_meter_run
  3b. Run ซ้ำ (ไม่สนตัวพิมพ์/ช่องว่าง) -> DuplicateMeterRunError, DB เก่าได้ unique index ด้วย
  4. insert_reading -> JSON ไป-กลับครบ, engine = "gemini", confidence_scores = None
  5. Run ใหม่ยังไม่มีประวัติ -> get_previous_reading คืน None
  6. get_previous_reading เรียงตามเวลาที่จด (recorded_at) ไม่ใช่ลำดับที่กดบันทึก
  7. list_readings กรองตาม Run ได้ + ใหม่สุดก่อน
  8. FK: บันทึก reading ของ Run ที่ไม่มีอยู่ -> error
  9. turbine_trim_digits นอก 0/1/2 -> error
  10. delete_meter_run: ลบได้เมื่อไม่มีประวัติ, มีประวัติ -> MeterRunInUseError, ไม่พบ -> False

รัน: python test_db.py
"""
import os
import sys
import sqlite3
import tempfile

import db

TMP_DIR = tempfile.mkdtemp(prefix="meterdb_test_")
DB = os.path.join(TMP_DIR, "test.db")

SAMPLE_RAW = {"correct_v_scm": 123456.0, "line_v_m3": 98765.0, "pressure_bar": 4.512,
              "turbine_v_m3": 12345.67}


def add_reading(run_id, recorded_at, **overrides):
    kwargs = dict(
        meter_run_id=run_id, recorded_at=recorded_at,
        raw_values=SAMPLE_RAW, final_values={**SAMPLE_RAW, "turbine_v_m3": 12345},
        edited_fields=[], low_confidence_fields=[], turbine_trim_digits=2,
        image_paths=[], engine="gemini", db_path=DB,
    )
    kwargs.update(overrides)
    return db.insert_reading(**kwargs)


def test_schema():
    db.init_db(DB)
    with db.get_conn(DB) as conn:
        runs_cols = [r["name"] for r in conn.execute("PRAGMA table_info(meter_runs)")]
        read_cols = [r["name"] for r in conn.execute("PRAGMA table_info(readings)")]
    assert runs_cols == ["id", "company", "run_label", "tag_fe", "serial_fe",
                         "tag_fy", "serial_fy", "created_at"], runs_cols
    assert read_cols == ["id", "meter_run_id", "recorded_at", "created_at", "raw_values",
                         "final_values", "edited_fields", "low_confidence_fields",
                         "turbine_trim_digits", "image_paths", "engine",
                         "confidence_scores"], read_cols
    print("✅ test_schema: คอลัมน์ครบตามสเปคทั้ง 2 ตาราง")


def test_seed_idempotent():
    runs = db.list_meter_runs(DB)
    assert len(runs) == 2, f"seed ควรได้ 2 Run แต่ได้ {len(runs)}"
    db.init_db(DB)
    assert len(db.list_meter_runs(DB)) == 2, "init_db รอบสองไม่ควร seed ซ้ำ"
    a = runs[0]
    assert (a["company"], a["run_label"], a["tag_fe"], a["tag_fy"]) == \
        ("ASM", "A", "ASM-0330-FE-057A", "ASM-0330-FY-057A"), a
    print("✅ test_seed_idempotent: seed 2 Run, รันซ้ำไม่เพิ่ม")


def test_create_meter_run():
    run = db.create_meter_run("COFCO", "A", "COFCO-FE-1", "S-FE-1", "COFCO-FY-1", "S-FY-1",
                              db_path=DB)
    assert run["id"] and run["company"] == "COFCO" and run["serial_fy"] == "S-FY-1", run
    assert run["created_at"], "created_at ต้องถูกใส่อัตโนมัติ"
    assert db.get_meter_run(run["id"], DB) == run
    assert db.get_meter_run(9999, DB) is None
    print(f"✅ test_create_meter_run: สร้าง Run #{run['id']} ได้")
    return run


def test_duplicate_run_rejected():
    for company, label in [("COFCO", "A"), ("cofco", "a"), ("  COFCO ", " A")]:
        try:
            db.create_meter_run(company, label, db_path=DB)
        except db.DuplicateMeterRunError:
            continue
        raise AssertionError(f"({company!r}, {label!r}) ควรถูกปฏิเสธเพราะซ้ำ COFCO/A")
    try:
        db.create_meter_run("  ", "A", db_path=DB)
    except db.DuplicateMeterRunError:
        raise AssertionError("company ว่างควรเป็น ValueError ธรรมดา ไม่ใช่ duplicate")
    except ValueError:
        pass
    else:
        raise AssertionError("company ว่างควรถูกปฏิเสธ")
    db.create_meter_run("COFCO", "B", db_path=DB)  # Run อื่นของบริษัทเดิมต้องเพิ่มได้
    print("✅ test_duplicate_run_rejected: ซ้ำ (ไม่สนตัวพิมพ์/ช่องว่าง) -> DuplicateMeterRunError")


def test_uppercase_codes():
    run = db.create_meter_run(" cofco ", "c", "cofco-fe-01c", "sn-fe-abc", "cofco-fy-01c", "sn-fy-abc",
                              db_path=DB)
    assert (run["company"], run["run_label"], run["tag_fe"], run["tag_fy"]) == \
        ("COFCO", "C", "COFCO-FE-01C", "COFCO-FY-01C"), run
    assert (run["serial_fe"], run["serial_fy"]) == ("sn-fe-abc", "sn-fy-abc"), "serial ต้องคงตามที่กรอก"
    print("✅ test_uppercase_codes: company/run_label/tag เป็นตัวใหญ่, serial คงเดิม")


def test_unique_added_to_existing_db():
    # DB ที่สร้างก่อนมี unique index (สถานะของ meter_records.db หลังเฟส 1) ต้องได้ constraint ด้วย
    old = os.path.join(TMP_DIR, "old.db")
    with db.get_conn(old) as conn:
        conn.executescript(db.SCHEMA.split("-- กัน Run ซ้ำ")[0])
    db.init_db(old)
    try:
        db.create_meter_run("ASM", "A", db_path=old)
    except db.DuplicateMeterRunError:
        print("✅ test_unique_added_to_existing_db: DB เก่าได้ unique index หลัง init_db")
        return
    raise AssertionError("DB เก่าไม่ได้ unique index")


def test_insert_roundtrip(run_id):
    r = add_reading(run_id, "2026-10-01T09:30",
                    edited_fields=["pressure_bar"], low_confidence_fields=["temperature_c"],
                    image_paths=["debug_images/x.jpg"])
    assert r["raw_values"] == SAMPLE_RAW, r["raw_values"]
    assert r["final_values"]["turbine_v_m3"] == 12345
    assert r["edited_fields"] == ["pressure_bar"]
    assert r["low_confidence_fields"] == ["temperature_c"]
    assert r["image_paths"] == ["debug_images/x.jpg"]
    assert r["turbine_trim_digits"] == 2
    assert r["engine"] == "gemini", r["engine"]
    assert r["confidence_scores"] is None
    assert r["recorded_at"] == "2026-10-01T09:30:00", r["recorded_at"]
    assert r["created_at"] != r["recorded_at"], "created_at ต้องเป็นเวลาที่กดบันทึก ไม่ใช่เวลาที่จด"
    print("✅ test_insert_roundtrip: JSON ไป-กลับครบ, engine=gemini, confidence_scores=None")
    return r


def test_no_previous_for_new_run():
    run = db.create_meter_run("NEW", "A", db_path=DB)
    assert db.get_previous_reading(run["id"], db_path=DB) is None
    print("✅ test_no_previous_for_new_run: Run ใหม่ไม่มีค่าครั้งก่อน")


def test_previous_by_recorded_at(run_id, first):
    # บันทึก "ทีหลัง" แต่วันที่จด "ก่อน" first -> ไม่ใช่ค่าล่าสุด
    backdated = add_reading(run_id, "2026-09-01T09:30")
    later = add_reading(run_id, "2026-10-08T09:30")

    latest = db.get_previous_reading(run_id, db_path=DB)
    assert latest["id"] == later["id"], "ค่าล่าสุดต้องเป็นอันที่วันที่จดใหม่สุด"
    assert db.get_previous_reading(run_id, later["id"], DB)["id"] == first["id"]
    assert db.get_previous_reading(run_id, first["id"], DB)["id"] == backdated["id"]
    assert db.get_previous_reading(run_id, backdated["id"], DB) is None
    print("✅ test_previous_by_recorded_at: ค่าครั้งก่อนเรียงตามเวลาที่จด")


def test_list_readings(run_id):
    seed_run = db.list_meter_runs(DB)[0]["id"]
    other = add_reading(seed_run, "2026-10-02T10:00")
    mine = db.list_readings(run_id, DB)
    assert len(mine) == 3 and all(r["meter_run_id"] == run_id for r in mine)
    times = [r["recorded_at"] for r in mine]
    assert times == sorted(times, reverse=True), times
    assert any(r["id"] == other["id"] for r in db.list_readings(db_path=DB))
    print("✅ test_list_readings: กรองตาม Run + ใหม่สุดก่อน")


def test_fk_enforced():
    try:
        add_reading(9999, "2026-10-01T09:30")
    except sqlite3.IntegrityError:
        print("✅ test_fk_enforced: Run ที่ไม่มีอยู่ -> IntegrityError")
        return
    raise AssertionError("ควรถูกปฏิเสธเพราะ meter_run_id ไม่มีอยู่")


def test_trim_digits_validated(run_id):
    try:
        add_reading(run_id, "2026-10-01T09:30", turbine_trim_digits=3)
    except ValueError:
        print("✅ test_trim_digits_validated: trim=3 -> ValueError")
        return
    raise AssertionError("turbine_trim_digits=3 ควรถูกปฏิเสธ")


def test_engine_required(run_id):
    """ไม่มีค่า default ของ engine — ลืมส่ง/ส่งค่าว่างต้อง error ทันที ไม่ถูกบันทึกเป็น 'gemini' เงียบๆ"""
    kwargs = dict(meter_run_id=run_id, recorded_at="2026-10-02T09:00", raw_values={}, final_values={},
                  edited_fields=[], low_confidence_fields=[], turbine_trim_digits=0, image_paths=[],
                  db_path=DB)
    try:
        db.insert_reading(**kwargs)
    except TypeError:
        pass
    else:
        raise AssertionError("ไม่ส่ง engine ต้อง TypeError")
    for bad in ("", "   ", None):
        try:
            db.insert_reading(**kwargs, engine=bad)
        except ValueError:
            continue
        raise AssertionError(f"engine={bad!r} ต้อง ValueError")

    with db.get_conn(DB) as conn:
        col = next(r for r in conn.execute("PRAGMA table_info(readings)") if r["name"] == "engine")
    assert col["dflt_value"] is None, f"schema ต้องไม่มี DEFAULT ของ engine: {col['dflt_value']}"
    assert col["notnull"] == 1
    print("✅ test_engine_required: ไม่ส่ง -> TypeError, ว่าง/None -> ValueError, schema ไม่มี DEFAULT")


def test_delete_meter_run(used_run_id):
    runs = {r["id"]: r for r in db.list_meter_runs(DB)}
    assert runs[used_run_id]["reading_count"] == 3, runs[used_run_id]
    try:
        db.delete_meter_run(used_run_id, DB)
    except db.MeterRunInUseError as e:
        assert e.reading_count == 3
    else:
        raise AssertionError("Run ที่มีประวัติต้องลบไม่ได้")
    assert db.get_meter_run(used_run_id, DB) is not None, "ต้องยังอยู่"

    fresh = db.create_meter_run("DEL", "A", db_path=DB)
    assert db.delete_meter_run(fresh["id"], DB) is True
    assert db.get_meter_run(fresh["id"], DB) is None
    assert db.delete_meter_run(fresh["id"], DB) is False, "ลบซ้ำ -> ไม่พบ"
    db.create_meter_run("DEL", "A", db_path=DB)  # ลบแล้วเพิ่มชื่อเดิมกลับได้
    print("✅ test_delete_meter_run: มีประวัติลบไม่ได้, ไม่มีประวัติลบได้, ไม่พบ -> False")


def main():
    failed = 0
    try:
        test_schema()
        test_seed_idempotent()
        run = test_create_meter_run()
        test_duplicate_run_rejected()
        test_uppercase_codes()
        test_unique_added_to_existing_db()
        first = test_insert_roundtrip(run["id"])
        test_no_previous_for_new_run()
        test_previous_by_recorded_at(run["id"], first)
        test_list_readings(run["id"])
        test_fk_enforced()
        test_trim_digits_validated(run["id"])
        test_engine_required(run["id"])
        test_delete_meter_run(run["id"])
    except AssertionError as e:
        failed += 1
        print(f"❌ {e}")
    print("─" * 60)
    if failed:
        print("❌ FAIL")
        sys.exit(1)
    print("🎉 ผ่านทั้งหมด")


if __name__ == "__main__":
    main()
