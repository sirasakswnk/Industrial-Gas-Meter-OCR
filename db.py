"""
ชั้นฐานข้อมูล (SQLite ไฟล์เดียว) — บันทึก Meter Run + ประวัติการอ่านค่า

ใช้ sqlite3 ของ standard library เท่านั้น (คงความเป็น standalone, ห้ามใช้ PostgreSQL)
เปิด connection ใหม่ทุกครั้งที่เรียก เพราะ FastAPI รัน endpoint แบบ sync ใน threadpool
และ sqlite3 connection ห้ามใช้ข้าม thread

ฟิลด์ JSON (raw_values, final_values, ...) เข้ารหัส/ถอดรหัสในไฟล์นี้ที่เดียว
ผู้เรียกส่ง/รับเป็น dict/list ของ Python ได้เลย
"""
import os
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, List

# override path ได้ผ่าน .env: METER_DB_PATH=... (เทสต์ใช้ไฟล์ชั่วคราวแทน)
DB_PATH = os.environ.get(
    "METER_DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "meter_records.db")
)

# คอลัมน์ของ readings ที่เก็บเป็น JSON text
JSON_COLUMNS = ("raw_values", "final_values", "edited_fields",
                "low_confidence_fields", "image_paths", "confidence_scores")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meter_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    company     TEXT NOT NULL,
    run_label   TEXT NOT NULL,
    tag_fe      TEXT,           -- Tag ของ Turbine Meter
    serial_fe   TEXT,
    tag_fy      TEXT,           -- Tag ของ Electronic Volume Corrector
    serial_fy   TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS readings (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    meter_run_id          INTEGER NOT NULL REFERENCES meter_runs(id),
    recorded_at           TEXT NOT NULL,   -- วัน/เวลาที่ผู้ใช้กรอก (เวลาที่จด)
    created_at            TEXT NOT NULL,   -- เวลาที่กดบันทึกจริง (ระบบใส่เอง)
    raw_values            TEXT NOT NULL,   -- JSON: ค่าดิบจากโมเดล ก่อนคนแก้
    final_values          TEXT NOT NULL,   -- JSON: ค่าหลังคนแก้ + หลังตัดทศนิยม Turbine
    edited_fields         TEXT NOT NULL,   -- JSON list
    low_confidence_fields TEXT NOT NULL,   -- JSON list (จากโมเดล)
    turbine_trim_digits   INTEGER NOT NULL DEFAULT 0 CHECK (turbine_trim_digits IN (0, 1, 2)),
    image_paths           TEXT NOT NULL,   -- JSON list
    engine                TEXT NOT NULL,   -- ชื่อ OCR engine ที่อ่านค่า (ต้องส่งมาเสมอ ไม่มีค่า default)
    confidence_scores     TEXT             -- JSON รายฟิลด์ หรือ NULL (ยังไม่มีค่า)
);

CREATE INDEX IF NOT EXISTS idx_readings_run_time
    ON readings (meter_run_id, recorded_at);

-- กัน Run ซ้ำ: ถ้าซ้ำได้ ประวัติจะแตกเป็นสองก้อนและ "ค่าครั้งก่อน" หายเงียบๆ
-- ใช้ unique index แทน UNIQUE ใน CREATE TABLE เพื่อให้ DB ที่สร้างไปแล้วได้ constraint ด้วย
-- NOCASE: "asm" กับ "ASM" นับเป็น Run เดียวกัน
CREATE UNIQUE INDEX IF NOT EXISTS uq_meter_runs_company_run
    ON meter_runs (company COLLATE NOCASE, run_label COLLATE NOCASE);
"""

# ตัวอย่างสำหรับเดโม — ใส่เฉพาะตอนตาราง meter_runs ว่าง (รัน init_db ซ้ำได้ไม่ซ้ำข้อมูล)
SEED_METER_RUNS = [
    {"company": "ASM", "run_label": "A",
     "tag_fe": "ASM-0330-FE-057A", "serial_fe": "DEMO-FE-057A",
     "tag_fy": "ASM-0330-FY-057A", "serial_fy": "DEMO-FY-057A"},
    {"company": "ASM", "run_label": "B",
     "tag_fe": "ASM-0330-FE-057B", "serial_fe": "DEMO-FE-057B",
     "tag_fy": "ASM-0330-FY-057B", "serial_fy": "DEMO-FY-057B"},
]


class DuplicateMeterRunError(ValueError):
    """เพิ่ม Meter Run ที่ company + run_label ซ้ำกับที่มีอยู่แล้ว"""


class MeterRunInUseError(ValueError):
    """ลบ Meter Run ที่มีประวัติการจดแล้ว — ห้ามลบ ไม่งั้นประวัติหลุดจาก Run"""

    def __init__(self, reading_count: int):
        super().__init__(f"มีประวัติการจด {reading_count} รายการ")
        self.reading_count = reading_count


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def get_conn(db_path: Optional[str] = None):
    """เปิด connection -> commit เมื่อสำเร็จ / rollback เมื่อพัง -> ปิดเสมอ"""
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")  # SQLite ปิด FK ไว้เป็น default
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[str] = None):
    """สร้างตาราง (ถ้ายังไม่มี) + ใส่ seed data (ถ้ายังไม่มี Meter Run เลย)"""
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)
        count = conn.execute("SELECT COUNT(*) FROM meter_runs").fetchone()[0]
        if count == 0:
            for run in SEED_METER_RUNS:
                _insert_meter_run(conn, **run)
            print(f"🗄️ DB seeded with {len(SEED_METER_RUNS)} demo meter runs")
    print(f"🗄️ DB ready: {db_path or DB_PATH}")


# ── meter_runs ────────────────────────────────────────────────────────

def _insert_meter_run(conn, company, run_label, tag_fe="", serial_fe="",
                      tag_fy="", serial_fy="") -> int:
    cur = conn.execute(
        "INSERT INTO meter_runs (company, run_label, tag_fe, serial_fe, tag_fy, serial_fy, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (company, run_label, tag_fe, serial_fe, tag_fy, serial_fy, _now()),
    )
    return cur.lastrowid


def create_meter_run(company: str, run_label: str, tag_fe: str = "", serial_fe: str = "",
                     tag_fy: str = "", serial_fy: str = "", db_path: Optional[str] = None) -> dict:
    """เพิ่ม Meter Run ใหม่ — ตัดช่องว่างหัวท้ายทุกช่อง (กัน " ASM" หลุด unique)

    company/run_label/tag ทั้งสองแปลงเป็นตัวใหญ่ (รหัสอุปกรณ์ของจริงเป็นตัวใหญ่หมด และไปโผล่ใน PDF)
    serial คงตามที่กรอก
    raise ValueError ถ้า company/run_label ว่าง, DuplicateMeterRunError ถ้าซ้ำ
    """
    company, run_label = (company or "").strip().upper(), (run_label or "").strip().upper()
    tag_fe, serial_fe = (tag_fe or "").strip().upper(), (serial_fe or "").strip()
    tag_fy, serial_fy = (tag_fy or "").strip().upper(), (serial_fy or "").strip()
    if not company or not run_label:
        raise ValueError("ต้องกรอกบริษัทและชื่อ Run")

    try:
        with get_conn(db_path) as conn:
            new_id = _insert_meter_run(conn, company, run_label, tag_fe, serial_fe, tag_fy, serial_fy)
    except sqlite3.IntegrityError as e:
        if "UNIQUE" not in str(e):
            raise
        # แสดงชื่อตามที่มีในระบบ (ผู้ใช้อาจพิมพ์ตัวพิมพ์เล็ก/ใหญ่ต่างไป)
        with get_conn(db_path) as conn:
            row = conn.execute(
                "SELECT company, run_label FROM meter_runs "
                "WHERE company = ? COLLATE NOCASE AND run_label = ? COLLATE NOCASE",
                (company, run_label),
            ).fetchone()
        name = f"{row['company']} Run {row['run_label']}" if row else f"{company} Run {run_label}"
        raise DuplicateMeterRunError(name) from e
    return get_meter_run(new_id, db_path)


def list_meter_runs(db_path: Optional[str] = None) -> List[dict]:
    """รายการ Run + reading_count (จำนวนการจด — หน้าแรกใช้ปิดปุ่มลบของ Run ที่มีประวัติ)"""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT m.*, (SELECT COUNT(*) FROM readings r WHERE r.meter_run_id = m.id) AS reading_count "
            "FROM meter_runs m ORDER BY m.company, m.run_label, m.id"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_meter_run(meter_run_id: int, db_path: Optional[str] = None) -> bool:
    """ลบ Meter Run — คืน False ถ้าไม่พบ, raise MeterRunInUseError ถ้ามีประวัติการจดแล้ว

    นับ + ลบใน transaction เดียวกัน และ FK (readings -> meter_runs) กันซ้ำอีกชั้น
    """
    with get_conn(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM readings WHERE meter_run_id = ?",
                             (meter_run_id,)).fetchone()[0]
        if count:
            raise MeterRunInUseError(count)
        cur = conn.execute("DELETE FROM meter_runs WHERE id = ?", (meter_run_id,))
        return cur.rowcount > 0


def get_meter_run(meter_run_id: int, db_path: Optional[str] = None) -> Optional[dict]:
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM meter_runs WHERE id = ?", (meter_run_id,)).fetchone()
    return dict(row) if row else None


# ── readings ──────────────────────────────────────────────────────────

def _reading_from_row(row) -> dict:
    d = dict(row)
    for col in JSON_COLUMNS:
        d[col] = json.loads(d[col]) if d[col] is not None else None
    return d


def insert_reading(meter_run_id: int, recorded_at: str, raw_values: dict, final_values: dict,
                   edited_fields: List[str], low_confidence_fields: List[str],
                   turbine_trim_digits: int, image_paths: List[str],
                   engine: str, confidence_scores: Optional[dict] = None,
                   db_path: Optional[str] = None) -> dict:
    """บันทึกการอ่านค่า 1 ครั้ง — created_at ระบบใส่เอง, recorded_at มาจากผู้ใช้

    engine ต้องส่งมาเสมอ (ไม่มีค่า default): มีหลาย engine แล้ว ถ้าลืมส่งแล้วถูกบันทึกเป็นชื่อใดชื่อหนึ่ง
    แบบเงียบๆ ข้อมูลเปรียบเทียบระหว่าง engine จะผิดโดยไม่มีใครรู้ -> ไม่ส่ง = TypeError, ว่าง = ValueError
    """
    if not isinstance(engine, str) or not engine.strip():
        raise ValueError(f"ต้องระบุชื่อ engine ที่อ่านค่า (เช่น 'gemini') — ได้ {engine!r}")
    if turbine_trim_digits not in (0, 1, 2):
        raise ValueError(f"turbine_trim_digits ต้องเป็น 0, 1 หรือ 2 (ได้ {turbine_trim_digits})")
    # ทำให้รูปแบบเดียวกันเสมอ (YYYY-MM-DDTHH:MM:SS) — การเรียง ORDER BY recorded_at เป็นการเรียง text
    recorded_at = datetime.fromisoformat(recorded_at).isoformat(timespec="seconds")

    def dump(v):
        return None if v is None else json.dumps(v, ensure_ascii=False)

    with get_conn(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO readings (meter_run_id, recorded_at, created_at, raw_values, final_values, "
            "edited_fields, low_confidence_fields, turbine_trim_digits, image_paths, engine, "
            "confidence_scores) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (meter_run_id, recorded_at, _now(), dump(raw_values), dump(final_values),
             dump(edited_fields), dump(low_confidence_fields), turbine_trim_digits,
             dump(image_paths), engine, dump(confidence_scores)),
        )
        new_id = cur.lastrowid
    return get_reading(new_id, db_path)


def get_reading(reading_id: int, db_path: Optional[str] = None) -> Optional[dict]:
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM readings WHERE id = ?", (reading_id,)).fetchone()
    return _reading_from_row(row) if row else None


def list_readings(meter_run_id: Optional[int] = None, db_path: Optional[str] = None) -> List[dict]:
    """ประวัติการอ่านค่า ใหม่สุดก่อน (เรียงตามเวลาที่จด) — กรองตาม Run ได้"""
    sql = "SELECT * FROM readings"
    params = ()
    if meter_run_id is not None:
        sql += " WHERE meter_run_id = ?"
        params = (meter_run_id,)
    sql += " ORDER BY recorded_at DESC, id DESC"
    with get_conn(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_reading_from_row(r) for r in rows]


def get_previous_reading(meter_run_id: int, before_reading_id: Optional[int] = None,
                         db_path: Optional[str] = None) -> Optional[dict]:
    """ค่าจดครั้งก่อนของ Run เดียวกัน (ใช้คำนวณผลต่าง)

    before_reading_id=None  -> reading ล่าสุดของ Run (หน้า OCR: ก่อนบันทึกครั้งใหม่)
    before_reading_id=X     -> reading ที่จดก่อน X (export PDF ซ้ำจากหน้าประวัติ)

    "ก่อน" ตัดสินจาก recorded_at (เวลาที่จด) ถ้าเวลาเท่ากันใช้ id ตัดสิน
    """
    with get_conn(db_path) as conn:
        if before_reading_id is None:
            row = conn.execute(
                "SELECT * FROM readings WHERE meter_run_id = ? "
                "ORDER BY recorded_at DESC, id DESC LIMIT 1",
                (meter_run_id,),
            ).fetchone()
        else:
            ref = conn.execute(
                "SELECT recorded_at, id FROM readings WHERE id = ?", (before_reading_id,)
            ).fetchone()
            if ref is None:
                return None
            row = conn.execute(
                "SELECT * FROM readings WHERE meter_run_id = ? "
                "AND (recorded_at < ? OR (recorded_at = ? AND id < ?)) "
                "ORDER BY recorded_at DESC, id DESC LIMIT 1",
                (meter_run_id, ref["recorded_at"], ref["recorded_at"], ref["id"]),
            ).fetchone()
    return _reading_from_row(row) if row else None


if __name__ == "__main__":
    # รันตรงๆ เพื่อสร้าง DB + seed: python db.py
    init_db()
    for r in list_meter_runs():
        print(f"  #{r['id']} {r['company']} Run {r['run_label']}: "
              f"FE={r['tag_fe']} / FY={r['tag_fy']}")
