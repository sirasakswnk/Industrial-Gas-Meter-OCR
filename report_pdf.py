"""
สร้าง PDF รายงานการจดค่ามิเตอร์ 1 ครั้ง (ค่าครั้งหลัง / ครั้งก่อน / ผลต่าง)

- ฝังฟอนต์ Sarabun จาก fonts/ ของโปรเจค (ไม่พึ่งฟอนต์ที่ติดตั้งในเครื่อง -> ย้ายเครื่องแล้วไทยไม่เพี้ยน)
- เปิด text shaping (uharfbuzz) -> สระบน/วรรณยุกต์ซ้อนกันวางตำแหน่งถูก เช่น "ที่", "ครั้ง"
- สร้างจากข้อมูลใน DB ทุกครั้ง (ไม่เก็บไฟล์ PDF) -> export ซ้ำจากหน้าประวัติได้ผลตรงกันเสมอ
- จัดตามฟอร์มจริง (ใบแจ้งปริมาณการใช้ก๊าซ) แต่ไม่เลียนแบบเป๊ะ (ไม่มีกล่องทีละหลัก/ช่องเซ็น):
    ค่าสะสม: แถว จดครั้งหลัง (บน) -> จดครั้งก่อน (กลาง) -> ค่าแตกต่าง (ล่าง), ฟิลด์เป็นคอลัมน์
    CF/Qm/Qb อยู่ในบรรทัด REMARK
- ไม่มีส่วนที่ระบบเติมไม่ได้ (เกจบนท่อ / ช่องติ๊กสถานะ) — PDF สร้างจาก DB ไม่ใช่ฟอร์มให้คนกรอก
"""
import os
from typing import Optional

from fpdf import FPDF
from fpdf.fonts import FontFace

import reading_fields

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONT_REGULAR = os.path.join(FONT_DIR, "Sarabun-Regular.ttf")
FONT_BOLD = os.path.join(FONT_DIR, "Sarabun-Bold.ttf")

REPORT_TITLE = "ใบแจ้งปริมาณการใช้ก๊าซ"

HEAD_FILL = (235, 239, 245)
MUTED = (110, 116, 128)

ROW_LABELS = ("จดครั้งหลัง", "จดครั้งก่อน", "ค่าแตกต่าง")

def format_datetime(iso: Optional[str]) -> str:
    """"2026-10-07T13:30:00" -> "07/10/2026 13:30" """
    if not iso:
        return ""
    date, _, time = iso.partition("T")
    y, m, d = date.split("-")
    return f"{d}/{m}/{y} {time[:5]}".strip()


def _field(key: str) -> dict:
    return next(f for f in reading_fields.FIELDS if f["key"] == key)


def remark_line(run: dict, final: dict) -> str:
    """บรรทัด REMARK ตามรูปแบบฟอร์มจริง — ค่า null ใส่ "-" ไม่มีหน่วยต่อท้าย (แบบ "CF คำนวณ = -"),
    CF คำนวณ = "-" เสมอ (ระบบไม่คำนวณ); ไม่มี "Remark =" ต่อท้าย (ระบบไม่มีช่องกรอกหมายเหตุ)"""
    def v(key, unit=""):
        value = final.get(key)
        return f"{value} {unit}".rstrip() if value else "-"
    return (f"Run {run['run_label']}: CF EVC = {v('correction_factor')}, CF คำนวณ = -, "
            f"Qm = {v('qm_m3_h', 'm3/hr')}, Qb = {v('qb_sm3_h', 'Sm3/hr')}")


def _cumulative_table(pdf: FPDF, keys, final: dict, prev_final: dict):
    """ตารางค่าสะสม: แถว จดครั้งหลัง / จดครั้งก่อน / ค่าแตกต่าง, ฟิลด์เป็นคอลัมน์
    ไม่มีค่าครั้งก่อน -> แถวครั้งก่อนและผลต่างปล่อยว่าง"""
    label_w, value_w = 36, 62
    with pdf.table(
        col_widths=(label_w,) + (value_w,) * len(keys),
        width=label_w + value_w * len(keys),
        align="LEFT",
        text_align=("LEFT",) + ("RIGHT",) * len(keys),
        headings_style=FontFace(emphasis="BOLD", fill_color=HEAD_FILL),
        line_height=7,
    ) as table:
        head = table.row()
        head.cell("")
        for key in keys:
            f = _field(key)
            head.cell(f"{f['label']} ({f['unit']})")

        rows = {
            ROW_LABELS[0]: [final.get(k) for k in keys],
            ROW_LABELS[1]: [prev_final.get(k) for k in keys],
            ROW_LABELS[2]: [reading_fields.difference(final.get(k), prev_final.get(k)) for k in keys],
        }
        for label, values in rows.items():
            row = table.row()
            row.cell(label, style=FontFace(emphasis="BOLD"))
            for value in values:
                row.cell(value or "")


def _measured_table(pdf: FPDF, keys, final: dict):
    """ค่า ณ ขณะจด (ไม่ใช่ค่าสะสม -> ไม่คำนวณผลต่าง)"""
    with pdf.table(
        col_widths=(36, 20, 42),
        width=98,
        align="LEFT",
        text_align=("LEFT", "CENTER", "RIGHT"),
        headings_style=FontFace(emphasis="BOLD", fill_color=HEAD_FILL),
        line_height=7,
    ) as table:
        head = table.row()
        for text in ("รายการ", "หน่วย", "ค่าที่วัดได้"):
            head.cell(text)
        for key in keys:
            f = _field(key)
            row = table.row()
            row.cell(f["label"])
            row.cell(f["unit"])
            row.cell(final.get(key) or "")


def _section_title(pdf: FPDF, title: str, tag: str, serial: str):
    pdf.set_font("Sarabun", "B", 12)
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Sarabun", "", 10)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 6, f"Tag: {tag or '—'}    Serial: {serial or '—'}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)


def _remark(pdf: FPDF, run: dict, final: dict):
    pdf.set_font("Sarabun", "B", 11)
    pdf.cell(0, 7, "REMARK", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Sarabun", "", 10.5)
    pdf.multi_cell(0, 6, remark_line(run, final), new_x="LMARGIN", new_y="NEXT")


def build_reading_pdf(run: dict, reading: dict, previous: Optional[dict]) -> bytes:
    """PDF ของการจดค่า 1 ครั้ง — previous = การจดครั้งก่อนหน้าของ Run เดียวกัน (หรือ None)"""
    final = reading["final_values"] or {}
    prev_final = (previous or {}).get("final_values") or {}

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(True, margin=16)
    pdf.add_font("Sarabun", "", FONT_REGULAR)
    pdf.add_font("Sarabun", "B", FONT_BOLD)
    pdf.set_text_shaping(True)  # ต้องเปิดเอง — ไม่งั้นสระ/วรรณยุกต์ไทยลอยผิดตำแหน่ง
    pdf.set_title(f"{REPORT_TITLE} {run['company']} Run {run['run_label']}")
    pdf.add_page()

    # ── หัวรายงาน ──
    pdf.set_font("Sarabun", "B", 18)
    pdf.cell(0, 10, REPORT_TITLE, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    info = [
        ("บริษัท", run["company"]),
        ("Meter Run", run["run_label"]),
        ("วัน/เวลาที่จด", format_datetime(reading["recorded_at"])),
        ("ค่าจดครั้งก่อน", format_datetime(previous["recorded_at"]) if previous else "— (ยังไม่มีประวัติ)"),
    ]
    for label, value in info:
        pdf.set_font("Sarabun", "B", 11)
        pdf.cell(32, 7, label)
        pdf.set_font("Sarabun", "", 11)
        pdf.cell(0, 7, value, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── ตารางที่ 1: Electronic Volume Corrector ──
    _section_title(pdf, "1. Electronic Volume Corrector", run.get("tag_fy"), run.get("serial_fy"))
    pdf.set_font("Sarabun", "", 10.5)
    _cumulative_table(pdf, ["correct_v_scm", "line_v_m3"], final, prev_final)
    pdf.ln(3)
    _measured_table(pdf, ["pressure_bar", "temperature_c"], final)
    pdf.ln(6)

    # ── ตารางที่ 2: Turbine Meter ──
    _section_title(pdf, "2. Turbine Meter", run.get("tag_fe"), run.get("serial_fe"))
    pdf.set_font("Sarabun", "", 10.5)
    _cumulative_table(pdf, [reading_fields.TURBINE_FIELD], final, prev_final)
    pdf.ln(6)

    # ── REMARK (CF/Qm/Qb) ──
    _remark(pdf, run, final)
    pdf.ln(8)

    # ── ท้ายรายงาน ──
    pdf.set_font("Sarabun", "", 8.5)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(0, 5, f"รายการที่ #{reading['id']} · บันทึกในระบบเมื่อ {format_datetime(reading['created_at'])}"
                         " · ค่าจดครั้งหลังผ่านการตรวจโดยผู้จดแล้ว")

    return bytes(pdf.output())


def pdf_filename(run: dict, reading: dict) -> str:
    """ชื่อไฟล์ ASCII ล้วน (ใช้ใน Content-Disposition ได้ทุก browser)"""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_"
                   for ch in f"{run['company']}_Run{run['run_label']}")
    stamp = (reading["recorded_at"] or "")[:16].replace("-", "").replace(":", "").replace("T", "_")
    return f"meter_reading_{safe}_{stamp}.pdf"
