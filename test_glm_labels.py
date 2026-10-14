"""
เทสต์ชั้นแปลงป้ายเป็นฟิลด์ (ocr_engines/glm_labels.py) — ไม่เรียกโมเดล ไม่ต้องเปิด Ollama

ครอบคลุม:
  1. ตารางป้ายขั้นต่ำตามสเปค -> ฟิลด์ถูก (ไม่สนตัวพิมพ์, มี ":" / ** / หน่วยในวงเล็บ)
  2. Zb/Z ถูกทิ้ง ไม่กลายเป็น correction_factor (spike พบโมเดลสับสน)
  3. ป้ายที่ไม่รู้จัก -> ทิ้ง ไม่เดา (รวม template ที่โมเดลลอกจาก prompt)
  4. ค่าแปลงเป็นตัวเลขไม่ได้ -> None ไม่ throw, ค่าเป็นสตริงตามที่เขียน (คงศูนย์ท้าย)
  5. ข้อความจริงของ bin/test5.jpg (หน้าค่าที่วัดได้ ไม่มียอดสะสม) -> correct_v/line_v = null อย่างถูกต้อง
  6. ฟิลด์ซ้ำ: ค่าเหมือน -> ใช้, ค่าต่าง -> null
  7. กฎอุปกรณ์: ภาพ analog ให้ได้แค่ turbine, ภาพจอ EVC ให้ turbine ไม่ได้
  8. เลย์เอาต์ ข. "ป้าย:" แล้วค่าบรรทัดถัดไป (ป้ายต้องรู้จัก, บรรทัดถัดไปเป็นป้ายอื่น -> ไม่มีค่า)
  9. เลย์เอาต์ ค. "ป้าย ค่า หน่วย" ไม่มีตัวคั่น (ป้ายต้องรู้จัก, ประโยคยาวไม่นับเป็นป้าย)
  10. เลย์เอาต์ ง. ตัวเลขเดี่ยวในภาพ analog -> turbine; หลายชุด -> null; ภาพจอ EVC ไม่รับตัวเลขเดี่ยว
  11. ป้ายใหม่ (CV Idx / UV Idx / alias สะกดผิด) + ห้าม fuzzy: Converted กับ Unconverted ต้องไม่สลับกัน

รัน: python test_glm_labels.py
"""
import sys

from ocr_engines import glm_labels as gl

# ข้อความจริงจาก glm-ocr:latest ภาพ bin/test5.jpg (ตัดช่วงวนซ้ำท้ายออกบางส่วน)
TEST5_TEXT = ("P = 9.656 bar\nT = 25.00 °C\nZb/Z = 1.0171\nC = 9.3648\nQm = 477.0 m³/h\n"
              "Qb = 4467.4 Sm³/h\nBat = 1500 days\n```markdown\n\n```\nINDEX VALUE ALARM DATA CONFIG\n"
              "P = 9.656 bar\nT = 25.00 °C\nZb/Z = 1.0171\nC = 9.3648\nQm = 477.0 m³/h\n"
              "Qb = 4467.4 Sm³/h\nBat = 1500 days\n```\n```\n")


def values_of(text, allowed=None):
    return {f: fv.value for f, fv in gl.collapse_fields(gl.parse_labeled_text(text, allowed)).items()}


def test_label_table():
    expected = {
        "P": "pressure_bar", "Pressure": "pressure_bar",
        "T": "temperature_c", "Temp": "temperature_c", "Temperature": "temperature_c",
        "C": "correction_factor", "Factor": "correction_factor", "Correction": "correction_factor",
        "Qm": "qm_m3_h", "Qb": "qb_sm3_h",
        "Converted Index": "correct_v_scm", "Vb": "correct_v_scm",
        "Unconverted Index": "line_v_m3", "Vm": "line_v_m3",
    }
    for label, field in expected.items():
        for variant in (label, label.upper(), label.lower(), f"{label}:", f"**{label}**", f"  {label} "):
            assert gl.field_for_label(variant) == field, (variant, gl.field_for_label(variant), field)
    assert gl.field_for_label("Pressure (bar)") == "pressure_bar"
    assert gl.field_for_label("Converted   index") == "correct_v_scm"
    print("✅ 1. ตารางป้ายขั้นต่ำครบ 14 ป้าย (ไม่สนตัวพิมพ์ / : / ** / หน่วยในวงเล็บ)")


def test_zb_z_discarded():
    for label in ("Zb/Z", "Zb / Z", "ZB/Z", "Z", "Zb"):
        assert gl.field_for_label(label) is None, label
    parsed = gl.parse_labeled_text("Zb/Z = 1.0171\nC = 9.3648\n")
    assert [v.field for v in parsed.values] == ["correction_factor"]
    assert parsed.discarded == ["Zb/Z"]
    assert values_of("Zb/Z = 1.0171\n") == {}, "Zb/Z อย่างเดียวต้องไม่ได้ correction_factor"
    print("✅ 2. Zb/Z ถูกทิ้ง ไม่กลายเป็น correction_factor")


def test_unknown_discarded():
    parsed = gl.parse_labeled_text("Bat = 1500 days\nK = 0.9291\nLABEL = VALUE UNIT\nQm = 477.0 m3/h\n")
    assert [v.field for v in parsed.values] == ["qm_m3_h"]
    assert parsed.unknown == ["Bat", "K", "LABEL"], parsed.unknown
    # ป้ายใกล้เคียงที่สะกดผิดแต่ไม่ได้เพิ่มเป็น alias -> ไม่เดา
    assert gl.field_for_label("Converled Index") is None
    # บรรทัดที่ไม่มีป้าย/ตัวคั่น ไม่นับเป็นป้าย
    assert gl.parse_labeled_text("INDEX VALUE ALARM DATA CONFIG\n```markdown\n67203344\n").label_count == 0
    print("✅ 3. ป้ายที่ไม่รู้จัก (Bat, K, template ที่ลอกจาก prompt, คำสะกดผิด) ถูกทิ้ง ไม่เดา")


def test_values_strings_and_bad_numbers():
    cases = {
        "T = 25.00 °C": "25.00",                 # คงศูนย์ท้าย
        "Vm = 00529778.000 m3": "529778.000",    # ตัดศูนย์นำหน้า (เหมือน to_plain_str ของ Gemini)
        "Qm = 477.0m³/h": "477.0",               # หน่วยติดตัวเลข
        "T: -3.50 °C": "-3.50",
        "P = 9,656 bar": None,                   # จุลภาคกำกวม -> ไม่เดา
        "P = - bar": None,
        "P = abc": None,
        "P = 1.2.3 bar": None,
        "Counter = DIGITS": None,               # template ที่โมเดลลอกจาก prompt
    }
    for line, expected in cases.items():
        parsed = gl.parse_labeled_text(line)
        assert len(parsed.values) == 1, (line, parsed)
        assert parsed.values[0].value == expected, (line, parsed.values[0].value, expected)
    # ป้ายที่รู้จักแต่ไม่มีค่าเลย -> ไม่ใช่ค่า None (ไม่ไปทำให้ค่าที่ถูกจากบรรทัดอื่นกลายเป็น null)
    parsed = gl.parse_labeled_text("P =\n")
    assert parsed.values == [] and parsed.no_value == ["P"], parsed
    print("✅ 4. ค่าเป็นสตริงตามที่เขียน (คงศูนย์ท้าย), แปลงไม่ได้ -> None ไม่ throw, ป้ายไม่มีค่า -> ไม่นับ")


def test_test5_real_text():
    got = values_of(TEST5_TEXT, gl.EVC_FIELDS)
    assert got == {"pressure_bar": "9.656", "temperature_c": "25.00", "correction_factor": "9.3648",
                   "qm_m3_h": "477.0", "qb_sm3_h": "4467.4"}, got
    assert "correct_v_scm" not in got and "line_v_m3" not in got, \
        "หน้าค่าที่วัดได้ไม่มียอดสะสม -> correct_v_scm/line_v_m3 ต้องไม่มีค่า"
    # span ชี้ที่ "ค่า" เท่านั้น (ไม่รวมป้าย/หน่วย) — ใช้หา confidence
    p = next(v for v in gl.parse_labeled_text(TEST5_TEXT).values if v.field == "pressure_bar")
    assert TEST5_TEXT[p.span[0]:p.span[1]] == "9.656"
    print("✅ 5. ข้อความจริง test5: 5 ฟิลด์ตรง spike, correct_v/line_v = null อย่างถูกต้อง, span = เฉพาะค่า")


def test_duplicates():
    fv = gl.collapse_fields(gl.parse_labeled_text("P = 9.656 bar\nP = 9.656 bar\n"))["pressure_bar"]
    assert fv.value == "9.656" and len(fv.spans) == 2 and not fv.conflict
    fv = gl.collapse_fields(gl.parse_labeled_text("P = 9.656 bar\nP = 9.665 bar\n"))["pressure_bar"]
    assert fv.value is None and fv.conflict, "ค่าต่างกันต้องเป็น null ไม่เลือกให้"
    fv = gl.collapse_fields(gl.parse_labeled_text("P = 9.656 bar\nP = 9,656 bar\n"))["pressure_bar"]
    assert fv.value is None and fv.conflict
    print("✅ 6. ฟิลด์ซ้ำ: ค่าเหมือน -> ใช้, ค่าต่าง/แปลงไม่ได้ปน -> null")


def test_device_rule():
    parsed = gl.parse_labeled_text("Counter = 01234567\nVm = 5\n", gl.ANALOG_FIELDS)
    assert [v.field for v in parsed.values] == ["turbine_v_m3"] and parsed.not_allowed == ["Vm"]
    parsed = gl.parse_labeled_text("Counter = 01234567\nVm = 5\n", gl.EVC_FIELDS)
    assert [v.field for v in parsed.values] == ["line_v_m3"] and parsed.not_allowed == ["Counter"]
    print("✅ 7. กฎอุปกรณ์: analog -> turbine เท่านั้น, จอ EVC -> ห้าม turbine")


# ข้อความจริงจาก glm-ocr (รอบทดลอง 66 รูป)
CORUS_INDEX_TEXT = ("INDEX VALUE ALERT DATA CONFIG\n\nUnconverted Index:\n000273155.00m³\nConverted Index:\n"
                    "149581472.698m³\n```markdown\n\nINDEX VALUE ALERT DATA CONFIG\n\nUnconverted Index:\n"
                    "000273155.00m³\nConverted Index:\n149581472.698m³\n```\n")       # eval_images/S01_p1.jpg
MACBAT_TEXT = ("MacBAT 5 30 16\nVkT 47.17 m³\nVm 000299666.40 m³\np1 1.0026 bar\nt 23.97 °C\n"
               "k1 1.000819\nC 0.906803")                                          # experiment_images/hard_1.jpeg
EK280_TEXT = ("Main LUET MUMM SERVE 09742\nUb 23629198.559 n3\nVn 00927753.600 n3\n"
              "P 25.609 bar K 0.95791\nT 31.40 °C C 24.95448\ne\nEK280\nYear:\nSerial")  # testpic/img1.jpeg
ANALOG_TEXT = "16897967m³\n\n\n16897967m³\n\n16897967m^{3}\n16897967m³\n"           # eval_images/S13_p1.jpg


def test_layout_next_line():
    parsed = gl.parse_labeled_text(CORUS_INDEX_TEXT, gl.EVC_FIELDS)
    got = {f: fv.value for f, fv in gl.collapse_fields(parsed).items()}
    assert got == {"line_v_m3": "273155.00", "correct_v_scm": "149581472.698"}, got
    assert all(v.layout == "next_line" for v in parsed.values)
    lv = parsed.values[0]
    assert CORUS_INDEX_TEXT[lv.span[0]:lv.span[1]] == "000273155.00", "span ต้องชี้ที่ค่าในบรรทัดถัดไป"
    # บรรทัดถัดไปเป็นป้ายอื่น -> ป้ายแรกไม่มีค่า (ไม่ข้ามไปเอาค่าของป้ายที่สอง)
    parsed = gl.parse_labeled_text("Converted Index:\nUnconverted Index:\n000273155.00m³\n")
    assert [(v.field, v.value) for v in parsed.values] == [("line_v_m3", "273155.00")], parsed.values
    assert parsed.no_value == ["Converted Index"]
    # ป้ายไม่รู้จัก -> ไม่จับคู่กับตัวเลขบรรทัดถัดไป แม้อยู่ติดกัน
    parsed = gl.parse_labeled_text("Year:\n2016\nSerial No:\n775900\n")
    assert parsed.values == [] and parsed.unknown == ["Year", "Serial No"], parsed
    # บรรทัดถัดไปไม่ใช่ตัวเลข / ไม่มีบรรทัดถัดไป / "=" ว่าง -> ไม่มีค่า (ไม่มองข้ามบรรทัด)
    parsed = gl.parse_labeled_text("Converted Index:\nINDEX VALUE\nP =\n9.656\nQm:")
    assert parsed.values == [] and parsed.no_value == ["Converted Index", "P", "Qm"], parsed
    # ค่าบรรทัดถัดไปแปลงไม่ได้ -> None
    parsed = gl.parse_labeled_text("Converted Index:\n1,234.5 m3\n")
    assert [(v.field, v.value) for v in parsed.values] == [("correct_v_scm", None)]
    print("✅ 8. ข. ป้าย: + ค่าบรรทัดถัดไป (S01_p1 จริง), ป้ายถัดไป/ป้ายไม่รู้จัก/ไม่ใช่ตัวเลข -> ไม่จับคู่")


def test_layout_space():
    parsed = gl.parse_labeled_text(MACBAT_TEXT, gl.EVC_FIELDS)
    got = {f: fv.value for f, fv in gl.collapse_fields(parsed).items()}
    assert got == {"line_v_m3": "299666.40", "temperature_c": "23.97", "correction_factor": "0.906803"}, got
    assert parsed.unknown == ["MacBAT", "VkT", "p1", "k1"], parsed.unknown
    v = next(x for x in parsed.values if x.field == "line_v_m3")
    assert MACBAT_TEXT[v.span[0]:v.span[1]] == "000299666.40" and v.layout == "space"

    parsed = gl.parse_labeled_text(EK280_TEXT, gl.EVC_FIELDS)
    got = {f: fv.value for f, fv in gl.collapse_fields(parsed).items()}
    # "Ub"/"Vn" คือ Vb/Vm ที่ OCR อ่านผิด -> ไม่รู้จัก ทิ้ง (ไม่เดา); P/T จับคู่ได้เพราะป้ายรู้จัก
    assert got == {"pressure_bar": "25.609", "temperature_c": "31.40"}, got
    assert {"Ub", "Vn"} <= set(parsed.unknown), parsed.unknown
    # ประโยคยาวที่มีตัวเลข ไม่ใช่ป้าย -> ไม่นับเลย
    parsed = gl.parse_labeled_text("The image shows a digital meter display with the numbers 1, 6, 8\n")
    assert parsed.values == [] and parsed.label_count == 0
    print("✅ 9. ค. ป้าย ค่า หน่วย (MacBAT / EK280 จริง), ป้ายไม่รู้จัก (Ub, VkT, p1) ทิ้ง, ประโยคยาวไม่นับ")


def test_layout_bare_analog():
    turbine = "turbine_v_m3"
    parsed = gl.parse_labeled_text(ANALOG_TEXT, gl.ANALOG_FIELDS, bare_number_field=turbine)
    fv = gl.collapse_fields(parsed)[turbine]
    assert fv.value == "16897967" and not fv.conflict and len(fv.spans) == 4, fv
    assert parsed.label_count == 0, "ตัวเลขเดี่ยวไม่นับเป็นป้าย"
    # ตัวเลขเดี่ยวหลายชุด -> กำกวม -> null (ไม่เดาว่าชุดไหนคือค่าจริง)
    fv = gl.collapse_fields(gl.parse_labeled_text("6876729\n\n1689796\n", gl.ANALOG_FIELDS,
                                                  bare_number_field=turbine))[turbine]
    assert fv.value is None and fv.conflict
    # ภาพจอ EVC: ไม่รับตัวเลขเดี่ยว (จับคู่ด้วยตำแหน่งอย่างเดียวไม่ได้)
    assert gl.parse_labeled_text("67203344\n", gl.EVC_FIELDS).values == []
    # หน่วยอื่น / ประโยค -> ไม่ใช่ตัวเลขเดี่ยว
    assert gl.parse_labeled_text("16897967 bar\nvolume 16897967 m3\n", gl.ANALOG_FIELDS,
                                 bare_number_field=turbine).values == []
    print("✅ 10. ง. analog ตัวเลขเดี่ยว -> turbine (S13 จริง), หลายชุด -> null, จอ EVC ไม่รับตัวเลขเดี่ยว")


def test_new_labels_no_fuzzy():
    assert gl.field_for_label("CV Idx") == "correct_v_scm" and gl.field_for_label("UV Idx") == "line_v_m3"
    assert gl.field_for_label("Convened Index") == "correct_v_scm"
    assert gl.field_for_label("Unconvened Index") == "line_v_m3", "alias ของ Unconverted ต้องไม่สลับเป็น Converted"
    # ไม่มี fuzzy: สะกดใกล้เคียงแต่ไม่อยู่ในตาราง -> ไม่รู้จัก
    for near in ("Convered Index", "Converted Indx", "Conerted Index", "Unconverted Indexx", "CV Id", "UV Index",
                 "Unonverted Index", "nconverted Index"):
        assert gl.field_for_label(near) is None, near
    # ตามจอจริงของ bin/ItronPage3.jpeg: "UV Idx: 00001102.000m³" / "CV Idx: 07546132.688Sm³"
    got = values_of("UV Idx: 00001102.000m³\nCV Idx: 07546132.688Sm³\n", gl.EVC_FIELDS)
    assert got == {"line_v_m3": "1102.000", "correct_v_scm": "7546132.688"}, got
    print("✅ 11. CV Idx / UV Idx / alias สะกดผิดทีละตัว, ห้าม fuzzy (Converted/Unconverted ไม่สลับ)")


def main():
    tests = [test_label_table, test_zb_z_discarded, test_unknown_discarded,
             test_values_strings_and_bad_numbers, test_test5_real_text, test_duplicates, test_device_rule,
             test_layout_next_line, test_layout_space, test_layout_bare_analog, test_new_labels_no_fuzzy]
    failed = 0
    for fn in tests:
        try:
            fn()
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
