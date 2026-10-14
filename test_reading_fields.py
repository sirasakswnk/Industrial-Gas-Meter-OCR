"""
เทสต์ logic จัดรูปแบบค่า + ตัดทศนิยม Turbine (reading_fields.py) — ไม่ต้องโหลดโมเดล

ครอบคลุม:
  1. ตัวอย่างในสเปค: "12345.67" ตัด 2 -> "12345", "1234567" ตัด 2 -> "12345"
  2. กับดัก float: 1123120.0 ตัด 1 -> "112312" (ไม่ใช่ "1123120" แบบ str())
  3. ตัด 0/1/2 หลัก + ตัดจนไม่เหลือ -> None
  4. format_fixed: เติม 0 / ตัดทศนิยมเกิน (ไม่ปัด) + flag / คงค่าดิบสตริง "4.500"
  5. to_plain_str: exponent, float จำนวนเต็ม, ค่าที่ไม่ใช่ตัวเลข
  6. initial_editor_values: รวมทุกฟิลด์ + truncated_fields + turbine_raw_decimals
  7. compute_final_values: normalize + ตัด Turbine + ปฏิเสธรูปแบบผิด
  8. find_edited_fields: เทียบหลัง normalize, ติ๊กตัด Turbine ไม่นับว่าแก้
  9. difference: Decimal, ไม่มีค่าใดค่าหนึ่ง -> None

รัน: python test_reading_fields.py
"""
import sys

from reading_fields import (FIELD_DECIMALS, compute_final_values, decreased_fields, difference,
                            find_edited_fields,
                            format_fixed, initial_editor_values, to_plain_str, trim_turbine,
                            turbine_digits)


def trim(value, n):
    digits, _ = turbine_digits(value)
    return trim_turbine(digits, n)


def test_spec_examples():
    assert trim("12345.67", 2) == "12345"
    assert trim("1234567", 2) == "12345"
    assert trim(12345.67, 2) == "12345", "float ที่มีทศนิยมต้องได้ผลเดียวกับสตริง"
    print("✅ test_spec_examples: 12345.67 / 1234567 ตัด 2 -> 12345 ทั้งคู่")


def test_float_dot_zero_trap():
    # str(1123120.0) = "1123120.0" -> ลบจุด "11231200" -> ตัด 1 -> "1123120" (ผิด)
    assert turbine_digits(1123120.0) == ("1123120", 0)
    assert trim(1123120.0, 1) == "112312"
    assert trim(1123120.0, 2) == "11231"
    assert trim("1123120", 1) == "112312"  # จากข้อความดิบ (parse_int=str)
    assert format_fixed(1123120.0, 0) == ("1123120", False)
    assert to_plain_str(1123120.0) == "1123120"
    print("✅ test_float_dot_zero_trap: 1123120.0 ตัด 1 -> 112312 (ไม่เติม .0)")


def test_trim_counts():
    assert trim_turbine("1234567", 0) == "1234567"
    assert trim_turbine("1234567", 1) == "123456"
    assert trim_turbine("0012", 1) == "1", "0 นำหน้าต้องถูกตัด"
    assert trim_turbine("12", 2) is None, "ตัดจนไม่เหลือหลัก -> None"
    assert trim_turbine("5", 1) is None
    assert trim_turbine(None, 1) is None
    assert trim_turbine("12a4", 1) is None
    try:
        trim_turbine("123", 3)
    except ValueError:
        pass
    else:
        raise AssertionError("trim=3 ต้อง ValueError")
    print("✅ test_trim_counts: 0/1/2 หลัก + กรณีขอบ")


def test_format_fixed():
    assert format_fixed("4.5", 3) == ("4.500", False), "ทศนิยมขาดต้องเติม 0"
    assert format_fixed("4.500", 3) == ("4.500", False)
    assert format_fixed(4.5, 3) == ("4.500", False)
    assert format_fixed("4.51279", 3) == ("4.512", True), "ทศนิยมเกินต้องตัดทิ้ง ไม่ปัด + flag"
    assert format_fixed("0.99876", 4) == ("0.9987", True)
    assert format_fixed("4.5000", 3) == ("4.500", False), "ศูนย์ท้ายที่เกินไม่นับว่าเสียข้อมูล"
    assert format_fixed("123456.7", 0) == ("123456", True)
    assert format_fixed("25", 2) == ("25.00", False)
    assert format_fixed("-3.456", 2) == ("-3.45", True)
    assert format_fixed("-0.001", 2) == ("0.00", True), "ไม่มี -0.00"
    assert format_fixed("007.5", 1) == ("7.5", False)
    assert format_fixed(None, 2) == (None, False)
    print("✅ test_format_fixed: เติม 0 / ตัดไม่ปัด / flag / ค่าลบ")


def test_to_plain_str():
    assert to_plain_str("4.500") == "4.500", "สตริงดิบต้องคงศูนย์ท้าย"
    assert to_plain_str("1.2e5") == "120000"
    assert to_plain_str(12345.67) == "12345.67"
    assert to_plain_str(0.1) == "0.1", "ไม่มี artifact 0.1000000000000000055"
    assert to_plain_str(42) == "42"
    assert to_plain_str(1e16) == "10000000000000000", "ไม่มี exponent"
    for bad in (None, True, "abc", "", float("nan"), "inf"):
        assert to_plain_str(bad) is None, bad
    print("✅ test_to_plain_str: exponent / float / ค่าที่ไม่ใช่ตัวเลข")


def test_initial_editor_values():
    raw = {"correct_v_scm": "123456", "pressure_bar": "4.5", "temperature_c": "25.678",
           "correction_factor": None, "turbine_v_m3": "12345.67"}
    out = initial_editor_values(raw)
    v = out["values"]
    assert set(v) == set(FIELD_DECIMALS), "ต้องมีครบทุกฟิลด์"
    assert v["correct_v_scm"] == "123456" and v["pressure_bar"] == "4.500"
    assert v["temperature_c"] == "25.67", "ตัดไม่ปัด"
    assert v["correction_factor"] is None and v["line_v_m3"] is None
    assert v["turbine_v_m3"] == "1234567", "Turbine = สตริงตัวเลขล้วน ยังไม่ตัด"
    assert out["truncated_fields"] == ["temperature_c"], out["truncated_fields"]
    assert out["turbine_raw_decimals"] == 2
    print("✅ test_initial_editor_values: ค่าเริ่มต้นช่องทีละหลักครบ")


def expect_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError(f"{fn.__name__}{args} ควร ValueError")


def test_compute_final_values():
    entered = {"correct_v_scm": "001234567", "pressure_bar": "4.500", "temperature_c": "25.67",
               "turbine_v_m3": "1234567", "qb_sm3_h": "", "line_v_m3": None}
    final = compute_final_values(entered, 2)
    assert final["correct_v_scm"] == "1234567", "ตัด 0 นำหน้า"
    assert final["pressure_bar"] == "4.500", "คงทศนิยมครบ"
    assert final["turbine_v_m3"] == "12345", "ตัด 2 หลักบนสตริง"
    assert final["qb_sm3_h"] is None and final["line_v_m3"] is None and final["qm_m3_h"] is None
    assert compute_final_values({"turbine_v_m3": "1123120"}, 1)["turbine_v_m3"] == "112312"
    expect_value_error(compute_final_values, {"pressure_bar": "4.5"}, 0)      # ทศนิยมไม่ครบ
    expect_value_error(compute_final_values, {"pressure_bar": "4.5001"}, 0)   # ทศนิยมเกิน
    expect_value_error(compute_final_values, {"correct_v_scm": "12.0"}, 0)    # ฟิลด์ 0 หลักห้ามมีจุด
    expect_value_error(compute_final_values, {"turbine_v_m3": "123.45"}, 0)   # Turbine ต้องตัวเลขล้วน
    expect_value_error(compute_final_values, {"turbine_v_m3": "12"}, 2)       # ตัดจนไม่เหลือ
    expect_value_error(compute_final_values, {"temperature_c": "abc"}, 0)
    print("✅ test_compute_final_values: normalize + ตัด Turbine + ปฏิเสธรูปแบบผิด")


def test_find_edited_fields():
    raw = {"pressure_bar": "4.5", "temperature_c": "25.678", "turbine_v_m3": "12345.67",
           "correct_v_scm": "1234567"}
    same = {"pressure_bar": "4.500", "temperature_c": "25.67", "turbine_v_m3": "1234567",
            "correct_v_scm": "1234567"}
    assert find_edited_fields(same, raw) == [], "ค่าเดียวกันหลัง normalize ต้องไม่นับว่าแก้"
    edited = dict(same, pressure_bar="4.505", qm_m3_h="120.5")
    assert find_edited_fields(edited, raw) == ["pressure_bar", "qm_m3_h"]
    cleared = dict(same, correct_v_scm=None)
    assert find_edited_fields(cleared, raw) == ["correct_v_scm"], "ลบค่าทิ้งนับว่าแก้"
    print("✅ test_find_edited_fields: เทียบหลัง normalize")


def test_difference():
    assert difference("1234567", "1234000") == "567"
    assert difference("100", "250") == "-150"
    assert difference("12345", None) is None and difference(None, "1") is None
    assert difference("99999999999999999", "1") == "99999999999999998", "ไม่ผ่าน float"
    print("✅ test_difference: Decimal / ค่าลด / ไม่มีค่า")


def test_decreased_fields():
    prev = {"correct_v_scm": "47715600", "line_v_m3": "43084", "turbine_v_m3": "12345",
            "pressure_bar": "4.500"}

    # ติดลบ -> เตือนพร้อมชื่อฟิลด์ (ข้อความตามสเปค) + ค่าที่ใช้เทียบ
    got = decreased_fields({"correct_v_scm": "40497286", "line_v_m3": "2086449",
                            "turbine_v_m3": "12344"}, prev)
    assert [d["key"] for d in got] == ["correct_v_scm", "turbine_v_m3"], got
    assert got[0]["message"] == ("CORRECT V. มีค่าน้อยกว่าครั้งก่อน "
                                 "ตรวจสอบว่าเป็นมิเตอร์ตัวเดียวกันหรืออ่านค่าถูกต้องหรือไม่"), got[0]
    assert (got[0]["current"], got[0]["previous"], got[0]["difference"]) == ("40497286", "47715600", "-7218314")
    assert got[1]["difference"] == "-1"

    # ศูนย์ (ไม่ได้ใช้ก๊าซ) -> ไม่เตือน
    same = {"correct_v_scm": "47715600", "line_v_m3": "43084", "turbine_v_m3": "12345"}
    assert decreased_fields(same, prev) == [], "ผลต่างเป็นศูนย์ต้องไม่เตือน"

    # ไม่มีค่าครั้งก่อน (Run ใหม่ / ครั้งก่อนไม่มีฟิลด์นั้น) -> ไม่เตือน
    assert decreased_fields({"correct_v_scm": "1"}, None) == []
    assert decreased_fields({"correct_v_scm": "1"}, {}) == []
    assert decreased_fields({"correct_v_scm": "1"}, {"correct_v_scm": None}) == []

    # ครั้งนี้ไม่มีค่า -> ไม่เตือน, ค่าไม่สะสม (pressure ลดลง) -> ไม่เตือน
    assert decreased_fields({"correct_v_scm": None, "pressure_bar": "1.000"}, prev) == []

    # เทียบด้วย Decimal ไม่ใช่สตริง ("9" > "10" ถ้าเทียบสตริง)
    assert decreased_fields({"correct_v_scm": "10"}, {"correct_v_scm": "9"}) == []
    assert [d["key"] for d in decreased_fields({"correct_v_scm": "9"}, {"correct_v_scm": "10"})] == ["correct_v_scm"]
    print("✅ test_decreased_fields: ติดลบเตือน+ชื่อฟิลด์ / ศูนย์ไม่เตือน / ไม่มีครั้งก่อนไม่เตือน")


def main():
    tests = [test_spec_examples, test_float_dot_zero_trap, test_trim_counts,
             test_format_fixed, test_to_plain_str, test_initial_editor_values,
             test_compute_final_values, test_find_edited_fields, test_difference,
             test_decreased_fields]
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
