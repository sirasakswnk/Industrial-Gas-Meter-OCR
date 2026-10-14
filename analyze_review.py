"""
วิเคราะห์ผลตรวจด้วยตาจาก review.csv -> ตารางสรุปสำหรับรายงาน
==============================================================
อ่าน review.csv (กรอกคอลัมน์ correct แล้ว) แล้วสรุปว่าแต่ละวิธี
หากรอบหน้าปัดได้ "ถูก / บางส่วน / ผิด / ไม่เจอ" กี่ภาพ

ค่าในคอลัมน์ correct: correct = ถูก, partial = บางส่วน, wrong = ผิด, none = ไม่เจอกรอบ

รัน: python analyze_review.py
"""
import csv
import os

REVIEW_CSV = "review.csv"
SUMMARY_METHOD_CSV = "summary_by_method.csv"   # สรุปรายวิธี (ตาราง 1+2)
SUMMARY_GROUP_CSV = "summary_by_group.csv"     # สรุปแยกกลุ่มภาพ (ตาราง 3)
SUMMARY_IMAGE_CSV = "summary_by_image.csv"     # รายภาพ เทียบ 5 วิธีในแถวเดียว
METHODS = ["canny", "otsu", "otsu_inv", "bright", "obb"]
LABELS = {"canny": "Canny", "otsu": "Otsu", "otsu_inv": "Otsu กลับสี",
          "bright": "Bright", "obb": "YOLO-OBB"}
GROUPS = ["normal", "tilt", "analog", "hard"]
VERDICTS = ["correct", "partial", "wrong", "none"]


def main():
    if not os.path.exists(REVIEW_CSV):
        print(f"[!] ไม่พบ {REVIEW_CSV}")
        return

    rows = [r for r in csv.DictReader(open(REVIEW_CSV, encoding="utf-8-sig"))
            if r.get("filename")]
    for r in rows:
        r["correct"] = (r.get("correct") or "").strip().lower()

    blank = [r for r in rows if r["correct"] not in VERDICTS]
    if blank:
        print(f"[!] มี {len(blank)} แถวที่ค่า correct ไม่ถูกต้อง/ว่าง:")
        for r in blank[:10]:
            print(f"    {r['filename']} {r['method']} -> '{r['correct']}'")
        print()

    n_img = len({r["filename"] for r in rows})
    print(f"ภาพทั้งหมด {n_img} ภาพ x {len(METHODS)} วิธี = {len(rows)} แถว\n")

    # ── ตาราง 1: ภาพรวมรายวิธี ──
    print("=" * 78)
    print("ตารางที่ 1 — ผลรายวิธี (จำนวนภาพ)")
    print("=" * 78)
    print(f"{'วิธี':<14}{'ถูก':>7}{'บางส่วน':>10}{'ผิด':>7}{'ไม่เจอ':>9}"
          f"{'ถูก%':>9}{'ถูก+บางส่วน%':>15}")
    print("-" * 78)
    for m in METHODS:
        mr = [r for r in rows if r["method"] == m]
        c = {v: sum(1 for r in mr if r["correct"] == v) for v in VERDICTS}
        tot = len(mr)
        print(f"{LABELS[m]:<14}{c['correct']:>7}{c['partial']:>10}{c['wrong']:>7}"
              f"{c['none']:>9}{c['correct']/tot*100:>8.1f}%"
              f"{(c['correct']+c['partial'])/tot*100:>14.1f}%")

    # ── ตาราง 2: "เจอแล้วถูกไหม" — precision เมื่อวิธีนั้นคืนกรอบมา ──
    print("\n" + "=" * 78)
    print("ตารางที่ 2 — เมื่อวิธีนั้น 'หากรอบเจอ' แล้วกรอบถูกกี่ครั้ง (precision)")
    print("=" * 78)
    print(f"{'วิธี':<14}{'เจอกรอบ':>10}{'ถูก':>7}{'บางส่วน':>10}{'ผิด':>7}{'แม่นยำ':>10}")
    print("-" * 78)
    for m in METHODS:
        mr = [r for r in rows if r["method"] == m and r["correct"] != "none"]
        c = {v: sum(1 for r in mr if r["correct"] == v) for v in VERDICTS}
        found = len(mr)
        prec = f"{c['correct']/found*100:.1f}%" if found else "-"
        print(f"{LABELS[m]:<14}{found:>10}{c['correct']:>7}{c['partial']:>10}"
              f"{c['wrong']:>7}{prec:>10}")

    # ── ตาราง 3: แยกตามกลุ่มภาพ (นับเฉพาะ correct) ──
    gtot = {g: len({r["filename"] for r in rows if r["group"] == g}) for g in GROUPS}
    print("\n" + "=" * 78)
    print("ตารางที่ 3 — จำนวนภาพที่ได้กรอบ 'ถูก' แยกตามกลุ่ม")
    print("=" * 78)
    print(f"{'วิธี':<14}" + "".join(f"{g+f' ({gtot[g]})':>14}" for g in GROUPS))
    print("-" * 78)
    for m in METHODS:
        line = f"{LABELS[m]:<14}"
        for g in GROUPS:
            gr = [r for r in rows if r["method"] == m and r["group"] == g]
            ok = sum(1 for r in gr if r["correct"] == "correct")
            line += f"{f'{ok}/{len(gr)}':>14}"
        print(line)

    # ── ตาราง 4: classical รวมทุกวิธี vs OBB ──
    print("\n" + "=" * 78)
    print("ตารางที่ 4 — Classical CV (รวม 4 วิธี) เทียบ YOLO-OBB")
    print("=" * 78)
    imgs = sorted({r["filename"] for r in rows})
    classical_any, obb_ok = [], []
    for fn in imgs:
        fr = [r for r in rows if r["filename"] == fn]
        if any(r["correct"] == "correct" for r in fr if r["method"] != "obb"):
            classical_any.append(fn)
        if any(r["correct"] == "correct" for r in fr if r["method"] == "obb"):
            obb_ok.append(fn)
    print(f"ภาพที่มี classical อย่างน้อย 1 วิธีได้กรอบถูก : "
          f"{len(classical_any)}/{n_img}  ({', '.join(classical_any) if classical_any else '-'})")
    print(f"ภาพที่ YOLO-OBB ได้กรอบถูก                  : {len(obb_ok)}/{n_img}")

    only_obb = [f for f in imgs if f not in classical_any]
    print(f"\nภาพที่ไม่มี classical วิธีไหนทำได้เลย ({len(only_obb)} ภาพ):")
    for fn in only_obb:
        o = next(r["correct"] for r in rows if r["filename"] == fn and r["method"] == "obb")
        print(f"    {fn:<16} OBB -> {o}")

    # ── รายภาพที่ OBB ไม่ได้ correct เต็ม ──
    imperfect = [r for r in rows if r["method"] == "obb" and r["correct"] != "correct"]
    if imperfect:
        print(f"\nภาพที่ OBB ไม่ได้ 'ถูกเต็ม' ({len(imperfect)} ภาพ):")
        for r in imperfect:
            print(f"    {r['filename']:<16} [{r['group']}] -> {r['correct']}")

    export_csv(rows, n_img, gtot)


def export_csv(rows, n_img, gtot):
    """export ตารางสรุปเป็น CSV สำหรับเอาไปใช้ต่อ/ส่งให้คนอื่น"""

    # 1) สรุปรายวิธี (ตารางที่ 1 + 2 รวมกัน)
    with open(SUMMARY_METHOD_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method", "total_images", "correct", "partial", "wrong", "none",
                    "correct_pct", "correct_or_partial_pct",
                    "found", "precision_when_found_pct"])
        for m in METHODS:
            mr = [r for r in rows if r["method"] == m]
            c = {v: sum(1 for r in mr if r["correct"] == v) for v in VERDICTS}
            tot = len(mr)
            found = tot - c["none"]
            w.writerow([
                m, tot, c["correct"], c["partial"], c["wrong"], c["none"],
                round(c["correct"] / tot * 100, 1),
                round((c["correct"] + c["partial"]) / tot * 100, 1),
                found,
                round(c["correct"] / found * 100, 1) if found else "",
            ])

    # 2) สรุปแยกกลุ่ม (ตารางที่ 3) — long format ไว้ pivot ต่อได้
    with open(SUMMARY_GROUP_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method", "group", "total_images",
                    "correct", "partial", "wrong", "none", "correct_pct"])
        for m in METHODS:
            for g in GROUPS:
                gr = [r for r in rows if r["method"] == m and r["group"] == g]
                if not gr:
                    continue
                c = {v: sum(1 for r in gr if r["correct"] == v) for v in VERDICTS}
                w.writerow([m, g, len(gr), c["correct"], c["partial"], c["wrong"],
                            c["none"], round(c["correct"] / len(gr) * 100, 1)])

    # 3) สรุปรายภาพ — 1 แถวต่อ 1 ภาพ เทียบ 5 วิธีในแถวเดียว
    with open(SUMMARY_IMAGE_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["filename", "group"] + METHODS + ["classical_any_correct"])
        for fn in sorted({r["filename"] for r in rows}):
            fr = {r["method"]: r["correct"] for r in rows if r["filename"] == fn}
            grp = next(r["group"] for r in rows if r["filename"] == fn)
            any_ok = any(fr.get(m) == "correct" for m in METHODS if m != "obb")
            w.writerow([fn, grp] + [fr.get(m, "") for m in METHODS] + [any_ok])

    print(f"\nexport CSV แล้ว:")
    for p in (SUMMARY_METHOD_CSV, SUMMARY_GROUP_CSV, SUMMARY_IMAGE_CSV):
        print(f"    {p}")


if __name__ == "__main__":
    main()
