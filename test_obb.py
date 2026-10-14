"""
Unit test สำหรับตรรกะ OBB deskew ใน server.py — รันได้โดยไม่ต้องมีโมเดล OBB / ไม่ยิง Gemini

ครอบคลุม:
  1. order_corners()     — เรียงมุมถูกต้องทุกองศา (โดยเฉพาะ 45°/135° ที่วิธี sum/diff เดิมพัง)
  2. detect_meter_obb()  — fail-safe: ภาพที่ไม่มีมิเตอร์ต้องคืน (None, None, 0.0) ไม่ throw
  3. expand_corners()    — เผื่อขอบรอบ centroid ถูกสัดส่วน ไม่ทำลำดับมุมพัง

รัน: python test_obb.py
หมายเหตุ: import server จะโหลด best_obb.pt จริงครั้งเดียว — ต้องมีไฟล์ + ultralytics
"""
import sys
import numpy as np

import server  # import ฟังก์ชันจริง เพื่อให้จับ regression ในโค้ดที่ deploy ได้ตรงๆ


# ── helpers ───────────────────────────────────────────────────────────
def make_rect(w: float, h: float, angle_deg: float, center=(500.0, 400.0)) -> np.ndarray:
    """สร้างสี่เหลี่ยม w×h หมุน angle_deg รอบ center (แกนภาพ y ชี้ลง) คืน 4 มุมเรียง TL,TR,BR,BL"""
    cx, cy = center
    base = np.array([[-w / 2, -h / 2], [w / 2, -h / 2],
                     [w / 2, h / 2], [-w / 2, h / 2]], dtype=np.float32)
    t = np.deg2rad(angle_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], dtype=np.float32)
    return (base @ R.T) + np.array([cx, cy], dtype=np.float32)


def dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def has_duplicate_point(rect: np.ndarray, tol: float = 1e-3) -> bool:
    for i in range(4):
        for j in range(i + 1, 4):
            if dist(rect[i], rect[j]) < tol:
                return True
    return False


# ── test 1: order_corners ทนทุกองศา ───────────────────────────────────
def test_order_corners_all_angles():
    W, H = 100.0, 60.0
    diag = np.hypot(W, H)  # เส้นทแยง — ห้ามถูกใช้เป็น "ด้าน" ของ quad ที่เรียงถูก
    rng = np.random.default_rng(0)

    for deg in [0, 10, 15, 30, 44, 45, 46, 60, 89, 90, 135, 179]:
        truth = make_rect(W, H, deg)
        shuffled = truth[rng.permutation(4)]        # จำลอง OBB ที่ส่งมุมมาไม่เรียง
        rect = server.order_corners(shuffled)

        assert rect.shape == (4, 2), f"[{deg}°] shape ผิด: {rect.shape}"
        # (ก) ไม่มีมุมซ้ำ — นี่คือ bug ของวิธีเดิมที่ 45°/135°
        assert not has_duplicate_point(rect), f"[{deg}°] มีมุมซ้ำ (order พัง)"
        # (ข) จุดที่คืนต้องเป็น permutation ของมุมจริง (ไม่ได้สร้างจุดใหม่)
        for p in truth:
            assert any(dist(p, r) < 1e-2 for r in rect), f"[{deg}°] มุมจริงหาย: {p}"
        # (ค) จุดแรกต้องเป็น top-left (ผลรวม x+y น้อยสุด)
        assert np.argmin(rect.sum(axis=1)) == 0, f"[{deg}°] จุดแรกไม่ใช่ top-left"
        # (ง) ด้านที่ต่อกันต้องเป็น "ด้านจริง" ไม่ใช่เส้นทแยง → กัน quad ไขว้/บิด
        for i in range(4):
            edge = dist(rect[i], rect[(i + 1) % 4])
            assert edge < diag - 1.0, f"[{deg}°] ด้าน {i} ยาวเท่าเส้นทแยง (quad ไขว้)"
    print("✅ test_order_corners_all_angles: ผ่านทุกองศา (รวม 45°/135°)")


# ── test 2: fail-safe ของ detect_meter_obb (ภาพที่ไม่มีมิเตอร์) ─────────
def test_detect_failsafe():
    from PIL import Image
    dummy = Image.new("RGB", (640, 480), "white")  # ภาพเปล่า — OBB ไม่ควรเจอกรอบ
    corners, cls_name, conf = server.detect_meter_obb(dummy)

    # ไม่ว่ามีโมเดลหรือไม่ ต้องคืน tuple 3 ค่าเสมอ และห้าม throw
    assert corners is None or corners.shape == (4, 2), "corners ต้องเป็น None หรือ ndarray (4,2)"
    if corners is None:
        assert cls_name is None and conf == 0.0, "ไม่เจอกรอบ ต้องคืน (None, None, 0.0)"
        print("✅ test_detect_failsafe: ภาพเปล่า → (None, None, 0.0) ไม่ throw")
    else:
        assert isinstance(cls_name, str) and conf > 0, "เจอกรอบต้องมี class + conf"
        print(f"✅ test_detect_failsafe: เจอกรอบ [{cls_name}] conf={conf:.2f} (คืนค่าถูกชนิด)")


# ── test 3: expand_corners เผื่อขอบถูกสัดส่วน ────────────────────────────
def test_expand_corners():
    W, H = 100.0, 60.0
    rect = make_rect(W, H, 0.0)               # สี่เหลี่ยมตั้งตรง กลางที่ (500,400)
    ratio = 0.06
    out = server.expand_corners(rect, ratio)

    # centroid ต้องไม่ขยับ
    assert np.allclose(out.mean(axis=0), rect.mean(axis=0), atol=1e-3), "centroid ขยับ"
    # ด้านต้องยาวขึ้นตามสัดส่วนพอดี
    w_new = dist(out[0], out[1])
    h_new = dist(out[1], out[2])
    assert abs(w_new - W * (1 + ratio)) < 0.1, f"ความกว้างเผื่อขอบผิด: {w_new}"
    assert abs(h_new - H * (1 + ratio)) < 0.1, f"ความสูงเผื่อขอบผิด: {h_new}"
    # ขยายแล้วต้องยังเรียงมุมได้ถูก (ไม่ทำให้ลำดับพัง) ทุกองศา
    for deg in [0, 45, 90, 135]:
        r = make_rect(W, H, deg)
        o = server.order_corners(server.expand_corners(r, ratio))
        assert not has_duplicate_point(o), f"[{deg}°] ขยายแล้วมุมซ้ำ"
    print(f"✅ test_expand_corners: เผื่อขอบ {ratio:.0%} ถูกต้อง centroid ไม่ขยับ")


if __name__ == "__main__":
    failed = 0
    for fn in [test_order_corners_all_angles, test_detect_failsafe, test_expand_corners]:
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
