"""Tao dataset cho he thong phien dich thu ngu.

1) Thu mau that tu webcam (BAT BUOC cho che do chu cai de dam bao do chinh xac):
    python dataset/collect_data.py camera                     # 23 chu cai, 2 vong
    python dataset/collect_data.py camera --group marks       # dau mu, moc + 6 dau dong (trang, 5 thanh)
    python dataset/collect_data.py camera --group controls    # cu chi xoa chu / xong tu
    python dataset/collect_data.py camera --group words       # 12 cu chi tu ngu
    python dataset/collect_data.py camera --labels M,N,S,T    # thu bo sung vai chu
    python dataset/collect_data.py camera --group letters,controls,marks --rounds 1 --min-rounds 3
    Phim: SPACE bat dau ghi | N bo qua | B quay lai | X xoa lan ghi gan nhat | C chup man hinh | Q thoat
    Moi lan ghi tu luu anh goc + anh khung xuong (va anh quy dao voi dau dong) vao dataset/images/.

2) Sinh du lieu tong hop cho nhom tu ngu (khong can camera):
    python dataset/collect_data.py synthetic --sequences 200

3) Xem so mau da thu:
    python dataset/collect_data.py stats
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config
from edge.hand_model import simulate_sequence
from edge.sensor_simulator import SensorSimulator, features_to_vector

RAW_COLUMNS = ["session", "label", "frame", "t", "handedness"] + [
    f"{axis}{i}" for i in range(21) for axis in "xyz"]
LANDMARK_COLUMNS = RAW_COLUMNS[5:]
WARMUP_FRAMES = 5


# ---------------------------------------------------------------------------
# Chuyen du lieu tho -> vector dac trung (dung chung cho train va danh gia)
# ---------------------------------------------------------------------------
def _augment(lm_seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Xoay nhe trong mat phang anh, co gian va nhieu khop: mo phong nguoi / goc may khac."""
    angle = np.radians(rng.uniform(-10, 10))
    scale = rng.uniform(0.85, 1.15)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    out = lm_seq.copy()
    wrist = out[:, :1, :2]
    out[:, :, :2] = (out[:, :, :2] - wrist) @ rot.T * scale + wrist
    out[:, :, 2] *= scale
    out += rng.normal(0, 0.0015, out.shape)
    return out


def raw_to_features(raw: pd.DataFrame, augment_copies: int = 0, seed: int = 0) -> pd.DataFrame:
    """Phat lai tung phien ghi qua SensorSimulator (giong het luong Edge thoi gian thuc)."""
    rng = np.random.default_rng(seed)
    rows = []
    for session, group in raw.groupby("session", sort=False):
        group = group.sort_values("frame")
        label = group["label"].iloc[0]
        times = group["t"].to_numpy()
        hands = group["handedness"].to_numpy()
        lm_seq = group[LANDMARK_COLUMNS].to_numpy(dtype=float).reshape(-1, 21, 3)
        variants = [("orig", lm_seq)] + [(f"aug{k}", _augment(lm_seq, rng)) for k in range(augment_copies)]
        for tag, seq in variants:
            sim = SensorSimulator()
            for i, (t, lm, hand) in enumerate(zip(times, seq, hands)):
                feats = sim.process(lm, t, hand)
                if i < WARMUP_FRAMES:
                    continue
                rows.append([*features_to_vector(feats), label, "camera", session, f"{session}#{i // 45}", tag])
    return pd.DataFrame(rows, columns=config.FEATURE_COLUMNS + ["label", "source", "session", "chunk", "variant"])


def _augment_trajectory(lm_seq: np.ndarray, times: np.ndarray, rng: np.random.Generator):
    """Xoay ca chuoi quanh diem bat dau, doi do lon net ve, bot khung (mo phong tan suat Edge thap hon)."""
    angle = np.radians(rng.uniform(-12, 12))
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    origin = lm_seq[0, 0, :2]
    stretch = rng.uniform(0.75, 1.3)
    out = lm_seq.copy()
    wrist = out[:, :1, :2]
    moved = origin + (wrist - origin) * stretch
    out[:, :, :2] = (out[:, :, :2] - wrist) @ rot.T + ((moved - origin) @ rot.T + origin)
    out += rng.normal(0, 0.0012, out.shape)
    keep = np.arange(rng.integers(0, 2), len(out), 2) if rng.random() < 0.5 else np.arange(len(out))
    return out[keep], times[keep], keep


def session_strokes(times, lm_seq, hands) -> list:
    """Phat lai 1 phien qua SensorSimulator + StrokeSegmenter (giong het Cloud thoi gian thuc)."""
    from cloud.trajectory import StrokeSegmenter, index_direction
    sim, seg, strokes = SensorSimulator(), StrokeSegmenter(), []
    for t, lm, hand in zip(times, lm_seq, hands):
        feats = sim.process(lm, t, hand)
        x, y, palm = feats["pointer"]
        s = seg.update(t, x, y, palm, index_direction(feats["shape"]), True)
        if s:
            strokes.append(s)
    s = seg.flush()
    if s:
        strokes.append(s)
    return strokes


def raw_to_strokes(raw: pd.DataFrame, augment_copies: int = 0, seed: int = 0) -> pd.DataFrame:
    """Moi phien ghi dau dong -> net ve dai nhat -> 1 mau dac trung quy dao."""
    from cloud.trajectory import FEATURE_NAMES, stroke_features
    rng = np.random.default_rng(seed)
    rows, skipped = [], []
    for session, group in raw.groupby("session", sort=False):
        group = group.sort_values("frame")
        label = group["label"].iloc[0]
        times = group["t"].to_numpy(dtype=float)
        hands = group["handedness"].to_numpy()
        lm_seq = group[LANDMARK_COLUMNS].to_numpy(dtype=float).reshape(-1, 21, 3)
        variants = [("orig", lm_seq, times, hands)]
        for k in range(augment_copies):
            seq, ts, keep = _augment_trajectory(lm_seq, times, rng)
            variants.append((f"aug{k}", seq, ts, hands[keep]))
        for tag, seq, ts, hs in variants:
            strokes = session_strokes(ts, seq, hs)
            if not strokes:
                if tag == "orig":
                    skipped.append(session)
                continue
            best = max(strokes, key=lambda s: s["path"])
            rows.append([*stroke_features(best), label, session, tag])
    if skipped:
        print(f"  Bo qua {len(skipped)} phien khong tach duoc net ve: {', '.join(skipped[:5])}{' ...' if len(skipped) > 5 else ''}")
    return pd.DataFrame(rows, columns=FEATURE_NAMES + ["label", "session", "variant"])


def load_raw() -> pd.DataFrame:
    if not config.CAMERA_RAW_PATH.exists():
        return pd.DataFrame(columns=RAW_COLUMNS)
    return pd.read_csv(config.CAMERA_RAW_PATH)


# ---------------------------------------------------------------------------
# Du lieu tong hop (chi cho nhom tu ngu)
# ---------------------------------------------------------------------------
def generate_synthetic(sequences_per_class: int, seed: int = 42, warmup_sec: float = 0.7,
                       every_n: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label in config.WORDS:
        for seq in range(sequences_per_class):
            fps = rng.uniform(15, 45)
            sim = SensorSimulator()
            session = f"syn_{label}_{seq}"
            for i, (t, lm, hand) in enumerate(simulate_sequence(label, duration=1.6, fps=fps, rng=rng)):
                feats = sim.process(lm, t, hand)
                if t >= warmup_sec and i % every_n == 0:
                    rows.append([*features_to_vector(feats), label, "synthetic", session, session, "orig"])
    return pd.DataFrame(rows, columns=config.FEATURE_COLUMNS + ["label", "source", "session", "chunk", "variant"])


def cmd_synthetic(args):
    start = time.time()
    df = generate_synthetic(args.sequences, args.seed)
    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.SYNTHETIC_DATASET_PATH, index=False)
    print(f"Da tao {len(df)} mau tong hop ({df['label'].nunique()} cu chi tu ngu) trong {time.time() - start:.1f}s")
    print(f"-> {config.SYNTHETIC_DATASET_PATH}")


def cmd_stats(_args=None):
    raw = load_raw()
    if raw.empty:
        print("Chua co mau camera nao. Chay: python dataset/collect_data.py camera")
        return
    summary = raw.groupby("label").agg(frames=("frame", "size"), sessions=("session", "nunique"))
    names = {"letters": "Chữ cái", "marks": "Dấu phụ / dấu thanh", "controls": "Cử chỉ điều khiển",
             "words": "Từ ngữ"}
    for group, labels in config.RECORD_GROUPS.items():
        print(f"\n== {names[group]} ==")
        for label in labels:
            if label in summary.index:
                s = summary.loc[label]
                if label in config.MARKS_DYNAMIC:
                    warn = "" if s.sessions >= 12 else "  <- nen thu them (>= 12 lan ve)"
                else:
                    warn = "" if s.sessions >= 2 and s.frames >= 200 else "  <- nen thu them"
                print(f"  {config.GESTURES[label][0]:>18s}: {s.frames:5d} khung, {s.sessions} phien{warn}")
            else:
                print(f"  {config.GESTURES[label][0]:>18s}: CHUA CO")


# ---------------------------------------------------------------------------
# Cong cu thu mau co huong dan
# ---------------------------------------------------------------------------
def _append_session(rows: list):
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    path = config.CAMERA_RAW_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _delete_session(session: str):
    raw = load_raw()
    raw[raw["session"] != session].to_csv(config.CAMERA_RAW_PATH, index=False)
    index = config.CAPTURE_INDEX_PATH
    if index.exists():
        df = pd.read_csv(index)
        for f in df.loc[df["session"] == session, "file"]:
            (config.CAPTURE_DIR / f).unlink(missing_ok=True)
        df[df["session"] != session].to_csv(index, index=False)


def _save_captures(session: str, label: str, shots: list, trail=None):
    """Luu anh goc + anh co khung xuong cua lan ghi vao dataset/images/<nhan>/ va ghi chi muc."""
    import cv2
    folder = config.CAPTURE_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for k, (kind, img) in enumerate(shots):
        name = f"{label}/{session}_{k // 2}_{kind}.jpg"
        cv2.imwrite(str(config.CAPTURE_DIR / name), img)
        entries.append((session, label, config.GESTURES[label][0], kind, name, time.strftime("%Y-%m-%d %H:%M:%S")))
    if trail is not None:
        name = f"{label}/{session}_trajectory.jpg"
        cv2.imwrite(str(config.CAPTURE_DIR / name), trail)
        entries.append((session, label, config.GESTURES[label][0], "trajectory", name, time.strftime("%Y-%m-%d %H:%M:%S")))
    df = pd.DataFrame(entries, columns=["session", "label", "text", "kind", "file", "time"])
    df.to_csv(config.CAPTURE_INDEX_PATH, mode="a", header=not config.CAPTURE_INDEX_PATH.exists(), index=False)


def _draw_trajectory(base, rows):
    """Ve duong di cua dau ngon tro (diem 8) len anh, tu xanh (bat dau) toi do (ket thuc)."""
    import cv2
    img = base.copy()
    h, w = img.shape[:2]
    pts = [(int(r[5 + 8 * 3] * h), int(r[5 + 8 * 3 + 1] * h)) for r in rows]  # x da nhan ti le khung (w/h)
    for i in range(1, len(pts)):
        c = i / max(1, len(pts) - 1)
        cv2.line(img, pts[i - 1], pts[i], (0, int(255 * (1 - c)), int(255 * c)), 4, cv2.LINE_AA)
    if pts:
        cv2.circle(img, pts[0], 8, (0, 255, 0), -1)
        cv2.circle(img, pts[-1], 8, (0, 0, 255), -1)
    return img


def cmd_camera(args):
    import cv2
    from edge.hand_tracker import HandTracker, open_camera
    from edge.ui_text import draw_texts

    if args.labels:
        labels = [l.strip().upper() for l in args.labels.split(",") if l.strip()]
        bad = [l for l in labels if l not in config.GESTURES]
        if bad:
            raise SystemExit(f"Nhan khong hop le: {bad}. Hop le: {', '.join(config.GESTURE_LABELS)}")
    else:
        groups = [g.strip() for g in args.group.split(",") if g.strip()]
        groups = list(config.RECORD_GROUPS) if "all" in groups else groups
        bad = [g for g in groups if g not in config.RECORD_GROUPS]
        if bad:
            raise SystemExit(f"Nhom khong hop le: {bad}. Hop le: {', '.join(config.RECORD_GROUPS)}, all")
        labels = [l for g in groups for l in config.RECORD_GROUPS[g]]
    bad = [l for l in labels if l == config.POSE_LABEL]
    if bad:
        raise SystemExit("Tu the dau duoc hoc tu dong tu cac lan ghi dau dong, khong can thu rieng.")
    existing = load_raw().groupby("label")["session"].nunique().to_dict()

    def rounds_for(l):
        have = existing.get(l, 0) // (args.reps if l in config.MARKS_DYNAMIC else 1)
        return max(args.rounds, args.min_rounds - have)

    plan = [l for r in range(max(rounds_for(l) for l in labels)) for l in labels if r < rounds_for(l)
            for _ in range(args.reps if l in config.MARKS_DYNAMIC else 1)]

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise SystemExit("Khong mo duoc camera.")
    for img, title, needed in ((config.REFERENCE_IMAGE, "Anh tham chieu bang chu cai", config.LETTERS),
                               (config.MARKS_REFERENCE_IMAGE, "Anh tham chieu dau",
                                {**config.MARKS_STATIC, **config.MARKS_DYNAMIC})):
        if img.exists() and any(l in needed for l in labels):
            cv2.imshow(title, cv2.imread(str(img)))

    tracker = HandTracker()
    counts = load_raw().groupby("label")["session"].nunique().to_dict()
    idx, state, last_session = 0, "ready", None
    rows, state_t, countdown = [], 0.0, 3
    print(f"Ke hoach thu: {len(plan)} luot. Xem huong dan tren cua so camera.")

    while idx < len(plan):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        now = time.time()
        lm, hand, raw = tracker.process(frame)
        clean = frame.copy()
        tracker.draw(frame, raw)
        skeleton = frame.copy()
        label = plan[idx]
        dynamic = label in config.MARKS_DYNAMIC
        duration = args.mark_seconds if dynamic else args.seconds
        text, desc = config.GESTURES[label]
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 110), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        group_name = {"mark": "Dấu", "control": "Điều khiển"}.get(
            config.LABEL_KIND[label], config.MODES[config.LABEL_MODE[label]]["name"])
        items = [
            (f"[{idx + 1}/{len(plan)}] {group_name}: {text}", (12, 6), 30, (0, 255, 255), True),
            (desc, (12, 46), 18, (255, 255, 255), False),
            (f"Đã có {counts.get(label, 0)} lần ghi cho '{text}'", (12, 76), 16, (180, 180, 180), False),
        ]

        if state == "ready":
            items.append(("SPACE: bắt đầu  N: bỏ qua  B: quay lại  X: xóa lần ghi trước  C: chụp màn hình  Q: thoát",
                          (12, h - 32), 16, (120, 255, 120), False))
        elif state == "countdown":
            remain = countdown - int(now - state_t)
            if dynamic:
                items.append(("Giữ tư thế ngón trỏ, chờ chữ VẼ NGAY rồi vẽ dấu 1 lần và giữ yên",
                              (12, h - 40), 16, (0, 255, 255), True))
            if remain <= 0:
                state, state_t, rows, shots, first_clean = "recording", now, [], [], None
                last_session = f"{label}_{int(now * 1000)}"
            else:
                items.append((str(remain), (w // 2 - 20, h // 2 - 50), 90, (0, 255, 255), True))
        if state == "recording":
            elapsed = now - state_t
            if lm is not None:
                rows.append([last_session, label, len(rows), now, hand, *lm.reshape(-1).tolist()])
                if first_clean is None:
                    first_clean = clean
                due = (len(shots) // 2 + 0.5) / config.CAPTURES_PER_SESSION * duration
                if not args.no_images and len(shots) < 2 * config.CAPTURES_PER_SESSION and elapsed >= due:
                    shots += [("raw", clean), ("skeleton", skeleton)]
            cv2.rectangle(frame, (0, h - 10), (int(w * min(1.0, elapsed / duration)), h), (0, 0, 255), -1)
            if dynamic:
                items.append(("VẼ NGAY!", (w // 2 - 80, h // 2 - 40), 44, (0, 0, 255), True))
            else:
                items.append((f"ĐANG GHI {len(rows)} khung  - di chuyển nhẹ tay: gần/xa, nghiêng ít, lệch trái/phải",
                              (12, h - 40), 16, (80, 80, 255), True))
            if lm is None:
                items.append(("Không thấy tay!", (w // 2 - 90, h // 2), 30, (0, 0, 255), True))
            if elapsed >= duration:
                state, countdown = "ready", 3
                ok_rows = len(rows) >= (10 if dynamic else 20)
                if ok_rows and dynamic:
                    seq = np.array([r[5:] for r in rows], dtype=float).reshape(-1, 21, 3)
                    strokes = session_strokes(np.array([r[3] for r in rows]), seq, [r[4] for r in rows])
                    if not strokes:
                        print(f"  Khong thay net ve ro rang, ve lai {text} (ve nhanh va dut khoat hon)")
                        ok_rows = None
                if ok_rows:
                    _append_session(rows)
                    if not args.no_images:
                        trail = _draw_trajectory(first_clean, rows) if dynamic and first_clean is not None else None
                        _save_captures(last_session, label, shots, trail)
                    counts[label] = counts.get(label, 0) + 1
                    print(f"  Da luu {len(rows)} khung cho {text}")
                    idx += 1
                    if dynamic and idx < len(plan) and plan[idx] == label:
                        state, state_t, countdown = "countdown", time.time(), 1
                elif ok_rows is not None:
                    print(f"  Qua it khung co tay ({len(rows)}), ghi lai {text}")

        draw_texts(frame, items)
        cv2.imshow("Thu mau thu ngu", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            shot_dir = config.CAPTURE_DIR / "screenshots"
            shot_dir.mkdir(parents=True, exist_ok=True)
            path = shot_dir / f"{label}_{int(now * 1000)}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"  Da chup man hinh: {path}")
        if state == "ready":
            if key == ord(" "):
                state, state_t = "countdown", now
            elif key == ord("n"):
                idx += 1
            elif key == ord("b"):
                idx = max(0, idx - 1)
            elif key == ord("x") and last_session:
                _delete_session(last_session)
                print(f"  Da xoa phien {last_session}")
                counts = load_raw().groupby("label")["session"].nunique().to_dict()
                last_session = None

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    print("\nTong ket du lieu camera:")
    cmd_stats()
    print("\nBuoc tiep theo: python cloud/train_model.py")


def main():
    parser = argparse.ArgumentParser(description="Tao dataset thu ngu")
    sub = parser.add_subparsers(dest="mode", required=True)
    p_cam = sub.add_parser("camera", help="Thu mau co huong dan tu webcam")
    p_cam.add_argument("--group", default="letters",
                       help="Nhom can thu, nhieu nhom cach nhau dau phay: letters,controls,marks,words hoac all")
    p_cam.add_argument("--labels", default="", help="Danh sach nhan, vd: A,B,DD,M hoac MU,DAU_SAC")
    p_cam.add_argument("--seconds", type=float, default=5.0, help="So giay ghi moi luot (tu the tinh)")
    p_cam.add_argument("--rounds", type=int, default=2, help="So vong lap qua toan bo danh sach")
    p_cam.add_argument("--reps", type=int, default=8, help="So lan ve moi dau dong trong 1 vong")
    p_cam.add_argument("--mark-seconds", type=float, default=2.5, help="Thoi gian moi lan ve dau dong")
    p_cam.add_argument("--min-rounds", type=int, default=0,
                       help="Nhan nao co it hon so vong nay se duoc thu bu cho du (vd ky hieu moi)")
    p_cam.add_argument("--no-images", action="store_true", help="Khong luu anh chup minh hoa")
    p_cam.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    p_syn = sub.add_parser("synthetic", help="Sinh du lieu tong hop cho nhom tu ngu")
    p_syn.add_argument("--sequences", type=int, default=200)
    p_syn.add_argument("--seed", type=int, default=42)
    sub.add_parser("stats", help="Thong ke so mau camera")
    args = parser.parse_args()
    {"camera": cmd_camera, "synthetic": cmd_synthetic, "stats": cmd_stats}[args.mode](args)


if __name__ == "__main__":
    main()
