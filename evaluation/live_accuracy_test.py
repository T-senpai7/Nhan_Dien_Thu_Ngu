"""Kiem thu do chinh xac tren camera that (khong dung du lieu mo phong).

Chuong trinh hien ngau nhien tung chu cai / cu chi, nguoi dung ra dau theo,
moi khung hinh duoc gui toi Cloud AI API (/api/predict) de cham diem.

    python evaluation/live_accuracy_test.py                    # 23 chu cai, moi chu 2 luot
    python evaluation/live_accuracy_test.py --group words
    python evaluation/live_accuracy_test.py --labels M,N,S,T --trials 3

Yeu cau: Cloud server dang chay (python run_demo.py --no-edge).
Ket qua: do chinh xac theo khung hinh va theo luot (bo phieu da so), ma tran nham lan.
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import requests

import config
from cloud.trajectory import MarkClassifier, StrokeSegmenter, index_direction
from edge.hand_tracker import HandTracker, open_camera
from edge.sensor_simulator import SensorSimulator, build_payload
from edge.ui_text import draw_texts

PREP_SEC = 2.0


def main():
    parser = argparse.ArgumentParser(description="Kiem thu do chinh xac tren camera that")
    parser.add_argument("--group", choices=["letters", "marks", "controls", "words"], default="letters")
    parser.add_argument("--labels", default="")
    parser.add_argument("--trials", type=int, default=2, help="So luot moi nhan")
    parser.add_argument("--seconds", type=float, default=2.5, help="Thoi gian cham diem moi luot")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = parser.parse_args()

    try:
        health = requests.get(f"{config.CLOUD_URL}/api/health", timeout=5).json()
    except requests.RequestException:
        raise SystemExit(f"Cloud server chua chay tai {config.CLOUD_URL} (python run_demo.py --no-edge)")
    api_mode = "letters" if args.group in ("marks", "controls") else args.group
    if api_mode not in health.get("modes", []):
        raise SystemExit(f"Cloud chua co mo hinh cho che do '{api_mode}'. Hay thu mau va train truoc.")
    labels = [l.strip().upper() for l in args.labels.split(",") if l.strip()] or config.RECORD_GROUPS[args.group]
    marks = None
    if any(l in config.MARKS_DYNAMIC for l in labels):
        if not config.MARK_MODEL_PATH.exists():
            raise SystemExit("Chua co mo hinh dau dong. Thu mau --group marks va train truoc.")
        marks = MarkClassifier()
    plan = labels * args.trials
    random.shuffle(plan)

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise SystemExit("Khong mo duoc camera.")
    ref = config.MARKS_REFERENCE_IMAGE if args.group == "marks" else config.REFERENCE_IMAGE
    if ref.exists() and args.group != "words":
        cv2.imshow("Anh tham chieu", cv2.imread(str(ref)))
    tracker, sim, http = HandTracker(), SensorSimulator(), requests.Session()
    segmenter = StrokeSegmenter()
    trials, idx, phase_t, preds, seq = [], 0, time.time(), [], 0
    stroke_pred = None
    quit_early = False

    while idx < len(plan):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        now = time.time()
        lm, hand, raw = tracker.process(frame)
        tracker.draw(frame, raw)
        target = plan[idx]
        dynamic = target in config.MARKS_DYNAMIC
        elapsed = now - phase_t
        scoring = elapsed >= PREP_SEC
        h, w = frame.shape[:2]

        last = None
        if lm is not None:
            feats = sim.process(lm, now, hand)
            if scoring:
                seq += 1
                payload = build_payload(feats, "live_test", seq, now * 1000, 0.0, target)
                resp = http.post(f"{config.CLOUD_URL}/api/predict", params={"mode": api_mode},
                                 json=payload, timeout=5).json()
                if dynamic:
                    pose = any(t["label"] == config.POSE_LABEL and t["confidence"] >= 0.3 for t in resp["top3"])
                    x, y, palm = feats["pointer"]
                    stroke = segmenter.update(now, x, y, palm, index_direction(feats["shape"]), pose)
                    if stroke is not None and stroke_pred is None:
                        stroke_pred = marks.predict(stroke)
                    if stroke_pred:
                        last = {"label": stroke_pred["label"], "text": config.GESTURES[stroke_pred["label"]][0],
                                "confidence": stroke_pred["confidence"]}
                else:
                    last = resp
                    preds.append(resp["label"])
        else:
            sim.reset()

        items = [
            (f"[{idx + 1}/{len(plan)}] Hãy ra dấu: {config.GESTURES[target][0]}", (12, 8), 34, (0, 255, 255), True),
            (config.GESTURES[target][1], (12, 52), 17, (255, 255, 255), False),
        ]
        if dynamic:
            items.append(("Duỗi ngón trỏ, khi thanh đỏ chạy thì vẽ dấu MỘT lần rồi giữ yên", (12, 80), 16,
                          (0, 255, 255), False))
        if not scoring:
            items.append((f"Chuẩn bị... {PREP_SEC - elapsed:.1f}s", (12, h - 40), 22, (0, 200, 255), True))
        else:
            cv2.rectangle(frame, (0, h - 8), (int(w * min(1, (elapsed - PREP_SEC) / args.seconds)), h), (0, 0, 255), -1)
            if last:
                good = last["label"] == target
                items.append((f"AI: {last['text']} ({last['confidence'] * 100:.0f}%)", (12, h - 44), 24,
                              (0, 255, 0) if good else (0, 0, 255), True))
        draw_texts(frame, items)
        cv2.imshow("Kiem thu do chinh xac", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            quit_early = True
            break

        if scoring and dynamic and stroke_pred is None and elapsed - PREP_SEC >= args.seconds:
            stroke = segmenter.flush()
            stroke_pred = marks.predict(stroke) if stroke else None
        if scoring and elapsed - PREP_SEC >= args.seconds:
            if dynamic:
                accepted = stroke_pred and stroke_pred["confidence"] >= config.MARK_MIN_CONFIDENCE
                preds = [stroke_pred["label"]] if accepted else []
                stroke_pred = None
                segmenter.reset()
            majority = Counter(preds).most_common(1)[0][0] if preds else None
            trials.append({"target": target, "frames": len(preds), "correct_frames": preds.count(target),
                           "majority": majority, "correct": majority == target})
            mark = "DUNG" if majority == target else "SAI"
            print(f"  {config.GESTURES[target][0]:>18s} -> {config.GESTURES[majority][0] if majority else '(khong thay tay)':<18s} "
                  f"{mark} | {preds.count(target)}/{len(preds)} khung dung")
            idx, phase_t, preds = idx + 1, time.time(), []
            sim.reset()

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    if not trials:
        return
    write_report(trials, args.group, quit_early)


def write_report(trials: list, group: str, partial: bool):
    frames = sum(t["frames"] for t in trials)
    frame_acc = sum(t["correct_frames"] for t in trials) / frames * 100 if frames else 0.0
    trial_acc = sum(t["correct"] for t in trials) / len(trials) * 100
    per_label = {}
    for t in trials:
        p = per_label.setdefault(t["target"], {"trials": 0, "correct": 0, "frames": 0, "correct_frames": 0})
        p["trials"] += 1
        p["correct"] += int(t["correct"])
        p["frames"] += t["frames"]
        p["correct_frames"] += t["correct_frames"]
    confusions = Counter((t["target"], t["majority"]) for t in trials if not t["correct"])

    print(f"\nDo chinh xac theo luot: {trial_acc:.1f}% ({sum(t['correct'] for t in trials)}/{len(trials)})")
    print(f"Do chinh xac theo khung hinh: {frame_acc:.1f}% ({frames} khung)")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    name = lambda l: config.GESTURES[l][0] if l else "(khong thay tay)"
    lines = [f"# Kiem thu do chinh xac tren camera that - { {'letters': 'Chữ cái', 'marks': 'Dấu phụ / dấu thanh', 'controls': 'Cử chỉ điều khiển', 'words': 'Từ ngữ'}[group] }", "",
             f"- Thoi gian: {datetime.now().isoformat(timespec='seconds')}{' (dung giua chung)' if partial else ''}",
             f"- Do chinh xac theo luot (bo phieu da so): **{trial_acc:.1f}%** ({len(trials)} luot)",
             f"- Do chinh xac theo khung hinh: **{frame_acc:.1f}%** ({frames} khung)", "",
             "| Nhan | Luot dung | Khung dung |", "|---|---|---|"]
    for label, p in per_label.items():
        lines.append(f"| {name(label)} | {p['correct']}/{p['trials']} | "
                     f"{p['correct_frames'] / p['frames'] * 100 if p['frames'] else 0:.0f}% |")
    if confusions:
        lines += ["", "## Nham lan", ""] + [f"- {name(a)} -> {name(b)}: {n} lan" for (a, b), n in confusions.most_common()]
    path = config.REPORT_DIR / f"live_accuracy_{group}_{stamp}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (path.with_suffix(".json")).write_text(json.dumps({"trials": trials, "trial_accuracy": trial_acc,
                                                       "frame_accuracy": frame_acc}, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print(f"Bao cao: {path}")


if __name__ == "__main__":
    main()
