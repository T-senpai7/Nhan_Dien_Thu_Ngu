"""Edge Gateway: Camera + MediaPipe -> cam bien mo phong (Flex + IMU) -> JSON -> MQTT.

    python edge/edge_gateway.py                       # dung webcam
    python edge/edge_gateway.py --source synthetic    # ban tay ao, khong can camera
    python edge/edge_gateway.py --source synthetic --script XIN_CHAO,CAM_ON --hold 3
"""
import argparse
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import config
from edge.hand_model import SyntheticHand
from edge.sensor_simulator import SensorSimulator, build_payload
from network.mqtt_client import MqttLink

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


class EdgeGateway:
    def __init__(self, device_id: str, publish_hz: float, source: str):
        self.device_id, self.source = device_id, source
        self.min_interval = 1.0 / publish_hz if publish_hz > 0 else 0.0
        self.sim = SensorSimulator()
        self.seq_id = 0
        self.last_publish = 0.0
        self.last_prediction = None
        self.rtt = deque(maxlen=300)
        self.fps = 0.0
        self.hand_detected = False
        self.mode = None
        self.spelling = {"current": "", "sentence": ""}
        self.last_word = None
        self.hold_ms = config.LETTER_HOLD_MS
        self.lock = threading.Lock()
        self.mqtt = MqttLink(f"edge-{device_id}", [config.TOPIC_PREDICTION], self.on_prediction)

    def on_prediction(self, topic: str, raw: bytes):
        try:
            msg = json.loads(raw)
        except ValueError:
            return
        if msg.get("device_id") != self.device_id:
            return
        with self.lock:
            self.mode = msg.get("mode", self.mode)
            if msg.get("spelling"):
                self.spelling = msg["spelling"]
            if msg.get("type") == "state":
                self.hold_ms = msg.get("hold_ms", self.hold_ms)
                if msg.get("word"):
                    self.last_word = (msg["word"], time.time())
                return
            msg["rtt_ms"] = time.time() * 1000 - msg["edge_timestamp_ms"]
            self.last_prediction = msg
            self.rtt.append(msg["rtt_ms"])

    def send_control(self, action: str, mode: str = None, value=None):
        self.mqtt.publish(config.TOPIC_CONTROL, json.dumps(
            {"device_id": self.device_id, "action": action, "mode": mode, "value": value}))

    def handle_key(self, key: int):
        """Phim tat tren cua so Edge: M doi che do, SPACE ket thuc tu, BACKSPACE xoa chu, C xoa het."""
        if key == ord("m"):
            self.send_control("toggle_mode")
        elif key == ord(" "):
            self.send_control("space")
        elif key == 8:
            self.send_control("backspace")
        elif key == ord("c"):
            self.send_control("clear")
        elif key in (ord("+"), ord("=")):
            self.hold_ms = min(config.LETTER_HOLD_RANGE[1], self.hold_ms + 100)
            self.send_control("set_hold", value=self.hold_ms)
        elif key in (ord("-"), ord("_")):
            self.hold_ms = max(config.LETTER_HOLD_RANGE[0], self.hold_ms - 100)
            self.send_control("set_hold", value=self.hold_ms)

    def handle_frame(self, landmarks, capture_t: float, ground_truth: str = None, handedness: str = "Right"):
        """Xu ly 1 khung hinh; tra ve dac trung cam bien (hoac None neu khong thay tay)."""
        self.hand_detected = landmarks is not None
        if landmarks is None:
            self.sim.reset()
            return None
        features = self.sim.process(landmarks, capture_t, handedness)
        if capture_t - self.last_publish >= self.min_interval * 0.9 and self.mqtt.connected.is_set():
            self.last_publish = capture_t
            self.seq_id += 1
            processing_ms = (time.time() - capture_t) * 1000
            payload = build_payload(features, self.device_id, self.seq_id, capture_t * 1000,
                                    processing_ms, ground_truth)
            self.mqtt.publish(config.TOPIC_SENSOR, json.dumps(payload))
        return features

    def publish_status(self):
        with self.lock:
            rtt = list(self.rtt)
        status = {
            "device_id": self.device_id, "source": self.source, "fps": round(self.fps, 1),
            "hand_detected": self.hand_detected, "sent": self.seq_id,
            "rtt_avg_ms": round(float(np.mean(rtt)), 1) if rtt else None,
            "ts_ms": time.time() * 1000,
        }
        self.mqtt.publish(config.TOPIC_STATUS, json.dumps(status))

    def draw_hud(self, frame, features):
        import cv2
        from edge.ui_text import draw_texts
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (20, 20, 20), -1)
        cv2.rectangle(overlay, (w - 190, 120), (w, 320), (20, 20, 20), -1)
        cv2.rectangle(overlay, (0, h - 64), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        mqtt_ok = self.mqtt.connected.is_set()
        with self.lock:
            pred = self.last_prediction
            rtt = list(self.rtt)
            mode, spelling, last_word = self.mode, dict(self.spelling), self.last_word
        mode_name = config.MODES[mode]["name"] if mode in config.MODES else "?"
        items = [(f"MQTT: {'OK' if mqtt_ok else 'MẤT KẾT NỐI'}   FPS: {self.fps:4.1f}   Gửi: {self.seq_id}   "
                  f"Chế độ: {mode_name} (M)   Giữ chữ: {self.hold_ms / 1000:.1f}s (+/-)", (10, 4), 15,
                  (0, 220, 0) if mqtt_ok else (0, 0, 255), False)]
        hold = None
        if pred and time.time() * 1000 - pred["edge_timestamp_ms"] < 2000:
            stable = pred.get("stable_text")
            hold = pred.get("hold") if mode == "letters" else None
            shown = stable or (hold["text"] if hold else pred["text"])
            color = (0, 255, 255) if stable else (160, 160, 160)
            items.append((f"{shown}  ({pred['confidence'] * 100:.0f}%)", (10, 26), 40, color, True))
        if hold and hold["progress"] < 1:
            cv2.rectangle(frame, (10, 80), (250, 88), (80, 80, 80), -1)
            cv2.rectangle(frame, (10, 80), (10 + int(240 * hold["progress"]), 88), (0, 200, 255), -1)
        if rtt:
            items.append((f"RTT Edge→Cloud→Edge: {rtt[-1]:.0f} ms (TB {np.mean(rtt):.0f} ms)",
                          (10, 92), 15, (255, 255, 255), False))
        if mode == "letters":
            word = spelling.get("current", "")
            items.append((f"Đang đánh vần: {word or '...'}", (10, h - 62), 24, (0, 255, 255), True))
            sentence = spelling.get("sentence", "")
            if last_word and time.time() - last_word[1] < 2.5:
                sentence = f"{sentence}   ✓ {last_word[0]}"
            items.append((f"Câu: {sentence[-50:]}   [Ngón cái chỉ xuống: xóa | Xòe 5 ngón: xong từ]",
                          (10, h - 28), 15, (220, 220, 220), False))
        draw_texts(frame, items)
        if features:
            y = 142
            for name in ("thumb", "index", "middle", "ring", "pinky"):
                v = features[f"flex_{name}"]
                cv2.putText(frame, f"{name[:5]:5s}", (w - 182, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.rectangle(frame, (w - 130, y - 10), (w - 130 + int(v / 180 * 110), y), (0, 180, 255), -1)
                y += 22
            for key in ("roll", "pitch", "speed", "touch_thumb_index"):
                cv2.putText(frame, f"{key[:9]}: {features[key]:7.2f}", (w - 182, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)
                y += 20
        if not self.hand_detected:
            cv2.putText(frame, "Dua ban tay vao khung hinh", (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def run_camera(gw: EdgeGateway, args):
    import cv2
    from edge.hand_tracker import HandTracker, open_camera

    cap = open_camera(args.camera)
    if not cap.isOpened():
        print("[edge] Khong mo duoc camera. Thu: --source synthetic")
        return
    tracker = HandTracker()
    start = time.time()
    last_status, frames, fps_t = 0.0, 0, start
    print("[edge] Dang doc camera. Phim: M doi che do | SPACE ket thuc tu | BACKSPACE xoa | C xoa het | Q thoat")
    try:
        while True:
            ok, frame = cap.read()
            capture_t = time.time()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            landmarks, handedness, raw = tracker.process(frame)
            features = gw.handle_frame(landmarks, capture_t, handedness=handedness)
            frames += 1
            if capture_t - fps_t >= 1.0:
                gw.fps, frames, fps_t = frames / (capture_t - fps_t), 0, capture_t
            if capture_t - last_status >= 1.0:
                gw.publish_status()
                last_status = capture_t
            if not args.no_window:
                tracker.draw(frame, raw)
                gw.draw_hud(frame, features)
                cv2.imshow("Edge Gateway - Thu ngu IoT", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                gw.handle_key(key)
            if args.duration and capture_t - start > args.duration:
                break
    finally:
        cap.release()
        tracker.close()
        if not args.no_window:
            cv2.destroyAllWindows()


def run_synthetic(gw: EdgeGateway, args):
    labels = [s.strip() for s in args.script.split(",") if s.strip()] if args.script else list(config.WORDS)
    unknown = set(labels) - set(config.WORDS)
    if unknown:
        raise SystemExit(f"Ban tay ao chi ho tro nhom tu ngu, khong hop le: {unknown}")
    gw.send_control("set_mode", "words")
    show = args.show
    if show:
        import cv2
    hand = SyntheticHand(np.random.default_rng(args.seed))
    frame_dt = 1.0 / args.fps
    start = time.time()
    idx, gesture_start = 0, start
    hand.set_gesture(labels[0], 0.0)
    last_status = 0.0
    print(f"[edge] Che do ban tay ao: {', '.join(labels)} (moi cu chi {args.hold}s)")
    next_frame = time.time()
    while True:
        t = time.time()
        if t - gesture_start >= args.hold:
            idx += 1
            if args.loops and idx >= len(labels) * args.loops:
                break
            gesture_start = t
            hand.set_gesture(labels[idx % len(labels)], t - start)
            gw.sim.reset()
        landmarks = hand.landmarks(t - start)
        warm = t - gesture_start >= config.MOTION_WINDOW_SEC
        features = gw.handle_frame(landmarks, t, ground_truth=hand.label if warm else None,
                                   handedness=hand.handedness)
        gw.fps = args.fps
        if t - last_status >= 1.0:
            gw.publish_status()
            last_status = t
        if show:
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            pts = [(int(p[0] / 1.333 * 640), int(p[1] * 480)) for p in landmarks]
            for a, b in HAND_CONNECTIONS:
                cv2.line(canvas, pts[a], pts[b], (0, 200, 255), 2)
            for p in pts:
                cv2.circle(canvas, p, 4, (255, 255, 255), -1)
            cv2.putText(canvas, f"Ground truth: {hand.label}", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            gw.draw_hud(canvas, features)
            cv2.imshow("Edge Gateway - Ban tay ao", canvas)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        if args.duration and t - start > args.duration:
            break
        next_frame += frame_dt
        sleep = next_frame - time.time()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_frame = time.time()
    if show:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Edge Gateway thu ngu IoT")
    parser.add_argument("--source", choices=["camera", "synthetic"], default="camera")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--device-id", default=config.DEVICE_ID)
    parser.add_argument("--hz", type=float, default=config.PUBLISH_HZ, help="Tan suat gui MQTT")
    parser.add_argument("--duration", type=float, default=0, help="Tu dung sau N giay (0 = chay mai)")
    parser.add_argument("--no-window", action="store_true", help="Khong mo cua so OpenCV (camera)")
    parser.add_argument("--show", action="store_true", help="Hien thi ban tay ao (synthetic)")
    parser.add_argument("--script", default="", help="Danh sach cu chi cho che do synthetic")
    parser.add_argument("--hold", type=float, default=3.0, help="So giay giu moi cu chi (synthetic)")
    parser.add_argument("--loops", type=int, default=0, help="So vong lap kich ban (0 = vo han)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    gw = EdgeGateway(args.device_id, args.hz, args.source)
    print(f"[edge] Ket noi MQTT {config.MQTT_HOST}:{config.MQTT_PORT} ...")
    if not gw.mqtt.start(wait_sec=10):
        print("[edge] Chua ket noi duoc broker, se tu thu lai trong nen.")
    try:
        run_camera(gw, args) if args.source == "camera" else run_synthetic(gw, args)
    except KeyboardInterrupt:
        pass
    finally:
        rtt = list(gw.rtt)
        if rtt:
            print(f"[edge] Da gui {gw.seq_id} goi | RTT trung binh {np.mean(rtt):.1f} ms | P95 {np.percentile(rtt, 95):.1f} ms")
        gw.mqtt.stop()


if __name__ == "__main__":
    main()
