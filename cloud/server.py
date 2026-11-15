"""Cloud Platform: FastAPI + MQTT subscriber + Cloud AI + SQLite + WebSocket Dashboard.

    python cloud/server.py
    -> Dashboard: http://127.0.0.1:8000
"""
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from cloud.ai_service import GestureClassifier, HoldStabilizer, PredictionStabilizer, Speller
from cloud.database import Database, _percentiles
from cloud.trajectory import MarkClassifier, StrokeSegmenter, index_direction
from edge.sensor_simulator import payload_to_features
from network.mqtt_client import MqttLink


def now_ms() -> float:
    return time.time() * 1000.0


class StreamStats:
    """Theo doi do on dinh luong MQTT: so goi nhan, mat goi (dua tren seq_id), toc do."""

    def __init__(self):
        self.lock = threading.Lock()
        self.devices = {}
        self.arrivals = deque(maxlen=5000)

    def record(self, device_id: str, seq_id: int, ts: float):
        with self.lock:
            self.arrivals.append(ts)
            d = self.devices.get(device_id)
            if d is None or seq_id < d["max_seq"] - 100:
                prev = d or {"sessions": 0, "prev_received": 0, "prev_expected": 0, "received": 0,
                             "min_seq": 1, "max_seq": 0}
                d = self.devices[device_id] = {
                    "min_seq": seq_id, "max_seq": seq_id, "received": 0, "last_ms": ts,
                    "sessions": prev["sessions"] + 1,
                    "prev_received": prev["prev_received"] + prev["received"],
                    "prev_expected": prev["prev_expected"] + prev["max_seq"] - prev["min_seq"] + 1,
                }
            d["min_seq"] = min(d["min_seq"], seq_id)
            d["max_seq"] = max(d["max_seq"], seq_id)
            d["received"] += 1
            d["last_ms"] = ts

    def snapshot(self) -> dict:
        with self.lock:
            t = now_ms()
            while self.arrivals and t - self.arrivals[0] > 5000:
                self.arrivals.popleft()
            received = sum(d["prev_received"] + d["received"] for d in self.devices.values())
            expected = sum(d["prev_expected"] + d["max_seq"] - d["min_seq"] + 1 for d in self.devices.values())
            lost = max(0, expected - received)
            return {
                "msgs_total": received,
                "msgs_lost": lost,
                "pdr": round(received / expected * 100, 2) if expected else 100.0,
                "msg_rate": round(len(self.arrivals) / 5.0, 2),
                "devices": {k: {**v, "online": t - v["last_ms"] < 3000} for k, v in self.devices.items()},
            }


class EdgeProcess:
    """Bat / tat Edge Gateway tu Dashboard (chay tren cung may voi Cloud)."""

    LOG_PATH = config.ROOT_DIR / "cloud" / "edge_process.log"

    def __init__(self):
        self.proc = None
        self.source = None
        self.started_ms = None
        self.stopped_by_user = False

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, source: str) -> dict:
        if self.running():
            return self.status()
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        if getattr(self, "log", None):
            self.log.close()
        self.log = open(self.LOG_PATH, "w", encoding="utf-8")
        self.proc = subprocess.Popen([sys.executable, "edge/edge_gateway.py", "--source", source],
                                     cwd=config.ROOT_DIR, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        self.source, self.started_ms, self.stopped_by_user = source, now_ms(), False
        return self.status()

    def stop(self) -> dict:
        if self.running():
            self.stopped_by_user = True
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        return self.status()

    def status(self) -> dict:
        info = {"running": self.running(), "source": self.source, "pid": self.proc.pid if self.proc else None,
                "started_ms": self.started_ms, "exit_code": None, "error": None}
        if self.proc is not None and not info["running"]:
            info["exit_code"] = self.proc.returncode
            if not self.stopped_by_user and self.proc.returncode != 0 and self.LOG_PATH.exists():
                lines = [l for l in self.LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
                         if l.strip() and not l.startswith(("W0000", "INFO:", "WARNING:"))]
                info["error"] = "\n".join(lines[-4:])
        return info


class DeviceState:
    """Trang thai rieng cua tung Edge: che do dang dung, bo on dinh du doan, bo danh van."""

    def __init__(self, mode: str):
        self.mode = mode
        self.stabilizers = {
            "letters": HoldStabilizer(config.LETTER_HOLD_MS, config.MODE_SETTINGS["letters"]["threshold"]),
            "words": PredictionStabilizer(config.MODE_SETTINGS["words"]["window"],
                                          config.MODE_SETTINGS["words"]["threshold"]),
        }
        self.speller = Speller()
        self.last_control_ms = 0.0
        self.prev_shape = None
        self.motion = (0.0, 0.0)
        self.pose_ready_ms = -1e12
        self.last_ms = 0.0
        self.last_edge_ms = 0.0
        self.segmenter = StrokeSegmenter()
        # Sau mot dau dong: chi nhan dau tiep theo khi tay roi tu the dau (tranh nhan nham luc rut tay ve).
        self.mark_armed = True
        self.mark_block_until = 0.0
        self.pose_lost = 0


class CloudApp:
    def __init__(self):
        self.db = Database()
        self.stats = StreamStats()
        self.devices = {}
        self.lock = threading.RLock()
        self.edge_status = {}
        self.e2e_live = deque(maxlen=500)
        self.clients = set()
        self.loop = None
        self.classifier = None
        self.model_error = None
        self.load_model()
        self.mqtt = MqttLink("cloud", [config.TOPIC_SENSOR, config.TOPIC_STATUS, config.TOPIC_CONTROL],
                             self.on_mqtt_message)

    def load_model(self):
        try:
            self.classifier = GestureClassifier()
            self.model_error = None
            print(f"[cloud] Da nap mo hinh cho che do: {', '.join(self.classifier.modes)}")
        except Exception as exc:
            self.classifier, self.model_error = None, str(exc)
            print("[cloud] CANH BAO:", exc)
        self.marks = None
        if config.MARK_MODEL_PATH.exists():
            try:
                self.marks = MarkClassifier()
                print(f"[cloud] Da nap mo hinh dau dong: {', '.join(self.marks.metadata['labels'])}")
            except Exception as exc:
                print("[cloud] CANH BAO mo hinh dau dong:", exc)

    @property
    def available_modes(self):
        return self.classifier.modes if self.classifier else []

    def resolve_mode(self, mode: str) -> str:
        modes = self.available_modes
        if mode in modes:
            return mode
        return config.DEFAULT_MODE if config.DEFAULT_MODE in modes else (modes[0] if modes else mode)

    def device(self, device_id: str) -> DeviceState:
        with self.lock:
            if device_id not in self.devices:
                self.devices[device_id] = DeviceState(self.resolve_mode(config.DEFAULT_MODE))
            return self.devices[device_id]

    # ------------------------------------------------------------- pipeline
    def on_mqtt_message(self, topic: str, raw: bytes):
        received = now_ms()
        try:
            payload = json.loads(raw)
        except ValueError:
            return
        if topic == config.TOPIC_SENSOR:
            try:
                result = self.process(payload, received)
            except (KeyError, TypeError, ValueError) as exc:
                print("[cloud] Goi tin loi:", exc)
                return
            if result:
                compact = {k: v for k, v in result.items() if k not in ("sensors", "top3")}
                self.mqtt.publish(config.TOPIC_PREDICTION, json.dumps(compact, ensure_ascii=False))
                self.broadcast(result)
        elif topic == config.TOPIC_STATUS:
            self.edge_status[payload.get("device_id", "?")] = {**payload, "received_ms": received}
            self.broadcast({"type": "edge_status", **payload})
        elif topic == config.TOPIC_CONTROL:
            self.control(payload.get("device_id", config.DEVICE_ID), payload.get("action"), payload.get("mode"),
                         payload.get("value"))

    def process(self, payload: dict, received_ms: float, store: bool = True, stabilize: bool = True,
                mode: str = None):
        if self.classifier is None:
            return None
        device_id = payload.get("device_id", "unknown")
        seq_id = int(payload.get("seq_id", 0))
        edge_ts = float(payload.get("edge_timestamp_ms", received_ms))
        state = self.device(device_id) if stabilize else None
        mode = self.resolve_mode(mode or payload.get("mode") or (state.mode if state else config.DEFAULT_MODE))
        pred = self.classifier.predict(payload_to_features(payload), mode)
        label, conf = pred["label"], pred["confidence"]

        stable, speak, speak_text, added, mark = None, False, None, None, None
        spelling, hold, action = None, None, None
        if stabilize:
            self.stats.record(device_id, seq_id, received_ms)
            with self.lock:
                state.last_ms, state.last_edge_ms = received_ms, edge_ts
                stab = state.stabilizers[mode]
                if mode == "letters":
                    stable, is_new, (cand, progress) = stab.update(label, conf, edge_ts,
                                                                   self._hand_moving(state, payload, edge_ts))
                    if cand:
                        hold = {"label": cand, "text": config.GESTURES[cand][0], "progress": round(progress, 2),
                                "hold_ms": stab.hold_ms}
                    if cand == config.POSE_LABEL and progress * stab.hold_ms >= config.MARK_POSE_READY_MS:
                        state.pose_ready_ms = edge_ts
                    mark = self._track_stroke(state, payload, pred, edge_ts)
                    drawing = state.segmenter.active and (
                        edge_ts - state.pose_ready_ms <= config.MARK_POSE_WINDOW_MS + config.MARK_MAX_STROKE_MS)
                    if stable and drawing:
                        if is_new:
                            stab.release()
                    elif stable:
                        state.speller.note_stable()
                        kind = config.LABEL_KIND.get(stable)
                        if is_new and kind == "letter":
                            added = config.GESTURES[stable][0]
                            state.speller.add(added)
                        elif is_new and kind == "mark" and state.speller.add_mark(stable):
                            mark = {"label": stable, "text": config.GESTURES[stable][0], "confidence": conf}
                        elif kind == "control" and (is_new or (
                                stable == "XOA" and edge_ts - state.last_control_ms >= config.DELETE_REPEAT_MS)):
                            state.last_control_ms = edge_ts
                            action = config.CONTROL_ACTIONS[stable]
                    elif cand:
                        state.speller.note_stable()
                    else:
                        state.speller.note_unstable(edge_ts, stab)
                else:
                    stable, is_new = stab.update(label, conf, edge_ts)
                    if stable and is_new:
                        speak, speak_text = True, config.GESTURES[stable][0]
            if action:
                self.control(device_id, action, source="gesture")
            if mode == "letters":
                spelling = {**state.speller.state(), "added": added, "mark": mark, "action": action}

        truth = payload.get("ground_truth")
        cloud_total = now_ms() - received_ms
        result = {
            "type": "prediction",
            "device_id": device_id,
            "seq_id": seq_id,
            "mode": mode,
            "label": label,
            "text": config.GESTURES[label][0],
            "confidence": round(conf, 4),
            "top3": pred["top3"],
            "stable_label": stable,
            "stable_text": config.GESTURES[stable][0] if stable else None,
            "speak": speak,
            "speak_text": speak_text,
            "spelling": spelling,
            "hold": hold,
            "ground_truth": truth,
            "edge_timestamp_ms": edge_ts,
            "edge_processing_ms": payload.get("edge_processing_ms"),
            "network_latency_ms": round(received_ms - edge_ts, 2),
            "ai_inference_ms": round(pred["inference_ms"], 3),
            "cloud_total_ms": round(cloud_total, 3),
            "cloud_send_ms": now_ms(),
            "sensors": {"flex": payload.get("flex_sensors"), "imu": payload.get("imu")},
        }
        if store:
            flex, imu = payload["flex_sensors"], payload["imu"]
            self.db.insert_log({
                "device_id": device_id, "seq_id": seq_id, "edge_ts_ms": edge_ts,
                "cloud_receive_ms": received_ms, "edge_processing_ms": payload.get("edge_processing_ms"),
                "network_latency_ms": result["network_latency_ms"], "ai_inference_ms": result["ai_inference_ms"],
                "cloud_total_ms": result["cloud_total_ms"], "label": label, "confidence": conf,
                "stable_label": stable, "ground_truth": truth,
                "correct": None if truth is None else int(truth == label),
                "flex_thumb": flex["thumb"], "flex_index": flex["index"], "flex_middle": flex["middle"],
                "flex_ring": flex["ring"], "flex_pinky": flex["pinky"],
                "roll": imu["roll"], "pitch": imu["pitch"], "yaw": imu["yaw"],
                "ax": imu["ax"], "ay": imu["ay"], "az": imu["az"], "speed": imu["speed"],
                "payload": payload,
            })
            if speak:
                self.db.insert_translation(received_ms, device_id, stable, speak_text,
                                           conf, result["network_latency_ms"] + cloud_total)
        return result

    @staticmethod
    def _hand_moving(state: DeviceState, payload: dict, ts_ms: float) -> bool:
        """Ban tay dang di chuyen hoac cac ngon dang doi hinh dang -> dang chuyen chu, chua tinh."""
        shape = payload["shape"][:len(config.SHAPE_FEATURES)]
        prev, state.prev_shape = state.prev_shape, (ts_ms, shape)
        shape_speed = 0.0
        if prev is not None and 0 < ts_ms - prev[0] < 300:
            diff = sum((a - b) ** 2 for a, b in zip(shape, prev[1])) ** 0.5
            shape_speed = diff / ((ts_ms - prev[0]) / 1000.0)
        speed = float(payload.get("imu", {}).get("speed", 0.0))
        a = config.LETTER_MOTION_EMA
        state.motion = (a * speed + (1 - a) * state.motion[0], a * shape_speed + (1 - a) * state.motion[1])
        return state.motion[0] > config.LETTER_STILL_SPEED or state.motion[1] > config.LETTER_STILL_SHAPE

    def _track_stroke(self, state: DeviceState, payload: dict, pred: dict, ts_ms: float):
        """Theo doi quy dao ngon tro khi tay o tu the dau; tra ve dau dong vua ghep (neu co)."""
        pointer = payload.get("pointer")
        if self.marks is None or not pointer:
            return None
        pose_conf = next((t["confidence"] for t in pred["top3"] if t["label"] == config.POSE_LABEL), 0.0)
        pose = pose_conf >= config.MARK_POSE_CONF
        state.pose_lost = 0 if pose else state.pose_lost + 1
        if state.pose_lost >= 3:
            state.mark_armed = True
        x, y, palm = pointer
        stroke = state.segmenter.update(ts_ms / 1000.0, x, y, palm, index_direction(payload["shape"]), pose)
        return self._classify_stroke(state, stroke, ts_ms)

    def _classify_stroke(self, state: DeviceState, stroke, ts_ms: float):
        if stroke is None or (not state.mark_armed and ts_ms < state.mark_block_until):
            return None
        if stroke["t"][0] * 1000 - state.pose_ready_ms > config.MARK_POSE_WINDOW_MS:
            return None
        res = self.marks.predict(stroke)
        if res["confidence"] < config.MARK_MIN_CONFIDENCE or not state.speller.add_mark(res["label"]):
            return None
        state.mark_armed, state.mark_block_until = False, ts_ms + 1500
        return {**res, "text": config.GESTURES[res["label"]][0]}

    # ------------------------------------------------------------- dieu khien
    def control(self, device_id: str, action: str, mode: str = None, value=None, source: str = "user") -> dict:
        state = self.device(device_id)
        word = None
        with self.lock:
            if action == "set_mode" and mode in self.available_modes:
                state.mode = mode
                for stab in state.stabilizers.values():
                    stab.reset()
            elif action == "set_hold" and value is not None:
                lo, hi = config.LETTER_HOLD_RANGE
                state.stabilizers["letters"].hold_ms = float(min(hi, max(lo, float(value))))
            elif action == "toggle_mode" and len(self.available_modes) > 1:
                modes = self.available_modes
                state.mode = modes[(modes.index(state.mode) + 1) % len(modes)] if state.mode in modes else modes[0]
            elif action == "space":
                word = state.speller.commit()
            elif action == "backspace":
                state.speller.backspace()
            elif action == "clear":
                state.speller.clear()
            spelling = state.speller.state()
            current_mode = state.mode
        if word:
            self.db.insert_translation(now_ms(), device_id, "SPELL", word, 1.0, None)
        message = {"type": "state", "device_id": device_id, "mode": current_mode, "event": action,
                   "source": source, "word": word, "spelling": spelling,
                   "hold_ms": state.stabilizers["letters"].hold_ms}
        self.broadcast(message)
        self.mqtt.publish(config.TOPIC_PREDICTION, json.dumps(message, ensure_ascii=False))
        return message

    async def spelling_loop(self):
        """Bo tay khoi khung hinh du lau -> tu dong ket thuc tu dang danh van va doc to."""
        while True:
            await asyncio.sleep(0.25)
            t = now_ms()
            for device_id, state in list(self.devices.items()):
                if state.segmenter.active and t - state.last_ms > 400:
                    with self.lock:
                        mark = self._classify_stroke(state, state.segmenter.flush(), state.last_edge_ms)
                    if mark:
                        message = {"type": "state", "device_id": device_id, "mode": state.mode,
                                   "event": "mark", "word": None, "mark": mark, "spelling": state.speller.state()}
                        self.broadcast(message)
                        self.mqtt.publish(config.TOPIC_PREDICTION, json.dumps(message, ensure_ascii=False))
                if (state.mode == "letters" and state.speller.current
                        and t - state.last_ms > config.SPELL_AUTO_COMMIT_MS):
                    await asyncio.to_thread(self.control, device_id, "space")

    # ------------------------------------------------------------- websocket
    def broadcast(self, message: dict):
        if self.loop is None or not self.clients:
            return
        text = json.dumps(message, ensure_ascii=False)
        asyncio.run_coroutine_threadsafe(self._send_all(text), self.loop)

    async def _send_all(self, text: str):
        for ws in list(self.clients):
            try:
                await ws.send_text(text)
            except Exception:
                self.clients.discard(ws)

    def mqtt_snapshot(self) -> dict:
        snap = self.stats.snapshot()
        snap.update({
            "ts_ms": now_ms(),
            "connected": self.mqtt.connected.is_set(),
            "reconnects": self.mqtt.reconnects,
            "broker": f"{config.MQTT_HOST}:{config.MQTT_PORT}",
            "e2e_live": _percentiles(list(self.e2e_live)),
        })
        return snap

    def device_modes(self) -> dict:
        return {d: s.mode for d, s in self.devices.items()}

    async def metrics_loop(self):
        while True:
            await asyncio.sleep(2)
            snap = self.mqtt_snapshot()
            self.db.insert_mqtt_metric(snap)
            await self._send_all(json.dumps({"type": "metrics", **snap}, ensure_ascii=False))


cloud = CloudApp()
edge_proc = EdgeProcess()


@asynccontextmanager
async def lifespan(app: FastAPI):
    cloud.loop = asyncio.get_running_loop()
    ok = await asyncio.to_thread(cloud.mqtt.start, 10)
    print(f"[cloud] MQTT {config.MQTT_HOST}:{config.MQTT_PORT} -> {'da ket noi' if ok else 'CHUA ket noi (se thu lai)'}")
    tasks = [asyncio.create_task(cloud.metrics_loop()), asyncio.create_task(cloud.spelling_loop())]
    print(f"[cloud] Dashboard: {config.CLOUD_URL}")
    yield
    for task in tasks:
        task.cancel()
    edge_proc.stop()
    cloud.mqtt.stop()
    cloud.db.close()


app = FastAPI(title="IoT Sign Language Cloud", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(config.DASHBOARD_DIR)), name="static")
config.CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/captures", StaticFiles(directory=str(config.CAPTURE_DIR)), name="captures")


@app.get("/")
def index():
    return FileResponse(config.DASHBOARD_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "mqtt_connected": cloud.mqtt.connected.is_set(),
            "model_loaded": cloud.classifier is not None, "modes": cloud.available_modes,
            "marks_loaded": cloud.marks is not None, "model_error": cloud.model_error}


@app.get("/api/gestures")
def gestures():
    return [{"label": k, "text": v[0], "description": v[1], "mode": config.LABEL_MODE[k],
             "kind": config.LABEL_KIND[k]} for k, v in config.GESTURES.items()]


@app.post("/api/predict")
def predict(payload: dict, mode: str = None, store: bool = False):
    """Cloud AI API: nhan goi JSON cam bien, tra ve nhan cu chi (dung cho tich hop HTTP va benchmark)."""
    if cloud.classifier is None:
        raise HTTPException(503, cloud.model_error)
    if mode and mode not in cloud.available_modes:
        raise HTTPException(400, f"Che do '{mode}' chua co mo hinh. Co: {cloud.available_modes}")
    try:
        result = cloud.process(payload, now_ms(), store=store, stabilize=False, mode=mode)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(422, f"Payload khong hop le: {exc}")
    result.pop("sensors", None)
    return result


@app.post("/api/control")
def control(body: dict):
    return cloud.control(body.get("device_id", config.DEVICE_ID), body.get("action"), body.get("mode"),
                         body.get("value"))


@app.get("/api/edge/status")
def edge_status():
    return edge_proc.status()


@app.post("/api/edge/start")
def edge_start(source: str = "camera"):
    if source not in ("camera", "synthetic"):
        raise HTTPException(400, "source phai la camera hoac synthetic")
    return edge_proc.start(source)


@app.post("/api/edge/stop")
def edge_stop():
    return edge_proc.stop()


@app.post("/api/reload_model")
def reload_model():
    cloud.load_model()
    return health()


@app.get("/api/stats")
def stats():
    summary = cloud.db.summary()
    summary["mqtt"] = cloud.mqtt_snapshot()
    summary["model"] = cloud.classifier.metadata if cloud.classifier else None
    summary["mark_model"] = cloud.marks.metadata if cloud.marks else None
    summary["edge_status"] = cloud.edge_status
    summary["device_modes"] = cloud.device_modes()
    return summary


@app.get("/api/history")
def history(limit: int = 100):
    return cloud.db.recent_translations(min(max(limit, 1), 1000))


@app.get("/api/logs")
def logs(limit: int = 100):
    return cloud.db.recent_logs(min(max(limit, 1), 1000))


@app.get("/api/dataset_summary")
def dataset_summary():
    """Thong ke du lieu thu thap (camera_raw.csv + dataset/images) cho tab Quan ly Data."""
    import csv
    sessions_by_label = {}
    frames_by_label = {}
    total_frames = 0
    all_sessions = set()
    if config.CAMERA_RAW_PATH.exists():
        with open(config.CAMERA_RAW_PATH, encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for r in reader:
                lb = r.get("label")
                ss = r.get("session")
                if not lb:
                    continue
                total_frames += 1
                frames_by_label[lb] = frames_by_label.get(lb, 0) + 1
                if ss:
                    all_sessions.add(ss)
                    sessions_by_label.setdefault(lb, set()).add(ss)
    images_by_label = {}
    recent_images = []
    if config.CAPTURE_INDEX_PATH.exists():
        with open(config.CAPTURE_INDEX_PATH, encoding="utf-8", errors="ignore") as f:
            rows = list(csv.DictReader(f))
            for r in rows:
                lb = r.get("label", "")
                images_by_label[lb] = images_by_label.get(lb, 0) + 1
            for r in rows[-24:][::-1]:
                recent_images.append(r)
    items = []
    for lb, (txt, desc) in config.GESTURES.items():
        if lb == config.POSE_LABEL:
            continue
        items.append({
            "label": lb,
            "text": txt,
            "description": desc,
            "mode": config.LABEL_MODE.get(lb, "letters"),
            "kind": config.LABEL_KIND.get(lb, "letter"),
            "sessions": len(sessions_by_label.get(lb, ())),
            "frames": frames_by_label.get(lb, 0),
            "images": images_by_label.get(lb, 0),
        })
    syn_rows = 0
    if config.SYNTHETIC_DATASET_PATH.exists():
        with open(config.SYNTHETIC_DATASET_PATH, encoding="utf-8", errors="ignore") as f:
            syn_rows = max(0, sum(1 for _ in f) - 1)
    return {
        "camera_sessions": len(all_sessions),
        "camera_frames": total_frames,
        "total_images": sum(images_by_label.values()),
        "synthetic_rows": syn_rows,
        "items": items,
        "recent_images": recent_images,
    }


@app.get("/api/mqtt_history")
def mqtt_history(limit: int = 120):
    return cloud.db.mqtt_history(limit)


@app.post("/api/reset")
def reset():
    cloud.db.reset()
    cloud.e2e_live.clear()
    cloud.stats = StreamStats()
    return {"status": "reset"}


def _tts_file(text: str) -> Path:
    config.TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TTS_CACHE_DIR / f"{hashlib.md5(text.encode()).hexdigest()}.mp3"
    if not path.exists():
        try:
            from gtts import gTTS
            gTTS(text, lang="vi").save(str(path))
        except Exception as exc:
            raise HTTPException(503, f"Khong tao duoc giong noi: {exc}")
    return path


@app.get("/api/tts/{label}")
def tts(label: str):
    """Tra ve file mp3 giong noi tieng Viet (gTTS, can Internet lan dau, sau do dung cache)."""
    if label not in config.GESTURES:
        raise HTTPException(404, "Khong co cu chi nay")
    return FileResponse(_tts_file(config.GESTURES[label][0]), media_type="audio/mpeg")


@app.get("/api/tts_text")
def tts_text(text: str):
    text = text.strip()[:200]
    if not text:
        raise HTTPException(400, "Thieu noi dung")
    return FileResponse(_tts_file(text), media_type="audio/mpeg")


@app.websocket("/ws")
async def websocket(ws: WebSocket):
    await ws.accept()
    cloud.clients.add(ws)
    await ws.send_text(json.dumps({
        "type": "hello",
        "gestures": {k: v[0] for k, v in config.GESTURES.items()},
        "modes": {m: {"name": info["name"], "available": m in cloud.available_modes}
                  for m, info in config.MODES.items()},
        "default_device": config.DEVICE_ID,
        "device_modes": cloud.device_modes(),
        "default_mode": cloud.resolve_mode(config.DEFAULT_MODE),
        "hold_ms": cloud.device(config.DEVICE_ID).stabilizers["letters"].hold_ms,
        "hold_range": config.LETTER_HOLD_RANGE,
        "model": cloud.classifier.metadata if cloud.classifier else None,
        "latency_target_ms": config.LATENCY_TARGET_MS,
        "mqtt": cloud.mqtt_snapshot(),
    }, ensure_ascii=False))
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get("type") == "ack" and msg.get("e2e_ms") is not None:
                e2e = float(msg["e2e_ms"])
                cloud.e2e_live.append(e2e)
                cloud.db.update_e2e(msg.get("device_id"), int(msg.get("seq_id", -1)), e2e)
            elif msg.get("type") == "control":
                await asyncio.to_thread(cloud.control, msg.get("device_id") or config.DEVICE_ID,
                                        msg.get("action"), msg.get("mode"), msg.get("value"))
    except (WebSocketDisconnect, RuntimeError, ValueError):
        pass
    finally:
        cloud.clients.discard(ws)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.CLOUD_PORT, log_level="warning")


if __name__ == "__main__":
    main()
