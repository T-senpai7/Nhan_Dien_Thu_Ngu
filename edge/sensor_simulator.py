"""Mo phong cam bien gang tay thong minh tu 21 khop tay cua MediaPipe.

- 5 Flex sensor: goc uon tung ngon (0 = duoi thang, 180 = gap sat long ban tay).
- IMU 6-DOF: goc nghieng ban tay (roll, pitch, yaw), van toc va gia toc cua tam ban tay
  (don vi: chieu dai long ban tay / s va / s^2 nen khong phu thuoc khoang cach toi camera).
- Cam bien tiep xuc (thumb-index) va dac trung chuyen dong theo cua so thoi gian.

Dau vao landmarks: mang (21, 3) da duoc dua ve cung ti le truc x/y (nhan x voi width/height).
"""
from collections import deque

import numpy as np

import config

FINGER_CHAINS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}
# Tong goc gap toi da (do) cua 3 khop ung voi ngon gap hoan toan.
FINGER_MAX_BEND = {"thumb": 180.0, "index": 250.0, "middle": 250.0, "ring": 250.0, "pinky": 250.0}


def _bend(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Goc lech (do) cua doan b->c so voi doan a->b; 0 khi ba diem thang hang."""
    u, v = b - a, c - b
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denom, -1.0, 1.0))))


def flex_angles(lm: np.ndarray) -> dict:
    result = {}
    for name, chain in FINGER_CHAINS.items():
        total = sum(_bend(lm[chain[i]], lm[chain[i + 1]], lm[chain[i + 2]]) for i in range(3))
        result[name] = float(np.clip(total * 180.0 / FINGER_MAX_BEND[name], 0.0, 180.0))
    return result


def orientation(lm: np.ndarray) -> tuple:
    direction = lm[9] - lm[0]
    dx, dy, dz = direction
    roll = np.degrees(np.arctan2(dx, -dy))
    pitch = np.degrees(np.arctan2(-dz, np.hypot(dx, dy)))
    normal = np.cross(lm[5] - lm[0], lm[17] - lm[0])
    yaw = np.degrees(np.arctan2(normal[0], -normal[2]))
    return float(roll), float(pitch), float(yaw)


def canonical_hand(landmarks, handedness: str = "Right") -> np.ndarray:
    """Lat guong tay trai thanh tay phai de mot mo hinh dung cho ca hai tay."""
    lm = np.array(landmarks, dtype=float).reshape(21, 3)
    if handedness == "Left":
        lm[:, 0] *= -1
    return lm


def shape_features(lm: np.ndarray, palm: float) -> list:
    """Vector hinh dang ban tay theo thu tu config.SHAPE_FEATURES + config.DISTANCE_FEATURES.

    Toa do lay goc o co tay va chia cho chieu dai long ban tay (khong phu thuoc khoang cach
    toi camera), nhung giu nguyen huong xoay vi nhieu chu cai chi khac nhau o huong tay (G/Q, K/P).
    """
    rel = (lm - lm[0]) / palm
    dist = lambda a, b: float(np.linalg.norm(rel[a] - rel[b]))
    values = rel[1:].reshape(-1).tolist()
    values += [dist(4, j) for j in config.THUMB_TARGETS]
    side = np.sign(rel[5, 0] - rel[9, 0]) or 1.0
    values += [
        dist(8, 12), dist(12, 16), dist(16, 20), dist(8, 20),
        float((rel[8, 0] - rel[12, 0]) * side),
        dist(8, 5), dist(12, 9),
    ]
    return values


DERIV_BASELINE_SEC = 1.0 / 30
REFERENCE_FRAME_SEC = 1.0 / 30


class SensorSimulator:
    """Giu trang thai theo thoi gian de tinh van toc, gia toc va dao dong.

    Dao ham duoc lay tren moc thoi gian co dinh (~33 ms) va bo loc EMA theo thoi gian,
    nen dac trung IMU khong phu thuoc toc do khung hinh cua camera / tan suat Edge.
    """

    def __init__(self, window_sec: float = config.MOTION_WINDOW_SEC, alpha: float = config.SMOOTHING_ALPHA):
        self.window_sec = window_sec
        self.alpha = alpha
        self.reset()

    def reset(self):
        self._prev_t = None
        self._history = deque()
        self._velocity = np.zeros(3)
        self._accel = np.zeros(3)
        self._palm = None
        self._window = deque()

    def _update_motion(self, center: np.ndarray, t: float):
        if self._prev_t is None or not 0 < t - self._prev_t < 0.3:
            self._history.clear()
            self._window.clear()
            self._velocity = np.zeros(3)
            self._accel = np.zeros(3)
        else:
            ref = next((h for h in reversed(self._history) if t - h[0] >= DERIV_BASELINE_SEC), None)
            if ref is not None:
                dt = t - ref[0]
                velocity = (center - ref[1]) / dt
                accel = (velocity - ref[2]) / dt
                frame_dt = t - self._prev_t
                a = 1 - (1 - self.alpha) ** (frame_dt / REFERENCE_FRAME_SEC)
                self._velocity = a * velocity + (1 - a) * self._velocity
                self._accel = np.clip(a * accel + (1 - a) * self._accel, -50, 50)
        self._prev_t = t
        self._history.append((t, center, self._velocity.copy()))
        while self._history and t - self._history[0][0] > 0.3:
            self._history.popleft()

    def process(self, landmarks: np.ndarray, t: float, handedness: str = "Right") -> dict:
        lm = canonical_hand(landmarks, handedness)
        palm_now = float(np.linalg.norm(lm[9] - lm[0])) or 1e-6
        self._palm = palm_now if self._palm is None else 0.8 * self._palm + 0.2 * palm_now
        center = (lm[0] + lm[9]) / 2.0 / self._palm
        self._update_motion(center, t)

        flex = flex_angles(lm)
        roll, pitch, yaw = orientation(lm)
        flex_mean = float(np.mean([flex[k] for k in ("index", "middle", "ring", "pinky")]))

        self._window.append((t, center[0], center[1], flex_mean))
        while self._window and t - self._window[0][0] > self.window_sec:
            self._window.popleft()
        win = np.array(self._window)

        return {
            "handedness": handedness,
            "shape": shape_features(lm, palm_now),
            "pointer": (float(lm[8, 0]), float(lm[8, 1]), palm_now),
            "flex_thumb": flex["thumb"],
            "flex_index": flex["index"],
            "flex_middle": flex["middle"],
            "flex_ring": flex["ring"],
            "flex_pinky": flex["pinky"],
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "vx": float(self._velocity[0]),
            "vy": float(self._velocity[1]),
            "vz": float(self._velocity[2]),
            "ax": float(self._accel[0]),
            "ay": float(self._accel[1]),
            "az": float(self._accel[2]),
            "speed": float(np.linalg.norm(self._velocity[:2])),
            "touch_thumb_index": float(np.linalg.norm(lm[4] - lm[8]) / self._palm),
            "osc_x": float(np.std(win[:, 1])),
            "osc_y": float(np.std(win[:, 2])),
            "flex_osc": float(np.std(win[:, 3])),
        }


def build_payload(features: dict, device_id: str, seq_id: int, edge_timestamp_ms: float,
                  edge_processing_ms: float, ground_truth: str = None) -> dict:
    r = lambda v: round(v, 3)
    payload = {
        "device_id": device_id,
        "seq_id": seq_id,
        "edge_timestamp_ms": round(edge_timestamp_ms, 2),
        "edge_processing_ms": round(edge_processing_ms, 2),
        "flex_sensors": {
            "thumb": r(features["flex_thumb"]),
            "index": r(features["flex_index"]),
            "middle": r(features["flex_middle"]),
            "ring": r(features["flex_ring"]),
            "pinky": r(features["flex_pinky"]),
        },
        "imu": {
            "roll": r(features["roll"]), "pitch": r(features["pitch"]), "yaw": r(features["yaw"]),
            "ax": r(features["ax"]), "ay": r(features["ay"]), "az": r(features["az"]),
            "vx": r(features["vx"]), "vy": r(features["vy"]), "vz": r(features["vz"]),
            "speed": r(features["speed"]),
        },
        "contact": {"thumb_index": r(features["touch_thumb_index"])},
        "motion": {
            "osc_x": r(features["osc_x"]), "osc_y": r(features["osc_y"]),
            "flex_osc": r(features["flex_osc"]),
        },
        "hand": {"handedness": features["handedness"]},
        "shape": [round(v, 4) for v in features["shape"]],
        # Dau ngon tro (toa do anh) + chieu dai long ban tay: de Cloud ve quy dao dau thanh.
        "pointer": [round(v, 5) for v in features["pointer"]],
    }
    if ground_truth:
        payload["ground_truth"] = ground_truth
    return payload


def payload_to_features(payload: dict) -> list:
    """Chuyen goi JSON ve vector dac trung theo thu tu config.FEATURE_COLUMNS."""
    flex, imu = payload["flex_sensors"], payload["imu"]
    contact, motion = payload.get("contact", {}), payload.get("motion", {})
    flat = {
        "flex_thumb": flex["thumb"], "flex_index": flex["index"], "flex_middle": flex["middle"],
        "flex_ring": flex["ring"], "flex_pinky": flex["pinky"],
        "roll": imu["roll"], "pitch": imu["pitch"], "yaw": imu["yaw"],
        "ax": imu["ax"], "ay": imu["ay"], "az": imu["az"], "speed": imu["speed"],
        "touch_thumb_index": contact.get("thumb_index", 1.0),
        "osc_x": motion.get("osc_x", 0.0), "osc_y": motion.get("osc_y", 0.0),
        "flex_osc": motion.get("flex_osc", 0.0),
    }
    shape = payload["shape"]
    expected = len(config.SHAPE_FEATURES) + len(config.DISTANCE_FEATURES)
    if len(shape) != expected:
        raise ValueError(f"shape co {len(shape)} gia tri, can {expected}")
    return [float(flat[c]) for c in config.SENSOR_FEATURES] + [float(v) for v in shape]


def features_to_vector(features: dict) -> list:
    """Vector dac trung truc tiep tu ket qua SensorSimulator.process (dung khi train)."""
    return [float(features[c]) for c in config.SENSOR_FEATURES] + [float(v) for v in features["shape"]]
