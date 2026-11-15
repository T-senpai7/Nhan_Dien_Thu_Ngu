"""Nhan dien dau dong (trang, sac, huyen, hoi, nga, nang) tu quy dao dau ngon tro.

Dung chung cho train (phat lai du lieu camera) va Cloud thoi gian thuc:
  StrokeSegmenter  - cat ra mot net ve dua tren van toc dau ngon (don vi: long ban tay / s)
  stroke_features  - vector dac trung co dinh do dai cho mot net
  MarkClassifier   - mo hinh da train (cloud/models/mark_classifier.pkl)
"""
from collections import deque

import joblib
import numpy as np

import config

RESAMPLE_POINTS = 12
V_START = 1.4
V_END = 0.5
END_RATIO = 0.25
END_HOLD_SEC = 0.15
MIN_DURATION_SEC = 0.15
MAX_DURATION_SEC = 2.0
MIN_PATH = 0.45
MAX_GAP_SEC = 0.3
PRE_ROLL = 3

FEATURE_NAMES = ([f"rx{i}" for i in range(RESAMPLE_POINTS)] + [f"ry{i}" for i in range(RESAMPLE_POINTS)] + [
    "disp_x", "disp_y", "path", "duration", "straightness", "width", "height",
    "rev_x", "rev_y", "turn", "abs_turn",
    "dir_mean_x", "dir_mean_y", "dir_mean_z", "dir_start_x", "dir_start_y", "dir_start_z",
])


def index_direction(shape) -> np.ndarray:
    """Huong ngon tro (khop 5 -> dau ngon 8) tu vector hinh dang ban tay."""
    p5 = np.asarray(shape[12:15], dtype=float)
    p8 = np.asarray(shape[21:24], dtype=float)
    d = p8 - p5
    n = np.linalg.norm(d)
    return d / n if n > 1e-6 else d


class StrokeSegmenter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.frames = []
        self.pre = deque(maxlen=PRE_ROLL)
        self.prev = None
        self.speed = 0.0
        self.peak = 0.0
        self.still_since = None

    @property
    def active(self) -> bool:
        return bool(self.frames)

    def update(self, t: float, x: float, y: float, palm: float, direction, pose: bool):
        """Them 1 khung; tra ve net ve vua ket thuc (dict) hoac None."""
        done = None
        if self.prev is not None and not 0 < t - self.prev[0] <= MAX_GAP_SEC:
            done = self._finish()
            self.pre.clear()
            self.prev, self.speed = None, 0.0
        if self.prev is not None:
            pt, px, py, ppalm = self.prev
            v = np.hypot(x - px, y - py) / (t - pt) / max(1e-6, (palm + ppalm) / 2)
            self.speed = 0.5 * v + 0.5 * self.speed
        frame = (t, x, y, palm, np.asarray(direction, dtype=float), bool(pose))
        self.prev = (t, x, y, palm)

        if not self.frames:
            if self.speed > V_START and pose:
                self.frames = list(self.pre) + [frame]
                self.still_since, self.peak = None, self.speed
            else:
                self.pre.append(frame)
            return done

        self.frames.append(frame)
        self.peak = max(self.peak, self.speed)
        if t - self.frames[0][0] > MAX_DURATION_SEC:
            self.frames = []
            return done
        if self.speed < max(V_END, END_RATIO * self.peak):
            self.still_since = self.still_since or t
            if t - self.still_since >= END_HOLD_SEC:
                return self._finish() or done
        else:
            self.still_since = None
        return done

    def flush(self):
        out = self._finish()
        self.reset()
        return out

    def _finish(self):
        frames, self.frames = self.frames, []
        self.still_since = None
        if len(frames) < 4:
            return None
        t = np.array([f[0] for f in frames])
        xy = np.array([[f[1], f[2]] for f in frames])
        palm = float(np.median([f[3] for f in frames]))
        path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) / palm
        pose_ratio = float(np.mean([f[5] for f in frames]))
        if not MIN_DURATION_SEC <= t[-1] - t[0] <= MAX_DURATION_SEC or path < MIN_PATH or pose_ratio < 0.5:
            return None
        return {"t": t, "xy": xy, "palm": palm, "dirs": np.array([f[4] for f in frames]), "path": path}


def stroke_features(stroke: dict) -> list:
    p = (stroke["xy"] - stroke["xy"][0]) / stroke["palm"]
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    path = float(cum[-1]) or 1e-6
    targets = np.linspace(0, path, RESAMPLE_POINTS + 1)[1:]
    rx, ry = np.interp(targets, cum, p[:, 0]), np.interp(targets, cum, p[:, 1])

    steps = np.diff(np.column_stack([np.r_[0.0, rx], np.r_[0.0, ry]]), axis=0)
    def reversals(v):
        s = np.sign(v[np.abs(v) > 0.02 * path])
        return float(np.sum(s[1:] != s[:-1])) if len(s) > 1 else 0.0
    heading = np.arctan2(steps[:, 1], steps[:, 0])
    dturn = (np.diff(heading) + np.pi) % (2 * np.pi) - np.pi
    dirs = stroke["dirs"]
    dir_mean = dirs.mean(axis=0)
    dir_start = dirs[:3].mean(axis=0)
    disp = p[-1]
    return [*rx, *ry,
            float(disp[0]), float(disp[1]), path, float(stroke["t"][-1] - stroke["t"][0]),
            float(np.linalg.norm(disp) / path), float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1])),
            reversals(steps[:, 0]), reversals(steps[:, 1]), float(dturn.sum()), float(np.abs(dturn).sum()),
            *dir_mean, *dir_start]


class MarkClassifier:
    def __init__(self, path=config.MARK_MODEL_PATH):
        bundle = joblib.load(path)
        if bundle["metadata"]["features"] != FEATURE_NAMES:
            raise ValueError("Mo hinh dau thanh khong khop dac trung hien tai, hay train lai.")
        self.model = bundle["model"]
        self.metadata = {k: v for k, v in bundle["metadata"].items() if k not in ("features", "candidates")}

    def predict(self, stroke: dict) -> dict:
        proba = self.model.predict_proba(np.asarray(stroke_features(stroke), dtype=float).reshape(1, -1))[0]
        i = int(np.argmax(proba))
        return {"label": str(self.model.classes_[i]), "confidence": float(proba[i])}
