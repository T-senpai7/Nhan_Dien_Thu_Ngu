"""Dich vu Cloud AI: nap mo hinh pre-trained theo che do, phan loai va ghep chu thanh tu."""
import time
from collections import Counter, deque

import joblib
import numpy as np

import config
from cloud.vietnamese import apply_mark


class GestureClassifier:
    def __init__(self, model_path=config.MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Chua co mo hinh tai {model_path}. Chay: python cloud/train_model.py")
        bundle = joblib.load(model_path)
        if "modes" not in bundle:
            raise ValueError("Mo hinh dinh dang cu, hay train lai: python cloud/train_model.py")
        self.models = {}
        for mode, item in bundle["modes"].items():
            if item["metadata"]["features"] != config.FEATURE_COLUMNS:
                raise ValueError(f"Mo hinh '{mode}' khong khop dac trung hien tai, hay train lai.")
            self.models[mode] = item
        self.metadata = {mode: {k: v for k, v in item["metadata"].items() if k not in ("features", "candidates")}
                         for mode, item in self.models.items()}

    @property
    def modes(self):
        return list(self.models)

    def predict(self, features, mode: str) -> dict:
        model = self.models[mode]["model"]
        start = time.perf_counter()
        proba = model.predict_proba(np.asarray(features, dtype=float).reshape(1, -1))[0]
        inference_ms = (time.perf_counter() - start) * 1000
        labels = model.classes_
        order = np.argsort(proba)[::-1]
        return {
            "label": str(labels[order[0]]),
            "confidence": float(proba[order[0]]),
            "top3": [{"label": str(labels[i]), "text": config.GESTURES[labels[i]][0],
                      "confidence": round(float(proba[i]), 4)} for i in order[:3]],
            "inference_ms": inference_ms,
        }


class PredictionStabilizer:
    """Bo phieu da so tren N du doan gan nhat + nguong tin cay de chong doc lap / nhay chu."""

    def __init__(self, window: int, threshold: float, reset_ms: float = config.HAND_LOST_RESET_MS):
        self.window, self.threshold, self.reset_ms = window, threshold, reset_ms
        self.history = deque(maxlen=window)
        self.last_spoken = None
        self.last_ts = None

    def update(self, label: str, confidence: float, ts_ms: float):
        """Tra ve (stable_label | None, is_new: bool)."""
        if self.last_ts is not None and ts_ms - self.last_ts > self.reset_ms:
            self.history.clear()
            self.last_spoken = None
        self.last_ts = ts_ms
        self.history.append((label, confidence))
        if len(self.history) < self.window:
            return None, False
        top, votes = Counter(l for l, _ in self.history).most_common(1)[0]
        mean_conf = np.mean([c for l, c in self.history if l == top])
        if votes < int(self.window * 0.7 + 0.5) or mean_conf < self.threshold:
            return None, False
        is_new = top != self.last_spoken
        if is_new:
            self.last_spoken = top
        return top, is_new

    def release(self):
        """Cho phep cung mot chu duoc nhan lai (vd danh van 'OO')."""
        self.last_spoken = None

    def reset(self):
        self.history.clear()
        self.last_spoken = None


class HoldStabilizer:
    """Chu cai chi duoc chap nhan khi tay giu yen cung mot tu the trong hold_ms.

    Bo phieu tren vai khung gan nhat loc nhieu 1-2 khung; khung nao tay dang di chuyen nhanh
    (dang chuyen sang chu khac) thi khong tinh, nen tu the trung gian khong bi nhan thanh chu.
    """

    def __init__(self, hold_ms: float, threshold: float, vote_window: int = 5,
                 reset_ms: float = config.HAND_LOST_RESET_MS):
        self.hold_ms, self.threshold, self.reset_ms = hold_ms, threshold, reset_ms
        self.history = deque(maxlen=vote_window)
        self.last_spoken = None
        self.last_ts = None
        self.candidate, self.since = None, None

    def update(self, label: str, confidence: float, ts_ms: float, moving: bool = False):
        """Tra ve (stable_label | None, is_new, (ung_vien, tien_do 0..1))."""
        if self.last_ts is not None and ts_ms - self.last_ts > self.reset_ms:
            self.history.clear()
            self.last_spoken = None
        self.last_ts = ts_ms
        self.history.append((None if moving else label, confidence))
        top, votes = Counter(l for l, _ in self.history).most_common(1)[0]
        mean_conf = np.mean([c for l, c in self.history if l == top])
        need = max(3, int(np.ceil(self.history.maxlen * 0.8)))
        candidate = top if top is not None and votes >= need and mean_conf >= self.threshold else None
        if candidate != self.candidate:
            self.candidate, self.since = candidate, ts_ms
        if candidate is None:
            return None, False, (None, 0.0)
        progress = min(1.0, (ts_ms - self.since) / self.hold_ms)
        if progress < 1.0:
            return None, False, (candidate, progress)
        is_new = candidate != self.last_spoken
        if is_new:
            self.last_spoken = candidate
        return candidate, is_new, (candidate, 1.0)

    def release(self):
        self.last_spoken = None

    def reset(self):
        self.history.clear()
        self.candidate, self.since, self.last_spoken = None, None, None


class Speller:
    """Ghep cac chu cai on dinh thanh tu, tu thanh cau."""

    def __init__(self):
        self.current = ""
        self.words = []
        self.unstable_since = None

    def add(self, letter: str):
        self.current += letter

    def add_mark(self, mark: str) -> bool:
        """Ghep dau phu / dau thanh vao tu dang danh van; False neu khong ap dung duoc."""
        new = apply_mark(self.current, mark)
        if new is None:
            return False
        self.current = new
        return True

    def note_unstable(self, ts_ms: float, stabilizer: PredictionStabilizer, release_ms: float = 400):
        """Tay o trang thai chuyen tiep du lau -> cho phep lap lai chu vua danh."""
        if self.unstable_since is None:
            self.unstable_since = ts_ms
        elif ts_ms - self.unstable_since >= release_ms:
            stabilizer.release()

    def note_stable(self):
        self.unstable_since = None

    def commit(self):
        word = self.current
        if word:
            self.words.append(word)
            self.words = self.words[-20:]
        self.current = ""
        return word

    def backspace(self):
        if self.current:
            self.current = self.current[:-1]
        elif self.words:
            self.current = self.words.pop()

    def clear(self):
        self.current, self.words = "", []

    def state(self) -> dict:
        return {"current": self.current, "sentence": " ".join(self.words)}
