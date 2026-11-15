"""Boc MediaPipe Hands: doc khung hinh BGR, tra ve 21 landmarks da hieu chinh ti le khung hinh."""
from collections import Counter, deque

import cv2
import numpy as np
import mediapipe as mp

import config

_mp_hands = mp.solutions.hands
_mp_draw = mp.solutions.drawing_utils
_mp_styles = mp.solutions.drawing_styles


class HandTracker:
    def __init__(self, model_complexity: int = 1):
        self.hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=model_complexity,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self._handedness = deque(maxlen=15)

    def process(self, frame_bgr):
        """Tra ve (landmarks (21,3) | None, handedness 'Left'/'Right', raw_result de ve).

        Khung hinh dau vao phai da lat guong (selfie) de nhan tay trai/phai dung voi nguoi dung.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            self._handedness.clear()
            return None, None, None
        raw = result.multi_hand_landmarks[0]
        if result.multi_handedness:
            self._handedness.append(result.multi_handedness[0].classification[0].label)
        handedness = Counter(self._handedness).most_common(1)[0][0] if self._handedness else "Right"
        aspect = w / h
        lm = np.array([[p.x * aspect, p.y, p.z * aspect] for p in raw.landmark], dtype=float)
        return lm, handedness, raw

    @staticmethod
    def draw(frame_bgr, raw):
        if raw is not None:
            _mp_draw.draw_landmarks(
                frame_bgr, raw, _mp_hands.HAND_CONNECTIONS,
                _mp_styles.get_default_hand_landmarks_style(),
                _mp_styles.get_default_hand_connections_style(),
            )

    def close(self):
        self.hands.close()


def open_camera(index: int = config.CAMERA_INDEX):
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    return cap
