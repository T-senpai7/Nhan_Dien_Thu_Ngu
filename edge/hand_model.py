"""Mo hinh dong hoc ban tay tao 21 landmarks gia lap theo dung dinh dang MediaPipe.

Duoc dung de: (1) sinh dataset tong hop, (2) chay Edge o che do khong camera,
(3) benchmark co nhan dung (ground truth). Moi landmarks sinh ra deu di qua cung
bo trich xuat edge/sensor_simulator.py nhu du lieu camera that.
"""
import numpy as np

import config

# Vi tri goc ngon (he toa do ban tay: x ve phia ngon cai, y huong len, z huong ra camera)
BASES = {
    "thumb": np.array([0.022, 0.025, 0.004]),
    "index": np.array([0.028, 0.090, 0.0]),
    "middle": np.array([0.008, 0.095, 0.0]),
    "ring": np.array([-0.012, 0.090, 0.0]),
    "pinky": np.array([-0.030, 0.080, 0.0]),
}
LENGTHS = {
    "thumb": (0.040, 0.032, 0.027),
    "index": (0.042, 0.025, 0.022),
    "middle": (0.047, 0.029, 0.024),
    "ring": (0.043, 0.027, 0.023),
    "pinky": (0.033, 0.021, 0.019),
}
SPREAD_DEG = {"index": 6.0, "middle": 0.0, "ring": -5.0, "pinky": -12.0}
JOINT_MAX_DEG = (85.0, 105.0, 65.0)
THUMB_MAX_DEG = (60.0, 50.0, 70.0)
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
LANDMARK_START = {"thumb": 1, "index": 5, "middle": 9, "ring": 13, "pinky": 17}

# curl: muc gap [thumb, index, middle, ring, pinky] (0 duoi thang, 1 gap hoan toan)
# pose: "up" ngon tay huong len, "side" huong ngang; motion: static / wave / pump
GESTURE_SPECS = {
    "XIN_CHAO": dict(curl=[0.0, 0.0, 0.0, 0.0, 0.0], pose="up", motion="static"),
    "TAM_BIET": dict(curl=[0.0, 0.0, 0.0, 0.0, 0.0], pose="up", motion="wave"),
    "CAM_ON": dict(curl=[0.1, 0.0, 0.0, 0.0, 0.0], pose="side", motion="static", spread=0.2),
    "TOI_CAN_GIUP_DO": dict(curl=[0.0, 1.0, 1.0, 1.0, 1.0], pose="side", motion="static"),
    "DONG_Y": dict(curl=[0.45, 0.55, 0.05, 0.05, 0.05], pose="up", motion="static", touch=True),
    "KHONG": dict(curl=[0.9, 0.0, 0.0, 1.0, 1.0], pose="up", motion="static", spread=2.0),
    "XIN_LOI": dict(curl=[0.9, 1.0, 1.0, 1.0, 1.0], pose="up", motion="static"),
    "UONG_NUOC": dict(curl=[0.3, 0.5, 0.5, 0.5, 0.5], pose="side", motion="static"),
    "DAU": dict(curl=[0.9, 0.0, 1.0, 1.0, 1.0], pose="side", motion="static"),
    "YEU_THUONG": dict(curl=[0.0, 0.0, 1.0, 1.0, 0.0], pose="up", motion="static"),
    "CHO_MOT_CHUT": dict(curl=[0.9, 0.0, 1.0, 1.0, 1.0], pose="up", motion="static"),
    "KHAN_CAP": dict(curl=[0.5, 0.5, 0.5, 0.5, 0.5], pose="up", motion="pump"),
}
assert set(GESTURE_SPECS) == set(config.WORDS)


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _unit(v):
    return v / np.linalg.norm(v)


def hand_local(curls, spread=1.0, touch=False, joint_jitter=None) -> np.ndarray:
    """Tinh 21 diem (he toa do ban tay) tu muc gap cua tung ngon."""
    pts = np.zeros((21, 3))
    z_axis = np.array([0.0, 0.0, 1.0])
    for f_idx, name in enumerate(FINGER_ORDER):
        curl = float(np.clip(curls[f_idx], 0.0, 1.0))
        start = LANDMARK_START[name]
        if name == "thumb":
            base_dir = _unit(np.array([0.55, 0.8, 0.25]))
            across = _unit(np.array([-0.8, 0.2, 0.6]))
            across = _unit(across - np.dot(across, base_dir) * base_dir)
            max_deg, pos = THUMB_MAX_DEG, BASES["thumb"].copy()
            pts[start] = pos
            phi = 0.0
            for j in range(3):
                jitter = joint_jitter[f_idx, j] if joint_jitter is not None else 0.0
                phi += np.radians(max(0.0, curl * max_deg[j] + jitter))
                direction = np.cos(phi) * base_dir + np.sin(phi) * across
                pos = pos + LENGTHS[name][j] * direction
                pts[start + j + 1] = pos
        else:
            s = np.radians(SPREAD_DEG[name] * spread)
            base_dir = np.array([np.sin(s), np.cos(s), 0.0])
            pos = BASES[name].copy()
            pts[start] = pos
            phi = 0.0
            for j in range(3):
                jitter = joint_jitter[f_idx, j] if joint_jitter is not None else 0.0
                phi += np.radians(max(0.0, curl * JOINT_MAX_DEG[j] + jitter))
                direction = np.cos(phi) * base_dir + np.sin(phi) * z_axis
                pos = pos + LENGTHS[name][j] * direction
                pts[start + j + 1] = pos
    if touch:
        tip = pts[8] + np.array([0.004, 0.0, 0.004])
        pts[4] = tip
        pts[3] = (pts[2] + tip) / 2.0 + np.array([0.006, 0.0, 0.006])
    return pts


class SyntheticHand:
    """Ban tay ao thuc hien mot cu chi voi tham so ngau nhien, sinh landmarks theo thoi gian."""

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()
        self.label = None

    def set_gesture(self, label: str, t0: float = 0.0):
        rng = self.rng
        spec = GESTURE_SPECS[label]
        self.label, self.spec, self.t0 = label, spec, t0
        self.curls = np.clip(np.array(spec["curl"]) + rng.uniform(-0.2, 0.2, 5), 0, 1)
        self.spread = spec.get("spread", 1.0) * rng.uniform(0.5, 1.5)
        if spec["pose"] == "up":
            self.roll = rng.uniform(-35, 35)
        else:
            self.roll = rng.choice([-1, 1]) * rng.uniform(50, 120)
        self.pitch = rng.uniform(-30, 30)
        self.yaw = rng.uniform(-40, 40)
        self.mirror = rng.random() < 0.5
        # Khung hinh da lat guong (selfie): ngon cai ben phai anh <=> tay trai
        self.handedness = "Right" if self.mirror else "Left"
        self.scale = rng.uniform(1.2, 2.6)
        self.center = np.array([rng.uniform(0.35, 0.95), rng.uniform(0.3, 0.7)])
        self.drift_amp = rng.uniform(0.0, 0.12)
        self.drift_freq = rng.uniform(0.2, 0.8)
        self.drift_phase = rng.uniform(0, 2 * np.pi)
        self.wave_amp = rng.uniform(0.5, 1.1)
        self.wave_freq = rng.uniform(1.5, 3.0)
        self.pump_freq = rng.uniform(2.0, 3.5)
        self.joint_jitter = rng.normal(0, 10.0, (5, 3))
        self.noise_xy = rng.uniform(0.0008, 0.0025)
        self.glitch_prob = rng.uniform(0.0, 0.08)

    def landmarks(self, t: float) -> np.ndarray:
        rng, spec = self.rng, self.spec
        tau = t - self.t0
        curls, roll = self.curls.copy(), self.roll
        palm_img = 0.095 * self.scale
        offset = self.drift_amp * palm_img * np.array([
            np.sin(2 * np.pi * self.drift_freq * tau + self.drift_phase),
            np.cos(2 * np.pi * self.drift_freq * 0.7 * tau + self.drift_phase),
        ])
        if spec["motion"] == "wave":
            offset[0] += self.wave_amp * palm_img * np.sin(2 * np.pi * self.wave_freq * tau)
            roll += 15 * np.sin(2 * np.pi * self.wave_freq * tau)
        elif spec["motion"] == "pump":
            c = 0.5 + 0.5 * np.sin(2 * np.pi * self.pump_freq * tau)
            curls = np.clip(np.full(5, c) + rng.normal(0, 0.03, 5), 0, 1)

        local = hand_local(curls, self.spread, spec.get("touch", False), self.joint_jitter)
        if self.mirror:
            local[:, 0] *= -1
        rot = _rot_z(np.radians(-roll)) @ _rot_x(np.radians(self.pitch)) @ _rot_y(np.radians(self.yaw))
        world = local @ rot.T * self.scale

        img = np.empty_like(world)
        img[:, 0] = self.center[0] + offset[0] + world[:, 0]
        img[:, 1] = self.center[1] + offset[1] - world[:, 1]
        img[:, 2] = -world[:, 2]
        img[:, :2] += rng.normal(0, self.noise_xy, (21, 2))
        img[:, 2] += rng.normal(0, self.noise_xy * 1.5, 21)
        if rng.random() < self.glitch_prob:
            # Mo phong loi tracking cua MediaPipe: mot dau ngon tay bi lech manh
            img[rng.choice([4, 8, 12, 16, 20])] += rng.normal(0, 0.25 * palm_img, 3)
        return img


def simulate_sequence(label: str, duration: float = 1.2, fps: float = 30.0, rng=None):
    """Sinh chuoi (t, landmarks, handedness) cho mot lan ra dau."""
    hand = SyntheticHand(rng)
    hand.set_gesture(label)
    n = int(duration * fps)
    for i in range(n):
        t = i / fps + (hand.rng.normal(0, 0.002) if i else 0.0)
        yield t, hand.landmarks(t), hand.handedness
