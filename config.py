"""Cau hinh tap trung cho toan bo he thong IoT phien dich thu ngu."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# MQTT (Network layer)
# Mac dinh dung broker noi bo (network/mini_broker.py). De dung broker cong cong:
#   set MQTT_HOST=broker.emqx.io   (hoac broker.hivemq.com)
# ---------------------------------------------------------------------------
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME") or None
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD") or None
MQTT_KEEPALIVE = 30
MQTT_QOS = int(os.getenv("MQTT_QOS", "0"))

# Prefix rieng de tranh trung topic voi nguoi khac khi dung broker cong cong.
TOPIC_PREFIX = os.getenv("MQTT_TOPIC_PREFIX", "iot/sign_language")
TOPIC_SENSOR = f"{TOPIC_PREFIX}/sensor_data"
TOPIC_PREDICTION = f"{TOPIC_PREFIX}/prediction"
TOPIC_STATUS = f"{TOPIC_PREFIX}/status"
TOPIC_CONTROL = f"{TOPIC_PREFIX}/control"

DEVICE_ID = os.getenv("DEVICE_ID", "edge_laptop_01")

# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PUBLISH_HZ = float(os.getenv("PUBLISH_HZ", "15"))
MOTION_WINDOW_SEC = 0.6
SMOOTHING_ALPHA = 0.5

# ---------------------------------------------------------------------------
# Cloud
# ---------------------------------------------------------------------------
CLOUD_HOST = os.getenv("CLOUD_HOST", "127.0.0.1")
CLOUD_PORT = int(os.getenv("CLOUD_PORT", "8000"))
CLOUD_URL = f"http://{CLOUD_HOST}:{CLOUD_PORT}"

HAND_LOST_RESET_MS = 1000
LATENCY_TARGET_MS = 500

# Moi che do co nguong rieng: chu cai can on dinh lau hon vi nhieu chu gan giong nhau.
MODE_SETTINGS = {
    "letters": {"window": 7, "threshold": 0.75},
    "words": {"window": 5, "threshold": 0.70},
}
DEFAULT_MODE = os.getenv("DEFAULT_MODE", "letters")
# Bo tay khoi khung hinh lau hon nguong nay -> tu dong ket thuc tu dang danh van.
SPELL_AUTO_COMMIT_MS = 1500

MODEL_DIR = ROOT_DIR / "cloud" / "models"
MODEL_PATH = MODEL_DIR / "gesture_classifier.pkl"
DB_PATH = ROOT_DIR / "cloud" / "iot_sign_language.db"
TTS_CACHE_DIR = ROOT_DIR / "cloud" / "tts_cache"

DATASET_DIR = ROOT_DIR / "dataset"
SYNTHETIC_DATASET_PATH = DATASET_DIR / "sign_language_dataset.csv"
CAMERA_RAW_PATH = DATASET_DIR / "camera_raw.csv"
CAPTURE_DIR = DATASET_DIR / "images"                 # anh chup minh hoa moi lan ghi (phuc vu bao cao)
CAPTURE_INDEX_PATH = CAPTURE_DIR / "index.csv"
CAPTURES_PER_SESSION = 3

DASHBOARD_DIR = ROOT_DIR / "dashboard"
REFERENCE_IMAGE = DASHBOARD_DIR / "assets" / "vsl_alphabet.png"
REPORT_DIR = ROOT_DIR / "evaluation" / "reports"

# ---------------------------------------------------------------------------
# Che do 1 - Bang chu cai ngon tay tieng Viet (23 chu, tu the tinh)
# ma -> (chu hien thi, mo ta cach ra dau theo anh tham chieu)
# ---------------------------------------------------------------------------
LETTERS = {
    "A": ("A", "Nắm tay, ngón cái áp sát cạnh ngón trỏ"),
    "B": ("B", "Bốn ngón duỗi thẳng khép sát hướng lên, ngón cái gập ngang lòng bàn tay"),
    "C": ("C", "Các ngón cong tạo hình chữ C"),
    "D": ("D", "Ngón trỏ giơ thẳng, các ngón còn lại cong chạm ngón cái"),
    "DD": ("Đ", "Giống chữ D nhưng ngón trỏ hơi cong, bàn tay nghiêng"),
    "E": ("E", "Các ngón gập, đầu ngón chạm vào ngón cái gập trước lòng bàn tay"),
    "G": ("G", "Ngón trỏ và ngón cái duỗi song song chỉ ngang, các ngón khác nắm"),
    "H": ("H", "Ngón trỏ và ngón giữa duỗi, các ngón khác nắm (theo ảnh)"),
    "I": ("I", "Nắm tay, chỉ giơ một ngón thẳng lên (theo ảnh)"),
    "K": ("K", "Ngón trỏ và ngón giữa giơ lên tách chữ V, ngón cái đặt giữa"),
    "L": ("L", "Ngón cái và ngón trỏ duỗi vuông góc tạo chữ L"),
    "M": ("M", "Ba ngón trỏ, giữa, áp út gập phủ lên ngón cái"),
    "N": ("N", "Hai ngón trỏ và giữa chúc xuống, phủ lên ngón cái"),
    "O": ("O", "Các ngón cong chạm đầu ngón cái tạo vòng tròn chữ O"),
    "P": ("P", "Giống chữ K nhưng bàn tay chỉ ngang / chúc xuống"),
    "Q": ("Q", "Ngón trỏ và ngón cái chúc xuống, các ngón khác nắm"),
    "R": ("R", "Ngón giữa bắt chéo lên ngón trỏ"),
    "S": ("S", "Nắm tay, ngón cái gập phủ ngang trước các ngón"),
    "T": ("T", "Ngón cái kẹp giữa ngón trỏ và ngón giữa (theo ảnh)"),
    "U": ("U", "Ngón trỏ và ngón giữa giơ thẳng khép sát"),
    "V": ("V", "Ngón trỏ và ngón giữa giơ thẳng tách chữ V"),
    "X": ("X", "Ngón trỏ cong móc câu, các ngón khác nắm"),
    "Y": ("Y", "Giơ ngón cái và ngón út, các ngón khác nắm"),
}

# ---------------------------------------------------------------------------
# Che do 2 - Cu chi tu / cau thong dung
# ---------------------------------------------------------------------------
WORDS = {
    "XIN_CHAO": ("Xin chào", "Bàn tay mở, các ngón duỗi thẳng hướng lên, giữ yên"),
    "TAM_BIET": ("Tạm biệt", "Bàn tay mở, vẫy qua lại sang hai bên"),
    "CAM_ON": ("Cảm ơn", "Bàn tay mở duỗi thẳng, các ngón hướng ngang"),
    "TOI_CAN_GIUP_DO": ("Tôi cần giúp đỡ", "Nắm tay, ngón cái giơ lên (thumbs up)"),
    "DONG_Y": ("Đồng ý", "Ngón cái chạm ngón trỏ tạo vòng tròn (OK)"),
    "KHONG": ("Không", "Giơ ngón trỏ và ngón giữa (chữ V), các ngón khác gập"),
    "XIN_LOI": ("Xin lỗi", "Nắm chặt tay, tất cả các ngón gập, giữ yên"),
    "UONG_NUOC": ("Tôi muốn uống nước", "Các ngón cong hình chữ C như cầm cốc, đặt ngang"),
    "DAU": ("Tôi bị đau", "Ngón trỏ duỗi chỉ sang ngang, các ngón khác gập"),
    "YEU_THUONG": ("Yêu thương", "Giơ ngón cái, ngón trỏ và ngón út (I Love You)"),
    "CHO_MOT_CHUT": ("Chờ một chút", "Ngón trỏ giơ thẳng lên, các ngón khác gập"),
    "KHAN_CAP": ("Khẩn cấp", "Nắm và mở bàn tay liên tục thật nhanh"),
}

# ---------------------------------------------------------------------------
# Dau phu va dau thanh (anh tham chieu thu 2), dung trong che do chu cai
# - Dau tinh: nhan dien bang tu the ban tay nhu chu cai.
# - Dau dong: nhan dien bang quy dao dau ngon tro (cloud/trajectory.py).
# ---------------------------------------------------------------------------
MARKS_STATIC = {
    "MU": ("^", "Dấu mũ (â ê ô): ngón trỏ và ngón giữa tách chữ V, hai đầu ngón cong gập xuống"),
    "MOC": ("Móc", "Dấu móc (ư ơ): ngón trỏ và ngón giữa cong móc câu, ngón cái đỡ phía dưới"),
}
MARKS_DYNAMIC = {
    "DAU_TRANG": ("˘", "Dấu trăng (ă): nắm tay giơ ngón trỏ, vẽ một vòng cung phía trên đầu ngón"),
    "DAU_SAC": ("Sắc", "Ngón trỏ chỉ chéo lên, vạch một nét chéo xuống theo mũi tên trong ảnh"),
    "DAU_HUYEN": ("Huyền", "Bàn tay nằm ngang, ngón trỏ vạch một nét chéo lên theo mũi tên trong ảnh"),
    "DAU_HOI": ("Hỏi", "Ngón trỏ giơ thẳng, vẽ hình dấu hỏi (?)"),
    "DAU_NGA": ("Ngã", "Ngón trỏ giơ thẳng, vẽ đường lượn sóng (~)"),
    "DAU_NANG": ("Nặng", "Nắm tay, ngón trỏ chỉ xuống, kéo một nét chéo xuống"),
}
# Tu the chung cua cac dau dong (ngon tro duoi) - mo hinh tinh hoc tu chinh cac lan ghi dau dong.
POSE_LABEL = "TU_THE_DAU"
POSE = {POSE_LABEL: ("Tư thế dấu", "Nắm tay, duỗi ngón trỏ để chuẩn bị vẽ dấu thanh")}
MARKS_REFERENCE_IMAGE = DASHBOARD_DIR / "assets" / "vsl_marks.png"
MARK_MODEL_PATH = MODEL_DIR / "mark_classifier.pkl"
MARK_MIN_CONFIDENCE = 0.7
MARK_POSE_CONF = 0.5            # xac suat "tu the dau" toi thieu de tinh khung do vao net ve
MARK_POSE_READY_MS = 200        # phai giu tu the ngon tro chua lau nay truoc khi ve thi net moi duoc tinh
MARK_POSE_WINDOW_MS = 1500      # net ve phai bat dau trong khoang nay sau khi tu the da san sang
MARK_MAX_STROKE_MS = 2000       # khop voi MAX_DURATION_SEC cua bo cat net trong cloud/trajectory.py

# ---------------------------------------------------------------------------
# Cu chi dieu khien khi danh van (khong nam trong bang chu cai nen khong bi nham voi chu)
# ---------------------------------------------------------------------------
CONTROLS = {
    "XOA": ("⌫ Xóa", "Nắm tay, chĩa ngón cái xuống đất (like ngược). Giữ tiếp để xóa thêm chữ"),
    "CACH": ("␣ Xong từ", "Xòe rộng cả 5 ngón tay, lòng bàn tay hướng về camera"),
}
CONTROL_ACTIONS = {"XOA": "backspace", "CACH": "space"}

# Chu cai chi duoc nhan khi tay GIU YEN cung mot tu the du lau -> khong nhan nham luc dang chuyen chu.
LETTER_HOLD_MS = int(os.getenv("LETTER_HOLD_MS", "700"))
LETTER_HOLD_RANGE = (300, 2000)
LETTER_STILL_SPEED = 3.0        # toc do tam ban tay toi da (long ban tay / s, da lam muot) de tinh la dang giu yen
LETTER_STILL_SHAPE = 1e9        # tat: hinh dang ngon tay rung manh voi mot so chu (G) du dang giu dung chu
LETTER_MOTION_EMA = 0.4         # he so lam muot toc do, tranh rung tay le te bi coi la dang chuyen chu
DELETE_REPEAT_MS = 900          # giu cu chi xoa -> xoa tiep moi khoang nay

MODES = {
    "letters": {"name": "Chữ cái",
                "labels": list(LETTERS) + list(MARKS_STATIC) + [POSE_LABEL] + list(CONTROLS)},
    "words": {"name": "Từ ngữ", "labels": list(WORDS)},
}
RECORD_GROUPS = {
    "letters": list(LETTERS),
    "marks": list(MARKS_STATIC) + list(MARKS_DYNAMIC),
    "controls": list(CONTROLS),
    "words": list(WORDS),
}
GESTURES = {**LETTERS, **MARKS_STATIC, **MARKS_DYNAMIC, **POSE, **CONTROLS, **WORDS}
GESTURE_LABELS = list(GESTURES)
LABEL_MODE = {label: mode for mode, m in MODES.items() for label in m["labels"]}
LABEL_MODE.update({label: "letters" for label in MARKS_DYNAMIC})
LABEL_KIND = {**{l: "letter" for l in LETTERS}, **{l: "mark" for l in MARKS_STATIC},
              **{l: "mark" for l in MARKS_DYNAMIC}, POSE_LABEL: "pose", **{l: "control" for l in CONTROLS},
              **{l: "word" for l in WORDS}}

# ---------------------------------------------------------------------------
# Dac trung dau vao cho mo hinh AI (phai trung voi edge/sensor_simulator.py)
# ---------------------------------------------------------------------------
SENSOR_FEATURES = [
    "flex_thumb", "flex_index", "flex_middle", "flex_ring", "flex_pinky",
    "roll", "pitch", "yaw",
    "ax", "ay", "az", "speed",
    "touch_thumb_index",
    "osc_x", "osc_y", "flex_osc",
]
# Hinh dang ban tay: 20 khop (bo co tay) da chuan hoa theo co tay va kich thuoc long ban tay.
SHAPE_FEATURES = [f"p{i}_{axis}" for i in range(1, 21) for axis in "xyz"]
# Khoang cach dac trung de phan biet cac chu cai gan giong nhau (A/E/M/N/S/T, U/V/R...).
THUMB_TARGETS = (3, 5, 6, 8, 9, 10, 12, 13, 14, 16, 18, 20)
DISTANCE_FEATURES = [f"d_thumb_{j}" for j in THUMB_TARGETS] + [
    "d_8_12", "d_12_16", "d_16_20", "d_8_20", "cross_index_middle", "d_8_5", "d_12_9",
]
FEATURE_COLUMNS = SENSOR_FEATURES + SHAPE_FEATURES + DISTANCE_FEATURES
