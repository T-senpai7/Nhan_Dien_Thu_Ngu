"""Ve chu tieng Viet co dau len khung hinh OpenCV (cv2.putText khong ho tro Unicode)."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]
FONT_BOLD_CANDIDATES = ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"]


@lru_cache(maxsize=32)
def _font(size: int, bold: bool):
    for path in (FONT_BOLD_CANDIDATES if bold else []) + FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_texts(frame_bgr, items):
    """items: list (text, (x, y), size, (b, g, r), bold). Ve tat ca trong 1 lan chuyen doi anh."""
    if not items:
        return frame_bgr
    image = Image.fromarray(frame_bgr[:, :, ::-1])
    draw = ImageDraw.Draw(image)
    for text, pos, size, color, bold in items:
        draw.text(pos, text, font=_font(size, bold), fill=(color[2], color[1], color[0]))
    frame_bgr[:] = np.asarray(image)[:, :, ::-1]
    return frame_bgr
