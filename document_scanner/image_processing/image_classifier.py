
from __future__ import annotations

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def _white_ratio(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    white_pixels = cv2.countNonZero(cv2.inRange(gray, 200, 255))
    total_pixels = gray.shape[0] * gray.shape[1]
    return white_pixels / total_pixels if total_pixels else 0.0


def _quick_text_stats(image: np.ndarray) -> dict:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > 800:
        scale = 800 / longest
        small = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        small = image

    data = pytesseract.image_to_data(small, config="--psm 11", output_type=Output.DICT)

    heights = []
    word_count = 0
    for text, conf, height in zip(data["text"], data["conf"], data["height"]):
        if text.strip() and float(conf) >= 0:
            word_count += 1
            heights.append(float(height))

    if not heights:
        return {"word_count": 0, "height_cv": 0.0}

    mean_h = float(np.mean(heights))
    std_h = float(np.std(heights))
    height_cv = (std_h / mean_h) if mean_h > 0 else 0.0

    return {"word_count": word_count, "height_cv": height_cv}


_INVOICE_KEYWORDS = (
    "invoice", "receipt", "bill", "total", "subtotal", "amount",
    "due", "paid", "balance", "qty", "quantity", "tax", "order",
    "customer", "bill to", "ship to", "date",
)


def _invoice_keyword_hits(image: np.ndarray) -> int:
    h, w = image.shape[:2]
    longest = max(h, w)
    small = cv2.resize(image, (int(w * 800 / longest), int(h * 800 / longest))) if longest > 800 else image
    text = pytesseract.image_to_string(small).lower()
    return sum(1 for kw in _INVOICE_KEYWORDS if kw in text)


def classify_document_type(image: np.ndarray) -> str:
    stats = _quick_text_stats(image)
    word_count = stats["word_count"]
    height_cv = stats["height_cv"]
    if word_count > 60 and height_cv < 0.35:
        return "document"
    if height_cv >= 0.5:
        return "poster"
    if word_count <= 60 and _invoice_keyword_hits(image) >= 1:
        return "invoice"
    if word_count <= 60:
        return "poster"
    white_ratio = _white_ratio(image)
    return "document" if white_ratio > 0.5 else "poster"
def detect_image_type(image: np.ndarray) -> str:
    return "document" if _white_ratio(image) > 0.5 else "general"