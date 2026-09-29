
from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Mes-Harini\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
from pytesseract import Output


@dataclass
class OCRResult:
    text: str
    confidence: float                
    word_count: int
    rotation_applied: int = 0        
    words: list = field(default_factory=list)  


def extract_text(image: np.ndarray, lang: str = "eng", config: str = "") -> str:
    return pytesseract.image_to_string(image, lang=lang, config=config)


def extract_text_with_confidence(image: np.ndarray, lang: str = "eng", config: str = "") -> OCRResult:
    data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=Output.DICT)

    words = []
    confidences = []
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        conf_val = float(conf)
        if text and conf_val >= 0:
            words.append((text, conf_val))
            confidences.append(conf_val)

    full_text = " ".join(w for w, _ in words)
    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    return OCRResult(
        text=full_text,
        confidence=round(avg_conf, 1),
        word_count=len(words),
        words=words,
    )


def detect_orientation(image: np.ndarray) -> int:
    try:
        osd = pytesseract.image_to_osd(image, output_type=Output.DICT)
        return int(osd.get("rotate", 0))
    except pytesseract.TesseractError:
        return 0


def correct_orientation(image: np.ndarray) -> tuple[np.ndarray, int]:
    angle = detect_orientation(image)
    if angle == 0:
        return image, 0

    rotation_map = {
        90: cv2.ROTATE_90_COUNTERCLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_CLOCKWISE,
    }
    flag = rotation_map.get(angle)
    if flag is None:
        return image, 0

    return cv2.rotate(image, flag), angle

def clean_text(raw_text: str) -> str:
    if not raw_text:
        return ""

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\x09\x0A\x20-\x7E]", "", text)

    lines = [line.strip() for line in text.split("\n")]

    lines = [re.sub(r"[ \t]+", " ", line) for line in lines]

    merged: list[str] = []
    for line in lines:
        if not line:
            merged.append("")  
            continue
        if (
            merged
            and merged[-1]
            and not re.search(r"[.!?:;,]$", merged[-1])
            and line[:1].islower()
        ):
            merged[-1] = f"{merged[-1]} {line}"
        else:
            merged.append(line)
    cleaned_lines: list[str] = []
    blank_run = 0
    for line in merged:
        if line == "":
            blank_run += 1
            if blank_run <= 1:
                cleaned_lines.append(line)
        else:
            blank_run = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def remove_special_characters(text: str, keep: str = ".,!?:;-@/$%()'\"") -> str:
    pattern = rf"[^\w\s{re.escape(keep)}]"
    return re.sub(pattern, "", text)


def postprocess_ocr_text(raw_text: str, strip_special_chars: bool = False) -> str:
    cleaned = clean_text(raw_text)
    if strip_special_chars:
        cleaned = remove_special_characters(cleaned)
    return cleaned
@dataclass
class PosterLine:
    text: str
    avg_height: float   
    confidence: float
    line_id: tuple       

def _group_words_into_lines(data: dict) -> list[PosterLine]:
    words = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        conf = float(data["conf"][i])
        height = float(data["height"][i])
        top = float(data["top"][i])
        left = float(data["left"][i])

        if not text or conf < 0:
            continue

        words.append({
            "text": text, "conf": conf, "height": height,
            "top": top, "left": left, "center_y": top + height / 2.0,
        })

    if not words:
        return []
    words.sort(key=lambda w: w["center_y"])

    clusters: list[list[dict]] = []
    for w in words:
        placed = False
        for cluster in clusters:
            cluster_center = np.mean([c["center_y"] for c in cluster])
            cluster_height = np.mean([c["height"] for c in cluster])
            if abs(w["center_y"] - cluster_center) <= 0.6 * max(cluster_height, w["height"]):
                cluster.append(w)
                placed = True
                break
        if not placed:
            clusters.append([w])

    result = []
    for idx, cluster in enumerate(clusters):
        cluster.sort(key=lambda w: w["left"])
        line_text = " ".join(w["text"] for w in cluster)
        avg_height = float(np.mean([w["height"] for w in cluster]))
        avg_conf = float(np.mean([w["conf"] for w in cluster]))
        avg_top = float(np.mean([w["top"] for w in cluster]))
        result.append(PosterLine(text=line_text, avg_height=avg_height, confidence=avg_conf, line_id=(idx, avg_top, 0)))

    return result


def _is_noise_line(text: str) -> bool:
    stripped = re.sub(r"[^A-Za-z0-9]", "", text)
    return len(stripped) < 2


def prepare_for_poster_ocr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest < 1200:
        scale = 1200 / longest
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    if float(np.mean(gray)) < 110:
        gray = cv2.bitwise_not(gray)

    return gray


def extract_poster_summary(
    image: np.ndarray,
    lang: str = "eng",
    max_lines: int = 3,
    min_confidence: float = 40.0,
) -> list[str]:
    prepared = prepare_for_poster_ocr(image)
    data = pytesseract.image_to_data(prepared, lang=lang, config="--psm 11", output_type=Output.DICT)
    lines = _group_words_into_lines(data)
    candidates = [
        line for line in lines
        if not _is_noise_line(line.text) and line.confidence >= min_confidence
    ]

    if not candidates:
        candidates = [line for line in lines if not _is_noise_line(line.text)]

    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda l: (l.avg_height, l.confidence), reverse=True)

    selected: list[PosterLine] = []
    seen_text: set[str] = set()
    for line in ranked:
        normalized = re.sub(r"\s+", " ", line.text).strip().lower()
        if normalized in seen_text:
            continue
        seen_text.add(normalized)
        selected.append(line)
        if len(selected) >= max_lines:
            break
    selected.sort(key=lambda l: l.line_id[1])

    return [postprocess_ocr_text(line.text) for line in selected]
_INVOICE_FIELD_PATTERN = re.compile(
    r"\b(invoice\s*(no\.?|number|#)?|receipt\s*(no\.?|#)?|order\s*(no\.?|#)?|"
    r"bill\s*to|ship\s*to|customer( name)?|total|subtotal|amount( due)?|"
    r"balance( due)?|tax|date|due date|qty|quantity)\b",
    re.IGNORECASE,
)
_CURRENCY_OR_DATE_PATTERN = re.compile(
    r"(\$\s?\d|₹\s?\d|€\s?\d|£\s?\d|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
)


def extract_invoice_highlights(image: np.ndarray, lang: str = "eng", max_lines: int = 5) -> list[str]:
    raw_text = extract_text(image, lang=lang)
    cleaned = postprocess_ocr_text(raw_text)
    lines = [ln.strip() for ln in cleaned.split("\n") if ln.strip()]

    matched: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if _INVOICE_FIELD_PATTERN.search(line) or _CURRENCY_OR_DATE_PATTERN.search(line):
            normalized = line.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            matched.append(line)
        if len(matched) >= max_lines:
            break

    if matched:
        return matched
    return lines[:max_lines]


def run_ocr_pipeline(image: np.ndarray, lang: str = "eng", auto_orient: bool = True) -> OCRResult:
    rotation_applied = 0
    working_image = image

    if auto_orient:
        working_image, rotation_applied = correct_orientation(image)

    result = extract_text_with_confidence(working_image, lang=lang)
    result.text = postprocess_ocr_text(result.text)
    result.rotation_applied = rotation_applied
    return result