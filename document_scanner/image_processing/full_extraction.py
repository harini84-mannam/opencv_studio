"""
full_extraction.py

A dedicated pipeline for *complete, verbatim* text extraction across every
page of a document - as opposed to the classify + summarize pipeline in
app.py used for the main "Scan Document" flow.

Rules this module follows (per the extraction spec it implements):
  - Every page is processed, first page to last - none are skipped.
  - Original reading order is preserved (top-to-bottom, left-to-right,
    grouped by OCR block/paragraph/line).
  - Content is not summarized or reworded - this returns the raw
    (cleaned of encoding noise only) text.
  - Low-confidence words are never guessed at blindly, but they are
    also never given up on early: each one is re-cropped from the
    winning image, upscaled further, sharpened, and re-OCR'd on its
    own before anything is finalized. A word is only ever marked
    "[unclear text]" if it is still unreadable after that rescue
    attempt.
  - All pages are combined into a single text block, with a page
    separator so reading order across pages stays unambiguous.
"""

from __future__ import annotations

import os

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from pytesseract import Output

from .scanner import detect_document_edges, get_fallback_corners
from .perspective import four_point_transform
from .enhancement import enhance_document
from .ocr import clean_text
from .document_loader import (
    SUPPORTED_IMAGE_EXTENSIONS,
    MIN_DIGITAL_TEXT_CHARS,
    load_docx_text,
    load_txt_text,
)

DEFAULT_MIN_CONFIDENCE = 45.0  # Lower = fewer "[unclear text]" markers, but
                                # more wrong/garbled words shown as if they
                                # were correctly read, with no way to tell
                                # which ones to double-check. 45.0 is roughly
                                # where Tesseract's own confidence score stops
                                # being trustworthy - lowering it trades
                                # visible uncertainty for invisible wrongness.
RASTERIZE_DPI = 300
SMALL_TEXT_MIN_DIMENSION = 2400  # upscale target for the longest side


def _ocr_words(image: np.ndarray, lang: str = "eng") -> list[dict]:
    """Raw per-word OCR output: text, confidence, and line-grouping key,
    in reading order. Shared by both the scoring pass and the final
    text-assembly pass so the two never disagree with each other."""
    data = pytesseract.image_to_data(image, lang=lang, config="--psm 6", output_type=Output.DICT)

    words = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        conf = float(data["conf"][i])
        if conf < 0:
            # Tesseract uses -1 conf for structural/whitespace entries
            continue
        words.append({
            "text": text,
            "conf": conf,
            "key": (data["block_num"][i], data["par_num"][i], data["line_num"][i]),
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i]),
        })
    return words


def _assemble_text(words: list[dict], min_confidence: float) -> str:
    """Turn a flat, ordered word list back into lines, marking anything
    below min_confidence as [unclear text] instead of guessing."""
    lines: dict[tuple, list[str]] = {}
    order: list[tuple] = []

    for w in words:
        key = w["key"]
        if key not in lines:
            lines[key] = []
            order.append(key)
        lines[key].append(w["text"] if w["conf"] >= min_confidence else "[unclear text]")

    return "\n".join(" ".join(lines[key]) for key in order if lines[key])


def _score_words(words: list[dict]) -> float:
    """Average confidence, used to pick the better of two OCR attempts.
    An empty result scores 0 so a non-empty attempt always wins."""
    if not words:
        return 0.0
    return sum(w["conf"] for w in words) / len(words)


def _upscale_if_small(image: np.ndarray, min_dimension: int = SMALL_TEXT_MIN_DIMENSION) -> np.ndarray:
    """Small screenshots/photos make small text (like code snippets)
    only a few pixels tall, which Tesseract struggles with. Upscale so
    the longest side is at least min_dimension before OCR."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest >= min_dimension:
        return image
    scale = min_dimension / longest
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)


def _gentle_preprocess(image: np.ndarray) -> np.ndarray:
    """
    A light-touch alternative to the full scanner pipeline, for content
    that's already flat and clean (screenshots, code snippets, digital
    exports) rather than a photographed paper document. Skips
    perspective warping and hard black/white thresholding - both of
    which can blur or break up small text - and just upscales +
    normalizes contrast.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    upscaled = _upscale_if_small(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(upscaled)


def _gentle_preprocess_thresholded(image: np.ndarray) -> np.ndarray:
    """
    Same as _gentle_preprocess, plus an Otsu threshold at the end.
    Testing against real, low-resolution/dense screenshots (small
    monospace code, compressed PNGs) showed this reads noticeably more
    accurately than either the full scanner pipeline or the untresholded
    gentle pass - the opposite of what's true for photographed paper,
    where adaptive thresholding tuned in enhance_document already
    covers this case. This targets already-flat, already-digital,
    but small/dense text.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    upscaled = _upscale_if_small(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(upscaled)
    _, otsu = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu


def _aggressive_preprocess(image: np.ndarray) -> np.ndarray:
    """
    A stronger fallback for text that's still small/faint/blurry after
    the normal scanner and gentle passes - tiny screenshots, heavily
    compressed photos, low-contrast scans. Upscales further than the
    gentle pass, boosts contrast harder, and sharpens - the same
    combination that reliably resolves tiny code screenshots when
    checked by hand.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    h, w = gray.shape[:2]
    longest = max(h, w)
    target = SMALL_TEXT_MIN_DIMENSION * 1.6
    if longest < target:
        scale = target / longest
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, sharpen_kernel)
    return gray


def _rescue_word(image: np.ndarray, box: dict, lang: str = "eng") -> tuple[str, float]:
    """
    Last-resort rescue for a single low-confidence word: crop just
    that word (plus a little padding) out of the winning image, blow
    it up to a solid pixel height, sharpen it, and re-OCR it alone
    with a single-line page mode. A word sitting by itself at high
    effective resolution reads far more reliably than the same word
    packed into a small, dense line - this is what turns a garbled
    line back into real text instead of "[unclear text]".
    """
    h, w = image.shape[:2]
    pad = max(6, int(box["height"] * 0.8))
    x0 = max(0, box["left"] - pad)
    y0 = max(0, box["top"] - pad)
    x1 = min(w, box["left"] + box["width"] + pad)
    y1 = min(h, box["top"] + box["height"] + pad)
    if x1 <= x0 or y1 <= y0:
        return "", -1.0

    crop = image[y0:y1, x0:x1]
    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    crop_h, crop_w = crop.shape[:2]
    target_h = 200
    if crop_h < target_h:
        scale = target_h / max(crop_h, 1)
        crop = cv2.resize(crop, (max(1, int(crop_w * scale)), target_h), interpolation=cv2.INTER_LANCZOS4)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    crop = clahe.apply(crop)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    crop = cv2.filter2D(crop, -1, sharpen_kernel)

    data = pytesseract.image_to_data(crop, lang=lang, config="--psm 7", output_type=Output.DICT)
    best_text, best_conf = "", -1.0
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        conf_val = float(conf)
        if text and conf_val > best_conf:
            best_text, best_conf = text, conf_val
    return best_text, best_conf


def _rescue_low_confidence_words(
    words: list[dict], image: np.ndarray, min_confidence: float, lang: str = "eng"
) -> list[dict]:
    """
    Give every word below the confidence bar one more chance before
    anything gets marked "[unclear text]". Only replaces a word's text
    when the rescue attempt actually comes back more confident than
    the original reading.
    """
    rescued = []
    for word in words:
        if word["conf"] >= min_confidence:
            rescued.append(word)
            continue
        text, conf = _rescue_word(image, word, lang=lang)
        if text and conf > word["conf"]:
            rescued.append({**word, "text": text, "conf": conf})
        else:
            rescued.append(word)
    return rescued


def _prepare_scanned_page(page_image: np.ndarray, mode: str = "scanner") -> np.ndarray:
    """Full pipeline tuned for photographed paper documents: boundary
    detection, perspective correction, shadow/noise cleanup, threshold."""
    corners = detect_document_edges(page_image)
    if corners is None:
        corners = get_fallback_corners(page_image)
    warped = four_point_transform(page_image, corners)
    enhanced = enhance_document(warped, mode=mode)
    return _upscale_if_small(enhanced)


def _ocr_page_verbatim(raw_image: np.ndarray, min_confidence: float = DEFAULT_MIN_CONFIDENCE, mode: str = "scanner", lang: str = "eng") -> str:
    """
    Run OCR with three different preprocessing strategies - the full
    "photographed paper" pipeline, a gentle upscale-only pipeline
    better suited to screenshots/small text/code, and an aggressive
    upscale+contrast+sharpen pass for text that's still small, faint,
    or blurry - and keep whichever produced the highest-confidence
    results overall. Then, before finalizing, every word that's still
    below the confidence bar gets one more individual rescue attempt
    (see _rescue_low_confidence_words) rather than being written off
    immediately. Only what's unreadable after all of that is marked
    "[unclear text]".
    """
    candidate_a = _prepare_scanned_page(raw_image, mode=mode)
    candidate_b = _gentle_preprocess(raw_image)
    candidate_c = _aggressive_preprocess(raw_image)

    words_a = _ocr_words(candidate_a, lang=lang)
    words_b = _ocr_words(candidate_b, lang=lang)
    words_c = _ocr_words(candidate_c, lang=lang)

    candidates = [
        (words_a, candidate_a),
        (words_b, candidate_b),
        (words_c, candidate_c),
    ]
    best_words, best_image = max(candidates, key=lambda pair: _score_words(pair[0]))

    best_words = _rescue_low_confidence_words(best_words, best_image, min_confidence, lang=lang)
    return _assemble_text(best_words, min_confidence=min_confidence)


def extract_full_text_from_pdf(
    pdf_path: str,
    mode: str = "scanner",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> tuple[str, int]:
    """
    Walk every page of the PDF, first to last, with no page cap.

    - If a page already has a real digital text layer, that text is
      used as-is (it's already exact, no OCR/confidence marking needed).
    - Otherwise the page is rasterized at RASTERIZE_DPI and run through
      the normal scanner pipeline + OCR, marking low-confidence words.

    Returns (combined_text, page_count).
    """
    doc = fitz.open(pdf_path)
    page_count = doc.page_count
    zoom = RASTERIZE_DPI / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    page_texts: list[str] = []

    for i in range(page_count):
        page = doc[i]
        digital_text = page.get_text().strip()

        if len(digital_text) >= MIN_DIGITAL_TEXT_CHARS:
            page_text = clean_text(digital_text)
        else:
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            raw = _ocr_page_verbatim(img_bgr, min_confidence=min_confidence, mode=mode)
            page_text = clean_text(raw) if raw.strip() else "[unclear text]"

        page_texts.append(page_text)

    doc.close()

    combined = "\n\n".join(
        f"--- Page {idx + 1} of {page_count} ---\n{text}"
        for idx, text in enumerate(page_texts)
    )
    return combined, page_count


def extract_full_text_from_image(
    image_path: str,
    mode: str = "scanner",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> tuple[str, int]:
    """Same verbatim extraction, for a single photographed/scanned image."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the uploaded image.")

    raw = _ocr_page_verbatim(image, min_confidence=min_confidence, mode=mode)
    page_text = clean_text(raw) if raw.strip() else "[unclear text]"
    return page_text, 1


def extract_full_text(
    file_path: str,
    category: str,
    mode: str = "scanner",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> tuple[str, int]:
    """
    Dispatches full-document text extraction by file category
    ("image", "pdf", "docx", "txt" - see document_loader.get_file_category).

    docx/txt already contain exact digital text, so they're returned
    as-is (cleaned of encoding noise) rather than re-OCR'd.
    """
    if category == "pdf":
        return extract_full_text_from_pdf(file_path, mode=mode, min_confidence=min_confidence)
    if category in SUPPORTED_IMAGE_EXTENSIONS or category == "image":
        return extract_full_text_from_image(file_path, mode=mode, min_confidence=min_confidence)
    if category == "docx":
        return clean_text(load_docx_text(file_path)), 1
    if category == "txt":
        return clean_text(load_txt_text(file_path)), 1

    raise ValueError(f"Unsupported file type for full extraction: {file_path}")