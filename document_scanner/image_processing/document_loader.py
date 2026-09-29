from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import fitz  # PyMuPDF
import numpy as np
from docx import Document as DocxDocument


SUPPORTED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | {"pdf", "docx", "txt"}
MIN_DIGITAL_TEXT_CHARS = 20

# Cap how many pages get rasterized into preview images. This only
# limits the *visual gallery* - the extracted text/summary still
# covers every page of the document regardless of this cap, since text
# extraction (digital PDFs) or OCR (scanned PDFs, per page during the
# app's own processing loop) is far cheaper than high-DPI rasterization.
MAX_PREVIEW_PAGES = 30


def get_file_category(filename: str) -> str:
    if "." not in filename:
        return "unsupported"
    ext = filename.rsplit(".", 1)[1].lower()
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    if ext in {"pdf", "docx", "txt"}:
        return ext
    return "unsupported"


@dataclass
class PdfContent:
    is_digital: bool
    text: str = ""
    page_images: list = field(default_factory=list)  # rasterized preview images (capped)
    page_count: int = 0            # true total page count in the PDF
    pages_rendered: int = 0        # how many pages were actually rasterized


def load_pdf(path: str, rasterize_dpi: int = 150, max_preview_pages: int = MAX_PREVIEW_PAGES) -> PdfContent:
    """
    Open a PDF, extract its full text (all pages), and rasterize page
    images for the preview gallery (capped at max_preview_pages to keep
    large documents fast/light).

    is_digital=True means a real text layer was found (content.text is
    populated, no OCR needed). is_digital=False means the PDF is
    essentially just page images (e.g. a scanned/photographed
    document) - the caller is expected to run OCR on content.page_images
    itself, since that happens per-page alongside the normal image
    pipeline (perspective correction, enhancement) rather than here.
    """
    doc = fitz.open(path)
    page_count = doc.page_count

    text_parts = [page.get_text() for page in doc]
    combined_text = "\n\n".join(t for t in text_parts if t.strip())
    is_digital = len(combined_text) >= MIN_DIGITAL_TEXT_CHARS

    zoom = rasterize_dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    pages_to_render = min(page_count, max_preview_pages)
    page_images = []
    for i in range(pages_to_render):
        page = doc[i]
        pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        page_images.append(img_bgr)

    doc.close()

    return PdfContent(
        is_digital=is_digital,
        text=combined_text if is_digital else "",
        page_images=page_images,
        page_count=page_count,
        pages_rendered=len(page_images),
    )


def load_docx_text(path: str) -> str:
    doc = DocxDocument(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    paragraphs.append(cell.text.strip())

    return "\n".join(paragraphs)


def load_txt_text(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")