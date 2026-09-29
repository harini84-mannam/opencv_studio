"""
Document Scanner blueprint.

This is the standalone Document Scanner app, mounted here as a Blueprint
under /document-scanner so it lives inside the same app/navbar as the
Object Detection (OpenCV) tools.
"""

import os
import uuid

import cv2
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    flash,
)
from werkzeug.utils import secure_filename

from .image_processing.document_preview import create_text_preview
from .image_processing.scanner import detect_document_edges, get_fallback_corners
from .image_processing.perspective import four_point_transform
from .image_processing.enhancement import enhance_document
from .image_processing.ocr import extract_text, extract_poster_summary, extract_invoice_highlights
from .image_processing.image_classifier import classify_document_type
from .image_processing.local_summary import summarize_text
from .image_processing.pdf import create_pdf
from .image_processing.document_loader import get_file_category, load_pdf, load_docx_text, load_txt_text
from .image_processing.text_pdf import create_text_only_pdf
from .image_processing.full_extraction import extract_full_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp", "pdf", "docx", "txt"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

document_scanner_bp = Blueprint(
    "document_scanner",
    __name__,
    url_prefix="/document-scanner",
    template_folder="templates",
    static_folder="static",
    static_url_path="/static",
)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def _summarize_by_type(image, full_text):
    """Shared classify + summarize step for a single image (a photo, or one PDF page)."""
    doc_type = classify_document_type(image)

    if doc_type == "poster":
        lines = extract_poster_summary(image)
        text = "\n".join(lines) if lines else "(no text detected)"
    elif doc_type == "invoice":
        lines = extract_invoice_highlights(image)
        text = "\n".join(lines) if lines else "(no text detected)"
    else:
        text = summarize_text(full_text, max_sentences=3) if full_text.strip() else "(no text detected)"

    return doc_type, text


def process_image_file(image_path, job_id, mode="scanner"):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the uploaded image.")

    corners = detect_document_edges(image)
    used_fallback = corners is None
    if corners is None:
        corners = get_fallback_corners(image)

    warped = four_point_transform(image, corners)
    enhanced = enhance_document(warped, mode=mode)

    full_text = extract_text(enhanced)
    doc_type, extracted_text = _summarize_by_type(warped, full_text)

    enhanced_filename = f"{job_id}_enhanced.png"
    enhanced_path = os.path.join(UPLOAD_DIR, enhanced_filename)
    cv2.imwrite(enhanced_path, enhanced)

    pdf_filename = f"{job_id}_scanned.pdf"
    pdf_path = os.path.join(OUTPUT_DIR, pdf_filename)
    create_pdf(enhanced_path, extracted_text, pdf_path)

    return {
        "original_filename": os.path.basename(image_path),
        "enhanced_filename": enhanced_filename,
        "pdf_filename": pdf_filename,
        "extracted_text": extracted_text,
        "used_fallback": used_fallback,
        "doc_type": doc_type,
        "has_image": True,
        "page_previews": [{"original": os.path.basename(image_path), "enhanced": enhanced_filename}],
        "page_count": 1,
        "pages_rendered": 1,
    }


def process_pdf_file(pdf_path_in, job_id, mode="scanner"):
    """
    Handles both digital PDFs (real text layer - no OCR needed) and
    scanned PDFs (rasterize each page, run the normal image pipeline on
    each). Builds a full page-by-page preview gallery either way.
    """
    content = load_pdf(pdf_path_in)
    page_previews = []

    if content.is_digital:
        # Real text layer - summarize directly, no OCR needed. Still
        # render every page (up to the cap) so there's a full preview
        # gallery, purely for display purposes.
        for i, page_image in enumerate(content.page_images):
            original_filename = f"{job_id}_page{i+1}_original.png"
            enhanced_filename = f"{job_id}_page{i+1}_enhanced.png"

            cv2.imwrite(os.path.join(UPLOAD_DIR, original_filename), page_image)
            enhanced = enhance_document(page_image, mode=mode)
            cv2.imwrite(os.path.join(UPLOAD_DIR, enhanced_filename), enhanced)

            page_previews.append({"original": original_filename, "enhanced": enhanced_filename})

        doc_type = "document"
        extracted_text = (
            summarize_text(content.text, max_sentences=4)
            if content.text.strip() else "(no text detected)"
        )

        pdf_filename = f"{job_id}_scanned.pdf"
        pdf_path = os.path.join(OUTPUT_DIR, pdf_filename)
        create_text_only_pdf(extracted_text, pdf_path, title="Extracted Summary")

        first_enhanced = page_previews[0]["enhanced"] if page_previews else None

        return {
            "original_filename": page_previews[0]["original"] if page_previews else None,
            "enhanced_filename": first_enhanced,
            "pdf_filename": pdf_filename,
            "extracted_text": extracted_text,
            "used_fallback": False,
            "doc_type": doc_type,
            "has_image": bool(page_previews),
            "page_previews": page_previews,
            "page_count": content.page_count,
            "pages_rendered": content.pages_rendered,
        }

    # Scanned PDF: run the normal image pipeline on every rendered page.
    page_texts = []
    doc_type = "document"

    for i, page_image in enumerate(content.page_images):
        corners = detect_document_edges(page_image)
        if corners is None:
            corners = get_fallback_corners(page_image)
        warped = four_point_transform(page_image, corners)
        enhanced = enhance_document(warped, mode=mode)

        page_full_text = extract_text(enhanced)
        page_texts.append(page_full_text)

        original_filename = f"{job_id}_page{i+1}_original.png"
        enhanced_filename = f"{job_id}_page{i+1}_enhanced.png"
        cv2.imwrite(os.path.join(UPLOAD_DIR, original_filename), page_image)
        cv2.imwrite(os.path.join(UPLOAD_DIR, enhanced_filename), enhanced)
        page_previews.append({"original": original_filename, "enhanced": enhanced_filename})

        if i == 0:
            doc_type, _ = _summarize_by_type(warped, page_full_text)

    combined_text = "\n\n".join(t for t in page_texts if t.strip())

    if doc_type == "poster":
        extracted_text = "\n".join(extract_poster_summary(content.page_images[0])) or "(no text detected)"
    elif doc_type == "invoice":
        extracted_text = "\n".join(extract_invoice_highlights(content.page_images[0])) or "(no text detected)"
    else:
        extracted_text = summarize_text(combined_text, max_sentences=4) if combined_text.strip() else "(no text detected)"

    pdf_filename = f"{job_id}_scanned.pdf"
    pdf_path = os.path.join(OUTPUT_DIR, pdf_filename)
    if page_previews:
        create_pdf(os.path.join(UPLOAD_DIR, page_previews[0]["enhanced"]), extracted_text, pdf_path)
    else:
        create_text_only_pdf(extracted_text, pdf_path, title="Extracted Summary")

    return {
        "original_filename": page_previews[0]["original"] if page_previews else None,
        "enhanced_filename": page_previews[0]["enhanced"] if page_previews else None,
        "pdf_filename": pdf_filename,
        "extracted_text": extracted_text,
        "used_fallback": False,
        "doc_type": doc_type,
        "has_image": bool(page_previews),
        "page_previews": page_previews,
        "page_count": content.page_count,
        "pages_rendered": content.pages_rendered,
    }


def process_text_file(file_path, job_id, category):
    """Handles .docx and .txt - a single synthetic preview page (no real scanned image)."""
    if category == "docx":
        raw_text = load_docx_text(file_path)
    else:
        raw_text = load_txt_text(file_path)

    preview_filename = create_text_preview(raw_text, UPLOAD_DIR, job_id)
    enhanced_filename = f"{job_id}_enhanced.png"

    img = cv2.imread(os.path.join(UPLOAD_DIR, preview_filename))
    enhanced = enhance_document(img)
    cv2.imwrite(os.path.join(UPLOAD_DIR, enhanced_filename), enhanced)

    extracted_text = (
        summarize_text(raw_text, max_sentences=4) if raw_text.strip() else "(no text detected)"
    )

    pdf_filename = f"{job_id}_scanned.pdf"
    pdf_path = os.path.join(OUTPUT_DIR, pdf_filename)
    create_text_only_pdf(extracted_text, pdf_path, title="Extracted Summary")

    return {
        "original_filename": preview_filename,
        "enhanced_filename": enhanced_filename,
        "pdf_filename": pdf_filename,
        "extracted_text": extracted_text,
        "used_fallback": False,
        "doc_type": "document",
        "has_image": True,
        "page_previews": [{"original": preview_filename, "enhanced": enhanced_filename}],
        "page_count": 1,
        "pages_rendered": 1,
    }


def process_document(file_path, job_id, mode="scanner"):
    """Dispatches to the right handler based on file type."""
    category = get_file_category(os.path.basename(file_path))

    if category == "image":
        return process_image_file(file_path, job_id, mode=mode)
    if category == "pdf":
        return process_pdf_file(file_path, job_id, mode=mode)
    if category in ("docx", "txt"):
        return process_text_file(file_path, job_id, category)

    raise ValueError(f"Unsupported file type: {file_path}")


def process_document_full_text(file_path, job_id, mode="scanner"):
    """
    Full, verbatim, page-by-page extraction (no summarization, no
    classification) - every page combined into one downloadable .txt.
    """
    category = get_file_category(os.path.basename(file_path))
    full_text, page_count = extract_full_text(file_path, category, mode=mode)

    txt_filename = f"{job_id}_extracted.txt"
    txt_path = os.path.join(OUTPUT_DIR, txt_filename)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    return {
        "original_filename": os.path.basename(file_path),
        "extracted_text": full_text,
        "txt_filename": txt_filename,
        "page_count": page_count,
    }


@document_scanner_bp.route("/", methods=["GET"])
def index():
    return render_template("document_scanner/index.html")


@document_scanner_bp.route("/scan", methods=["POST"])
def scan():
    if "image" not in request.files or request.files["image"].filename == "":
        flash("Please choose a file to scan.")
        return redirect(url_for("document_scanner.index"))

    file = request.files["image"]

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload an image, PDF, Word document, or text file.")
        return redirect(url_for("document_scanner.index"))

    mode = request.form.get("mode", "scanner")
    extraction_mode = request.form.get("extraction_mode", "summary")

    job_id = uuid.uuid4().hex[:10]
    filename = secure_filename(file.filename)
    saved_name = f"{job_id}_{filename}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)
    file.save(saved_path)

    if extraction_mode == "full":
        try:
            result = process_document_full_text(saved_path, job_id, mode=mode)
        except Exception as exc:  # noqa: BLE001
            flash(f"Something went wrong while extracting text: {exc}")
            return redirect(url_for("document_scanner.index"))
        return render_template("document_scanner/extract_result.html", **result)

    try:
        result = process_document(saved_path, job_id, mode=mode)
    except Exception as exc:  # noqa: BLE001
        flash(f"Something went wrong while scanning: {exc}")
        return redirect(url_for("document_scanner.index"))

    return render_template("document_scanner/result.html", **result)


@document_scanner_bp.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@document_scanner_bp.route("/outputs/<path:filename>")
def download_pdf(filename):
    # Serves any generated output file from OUTPUT_DIR - PDFs and the
    # .txt files produced by the full-text extraction mode alike.
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)