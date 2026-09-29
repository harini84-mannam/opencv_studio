import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from document_scanner import document_scanner_bp

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
for directory in (UPLOAD_DIR, OUTPUT_DIR):
    directory.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
app.secret_key = "dev-secret-key-change-in-production"  # needed for the document_scanner blueprint's flash() messages
app.register_blueprint(document_scanner_bp)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

face_net = cv2.dnn.readNetFromCaffe(
    str(MODEL_DIR / "deploy.prototxt"),
    str(MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"),
)
eye_cascade = cv2.CascadeClassifier(str(MODEL_DIR / "haarcascade_eye.xml"))
face_cascade = cv2.CascadeClassifier(str(MODEL_DIR / "haarcascade_frontalface_default.xml"))

TOOLS = [
    ("image-editor", "Image Editor", "Upload an image and apply a processing operation."),
    ("grayscale", "Grayscale", "Convert an image to grayscale."),
    ("blur", "Blur & Smoothing", "Gaussian, median, or bilateral smoothing."),
    ("edges", "Edge Detection", "Canny and Sobel gradient edges."),
    ("brightness", "Brightness", "Adjust brightness and contrast."),
    ("color-filter", "Color Filtering", "Keep a selected HSV color range."),
    ("threshold", "Thresholding", "Binary, Otsu, and adaptive thresholding."),
    ("morphology", "Morphology", "Erosion, dilation, opening, and closing."),
    ("background", "Background Removal", "GrabCut foreground extraction."),
    ("corners", "Corner Detection", "Detect strong corners with Shi-Tomasi."),
    ("template-matching", "Template Matching", "Find a template inside a source image."),
    ("feature-matching", "Feature Matching", "Match ORB features across two images."),
    ("shapes-text", "Shapes & Text", "Draw a rectangle, circle, and label."),
    ("image-arithmetic", "Image Arithmetic", "Add, subtract, and blend two images."),
    ("splitting-combining", "Splitting & Combining", "Split images into halves or quadrants and combine them."),
    ("video-frames", "Video Frames", "Extract selected frames from a video."),
]


def save_upload(file_storage, allowed):
    if not file_storage or not file_storage.filename:
        raise ValueError("Please choose a file first.")
    ext = Path(file_storage.filename).suffix.lower()
    if ext not in allowed:
        raise ValueError(f"Unsupported file type: {ext or 'none'}")
    filename = f"{uuid.uuid4().hex}_{secure_filename(file_storage.filename)}"
    path = UPLOAD_DIR / filename
    file_storage.save(path)
    return path


def save_image(image, prefix="result"):
    filename = f"{prefix}_{uuid.uuid4().hex}.png"
    path = OUTPUT_DIR / filename
    cv2.imwrite(str(path), image)
    return filename


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("The uploaded file could not be read as an image.")
    return image


def load_single_image():
    return read_image(save_upload(request.files.get("image"), IMAGE_EXTENSIONS))


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def concat_side_by_side(left, right, target_height=None):
    """Normalize two images so OpenCV can concatenate them horizontally."""
    def color(image):
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    left, right = color(left), color(right)
    target_height = target_height or max(left.shape[0], right.shape[0])
    left = cv2.resize(left, (max(1, int(left.shape[1] * target_height / left.shape[0])), target_height), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (max(1, int(right.shape[1] * target_height / right.shape[0])), target_height), interpolation=cv2.INTER_AREA)
    return cv2.hconcat([left, right])


def process_image(image, operation, form):
    if operation == "image-editor":
        g = gray(image)
        blurred = cv2.GaussianBlur(image, (9, 9), 0)
        edges = cv2.cvtColor(cv2.Canny(g, 50, 150), cv2.COLOR_GRAY2BGR)
        h, w = image.shape[:2]
        tile_h, tile_w = max(1, h // 2), max(1, w // 2)
        tiles = [image, cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), blurred, edges]
        tiles = [cv2.resize(tile, (tile_w, tile_h)) for tile in tiles]
        return cv2.vconcat([cv2.hconcat(tiles[:2]), cv2.hconcat(tiles[2:])])
    if operation == "grayscale":
        return gray(image)
    if operation == "blur":
        method = form.get("method", "gaussian")
        if method == "median":
            return cv2.medianBlur(image, 9)
        if method == "bilateral":
            return cv2.bilateralFilter(image, 9, 75, 75)
        return cv2.GaussianBlur(image, (9, 9), 0)
    if operation == "edges":
        method = form.get("method", "canny")
        g = gray(image)
        if method == "sobel":
            sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
            return cv2.convertScaleAbs(cv2.magnitude(sx, sy))
        return cv2.Canny(g, int(form.get("low", 50)), int(form.get("high", 150)))
    if operation == "brightness":
        alpha = float(form.get("contrast", 1.2))
        beta = int(form.get("brightness", 25))
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    if operation == "color-filter":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color = form.get("color", "green")
        ranges = {
            "red": ((0, 80, 50), (10, 255, 255)),
            "blue": ((90, 60, 40), (135, 255, 255)),
            "green": ((35, 50, 40), (85, 255, 255)),
            "yellow": ((18, 70, 50), (38, 255, 255)),
        }
        lower, upper = ranges.get(color, ranges["green"])
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        return cv2.bitwise_and(image, image, mask=mask)
    if operation == "threshold":
        g = gray(image)
        method = form.get("method", "otsu")
        if method == "adaptive":
            return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        threshold_type = cv2.THRESH_BINARY + (cv2.THRESH_OTSU if method == "otsu" else 0)
        _, result = cv2.threshold(g, int(form.get("value", 127)), 255, threshold_type)
        return result
    if operation == "morphology":
        g = gray(image)
        kernel = np.ones((5, 5), np.uint8)
        method = form.get("method", "opening")
        operations = {"erosion": cv2.MORPH_ERODE, "dilation": cv2.MORPH_DILATE, "opening": cv2.MORPH_OPEN, "closing": cv2.MORPH_CLOSE}
        return cv2.morphologyEx(g, operations.get(method, cv2.MORPH_OPEN), kernel)
    if operation == "background":
        mask = np.zeros(image.shape[:2], np.uint8)
        rect = (10, 10, max(1, image.shape[1] - 20), max(1, image.shape[0] - 20))
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(image, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        binary = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")
        return cv2.bitwise_and(image, image, mask=binary)
    if operation == "corners":
        result = image.copy()
        corners = cv2.goodFeaturesToTrack(gray(image), 100, 0.01, 10)
        if corners is not None:
            for corner in np.intp(corners):
                x, y = corner.ravel()
                cv2.circle(result, (x, y), 5, (0, 0, 255), -1)
        return result
    if operation == "shapes-text":
        result = image.copy()
        h, w = result.shape[:2]
        shape = form.get("shape", "all")
        text = form.get("text", "My Text").strip() or "My Text"
        color_name = form.get("color", "blue")
        colors = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255), "yellow": (0, 255, 255), "white": (255, 255, 255)}
        color = colors.get(color_name, colors["blue"])
        thickness = max(1, int(form.get("thickness", 4)))
        if shape in {"rectangle", "all"}:
            cv2.rectangle(result, (w // 10, h // 10), (w * 4 // 10, h * 4 // 10), color, thickness)
        if shape in {"circle", "all"}:
            cv2.circle(result, (w * 7 // 10, h * 5 // 10), max(10, min(w, h) // 8), color, thickness)
        if shape in {"line", "all"}:
            cv2.line(result, (w // 10, h // 2), (w * 9 // 10, h // 2), color, thickness)
        if shape in {"arrow", "all"}:
            cv2.arrowedLine(result, (w // 10, h * 7 // 10), (w * 8 // 10, h * 7 // 10), color, thickness, tipLength=0.08)
        if form.get("add_text", "yes") == "yes":
            font_scale = max(0.4, min(3.0, float(form.get("font_scale", 1.2))))
            cv2.putText(result, text, (w // 10, max(35, h * 9 // 10)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return result
    raise ValueError("Unknown image operation.")


def two_image_operation(first, second, operation):
    first = read_image(first)
    second = read_image(second)
    if operation == "arithmetic":
        second = cv2.resize(second, (first.shape[1], first.shape[0]))
        return cv2.addWeighted(first, 0.5, second, 0.5, 0)
    if operation == "template":
        source = first
        template = second
        if template.shape[0] > source.shape[0] or template.shape[1] > source.shape[1]:
            scale = min(source.shape[1] / template.shape[1], source.shape[0] / template.shape[0])
            template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
        _, max_score, _, max_loc = cv2.minMaxLoc(result)
        th, tw = template.shape[:2]
        x, y = max_loc
        annotated_source = source.copy()
        cv2.rectangle(annotated_source, (x, y), (x + tw, y + th), (0, 255, 0), 4)
        source_label = annotated_source.copy()
        return source_label
    orb = cv2.ORB_create(nfeatures=1000)
    k1, d1 = orb.detectAndCompute(first, None)
    k2, d2 = orb.detectAndCompute(second, None)
    if d1 is None or d2 is None:
        return concat_side_by_side(first, second)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    matches = sorted(matches, key=lambda m: m.distance)[:50]
    return cv2.drawMatches(first, k1, second, k2, matches, None, flags=2)


@app.context_processor
def inject_tools():
    return {"tools": TOOLS}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/face-eye")
def face_eye():
    return render_template("video.html")


@app.route("/upload-video", methods=["POST"])
def upload_video():
    try:
        path = save_upload(request.files.get("video"), VIDEO_EXTENSIONS)
        return jsonify({"filename": path.name})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/video-feed/<filename>")
def video_feed(filename):
    path = UPLOAD_DIR / secure_filename(filename)
    if not path.exists():
        abort(404)
    def frames():
        cap = cv2.VideoCapture(str(path))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
                face_net.setInput(blob)
                detections = face_net.forward()
                gray_frame = gray(frame)
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence < 0.35:
                        continue
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = np.clip(box.astype(int), [0, 0, 0, 0], [w, h, w, h])
                    if x2 <= x1 or y2 <= y1:
                        continue
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    roi = gray_frame[y1:y1 + max(1, (y2 - y1) // 2), x1:x2]
                    eyes = eye_cascade.detectMultiScale(roi, 1.05, 5, minSize=(15, 15))
                    for ex, ey, ew, eh in eyes:
                        cv2.rectangle(frame, (x1 + ex, y1 + ey), (x1 + ex + ew, y1 + ey + eh), (0, 255, 0), 2)
                ok, encoded = cv2.imencode(".jpg", frame)
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
        finally:
            cap.release()
    from flask import Response
    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/tool/<operation>", methods=["GET", "POST"])
def tool(operation):
    valid = {item[0] for item in TOOLS}
    if operation not in valid:
        abort(404)
    if operation in {"template-matching", "feature-matching", "image-arithmetic"}:
        if request.method == "POST":
            try:
                first = save_upload(request.files.get("image"), IMAGE_EXTENSIONS)
                second = save_upload(request.files.get("second_image"), IMAGE_EXTENSIONS)
                kind = {"template-matching": "template", "feature-matching": "features", "image-arithmetic": "arithmetic"}[operation]
                result = two_image_operation(first, second, kind)
                return render_template("two_image_tool.html", operation=operation, result=save_image(result, operation), error=None)
            except ValueError as exc:
                return render_template("two_image_tool.html", operation=operation, result=None, error=str(exc))
        return render_template("two_image_tool.html", operation=operation, result=None, error=None)
    if operation == "video-frames":
        return render_template("video_frames.html")
    if operation == "splitting-combining":
        if request.method == "POST":
            try:
                mode = request.form.get("mode", "left-right")
                outputs = {}
                image = None
                if mode == "left-right":
                    image = load_single_image()
                    height, width = image.shape[:2]
                    outputs["left"] = save_image(image[:, :width // 2], "left_half")
                    outputs["right"] = save_image(image[:, width // 2:], "right_half")
                elif mode == "quadrants":
                    image = load_single_image()
                    height, width = image.shape[:2]
                    middle_y, middle_x = height // 2, width // 2
                    outputs["top_left"] = save_image(image[:middle_y, :middle_x], "top_left")
                    outputs["top_right"] = save_image(image[:middle_y, middle_x:], "top_right")
                    outputs["bottom_left"] = save_image(image[middle_y:, :middle_x], "bottom_left")
                    outputs["bottom_right"] = save_image(image[middle_y:, middle_x:], "bottom_right")
                elif mode in {"horizontal", "vertical"}:
                    image = load_single_image()
                    second = read_image(save_upload(request.files.get("second_image"), IMAGE_EXTENSIONS))
                    if mode == "horizontal":
                        second = cv2.resize(second, (max(1, int(second.shape[1] * image.shape[0] / second.shape[0])), image.shape[0]))
                        combined = concat_side_by_side(image, second, image.shape[0])
                    else:
                        second = cv2.resize(second, (image.shape[1], image.shape[1] * second.shape[0] // max(1, second.shape[1])))
                        target_width = max(image.shape[1], second.shape[1])
                        image = cv2.resize(image, (target_width, max(1, image.shape[0] * target_width // image.shape[1])))
                        second = cv2.resize(second, (target_width, max(1, second.shape[0] * target_width // second.shape[1])))
                        combined = cv2.vconcat([image, second])
                    outputs["combined"] = save_image(combined, f"combined_{mode}")
                elif mode == "recombine":
                    parts = [request.files.get(name) for name in ("top_left", "top_right", "bottom_left", "bottom_right")]
                    images = [read_image(save_upload(part, IMAGE_EXTENSIONS)) for part in parts]
                    top_left, top_right, bottom_left, bottom_right = images
                    row_width = max(top_left.shape[1], top_right.shape[1])
                    top_left = cv2.resize(top_left, (row_width // 2, max(1, top_left.shape[0] * (row_width // 2) // top_left.shape[1])))
                    top_right = cv2.resize(top_right, (row_width - top_left.shape[1], top_left.shape[0]))
                    bottom_left = cv2.resize(bottom_left, (top_left.shape[1], max(1, bottom_left.shape[0] * top_left.shape[1] // bottom_left.shape[1])))
                    bottom_right = cv2.resize(bottom_right, (top_right.shape[1], bottom_left.shape[0]))
                    outputs["combined"] = save_image(cv2.vconcat([cv2.hconcat([top_left, top_right]), cv2.hconcat([bottom_left, bottom_right])]), "combined_quadrants")
                return render_template("splitting_combining.html", outputs=outputs, error=None)
            except (ValueError, cv2.error) as exc:
                return render_template("splitting_combining.html", outputs=None, error=str(exc))
        return render_template("splitting_combining.html", outputs=None, error=None)
    if request.method == "POST":
        try:
            image = load_single_image()
            result = process_image(image, operation, request.form)
            return render_template("image_tool.html", operation=operation, result=save_image(result, operation), error=None)
        except (ValueError, cv2.error) as exc:
            return render_template("image_tool.html", operation=operation, result=None, error=str(exc))
    return render_template("image_tool.html", operation=operation, result=None, error=None)


@app.route("/tool/video-frames", methods=["POST"])
def extract_video_frames():
    try:
        path = save_upload(request.files.get("video"), VIDEO_EXTENSIONS)
        cap = cv2.VideoCapture(str(path))
        count = max(1, int(request.form.get("count", 6)))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or count
        indexes = np.linspace(0, max(0, total - 1), count).astype(int)
        results = []
        for index in indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                results.append(save_image(frame, "video_frame"))
        cap.release()
        return render_template("video_frames.html", results=results, error=None)
    except (ValueError, cv2.error) as exc:
        return render_template("video_frames.html", results=[], error=str(exc))


@app.route("/outputs/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_DIR, secure_filename(filename))


if __name__ == "__main__":
    app.run(debug=True, threaded=True)