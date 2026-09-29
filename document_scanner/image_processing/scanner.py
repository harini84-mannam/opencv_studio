
import cv2
import numpy as np


def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  
    rect[2] = pts[np.argmax(s)]  

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  
    rect[3] = pts[np.argmax(diff)]  

    return rect


def detect_document_edges(image):
    orig_h, orig_w = image.shape[:2]
    resize_ratio = 500.0 / orig_h if orig_h > 500 else 1.0
    resized = cv2.resize(
        image, (int(orig_w * resize_ratio), int(orig_h * resize_ratio))
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    doc_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.2 * (
            resized.shape[0] * resized.shape[1]
        ):
            doc_contour = approx
            break

    if doc_contour is None:
        return None

    pts = doc_contour.reshape(4, 2).astype("float32")
    pts = pts / resize_ratio  

    return _order_points(pts)


def get_fallback_corners(image):
    h, w = image.shape[:2]
    return np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32"
    )
