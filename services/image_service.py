"""
services/image_service.py — Auto crop, deskew, enhance images
"""
import io
import math
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from config import MAX_IMAGE_DIMENSION, ENHANCED_IMAGE_QUALITY


def process_image(file_bytes: bytes, filename: str) -> bytes:
    """
    Full pipeline:
      1. Decode / convert to RGB
      2. Resize if too large
      3. Deskew
      4. Auto-crop (detect document edges)
      5. Enhance (contrast + sharpness)
      6. Return as JPEG bytes
    """
    img = _load_image(file_bytes, filename)
    img = _resize_if_large(img)
    img = _deskew(img)
    img = _auto_crop(img)
    img = _enhance(img)
    return _to_jpeg_bytes(img)


# ──────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ──────────────────────────────────────────────────────────────────

def _load_image(file_bytes: bytes, filename: str) -> Image.Image:
    """Load image from bytes; handle PDF first page separately."""
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        # Convert first page of PDF to image using PyMuPDF
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page = doc[0]
        mat = fitz.Matrix(2, 2)          # 2x zoom for quality
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("jpeg")
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    img = Image.open(io.BytesIO(file_bytes))
    if img.mode in ("RGBA", "P", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        return bg
    return img.convert("RGB")


def _resize_if_large(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIMENSION:
        ratio = MAX_IMAGE_DIMENSION / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    return img


def _deskew(img: Image.Image) -> Image.Image:
    """Detect skew angle via Hough lines and rotate to correct."""
    try:
        cv_img = _pil_to_cv(img)
        gray   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges  = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines  = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None:
            return img

        angles = []
        for line in lines[:20]:
            rho, theta = line[0]
            angle = math.degrees(theta) - 90
            if -45 < angle < 45:
                angles.append(angle)

        if not angles:
            return img

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:        # negligible skew
            return img

        return img.rotate(-median_angle, expand=True, fillcolor=(255, 255, 255))
    except Exception:
        return img                          # silently skip on error


def _auto_crop(img: Image.Image) -> Image.Image:
    """
    Detect the largest quadrilateral (document) in the image
    and perspective-crop it. Falls back to simple border trim.
    """
    try:
        cv_img = _pil_to_cv(img)
        gray   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges  = cv2.Canny(blurred, 30, 100)
        kernel = np.ones((5, 5), np.uint8)
        edges  = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return _trim_borders(img)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        doc_contour = None

        for c in contours[:5]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_contour = approx
                break

        if doc_contour is None:
            return _trim_borders(img)

        # Perspective transform
        pts = doc_contour.reshape(4, 2).astype("float32")
        warped = _four_point_transform(cv_img, pts)
        return _cv_to_pil(warped)

    except Exception:
        return _trim_borders(img)


def _enhance(img: Image.Image) -> Image.Image:
    """Increase contrast and sharpness for better OCR / readability."""
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = ImageEnhance.Sharpness(img).enhance(1.6)
    img = ImageEnhance.Brightness(img).enhance(1.05)
    return img


def _to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=ENHANCED_IMAGE_QUALITY, optimize=True)
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────

def _pil_to_cv(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _cv_to_pil(cv_img) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def _trim_borders(img: Image.Image) -> Image.Image:
    """Simple border trim using PIL getbbox on grayscale."""
    gray = img.convert("L")
    inverted = Image.eval(gray, lambda p: 255 - p)
    bbox = inverted.getbbox()
    if bbox:
        margin = 10
        w, h = img.size
        bbox = (
            max(0, bbox[0] - margin),
            max(0, bbox[1] - margin),
            min(w, bbox[2] + margin),
            min(h, bbox[3] + margin),
        )
        return img.crop(bbox)
    return img


def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def _four_point_transform(image, pts):
    rect = _order_points(pts)
    tl, tr, br, bl = rect

    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    max_w = max(int(wA), int(wB))

    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)
    max_h = max(int(hA), int(hB))

    dst = np.array([[0,0],[max_w-1,0],[max_w-1,max_h-1],[0,max_h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_w, max_h))
