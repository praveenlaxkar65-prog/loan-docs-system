"""
services/image_service.py — Auto crop, deskew, enhance images
Conservative crop — prefer full document over aggressive edge detection.
"""
import io
import math
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from config import MAX_IMAGE_DIMENSION, ENHANCED_IMAGE_QUALITY


def process_image(file_bytes: bytes, filename: str) -> bytes:
    img = _load_image(file_bytes, filename)
    img = _resize_if_large(img)
    img = _deskew(img)
    img = _conservative_crop(img)
    img = _enhance(img)
    return _to_jpeg_bytes(img)


def _load_image(file_bytes: bytes, filename: str) -> Image.Image:
    ext = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()
    if ext == "pdf":
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page = doc[0]
        mat = fitz.Matrix(2, 2)
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
    try:
        cv_img = _pil_to_cv(img)
        gray  = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
        if lines is None:
            return img
        angles = []
        for line in lines[:30]:
            rho, theta = line[0]
            angle = math.degrees(theta) - 90
            if -15 < angle < 15 and abs(angle) > 1:
                angles.append(angle)
        if not angles:
            return img
        median_angle = float(np.median(angles))
        if abs(median_angle) < 1 or abs(median_angle) > 15:
            return img
        return img.rotate(-median_angle, expand=True, fillcolor=(255, 255, 255))
    except Exception:
        return img


def _conservative_crop(img: Image.Image) -> Image.Image:
    try:
        w, h = img.size
        min_area = w * h * 0.60
        cv_img  = _pil_to_cv(img)
        gray    = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges   = cv2.Canny(blurred, 20, 80)
        kernel  = np.ones((7, 7), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return _trim_white_border(img)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours[:3]:
            area = cv2.contourArea(c)
            if area < min_area:
                break
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                pts    = approx.reshape(4, 2).astype("float32")
                warped = _four_point_transform(cv_img, pts)
                ww, wh = warped.shape[1], warped.shape[0]
                if ww >= w * 0.70 and wh >= h * 0.70:
                    return _cv_to_pil(warped)
        return _trim_white_border(img)
    except Exception:
        return img


def _trim_white_border(img: Image.Image) -> Image.Image:
    try:
        arr      = np.array(img.convert("L"))
        row_mask = np.any(arr < 240, axis=1)
        col_mask = np.any(arr < 240, axis=0)
        if not row_mask.any() or not col_mask.any():
            return img
        r_min, r_max = np.where(row_mask)[0][[0, -1]]
        c_min, c_max = np.where(col_mask)[0][[0, -1]]
        margin = 20
        iw, ih = img.size
        c_min = max(0, c_min - margin)
        r_min = max(0, r_min - margin)
        c_max = min(iw, c_max + margin)
        r_max = min(ih, r_max + margin)
        if (c_min < iw * 0.02 and r_min < ih * 0.02 and
                c_max > iw * 0.98 and r_max > ih * 0.98):
            return img
        cropped = img.crop((c_min, r_min, c_max, r_max))
        cw, ch  = cropped.size
        if cw < iw * 0.80 or ch < ih * 0.80:
            return img
        return cropped
    except Exception:
        return img


def _enhance(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    img = ImageEnhance.Brightness(img).enhance(1.02)
    return img


def _to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=ENHANCED_IMAGE_QUALITY, optimize=True)
    return buf.getvalue()


def _pil_to_cv(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _cv_to_pil(cv_img) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s    = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image, pts):
    rect  = _order_points(pts)
    tl, tr, br, bl = rect
    wA    = np.linalg.norm(br - bl)
    wB    = np.linalg.norm(tr - tl)
    max_w = max(int(wA), int(wB))
    hA    = np.linalg.norm(tr - br)
    hB    = np.linalg.norm(tl - bl)
    max_h = max(int(hA), int(hB))
    dst   = np.array([[0,0],[max_w-1,0],[max_w-1,max_h-1],[0,max_h-1]], dtype="float32")
    M     = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_w, max_h))
