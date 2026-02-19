"""
Core image processing engine for PicsMake.
Detects individual photos in a scanned image and extracts them at highest quality.
"""

import cv2
import numpy as np
from PIL import Image
import io
import os
import gc


MAX_DETECTION_DIM = 2000  # Max dimension used for detection (saves memory)


def detect_and_extract_photos(image_bytes, min_area_ratio=0.02, padding=5):
    """
    Detect individual photos in a scanned image and extract them.

    Args:
        image_bytes: Raw bytes of the uploaded scan image.
        min_area_ratio: Minimum area of a detected region relative to the
                        full image to be considered a photo (filters noise).
        padding: Pixels of padding to add around each detected photo.

    Returns:
        List of PIL Image objects, one per detected photo.
    """
    # Decode image from bytes
    np_arr = np.frombuffer(image_bytes, np.uint8)
    original = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError("Could not decode the uploaded image.")

    height, width = original.shape[:2]

    # Downscale a working copy for detection to reduce memory usage.
    # Detection only needs to find bounding boxes; we crop from the original.
    scale = min(1.0, MAX_DETECTION_DIM / max(height, width))
    if scale < 1.0:
        work = cv2.resize(original, (int(width * scale), int(height * scale)),
                          interpolation=cv2.INTER_AREA)
    else:
        work = original

    work_h, work_w = work.shape[:2]
    total_area = work_h * work_w

    # Convert to grayscale
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Use adaptive thresholding to handle varying scan backgrounds
    # Try edge-based detection first (works better for photos on a scanner bed)
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate edges to close gaps between edge fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and sort candidate regions
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < total_area * min_area_ratio:
            continue

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h

        # Filter out regions that are too narrow (likely scanner artifacts)
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 8:
            continue

        # Filter out regions that span nearly the entire image (that's the scan itself)
        if rect_area > total_area * 0.95:
            continue

        candidates.append((x, y, w, h, rect_area))

    # If edge detection found nothing useful, fall back to threshold-based detection
    if len(candidates) == 0:
        candidates = _threshold_based_detection(gray, blurred, total_area, min_area_ratio)

    # If still nothing, try a more aggressive approach
    if len(candidates) == 0:
        candidates = _aggressive_detection(gray, total_area, min_area_ratio)

    # Merge overlapping rectangles
    candidates = _merge_overlapping(candidates, overlap_thresh=0.3)

    # Sort top-to-bottom, then left-to-right (using work image coordinates)
    candidates.sort(key=lambda r: (r[1] // (work_h // 4), r[0]))

    # Scale bounding boxes back to original image coordinates
    if scale < 1.0:
        candidates = [
            (int(x / scale), int(y / scale), int(w / scale), int(h / scale), int(a / (scale * scale)))
            for (x, y, w, h, a) in candidates
        ]

    # Free all working arrays before extracting from the (potentially large) original
    del work, gray, blurred, edges, dilated, contours, kernel
    gc.collect()

    # Extract each photo region from the original high-quality image
    extracted = []
    for (x, y, w, h, _) in candidates:
        # Add padding but stay within bounds
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)

        crop = original[y1:y2, x1:x2]

        # Convert BGR to RGB for PIL
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(crop_rgb)
        extracted.append(pil_image)

    del original, np_arr
    gc.collect()

    return extracted


def _threshold_based_detection(gray, blurred, total_area, min_area_ratio):
    """Fallback detection using Otsu's thresholding."""
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < total_area * min_area_ratio:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 8 or rect_area > total_area * 0.95:
            continue
        candidates.append((x, y, w, h, rect_area))

    return candidates


def _aggressive_detection(gray, total_area, min_area_ratio):
    """More aggressive detection for difficult scans."""
    # Try multiple threshold values
    candidates = []
    for thresh_val in [60, 100, 140, 180]:
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < total_area * min_area_ratio:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > 8 or rect_area > total_area * 0.95:
                continue
            candidates.append((x, y, w, h, rect_area))

    return _merge_overlapping(candidates, overlap_thresh=0.3)


def _merge_overlapping(rects, overlap_thresh=0.3):
    """Merge overlapping bounding rectangles."""
    if len(rects) == 0:
        return []

    # Sort by area descending
    rects = sorted(rects, key=lambda r: r[4], reverse=True)
    merged = []

    for rect in rects:
        x, y, w, h, area = rect
        should_merge = False

        for i, (mx, my, mw, mh, ma) in enumerate(merged):
            # Calculate overlap
            ix1 = max(x, mx)
            iy1 = max(y, my)
            ix2 = min(x + w, mx + mw)
            iy2 = min(y + h, my + mh)

            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                smaller_area = min(area, ma)
                if intersection / smaller_area > overlap_thresh:
                    # Merge by expanding the existing rectangle
                    nx = min(x, mx)
                    ny = min(y, my)
                    nx2 = max(x + w, mx + mw)
                    ny2 = max(y + h, my + mh)
                    nw = nx2 - nx
                    nh = ny2 - ny
                    merged[i] = (nx, ny, nw, nh, nw * nh)
                    should_merge = True
                    break

        if not should_merge:
            merged.append(rect)

    return merged


def save_photo_to_bytes(pil_image, format="PNG", quality=100):
    """Save a PIL Image to bytes at maximum quality."""
    buf = io.BytesIO()
    if format.upper() == "JPEG":
        pil_image.save(buf, format="JPEG", quality=100, subsampling=0)
    else:
        pil_image.save(buf, format="PNG", optimize=False)
    buf.seek(0)
    return buf.getvalue()
