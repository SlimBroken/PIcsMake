"""
Core image processing engine for PicsMake.
Detects individual photos in a scanned image and extracts them at highest quality.
"""

import cv2
import numpy as np
from PIL import Image
import io
import gc


MAX_DETECTION_DIM = 2000  # Max dimension used for detection (saves memory)
BG_THRESHOLD = 235        # Pixels brighter than this are considered scanner background


def detect_and_extract_photos(image_bytes, min_area_ratio=0.02, padding=4):
    """
    Detect individual photos in a scanned image and extract them.

    Returns a list of PIL Image objects, one per detected photo.
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    original = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError("Could not decode the uploaded image.")

    height, width = original.shape[:2]

    # Downscale a working copy for detection to keep memory low.
    # Bounding boxes are scaled back up before cropping from the original.
    scale = min(1.0, MAX_DETECTION_DIM / max(height, width))
    if scale < 1.0:
        work = cv2.resize(original, (int(width * scale), int(height * scale)),
                          interpolation=cv2.INTER_AREA)
    else:
        work = original

    work_h, work_w = work.shape[:2]
    total_area = work_h * work_w

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    # ── Primary: background subtraction ──────────────────────────────────────
    # Scanner background is bright white. Threshold it away, leaving only photos.
    candidates = _background_detection(gray, work_h, work_w, total_area, min_area_ratio)

    # ── Fallback: edge + small dilation ──────────────────────────────────────
    if len(candidates) == 0:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        candidates = _edge_detection(blurred, total_area, min_area_ratio)

    # ── Fallback: Otsu threshold ──────────────────────────────────────────────
    if len(candidates) == 0:
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        candidates = _otsu_detection(blurred, total_area, min_area_ratio)

    # Merge only truly overlapping boxes (low threshold avoids merging neighbours)
    candidates = _merge_overlapping(candidates, overlap_thresh=0.5)

    # Sort top-to-bottom, left-to-right
    candidates.sort(key=lambda r: (r[1] // (work_h // 5 or 1), r[0]))

    # Refine each bounding box to the exact photo edges
    candidates = [_refine_crop(gray, x, y, w, h) for (x, y, w, h, _) in candidates]

    # Scale back to original image coordinates
    if scale < 1.0:
        candidates = [
            (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
            for (x, y, w, h) in candidates
        ]

    # Free working arrays before touching the (potentially large) original
    del work, gray
    gc.collect()

    # Extract crops from the full-resolution original
    extracted = []
    for (x, y, w, h) in candidates:
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width,  x + w + padding)
        y2 = min(height, y + h + padding)

        crop_rgb = cv2.cvtColor(original[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        extracted.append(Image.fromarray(crop_rgb))

    del original, np_arr
    gc.collect()

    return extracted


# ── Detection helpers ─────────────────────────────────────────────────────────

def _background_detection(gray, work_h, work_w, total_area, min_area_ratio):
    """
    Primary method: find non-background (non-white) blobs.
    Works well for photos on a bright scanner bed.
    """
    # Mark every pixel darker than BG_THRESHOLD as foreground
    _, mask = cv2.threshold(gray, BG_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # Close small gaps *within* a photo (e.g. white sky areas) without bridging
    # the gap between two adjacent photos.  Keep the kernel small.
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=2)

    # Remove tiny noise specks
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _filter_contours(contours, total_area, min_area_ratio)


def _edge_detection(blurred, total_area, min_area_ratio):
    """
    Fallback: Canny edges with a small dilation so nearby-but-separate
    photos are not merged together.
    """
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _filter_contours(contours, total_area, min_area_ratio)


def _otsu_detection(blurred, total_area, min_area_ratio):
    """Fallback: Otsu global threshold."""
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _filter_contours(contours, total_area, min_area_ratio)


def _filter_contours(contours, total_area, min_area_ratio):
    candidates = []
    for contour in contours:
        if cv2.contourArea(contour) < total_area * min_area_ratio:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 8:
            continue
        if rect_area > total_area * 0.95:
            continue
        candidates.append((x, y, w, h, rect_area))
    return candidates


def _refine_crop(gray, x, y, w, h):
    """
    Tighten a bounding box by scanning row/column intensities.
    Finds exactly where photo content starts and ends, removing the
    over-estimation introduced by morphological operations.
    """
    # Add a small search margin around the rough box
    margin = 10
    sx = max(0, x - margin)
    sy = max(0, y - margin)
    ex = min(gray.shape[1], x + w + margin)
    ey = min(gray.shape[0], y + h + margin)

    region = gray[sy:ey, sx:ex]

    row_content = np.any(region < BG_THRESHOLD, axis=1)
    col_content = np.any(region < BG_THRESHOLD, axis=0)

    rows = np.where(row_content)[0]
    cols = np.where(col_content)[0]

    if len(rows) == 0 or len(cols) == 0:
        return x, y, w, h

    new_y = sy + int(rows[0])
    new_h = int(rows[-1]) - int(rows[0]) + 1
    new_x = sx + int(cols[0])
    new_w = int(cols[-1]) - int(cols[0]) + 1

    return new_x, new_y, new_w, new_h


def _merge_overlapping(rects, overlap_thresh=0.5):
    """Merge rectangles whose overlap exceeds the threshold fraction of the smaller one."""
    if not rects:
        return []

    rects = sorted(rects, key=lambda r: r[4], reverse=True)
    merged = []

    for (x, y, w, h, area) in rects:
        absorbed = False
        for i, (mx, my, mw, mh, ma) in enumerate(merged):
            ix1, iy1 = max(x, mx), max(y, my)
            ix2, iy2 = min(x + w, mx + mw), min(y + h, my + mh)
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                if inter / min(area, ma) > overlap_thresh:
                    nx, ny = min(x, mx), min(y, my)
                    nx2, ny2 = max(x + w, mx + mw), max(y + h, my + mh)
                    nw, nh = nx2 - nx, ny2 - ny
                    merged[i] = (nx, ny, nw, nh, nw * nh)
                    absorbed = True
                    break
        if not absorbed:
            merged.append((x, y, w, h, area))

    return merged
