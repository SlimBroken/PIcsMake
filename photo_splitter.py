"""
Core image processing engine for PicsMake.
Detects individual photos in a scanned image and extracts them at highest quality.
"""

import cv2
import numpy as np
from PIL import Image
import gc


MAX_DETECTION_DIM = 2000  # px — working copy for detection
BG_THRESHOLD      = 235   # pixels brighter than this = scanner background
VALLEY_RATIO      = 0.04  # column/row must have <4% of max fg-pixels to be a gap
MIN_GAP_PX        = 2     # gap must be at least this many pixels wide to split


# ── Public API ────────────────────────────────────────────────────────────────

def detect_and_extract_photos(image_bytes, min_area_ratio=0.02, padding=4):
    """
    Detect individual photos in a scanned image and extract them.
    Returns a list of PIL Image objects, one per detected photo.
    """
    np_arr   = np.frombuffer(image_bytes, np.uint8)
    original = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError("Could not decode the uploaded image.")

    height, width = original.shape[:2]

    # Downscale for detection; scale bounding boxes back before cropping.
    scale = min(1.0, MAX_DETECTION_DIM / max(height, width))
    work  = (cv2.resize(original, (int(width * scale), int(height * scale)),
                        interpolation=cv2.INTER_AREA)
             if scale < 1.0 else original)

    work_h, work_w = work.shape[:2]
    total_area     = work_h * work_w
    gray           = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    # ── 1. Two masks with different purposes ─────────────────────────────────
    _, raw_mask = cv2.threshold(gray, BG_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # proj_mask: minimal cleanup only — preserves the thin gaps BETWEEN photos
    # so projection analysis can find valleys there.
    denoise_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    proj_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, denoise_k, iterations=1)

    # blob_mask: aggressively closed — fills white areas INSIDE a photo
    # (e.g. bright sky) so each photo appears as one solid blob for contour detection.
    close_k   = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blob_mask = cv2.morphologyEx(proj_mask, cv2.MORPH_CLOSE, close_k, iterations=2)

    # ── 2. Find coarse blobs from the closed mask ─────────────────────────────
    candidates = _contour_candidates(blob_mask, total_area, min_area_ratio)

    # Fallback: edge-based if background approach found nothing
    if not candidates:
        blurred    = cv2.GaussianBlur(gray, (5, 5), 0)
        edges      = cv2.Canny(blurred, 30, 100)
        kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated    = cv2.dilate(edges, kernel, iterations=2)
        candidates = _contour_candidates(dilated, total_area, min_area_ratio)

    # Fallback 2: Otsu
    if not candidates:
        blurred  = cv2.GaussianBlur(gray, (7, 7), 0)
        _, otsu  = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        k        = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        otsu     = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, k, iterations=3)
        otsu     = cv2.morphologyEx(otsu, cv2.MORPH_OPEN,  k, iterations=1)
        candidates = _contour_candidates(otsu, total_area, min_area_ratio)

    # ── 3. Projection-split on the RAW mask (gaps intact) ────────────────────
    # proj_mask has no CLOSE applied, so white gaps between touching photos
    # still appear as near-zero valleys in the column/row projections.
    split = []
    for (x, y, w, h, _) in candidates:
        split.extend(_project_split(proj_mask, x, y, w, h))
    candidates = [(x, y, w, h, w * h) for (x, y, w, h) in split
                  if w * h >= total_area * min_area_ratio]

    # ── 4. Merge only genuinely overlapping boxes ─────────────────────────────
    candidates = _merge_overlapping(candidates, overlap_thresh=0.6)

    # Sort top-to-bottom, left-to-right
    candidates.sort(key=lambda r: (r[1] // (work_h // 6 or 1), r[0]))

    # ── 5. Refine each box to exact photo edges (using raw proj_mask) ─────────
    candidates = [_refine_crop(proj_mask, x, y, w, h) for (x, y, w, h, _) in candidates]

    # Scale back to original coordinates
    if scale < 1.0:
        candidates = [(int(x / scale), int(y / scale),
                       int(w / scale), int(h / scale))
                      for (x, y, w, h) in candidates]

    del work, gray, raw_mask, proj_mask, blob_mask
    gc.collect()

    # ── 6. Extract from full-resolution original ──────────────────────────────
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

def _contour_candidates(mask, total_area, min_area_ratio):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in contours:
        if cv2.contourArea(c) < total_area * min_area_ratio:
            continue
        x, y, w, h = cv2.boundingRect(c)
        area   = w * h
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 8 or area > total_area * 0.95:
            continue
        results.append((x, y, w, h, area))
    return results


def _project_split(fg_mask, x, y, w, h):
    """
    Split a bounding box into sub-boxes by finding low-projection valleys.
    Works for photos arranged in a grid that touch each other.
    Returns a list of (x, y, w, h) boxes.
    """
    region   = fg_mask[y:y + h, x:x + w]
    col_proj = np.sum(region, axis=0).astype(np.float32) / 255.0
    row_proj = np.sum(region, axis=1).astype(np.float32) / 255.0

    v_cuts = _valley_cuts(col_proj)   # vertical splits (separate left/right photos)
    h_cuts = _valley_cuts(row_proj)   # horizontal splits (separate top/bottom photos)

    x_segs = _to_segments(v_cuts, w)
    y_segs = _to_segments(h_cuts, h)

    results = []
    for (sy, sh) in y_segs:
        for (sx, sw) in x_segs:
            if np.any(region[sy:sy + sh, sx:sx + sw]):
                results.append((x + sx, y + sy, sw, sh))

    return results or [(x, y, w, h)]


def _valley_cuts(proj):
    """
    Find midpoints of "valley" segments in a 1-D projection array.
    A valley is a run of values below VALLEY_RATIO * max, at least MIN_GAP_PX wide.
    """
    if len(proj) < 2 or proj.max() == 0:
        return []

    thresh    = proj.max() * VALLEY_RATIO
    is_low    = proj <= thresh
    cuts      = []
    in_valley = False
    start     = 0

    for i, low in enumerate(is_low):
        if low and not in_valley:
            start     = i
            in_valley = True
        elif not low and in_valley:
            if i - start >= MIN_GAP_PX:
                cuts.append((start + i) // 2)
            in_valley = False

    if in_valley and len(proj) - start >= MIN_GAP_PX:
        cuts.append((start + len(proj)) // 2)

    return cuts


def _to_segments(cuts, total):
    pts = [0] + cuts + [total]
    return [(pts[i], pts[i + 1] - pts[i])
            for i in range(len(pts) - 1)
            if pts[i + 1] > pts[i]]


def _refine_crop(fg_mask, x, y, w, h):
    """Tighten a bounding box to the exact foreground extent."""
    margin = 8
    sx = max(0, x - margin);  ex = min(fg_mask.shape[1], x + w + margin)
    sy = max(0, y - margin);  ey = min(fg_mask.shape[0], y + h + margin)

    region      = fg_mask[sy:ey, sx:ex]
    row_content = np.any(region, axis=1)
    col_content = np.any(region, axis=0)

    rows = np.where(row_content)[0]
    cols = np.where(col_content)[0]
    if not len(rows) or not len(cols):
        return x, y, w, h

    return (sx + int(cols[0]),
            sy + int(rows[0]),
            int(cols[-1]) - int(cols[0]) + 1,
            int(rows[-1]) - int(rows[0]) + 1)


def _merge_overlapping(rects, overlap_thresh=0.6):
    """Merge rectangles whose overlap/smaller-area exceeds the threshold."""
    if not rects:
        return []
    rects  = sorted(rects, key=lambda r: r[4], reverse=True)
    merged = []
    for (x, y, w, h, area) in rects:
        absorbed = False
        for i, (mx, my, mw, mh, ma) in enumerate(merged):
            ix1, iy1 = max(x, mx), max(y, my)
            ix2, iy2 = min(x + w, mx + mw), min(y + h, my + mh)
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                if inter / min(area, ma) > overlap_thresh:
                    nx, ny   = min(x, mx), min(y, my)
                    nx2, ny2 = max(x + w, mx + mw), max(y + h, my + mh)
                    nw, nh   = nx2 - nx, ny2 - ny
                    merged[i] = (nx, ny, nw, nh, nw * nh)
                    absorbed  = True
                    break
        if not absorbed:
            merged.append((x, y, w, h, area))
    return merged
