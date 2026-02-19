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

# Projection split: white-gap detection
VALLEY_RATIO  = 0.04   # column/row must have <4% of max fg-pixels to count as a gap
MIN_GAP_PX    = 2      # minimum gap width in pixels

# Seam split: touching-photo boundary detection
SEAM_COVERAGE = 0.60   # seam edge must span this fraction of the blob's height/width
SEAM_SMOOTH   = 9      # box-filter size for smoothing the edge projection
SEAM_CENTER   = 0.15   # ignore the outer N% of the blob when searching for seams


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

    scale = min(1.0, MAX_DETECTION_DIM / max(height, width))
    work  = (cv2.resize(original, (int(width * scale), int(height * scale)),
                        interpolation=cv2.INTER_AREA)
             if scale < 1.0 else original)

    work_h, work_w = work.shape[:2]
    total_area     = work_h * work_w
    gray           = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    # ── 1. Two masks ──────────────────────────────────────────────────────────
    _, raw_mask = cv2.threshold(gray, BG_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # proj_mask: minimal cleanup – preserves thin gaps BETWEEN photos
    dn_k      = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    proj_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, dn_k, iterations=1)

    # blob_mask: heavy close – fills bright areas INSIDE a photo so each photo
    # appears as one solid blob for contour detection
    cl_k      = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blob_mask = cv2.morphologyEx(proj_mask, cv2.MORPH_CLOSE, cl_k, iterations=2)

    # ── 2. Coarse blob detection ──────────────────────────────────────────────
    candidates = _contour_candidates(blob_mask, total_area, min_area_ratio)

    if not candidates:
        blurred    = cv2.GaussianBlur(gray, (5, 5), 0)
        edges      = cv2.Canny(blurred, 30, 100)
        kern       = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        candidates = _contour_candidates(
            cv2.dilate(edges, kern, iterations=2), total_area, min_area_ratio)

    if not candidates:
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        k       = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        otsu    = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, k, iterations=3)
        otsu    = cv2.morphologyEx(otsu, cv2.MORPH_OPEN,  k, iterations=1)
        candidates = _contour_candidates(otsu, total_area, min_area_ratio)

    # ── 3. Two-pass split ─────────────────────────────────────────────────────
    #
    # Pass 1 – projection split on the raw mask.
    #   Finds near-zero valleys in row/col projections → white-background gaps.
    #
    # Pass 2 – seam split on each result from pass 1.
    #   Finds very tall/wide Canny-edge columns/rows → boundary between two
    #   photos that are directly touching (no white gap).
    #
    # Running pass 2 on the OUTPUT of pass 1 handles the common 2x2 grid where
    # one axis has a white gap (found by projection) and the other axis has
    # touching photos (found by seam).

    final = []
    for (x, y, w, h, _) in candidates:
        for (sx, sy, sw, sh) in _project_split(proj_mask, x, y, w, h):
            for box in _seam_split(gray, sx, sy, sw, sh, total_area, min_area_ratio):
                final.append(box + (box[2] * box[3],))

    candidates = [(x, y, w, h, a) for (x, y, w, h, a) in final
                  if a >= total_area * min_area_ratio]

    # ── 4. Merge truly overlapping boxes, sort, refine ────────────────────────
    candidates = _merge_overlapping(candidates, overlap_thresh=0.6)
    candidates.sort(key=lambda r: (r[1] // (work_h // 6 or 1), r[0]))
    candidates = [_refine_crop(proj_mask, x, y, w, h) for (x, y, w, h, _) in candidates]

    if scale < 1.0:
        candidates = [(int(x / scale), int(y / scale),
                       int(w / scale), int(h / scale))
                      for (x, y, w, h) in candidates]

    del work, gray, raw_mask, proj_mask, blob_mask
    gc.collect()

    # ── 5. Extract crops from the full-resolution original ────────────────────
    extracted = []
    for (x, y, w, h) in candidates:
        x1 = max(0, x - padding);  x2 = min(width,  x + w + padding)
        y1 = max(0, y - padding);  y2 = min(height, y + h + padding)
        crop_rgb = cv2.cvtColor(original[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        extracted.append(Image.fromarray(crop_rgb))

    del original, np_arr
    gc.collect()
    return extracted


# ── Split helpers ─────────────────────────────────────────────────────────────

def _project_split(proj_mask, x, y, w, h):
    """
    Split a bounding box at near-zero valleys in the foreground projection.
    Works when there is a white background gap between photos.
    Returns list of (x, y, w, h).
    """
    region   = proj_mask[y:y + h, x:x + w]
    col_proj = np.sum(region, axis=0).astype(np.float32) / 255.0
    row_proj = np.sum(region, axis=1).astype(np.float32) / 255.0

    v_cuts = _valley_cuts(col_proj)
    h_cuts = _valley_cuts(row_proj)

    x_segs = _to_segments(v_cuts, w)
    y_segs = _to_segments(h_cuts, h)

    results = []
    for (sy, sh) in y_segs:
        for (sx, sw) in x_segs:
            if np.any(region[sy:sy + sh, sx:sx + sw]):
                results.append((x + sx, y + sy, sw, sh))
    return results or [(x, y, w, h)]


def _seam_split(gray, x, y, w, h, total_area, min_area_ratio):
    """
    Split a bounding box at tall/wide Canny-edge seams.
    Works when two photos are directly touching (no white gap between them).

    A seam between two photos creates a strong vertical (or horizontal) edge
    that spans most of the blob's height (or width) — unlike in-photo edges
    which are rarely that tall or perfectly aligned.
    Returns list of (x, y, w, h).
    """
    region  = gray[y:y + h, x:x + w]
    blurred = cv2.GaussianBlur(region, (3, 3), 0)
    edges   = cv2.Canny(blurred, 20, 80)

    # Smooth projections to handle slight scan misalignment
    kernel   = np.ones(SEAM_SMOOTH) / SEAM_SMOOTH
    col_proj = np.convolve(np.sum(edges > 0, axis=0).astype(float), kernel, 'same')
    row_proj = np.convolve(np.sum(edges > 0, axis=1).astype(float), kernel, 'same')

    # A real seam must span at least SEAM_COVERAGE of the blob's dimension
    v_cuts = _seam_peaks(col_proj, h * SEAM_COVERAGE, w)
    h_cuts = _seam_peaks(row_proj, w * SEAM_COVERAGE, h)

    # Each resulting sub-box must be large enough and balanced (20-80% split)
    valid_v = [c for c in v_cuts
               if 0.20 <= c / w <= 0.80
               and min(c, w - c) * h >= total_area * min_area_ratio]
    valid_h = [c for c in h_cuts
               if 0.20 <= c / h <= 0.80
               and w * min(c, h - c) >= total_area * min_area_ratio]

    if not valid_v and not valid_h:
        return [(x, y, w, h)]

    x_segs = _to_segments(valid_v, w)
    y_segs = _to_segments(valid_h, h)

    results = []
    for (sy, sh) in y_segs:
        for (sx, sw) in x_segs:
            results.append((x + sx, y + sy, sw, sh))
    return results or [(x, y, w, h)]


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _valley_cuts(proj):
    """Midpoints of runs where proj < VALLEY_RATIO * max and >= MIN_GAP_PX wide."""
    if len(proj) < 2 or proj.max() == 0:
        return []
    thresh    = proj.max() * VALLEY_RATIO
    is_low    = proj <= thresh
    cuts      = []
    in_valley = False
    start     = 0
    for i, low in enumerate(is_low):
        if low and not in_valley:
            start = i; in_valley = True
        elif not low and in_valley:
            if i - start >= MIN_GAP_PX:
                cuts.append((start + i) // 2)
            in_valley = False
    if in_valley and len(proj) - start >= MIN_GAP_PX:
        cuts.append((start + len(proj)) // 2)
    return cuts


def _seam_peaks(proj, threshold, total):
    """
    Midpoints of runs above threshold in the center region of the projection.
    """
    lo, hi = int(total * SEAM_CENTER), int(total * (1 - SEAM_CENTER))
    center  = proj[lo:hi]
    if len(center) == 0 or center.max() < threshold:
        return []
    above = np.where(center >= threshold)[0]
    if len(above) == 0:
        return []
    groups, cur = [], [above[0]]
    for i in range(1, len(above)):
        if above[i] - above[i - 1] <= 5:
            cur.append(above[i])
        else:
            groups.append(cur); cur = [above[i]]
    groups.append(cur)
    return [lo + int(np.mean(g)) for g in groups]


def _to_segments(cuts, total):
    pts = [0] + cuts + [total]
    return [(pts[i], pts[i + 1] - pts[i])
            for i in range(len(pts) - 1) if pts[i + 1] > pts[i]]


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


def _refine_crop(fg_mask, x, y, w, h):
    """Tighten a bounding box to the exact foreground extent."""
    margin = 8
    sx = max(0, x - margin);  ex = min(fg_mask.shape[1], x + w + margin)
    sy = max(0, y - margin);  ey = min(fg_mask.shape[0], y + h + margin)
    region = fg_mask[sy:ey, sx:ex]
    rows   = np.where(np.any(region, axis=1))[0]
    cols   = np.where(np.any(region, axis=0))[0]
    if not len(rows) or not len(cols):
        return x, y, w, h
    return (sx + int(cols[0]), sy + int(rows[0]),
            int(cols[-1]) - int(cols[0]) + 1,
            int(rows[-1]) - int(rows[0]) + 1)


def _merge_overlapping(rects, overlap_thresh=0.6):
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
