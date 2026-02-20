"""
Core image processing engine for PicsMake.
Detects individual photos in a scanned image and extracts them at highest quality.
"""

import cv2
import numpy as np
from PIL import Image
import gc


MAX_DETECTION_DIM = 2000  # px — working copy for detection
BG_THRESHOLD      = 235   # binary mask: pixels brighter than this = background
BRIGHT_THRESH     = 218   # intensity mean: columns/rows above this = white gap
MIN_GAP_PX        = 2     # minimum gap width in pixels
VALLEY_RATIO      = 0.04  # binary projection: gap threshold as fraction of max

SEAM_CENTER       = 0.15  # ignore outer N% of blob when searching for seams
SEAM_SMOOTH       = 21    # Sobel projection smoothing kernel (wider = better for spread seams)
SEAM_PEAK_RATIO   = 1.5   # seam peak must be >= this × mean of the projection


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

    # proj_mask: minimal cleanup – preserves thin gaps between photos
    dn_k      = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    proj_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, dn_k, iterations=1)

    # blob_mask: heavy close – fills bright areas inside a single photo
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

    # ── 3. Three-pass split ───────────────────────────────────────────────────
    #
    # Pass 1 – gap split: detects white-background gaps between photos.
    #   Uses both binary mask valleys AND intensity-mean brightness peaks so
    #   JPEG-degraded white borders (pixel values 215–234) are still caught.
    #
    # Pass 2 – seam split: detects boundaries between directly-touching photos.
    #   Uses Sobel gradient magnitude projection (not Canny) so the energy
    #   spread over 5-10px at a compressed boundary is fully captured.
    #   A relative threshold (peak vs column mean) adapts to content type.
    #
    # Pass 2 runs on the OUTPUT of pass 1, handling the common 2×2 grid where
    # one axis has a white gap and the other axis is touching.

    final = []
    for (x, y, w, h, _) in candidates:
        for (sx, sy, sw, sh) in _gap_split(gray, proj_mask, x, y, w, h):
            for box in _seam_split(gray, sx, sy, sw, sh, total_area, min_area_ratio):
                final.append(box + (box[2] * box[3],))

    candidates = [(x, y, w, h, a) for (x, y, w, h, a) in final
                  if a >= total_area * min_area_ratio]

    # ── 4. Merge, sort, refine ────────────────────────────────────────────────
    candidates = _merge_overlapping(candidates, overlap_thresh=0.6)
    candidates.sort(key=lambda r: (r[1] // (work_h // 6 or 1), r[0]))
    candidates = [_refine_crop(proj_mask, x, y, w, h) for (x, y, w, h, _) in candidates]

    if scale < 1.0:
        candidates = [(int(x / scale), int(y / scale),
                       int(w / scale), int(h / scale))
                      for (x, y, w, h) in candidates]

    del work, gray, raw_mask, proj_mask, blob_mask
    gc.collect()

    # ── 5. Extract crops ──────────────────────────────────────────────────────
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

def _gap_split(gray, proj_mask, x, y, w, h):
    """
    Detect white-background gaps between photos using two methods:

    1. Binary projection (proj_mask valleys): exact white pixels → near-zero count
    2. Intensity mean: average column/row brightness > BRIGHT_THRESH → white gap
       Catches JPEG-degraded borders whose pixels dip to 215-234.

    Returns list of (x, y, w, h).
    """
    # ── Method 1: binary projection ──
    region_mask = proj_mask[y:y + h, x:x + w]
    col_bin = np.sum(region_mask, axis=0).astype(np.float32) / 255.0
    row_bin = np.sum(region_mask, axis=1).astype(np.float32) / 255.0
    v_cuts = _valley_cuts(col_bin)
    h_cuts = _valley_cuts(row_bin)

    # ── Method 2: intensity mean ──
    region_gray = gray[y:y + h, x:x + w].astype(np.float32)
    col_mean = np.mean(region_gray, axis=0)
    row_mean = np.mean(region_gray, axis=1)
    v_cuts = _merge_cut_lists(v_cuts, _bright_cuts(col_mean))
    h_cuts = _merge_cut_lists(h_cuts, _bright_cuts(row_mean))

    x_segs = _to_segments(v_cuts, w)
    y_segs = _to_segments(h_cuts, h)

    results = []
    for (sy, sh) in y_segs:
        for (sx, sw) in x_segs:
            if np.any(region_mask[sy:sy + sh, sx:sx + sw]):
                results.append((x + sx, y + sy, sw, sh))
    return results or [(x, y, w, h)]


def _seam_split(gray, x, y, w, h, total_area, min_area_ratio):
    """
    Detect the boundary between directly-touching photos using Sobel gradient.

    Canny thins edges to 1px. A compressed photo boundary spreads over 5-10px,
    so Canny misses most of it. Sobel keeps the full gradient magnitude across
    the whole transition, giving a much stronger and wider signal.

    Uses a relative threshold: the peak must be >= SEAM_PEAK_RATIO × mean
    of the projection, so the detector adapts to the content's gradient level.

    Returns list of (x, y, w, h).
    """
    region = gray[y:y + h, x:x + w].astype(np.float32)

    # sobelx detects vertical seams; sobely detects horizontal seams
    sobelx = np.abs(cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3))
    sobely = np.abs(cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3))

    kernel   = np.ones(SEAM_SMOOTH) / SEAM_SMOOTH
    col_proj = np.convolve(np.sum(sobelx, axis=0), kernel, 'same')
    row_proj = np.convolve(np.sum(sobely, axis=1), kernel, 'same')

    v_cut = _best_seam(col_proj, w, h, total_area, min_area_ratio)
    h_cut = _best_seam(row_proj, h, w, total_area, min_area_ratio)

    valid_v = [v_cut] if v_cut is not None else []
    valid_h = [h_cut] if h_cut is not None else []

    if not valid_v and not valid_h:
        return [(x, y, w, h)]

    x_segs = _to_segments(valid_v, w)
    y_segs = _to_segments(valid_h, h)

    results = []
    for (sy, sh) in y_segs:
        for (sx, sw) in x_segs:
            results.append((x + sx, y + sy, sw, sh))
    return results or [(x, y, w, h)]


def _best_seam(proj, total, other_dim, total_area, min_area_ratio):
    """
    Find the single best seam position in the center of the projection.
    Returns the position or None if no convincing seam is found.
    """
    lo = int(total * SEAM_CENTER)
    hi = int(total * (1 - SEAM_CENTER))
    center = proj[lo:hi]
    if len(center) == 0:
        return None

    peak_local = int(np.argmax(center))
    peak_abs   = lo + peak_local
    peak_val   = float(proj[peak_abs])
    mean_val   = float(np.mean(center))

    # Relative: peak must be significantly above the mean
    if mean_val == 0 or peak_val < mean_val * SEAM_PEAK_RATIO:
        return None

    # Balance: 20–80% of total
    if not (0.20 <= peak_abs / total <= 0.80):
        return None

    # Both halves large enough
    if (peak_abs * other_dim < total_area * 0.02 or
            (total - peak_abs) * other_dim < total_area * 0.02):
        return None

    return peak_abs


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _valley_cuts(proj):
    """Midpoints of low-projection runs (binary mask approach)."""
    if len(proj) < 2 or proj.max() == 0:
        return []
    thresh    = proj.max() * VALLEY_RATIO
    is_low    = proj <= thresh
    cuts, in_valley, start = [], False, 0
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


def _bright_cuts(means):
    """Midpoints of high-intensity runs (intensity mean approach)."""
    is_bright = means > BRIGHT_THRESH
    cuts, in_gap, start = [], False, 0
    for i, bright in enumerate(is_bright):
        if bright and not in_gap:
            start = i; in_gap = True
        elif not bright and in_gap:
            if i - start >= MIN_GAP_PX:
                cuts.append((start + i) // 2)
            in_gap = False
    if in_gap and len(means) - start >= MIN_GAP_PX:
        cuts.append((start + len(means)) // 2)
    return cuts


def _merge_cut_lists(cuts_a, cuts_b, proximity=8):
    """Merge two sorted cut lists, deduplicating cuts within proximity px."""
    merged = sorted(set(cuts_a) | set(cuts_b))
    if not merged:
        return merged
    deduped = [merged[0]]
    for c in merged[1:]:
        if c - deduped[-1] >= proximity:
            deduped.append(c)
    return deduped


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
