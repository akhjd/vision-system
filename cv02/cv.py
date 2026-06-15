"""
Three-dot vertical marker detector.
No color masking — works on any background.
Detects 3 vertically stacked circles by geometry alone.
Runs on phone or PC camera.

Usage:
    python3 dot_detector.py
    python3 dot_detector.py --camera 1      # use a different camera index
    python3 dot_detector.py --no-debug      # hide debug windows

Press q to quit.
Press s to save debug images.
"""

import cv2
import numpy as np
from collections import deque, Counter
import time
import os
import argparse


# ============================================================
# SYMBOL MAPPING
# 0 = white circle
# 1 = black circle
# 2 = half black left
# 3 = half black right
# 4 = half black top
# 5 = half black bottom
# ============================================================

LABEL_TO_NUMBER = {
    "white_circle":       0,
    "black_circle":       1,
    "half_black_left":    2,
    "half_black_right":   3,
    "half_black_top":     4,
    "half_black_bottom":  5,
}

NUMBER_TO_LABEL = {v: k for k, v in LABEL_TO_NUMBER.items()}


# ============================================================
# CONFIG — tune these if detection is too noisy or too strict
# ============================================================

CAMERA_INDEX = 1

# Blob shape filtering
MIN_BLOB_AREA        = 60       # px^2 — ignore specks
MIN_CIRCULARITY      = 0.45     # 1.0 = perfect circle; lower = allow more distortion
MIN_ASPECT           = 0.55     # bounding box w/h ratio lower bound
MAX_ASPECT           = 1.45     # bounding box w/h ratio upper bound

# Group geometry — how the 3 dots must relate to each other
MAX_X_SPREAD_FACTOR  = 0.90     # max horizontal spread as fraction of avg radius
MIN_GAP_RATIO        = 1.6      # min gap between dot centers / avg radius
MAX_GAP_RATIO        = 5.5      # max gap between dot centers / avg radius
MIN_GAP_SIMILARITY   = 0.60     # how equal the two gaps must be (0–1)
MIN_RADIUS_SIMILARITY= 0.60     # how equal the three radii must be (0–1)
MIN_AREA_SIMILARITY  = 0.40     # how equal the three areas must be (0–1)
MIN_LINE_ERROR_SCALE = 0.50     # max x deviation from vertical center line / avg_r
MIN_MARKER_ASPECT    = 1.8      # marker height/width must be at least this
MAX_MARKER_ASPECT    = 6.0      # and at most this

# Overall geometry confidence threshold to accept a group
MIN_GEOMETRY_CONF    = 0.70

# ROI lock — once found, search only near the last known marker
ROI_EXPAND           = 2.8      # how much larger to make the search ROI
MAX_MISSING_FRAMES   = 15       # frames to lose marker before unlocking ROI

# Temporal voting — require stable code across many frames
HISTORY_LEN          = 18
MIN_STABLE_COUNT     = 11       # out of HISTORY_LEN frames

# Circle normalization
CIRCLE_RADIUS_SCALE  = 0.80     # fraction of detected radius to actually sample

# Debug output folder
DEBUG_DIR = "debug_saves"


# ============================================================
# BLOB DETECTION — no color mask, uses adaptive threshold
# ============================================================

def find_blobs(frame):
    """
    Finds candidate circular blobs in the frame using adaptive thresholding.
    No color assumptions — works on any background.
    Returns list of blob dicts and the binary mask used.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold finds local dark/light transitions regardless of background
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=8
    )

    # Mild cleanup — remove salt & pepper, close small gaps
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel_open)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_BLOB_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 5:
            continue

        aspect = w / float(h)
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue

        cx = x + w // 2
        cy = y + h // 2
        radius = int((w + h) / 4)

        blobs.append({
            "center":      (cx, cy),
            "bbox":        (x, y, w, h),
            "radius":      radius,
            "area":        area,
            "circularity": circularity,
        })

    return blobs, binary


# ============================================================
# FIND BEST VERTICAL THREE-CIRCLE GROUP
# ============================================================

def find_best_vertical_three(blobs):
    """
    Finds the best 3-blob group that looks like a vertical marker.
    No hard minimum radius — small circles are allowed if geometry is clean.
    Returns (group, info_dict) or (None, None).
    """
    if len(blobs) < 3:
        return None, None

    best_group  = None
    best_info   = None
    best_score  = -1

    blobs_sorted = sorted(blobs, key=lambda b: b["center"][1])

    for i in range(len(blobs_sorted)):
        for j in range(i + 1, len(blobs_sorted)):
            for k in range(j + 1, len(blobs_sorted)):
                group = [blobs_sorted[i], blobs_sorted[j], blobs_sorted[k]]

                xs    = [b["center"][0] for b in group]
                ys    = [b["center"][1] for b in group]
                rs    = [max(b["radius"], 1) for b in group]
                areas = [max(b["area"],   1) for b in group]

                avg_r = float(np.mean(rs))
                min_r = min(rs)
                max_r = max(rs)

                # ---- radius similarity ----
                radius_sim = min_r / max_r
                if radius_sim < MIN_RADIUS_SIMILARITY:
                    continue

                # ---- horizontal alignment ----
                x_spread = max(xs) - min(xs)
                if x_spread > avg_r * MAX_X_SPREAD_FACTOR:
                    continue

                mean_x    = float(np.mean(xs))
                line_error = float(np.mean([abs(x - mean_x) for x in xs]))
                if line_error > avg_r * MIN_LINE_ERROR_SCALE:
                    continue

                # ---- gap ratios ----
                gap1 = ys[1] - ys[0]
                gap2 = ys[2] - ys[1]
                if gap1 <= 0 or gap2 <= 0:
                    continue

                g1r = gap1 / avg_r
                g2r = gap2 / avg_r
                if g1r < MIN_GAP_RATIO or g1r > MAX_GAP_RATIO:
                    continue
                if g2r < MIN_GAP_RATIO or g2r > MAX_GAP_RATIO:
                    continue

                gap_sim = min(gap1, gap2) / max(gap1, gap2)
                if gap_sim < MIN_GAP_SIMILARITY:
                    continue

                # ---- area similarity ----
                area_sim = min(areas) / max(areas)
                if area_sim < MIN_AREA_SIMILARITY:
                    continue

                # ---- marker aspect ratio (must look like a tall vertical strip) ----
                marker_h = (ys[2] + avg_r) - (ys[0] - avg_r)
                marker_w = (max(xs) + avg_r) - (min(xs) - avg_r)
                if marker_w <= 0:
                    continue
                marker_aspect = marker_h / marker_w
                if marker_aspect < MIN_MARKER_ASPECT or marker_aspect > MAX_MARKER_ASPECT:
                    continue

                # ---- soft size score (no hard rejection) ----
                size_score = min(avg_r / 22.0, 1.0)

                x_align_score = 1.0 - min(x_spread / max(avg_r * 1.5, 1), 1.0)
                line_score    = 1.0 - min(line_error / max(avg_r, 1),   1.0)

                geometry_conf = (
                    0.25 * radius_sim    +
                    0.25 * gap_sim       +
                    0.20 * x_align_score +
                    0.15 * line_score    +
                    0.10 * area_sim      +
                    0.05 * size_score
                )

                if geometry_conf < MIN_GEOMETRY_CONF:
                    continue

                score = (
                    10000 * geometry_conf
                    + 300 * marker_aspect
                    - 50  * x_spread
                    - 40  * line_error
                )

                if score > best_score:
                    best_score = score
                    best_group = group
                    best_info  = {
                        "avg_radius":       avg_r,
                        "gap1":             gap1,
                        "gap2":             gap2,
                        "gap1_ratio":       g1r,
                        "gap2_ratio":       g2r,
                        "gap_similarity":   gap_sim,
                        "radius_sim":       radius_sim,
                        "area_sim":         area_sim,
                        "x_spread":         x_spread,
                        "line_error":       line_error,
                        "marker_aspect":    marker_aspect,
                        "geometry_conf":    geometry_conf,
                    }

    return best_group, best_info


# ============================================================
# CROP MARKER REGION
# ============================================================

def crop_marker_region(frame, group, margin_scale=0.80):
    xs, ys = [], []
    for b in group:
        cx, cy = b["center"]
        r = b["radius"]
        xs += [cx - r, cx + r]
        ys += [cy - r, cy + r]

    avg_r  = int(np.mean([b["radius"] for b in group]))
    margin = int(avg_r * margin_scale)

    x1 = max(int(min(xs)) - margin, 0)
    y1 = max(int(min(ys)) - margin, 0)
    x2 = min(int(max(xs)) + margin, frame.shape[1])
    y2 = min(int(max(ys)) + margin, frame.shape[0])

    crop = frame[y1:y2, x1:x2].copy()

    local_group = [
        {**b, "center": (b["center"][0] - x1, b["center"][1] - y1)}
        for b in group
    ]

    return crop, local_group, (x1, y1, x2, y2)


# ============================================================
# NORMALIZE — grey outside, black/white inside circles
# ============================================================

def normalize_crop(crop, group):
    gray       = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    normalized = np.full_like(gray, 127)     # everything starts grey

    circle_infos = []
    group = sorted(group, key=lambda b: b["center"][1])

    for b in group:
        cx, cy = b["center"]
        r = int(b["radius"] * CIRCLE_RADIUS_SCALE)
        if r <= 2:
            continue

        mask   = np.zeros_like(gray)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        pixels = gray[mask == 255]

        if len(pixels) == 0:
            continue

        p10     = float(np.percentile(pixels, 10))
        p50     = float(np.percentile(pixels, 50))
        p90     = float(np.percentile(pixels, 90))
        contrast = p90 - p10

        if p50 < 85 and contrast < 60:
            # Solid dark circle
            normalized[mask == 255] = 0
            threshold_used = p50

        elif p50 > 175 and contrast < 60:
            # Solid light circle
            normalized[mask == 255] = 255
            threshold_used = p50

        else:
            # Mixed — use Otsu only on pixels inside the circle
            _, binary_circle = cv2.threshold(
                pixels.astype(np.uint8),
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            normalized[mask == 255] = binary_circle.flatten()
            threshold_used, _ = cv2.threshold(
                pixels.astype(np.uint8), 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            threshold_used = float(threshold_used)

        circle_infos.append({
            "center":    (cx, cy),
            "radius":    r,
            "mask":      mask,
            "threshold": threshold_used,
            "p10": p10, "p50": p50, "p90": p90,
            "contrast":  contrast,
        })

    normalized_bgr = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    return normalized, normalized_bgr, circle_infos


# ============================================================
# CLASSIFY EACH NORMALIZED CIRCLE
# ============================================================

def classify_circle(normalized_gray, info):
    cx, cy = info["center"]
    r      = info["radius"]

    mask = np.zeros_like(normalized_gray)
    cv2.circle(mask, (cx, cy), r, 255, -1)

    inside = normalized_gray[mask == 255]
    if len(inside) == 0:
        return "unknown", 0.0, {}

    black_ratio = float(np.mean(inside < 128))

    if black_ratio < 0.18:
        return "white_circle", 1.0 - black_ratio, {"black_ratio": black_ratio}

    if black_ratio > 0.82:
        return "black_circle", black_ratio, {"black_ratio": black_ratio}

    # Half-split detection
    left_mask   = np.zeros_like(mask); left_mask[:,  :cx]  = mask[:,  :cx]
    right_mask  = np.zeros_like(mask); right_mask[:, cx:]  = mask[:, cx:]
    top_mask    = np.zeros_like(mask); top_mask[   :cy, :] = mask[:cy, :]
    bottom_mask = np.zeros_like(mask); bottom_mask[cy:, :] = mask[cy:, :]

    def br(m):
        px = normalized_gray[m == 255]
        return float(np.mean(px < 128)) if len(px) else 0.0

    lb, rb, tb, bb = br(left_mask), br(right_mask), br(top_mask), br(bottom_mask)

    scores = {
        "half_black_left":   lb - rb,
        "half_black_right":  rb - lb,
        "half_black_top":    tb - bb,
        "half_black_bottom": bb - tb,
    }

    label      = max(scores, key=scores.get)
    confidence = float(scores[label])

    if confidence < 0.20:
        return "unknown", confidence, {"black_ratio": black_ratio}

    return label, confidence, {
        "black_ratio": black_ratio,
        "left": lb, "right": rb, "top": tb, "bottom": bb,
        "split_conf": confidence,
    }


# ============================================================
# FULL PIPELINE FOR ONE FRAME / ROI
# ============================================================

def read_frame(frame):
    blobs, binary_mask = find_blobs(frame)
    group, group_info  = find_best_vertical_three(blobs)

    if group is None:
        return {
            "numbers": None, "detections": [],
            "crop": None, "normalized_bgr": None,
            "binary_mask": binary_mask, "blobs": blobs,
            "crop_box": None, "group_info": None,
        }

    crop, local_group, crop_box = crop_marker_region(frame, group)
    norm_gray, norm_bgr, circle_infos = normalize_crop(crop, local_group)

    circle_infos = sorted(circle_infos, key=lambda c: c["center"][1])
    detections   = []

    for info in circle_infos:
        label, conf, details = classify_circle(norm_gray, info)
        detections.append({
            "label":      label,
            "number":     LABEL_TO_NUMBER.get(label, -1),
            "center":     info["center"],
            "radius":     info["radius"],
            "confidence": conf,
            "details":    details,
        })

    if len(detections) == 3 and all(d["number"] != -1 for d in detections):
        numbers = [d["number"] for d in detections]
    else:
        numbers = None

    return {
        "numbers":       numbers,
        "detections":    detections,
        "crop":          crop,
        "normalized_bgr": norm_bgr,
        "binary_mask":   binary_mask,
        "blobs":         blobs,
        "crop_box":      crop_box,
        "group_info":    group_info,
    }


# ============================================================
# TEMPORAL STABILITY
# ============================================================

def stable_code(history, min_count=MIN_STABLE_COUNT):
    if not history:
        return None
    code, count = Counter(history).most_common(1)[0]
    return list(code) if count >= min_count else None


# ============================================================
# ROI HELPERS
# ============================================================

def expanded_roi(frame_shape, crop_box):
    fh, fw = frame_shape[:2]
    x1, y1, x2, y2 = crop_box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    w, h   = x2 - x1, y2 - y1
    nw, nh = int(w * ROI_EXPAND), int(h * ROI_EXPAND)
    return (
        max(cx - nw // 2, 0),
        max(cy - nh // 2, 0),
        min(cx + nw // 2, fw),
        min(cy + nh // 2, fh),
    )

def shift_box(box, ox, oy):
    if box is None: return None
    x1, y1, x2, y2 = box
    return x1 + ox, y1 + oy, x2 + ox, y2 + oy

def shift_blobs(blobs, ox, oy):
    out = []
    for b in blobs:
        cx, cy = b["center"]
        bx, by, bw, bh = b["bbox"]
        nb = dict(b)
        nb["center"] = (cx + ox, cy + oy)
        nb["bbox"]   = (bx + ox, by + oy, bw, bh)
        out.append(nb)
    return out


# ============================================================
# DEBUG DRAWING
# ============================================================

def draw_debug(frame, result, code, locked, last_box, roi_box, blobs):
    out = frame.copy()

    if roi_box:
        x1, y1, x2, y2 = roi_box
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 220, 0), 1)
        cv2.putText(out, "ROI", (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 0), 1)

    for b in blobs:
        cv2.circle(out, b["center"], b["radius"], (180, 180, 180), 1)

    crop_box = result.get("crop_box")
    if crop_box:
        cv2.rectangle(out, crop_box[:2], crop_box[2:], (0, 230, 0), 2)

    if last_box:
        cv2.rectangle(out, last_box[:2], last_box[2:], (0, 140, 255), 1)

    status = "LOCKED" if locked else "SEARCHING"
    if code:
        txt   = f"{status}  CODE: {code[0]} {code[1]} {code[2]}"
        color = (0, 230, 0)
    elif result.get("numbers"):
        txt   = f"{status}  reading: {result['numbers']}"
        color = (0, 220, 220)
    else:
        txt   = f"{status}  ..."
        color = (0, 60, 220)

    cv2.putText(out, txt, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)

    gi = result.get("group_info")
    if gi:
        info = (f"r={gi['avg_radius']:.1f}  "
                f"gaps={gi['gap1_ratio']:.2f},{gi['gap2_ratio']:.2f}  "
                f"conf={gi['geometry_conf']:.2f}")
        cv2.putText(out, info, (16, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)

    return out


def draw_norm_debug(norm_bgr, detections):
    if norm_bgr is None:
        return None
    out = norm_bgr.copy()
    for d in detections:
        cx, cy = d["center"]
        r      = d["radius"]
        cv2.circle(out, (cx, cy), r, (0, 220, 0), 2)
        cv2.putText(out, f"{d['number']}:{d['label'][:4]}",
                    (max(cx - r, 0), max(cy - r - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 0), 1)
    return out


# ============================================================
# DEBUG SAVE
# ============================================================

def save_debug(frame, result, debug_frame):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    ts = int(time.time() * 1000)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{ts}_frame.png"),       frame)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{ts}_debug.png"),       debug_frame)
    if result.get("binary_mask") is not None:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{ts}_mask.png"),    result["binary_mask"])
    if result.get("crop") is not None:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{ts}_crop.png"),    result["crop"])
    if result.get("normalized_bgr") is not None:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{ts}_norm.png"),    result["normalized_bgr"])
    print(f"[s] Saved debug images → {DEBUG_DIR}/")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",   type=int, default=CAMERA_INDEX)
    parser.add_argument("--no-debug", action="store_true",
                        help="Suppress debug windows (terminal output only)")
    args = parser.parse_args()

    show_debug = not args.no_debug

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return

    locked         = False
    last_box       = None
    missing_frames = 0

    history         = deque(maxlen=HISTORY_LEN)
    last_printed    = None

    print("Running  — q to quit, s to save debug images")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        # --- ROI locking ---
        roi_box    = None
        roi_offset = (0, 0)
        search     = frame

        if locked and last_box is not None:
            rx1, ry1, rx2, ry2 = expanded_roi(frame.shape, last_box)
            roi_box    = (rx1, ry1, rx2, ry2)
            search     = frame[ry1:ry2, rx1:rx2]
            roi_offset = (rx1, ry1)

        result = read_frame(search)

        # Shift results back to full-frame coords
        if roi_offset != (0, 0):
            ox, oy = roi_offset
            result["crop_box"] = shift_box(result["crop_box"], ox, oy)
            result["blobs"]    = shift_blobs(result["blobs"],  ox, oy)

        numbers  = result["numbers"]
        crop_box = result["crop_box"]

        if numbers is not None and crop_box is not None:
            locked         = True
            last_box       = crop_box
            missing_frames = 0
            history.append(tuple(numbers))

            gi = result["group_info"]
            if gi:
                print(
                    f"  code={numbers}  "
                    f"r={gi['avg_radius']:.1f}  "
                    f"gaps={gi['gap1_ratio']:.2f},{gi['gap2_ratio']:.2f}  "
                    f"conf={gi['geometry_conf']:.2f}"
                )
        else:
            missing_frames += 1
            if missing_frames > MAX_MISSING_FRAMES:
                locked   = False
                last_box = None
                history.clear()

        code = stable_code(history)

        if code is not None and code != last_printed:
            print(f"Stable detected code: {code}")
            last_printed = code

        if show_debug:
            debug_frame = draw_debug(
                frame, result, code, locked, last_box, roi_box, result["blobs"]
            )
            norm_debug = draw_norm_debug(result.get("normalized_bgr"), result.get("detections", []))

            cv2.imshow("Dot Detector",           debug_frame)
            cv2.imshow("Binary Mask",            result["binary_mask"])
            if result.get("crop")          is not None: cv2.imshow("Crop",       result["crop"])
            if norm_debug                  is not None: cv2.imshow("Normalized",  norm_debug)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            if show_debug:
                save_debug(frame, result, debug_frame)
            else:
                save_debug(frame, result, frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()