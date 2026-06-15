"""
Combined circle marker detector + CNN classifier.

Usage:
    python3 run.py
    python3 run.py --model model_best.pth
    python3 run.py --camera 1
    python3 run.py --no-debug
    python3 run.py --verbose     # prints why each blob group is rejected

Press q to quit.
Press s to save debug images.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from collections import deque, Counter
import argparse
import time
import os


# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

CAMERA_INDEX          = 0
MODEL_PATH            = "model_best.pth"
IMAGE_SIZE            = 64

# Blob detection — loosened to catch real printed circles
MIN_BLOB_AREA         = 50
MIN_CIRCULARITY       = 0.40     # printed circles aren't perfect
MIN_ASPECT            = 0.50
MAX_ASPECT            = 1.50
MIN_FILL_RATIO        = 0.55     # rejects half-circle dark blobs
MAX_BLOB_RADIUS       = 80       # rejects giant sticker outline blobs

# Group geometry
MAX_X_SPREAD_FACTOR   = 1.00     # allow slight horizontal offset
MIN_GAP_RATIO         = 1.4      # gap / avg_radius lower bound
MAX_GAP_RATIO         = 6.0
MIN_GAP_SIMILARITY    = 0.55
MIN_RADIUS_SIMILARITY = 0.55
MIN_AREA_SIMILARITY   = 0.35
MIN_LINE_ERROR_SCALE  = 0.60
MIN_MARKER_ASPECT     = 1.6
MAX_MARKER_ASPECT     = 7.0
MIN_GEOMETRY_CONF     = 0.68

# ROI lock
ROI_EXPAND            = 2.8
MAX_MISSING_FRAMES    = 15

# Temporal voting
HISTORY_LEN           = 18
MIN_STABLE_COUNT      = 11

# Circle crop margin
CROP_MARGIN_SCALE     = 0.25

# Model confidence threshold
MIN_MODEL_CONFIDENCE  = 0.65

DEBUG_DIR             = "debug_saves"

CLASS_NAMES = {
    0: "white",
    1: "dark",
    2: "half_L",
    3: "half_R",
    4: "half_T",
    5: "half_B",
}

# Global verbose flag — set by --verbose arg
VERBOSE = False


# ════════════════════════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════════════════════════

class CircleClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_model(model_path):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    checkpoint = torch.load(model_path, map_location=device)
    model = CircleClassifier(num_classes=6).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Loaded: {model_path}")
    print(f"  Val accuracy: {checkpoint.get('val_acc', '?'):.1%}")
    print(f"  Classes: {checkpoint.get('class_names', '?')}\n")

    return model, device


# ════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ════════════════════════════════════════════════════════════════════════════

infer_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def classify_circle_crop(model, device, crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor   = infer_transform(crop_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
    return int(pred.item()), float(conf.item())


# ════════════════════════════════════════════════════════════════════════════
# BLOB DETECTION
# ════════════════════════════════════════════════════════════════════════════

def find_blobs(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31, C=8,
    )

    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel_open)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # ── Fill all contours to turn rings/partial arcs into solid blobs ─────
    # Strategy:
    #   1. Find all contours with RETR_CCOMP (2-level: outer + holes)
    #   2. Fill every hole (parent >= 0) regardless of parent shape —
    #      this handles white circles which appear as holes in the mask
    #   3. Also draw every outer contour filled — handles dark circles
    #   4. Apply a closing pass to bridge gaps in half-circle arcs
    filled = binary.copy()
    contours_ccomp, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is not None:
        for idx, h_entry in enumerate(hierarchy[0]):
            # Fill both holes (parent>=0) and outer contours (parent<0)
            # that are roughly circular — this catches all circle types
            cnt  = contours_ccomp[idx]
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue
            circ = 4 * np.pi * area / (peri * peri)
            # Fill anything reasonably circular — loose threshold
            if circ > 0.30:
                cv2.drawContours(filled, contours_ccomp, idx, 255, -1)

    # Close small gaps (helps half-circle arcs merge into full blobs)
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel_fill)
    # ─────────────────────────────────────────────────────────────────────

    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

        radius     = int((w + h) / 4)
        if radius > MAX_BLOB_RADIUS:
            continue

        fill_ratio = area / max(np.pi * radius * radius, 1)
        if fill_ratio < MIN_FILL_RATIO:
            continue

        cx = x + w // 2
        cy = y + h // 2

        blobs.append({
            "center":      (cx, cy),
            "bbox":        (x, y, w, h),
            "radius":      radius,
            "area":        area,
            "circularity": circularity,
            "fill_ratio":  fill_ratio,
        })

    if VERBOSE and len(blobs) > 0:
        print(f"  [blobs] found {len(blobs)}: "
              + "  ".join(f"r={b['radius']} circ={b['circularity']:.2f} "
                          f"fill={b['fill_ratio']:.2f} @ {b['center']}"
                          for b in blobs[:6]))

    return blobs, filled


# ════════════════════════════════════════════════════════════════════════════
# FIND BEST VERTICAL THREE-CIRCLE GROUP
# ════════════════════════════════════════════════════════════════════════════

def find_best_vertical_three(blobs):
    if len(blobs) < 3:
        if VERBOSE:
            print(f"  [group] rejected: only {len(blobs)} blobs")
        return None, None

    best_group = None
    best_info  = None
    best_score = -1

    blobs_sorted = sorted(blobs, key=lambda b: b["center"][1])
    reject_reasons = Counter()

    for i in range(len(blobs_sorted)):
        for j in range(i + 1, len(blobs_sorted)):
            for k in range(j + 1, len(blobs_sorted)):
                group = [blobs_sorted[i], blobs_sorted[j], blobs_sorted[k]]

                xs    = [b["center"][0] for b in group]
                ys    = [b["center"][1] for b in group]
                rs    = [max(b["radius"], 1) for b in group]
                areas = [max(b["area"],   1) for b in group]

                avg_r      = float(np.mean(rs))
                radius_sim = min(rs) / max(rs)
                if radius_sim < MIN_RADIUS_SIMILARITY:
                    reject_reasons["radius_sim"] += 1
                    continue

                x_spread = max(xs) - min(xs)
                if x_spread > avg_r * MAX_X_SPREAD_FACTOR:
                    reject_reasons["x_spread"] += 1
                    continue

                mean_x     = float(np.mean(xs))
                line_error = float(np.mean([abs(x - mean_x) for x in xs]))
                if line_error > avg_r * MIN_LINE_ERROR_SCALE:
                    reject_reasons["line_error"] += 1
                    continue

                gap1 = ys[1] - ys[0]
                gap2 = ys[2] - ys[1]
                if gap1 <= 0 or gap2 <= 0:
                    reject_reasons["gap_zero"] += 1
                    continue

                g1r = gap1 / avg_r
                g2r = gap2 / avg_r
                if g1r < MIN_GAP_RATIO or g1r > MAX_GAP_RATIO:
                    reject_reasons["gap1_ratio"] += 1
                    continue
                if g2r < MIN_GAP_RATIO or g2r > MAX_GAP_RATIO:
                    reject_reasons["gap2_ratio"] += 1
                    continue

                gap_sim = min(gap1, gap2) / max(gap1, gap2)
                if gap_sim < MIN_GAP_SIMILARITY:
                    reject_reasons["gap_sim"] += 1
                    continue

                area_sim = min(areas) / max(areas)
                if area_sim < MIN_AREA_SIMILARITY:
                    reject_reasons["area_sim"] += 1
                    continue

                marker_h = (ys[2] + avg_r) - (ys[0] - avg_r)
                marker_w = (max(xs) + avg_r) - (min(xs) - avg_r)
                if marker_w <= 0:
                    reject_reasons["marker_w"] += 1
                    continue
                marker_aspect = marker_h / marker_w
                if marker_aspect < MIN_MARKER_ASPECT or marker_aspect > MAX_MARKER_ASPECT:
                    reject_reasons["marker_aspect"] += 1
                    continue

                size_score    = min(avg_r / 22.0, 1.0)
                x_align_score = 1.0 - min(x_spread / max(avg_r * 1.5, 1), 1.0)
                line_score    = 1.0 - min(line_error / max(avg_r, 1),     1.0)

                geometry_conf = (
                    0.25 * radius_sim    +
                    0.25 * gap_sim       +
                    0.20 * x_align_score +
                    0.15 * line_score    +
                    0.10 * area_sim      +
                    0.05 * size_score
                )

                if geometry_conf < MIN_GEOMETRY_CONF:
                    reject_reasons["geometry_conf"] += 1
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
                        "avg_radius":    avg_r,
                        "gap1_ratio":    g1r,
                        "gap2_ratio":    g2r,
                        "gap_sim":       gap_sim,
                        "radius_sim":    radius_sim,
                        "geometry_conf": geometry_conf,
                    }

    if VERBOSE and best_group is None and reject_reasons:
        top = reject_reasons.most_common(3)
        print(f"  [group] no valid group. top rejections: "
              + "  ".join(f"{r}={c}" for r, c in top))

    return best_group, best_info


# ════════════════════════════════════════════════════════════════════════════
# CROP HELPERS
# ════════════════════════════════════════════════════════════════════════════

def crop_circle(frame, blob):
    cx, cy = blob["center"]
    r      = blob["radius"]
    margin = int(r * CROP_MARGIN_SCALE)
    x1 = max(cx - r - margin, 0)
    y1 = max(cy - r - margin, 0)
    x2 = min(cx + r + margin, frame.shape[1])
    y2 = min(cy + r + margin, frame.shape[0])
    return frame[y1:y2, x1:x2].copy()


def crop_group_region(frame, group, margin_scale=0.75):
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

    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


# ════════════════════════════════════════════════════════════════════════════
# ROI HELPERS
# ════════════════════════════════════════════════════════════════════════════

def expanded_roi(frame_shape, crop_box):
    fh, fw = frame_shape[:2]
    x1, y1, x2, y2 = crop_box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    w, h   = x2 - x1, y2 - y1
    nw     = int(w * ROI_EXPAND)
    nh     = int(h * ROI_EXPAND)
    return (
        max(cx - nw // 2, 0),
        max(cy - nh // 2, 0),
        min(cx + nw // 2, fw),
        min(cy + nh // 2, fh),
    )

def shift_box(box, ox, oy):
    if box is None:
        return None
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


# ════════════════════════════════════════════════════════════════════════════
# TEMPORAL STABILITY
# ════════════════════════════════════════════════════════════════════════════

def stable_code(history, min_count=MIN_STABLE_COUNT):
    if not history:
        return None
    code, count = Counter(history).most_common(1)[0]
    return list(code) if count >= min_count else None


# ════════════════════════════════════════════════════════════════════════════
# DEBUG DRAWING
# ════════════════════════════════════════════════════════════════════════════

def draw_debug(frame, blobs, crop_box, last_box, roi_box,
               detections, code, locked):
    out = frame.copy()

    if roi_box:
        rx1, ry1, rx2, ry2 = roi_box
        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (255, 220, 0), 1)
        cv2.putText(out, "ROI", (rx1 + 4, ry1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 0), 1)

    for b in blobs:
        cv2.circle(out, b["center"], b["radius"], (160, 160, 160), 1)

    if crop_box:
        cv2.rectangle(out, crop_box[:2], crop_box[2:], (0, 230, 0), 2)

    if last_box:
        cv2.rectangle(out, last_box[:2], last_box[2:], (0, 140, 255), 1)

    for d in detections:
        cx, cy = d["center_full"]
        r      = d["radius"]
        label  = CLASS_NAMES.get(d["class_id"], "?")
        conf   = d["confidence"]
        color  = (0, 230, 0) if conf >= MIN_MODEL_CONFIDENCE else (0, 100, 230)
        cv2.circle(out, (cx, cy), r, color, 2)
        cv2.putText(out, f"{d['class_id']}:{label}({conf:.0%})",
                    (cx - r, cy - r - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    status = "LOCKED" if locked else "SEARCHING"
    if code:
        txt   = f"{status}  CODE: {code[0]} {code[1]} {code[2]}"
        color = (0, 230, 0)
    elif detections:
        nums  = [d["class_id"] for d in detections]
        txt   = f"{status}  reading: {nums}"
        color = (0, 220, 220)
    else:
        txt   = f"{status}  ..."
        color = (0, 60, 220)

    cv2.putText(out, txt, (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)

    # Show blob count
    cv2.putText(out, f"blobs: {len(blobs)}", (16, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    return out


def save_debug(frame, debug_frame, binary_mask, group_crop):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    ts = int(time.time() * 1000)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{ts}_frame.png"),  frame)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{ts}_debug.png"),  debug_frame)
    if binary_mask is not None:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{ts}_mask.png"),   binary_mask)
    if group_crop is not None:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{ts}_crop.png"),   group_crop)
    print(f"[s] Saved to {DEBUG_DIR}/")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    global VERBOSE

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str,  default=MODEL_PATH)
    parser.add_argument("--camera",   type=int,  default=CAMERA_INDEX)
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--verbose",  action="store_true",
                        help="Print per-frame rejection reasons")
    args = parser.parse_args()

    VERBOSE    = args.verbose
    show_debug = not args.no_debug

    model, device = load_model(args.model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return

    locked         = False
    last_box       = None
    missing_frames = 0
    history        = deque(maxlen=HISTORY_LEN)
    last_printed   = None

    print("Running  —  q to quit, s to save debug images")
    print("Tip: run with --verbose to see why groups are being rejected\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        # ── ROI lock ──────────────────────────────────────────────────────
        roi_box    = None
        roi_offset = (0, 0)
        search     = frame

        if locked and last_box is not None:
            rx1, ry1, rx2, ry2 = expanded_roi(frame.shape, last_box)
            roi_box    = (rx1, ry1, rx2, ry2)
            search     = frame[ry1:ry2, rx1:rx2]
            roi_offset = (rx1, ry1)

        # ── Blob detection ────────────────────────────────────────────────
        local_blobs, binary_mask = find_blobs(search)
        full_blobs = shift_blobs(local_blobs, *roi_offset) if roi_offset != (0, 0) else local_blobs

        # ── Find 3-dot group ──────────────────────────────────────────────
        group, group_info = find_best_vertical_three(local_blobs)

        crop_box   = None
        group_crop = None
        detections = []

        if group is not None:
            ox, oy = roi_offset
            for b in group:
                cx, cy = b["center"]
                b["center"] = (cx + ox, cy + oy)

            group_crop, crop_box = crop_group_region(frame, group)

            group_sorted = sorted(group, key=lambda b: b["center"][1])
            for b in group_sorted:
                circle_crop = crop_circle(frame, b)
                if circle_crop.size == 0:
                    continue
                class_id, confidence = classify_circle_crop(model, device, circle_crop)
                detections.append({
                    "class_id":    class_id,
                    "confidence":  confidence,
                    "center_full": b["center"],
                    "radius":      b["radius"],
                })

        # ── Update lock & history ─────────────────────────────────────────
        if len(detections) == 3 and crop_box is not None:
            all_confident = all(
                d["confidence"] >= MIN_MODEL_CONFIDENCE for d in detections
            )
            if all_confident:
                locked         = True
                last_box       = crop_box
                missing_frames = 0
                code_tuple     = tuple(d["class_id"] for d in detections)
                history.append(code_tuple)
                gi = group_info
                print(
                    f"  code={list(code_tuple)}  "
                    f"r={gi['avg_radius']:.1f}  "
                    f"conf={gi['geometry_conf']:.2f}  "
                    f"model={[round(d['confidence'], 2) for d in detections]}"
                )
            else:
                missing_frames += 1
        else:
            missing_frames += 1

        if missing_frames > MAX_MISSING_FRAMES:
            locked         = False
            last_box       = None
            missing_frames = 0
            history.clear()

        code = stable_code(history)
        if code is not None and code != last_printed:
            print(f"Stable detected code: {code}")
            last_printed = code

        # ── Display ───────────────────────────────────────────────────────
        if show_debug:
            debug_frame = draw_debug(
                frame, full_blobs, crop_box, last_box,
                roi_box, detections, code, locked,
            )
            cv2.imshow("Dot Detector", debug_frame)
            cv2.imshow("Binary Mask",  binary_mask)
            if group_crop is not None:
                cv2.imshow("Marker Crop", group_crop)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            save_debug(
                frame,
                debug_frame if show_debug else frame,
                binary_mask,
                group_crop,
            )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()