"""
Scans a grid image of cup lids and detects the 3-circle code on each.

Pipeline per lid:
  1. Crop the lid from the full image
  2. Find the marker region (right side of sticker where circles always are)
  3. Scale that region up to a fixed size (NORMALIZED_SIZE)
  4. Run blob detection + group finder + model at consistent scale

Usage:
    python3 scan_grid.py image.png
    python3 scan_grid.py image.png --out annotated.png
    python3 scan_grid.py image.png --model model_best.pth
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import argparse
import sys
from pathlib import Path


# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

MODEL_PATH          = "model_best.pth"
IMAGE_SIZE          = 64

# The marker region (right strip of each lid) is scaled to this fixed size
# before running blob detection. All thresholds below are calibrated for this.
NORMALIZED_W        = 100   # px — width of the normalized marker strip
NORMALIZED_H        = 300   # px — height of the normalized marker strip

# Lid detection
MIN_LID_AREA_RATIO  = 0.003
MAX_LID_AREA_RATIO  = 0.25
MIN_LID_ASPECT      = 0.5
MAX_LID_ASPECT      = 2.0

# Blob detection — calibrated for 100x300 normalized strip
# Circles are r~10-15px in this space
MIN_BLOB_AREA       = 60
MIN_CIRCULARITY     = 0.35
MIN_BLOB_ASPECT     = 0.50
MAX_BLOB_ASPECT     = 1.50
MIN_FILL_RATIO      = 0.40
MAX_BLOB_RADIUS     = 40    # r~10-15px expected, 40 gives headroom
MIN_BLOB_RADIUS     = 4

# Group geometry
MAX_X_SPREAD_FACTOR  = 1.00
MIN_GAP_RATIO        = 1.4
MAX_GAP_RATIO        = 6.0
MIN_GAP_SIMILARITY   = 0.55
MIN_RADIUS_SIMILARITY= 0.55
MIN_AREA_SIMILARITY  = 0.35
MIN_LINE_ERROR_SCALE = 0.60
MIN_MARKER_ASPECT    = 1.6
MAX_MARKER_ASPECT    = 7.0
MIN_GEOMETRY_CONF    = 0.65

MIN_MODEL_CONFIDENCE = 0.60

CLASS_NAMES = {0:"W", 1:"D", 2:"L", 3:"R", 4:"T", 5:"B"}
FONT        = cv2.FONT_HERSHEY_SIMPLEX


# ════════════════════════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════════════════════════

class CircleClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


def load_model(model_path):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    ckpt  = torch.load(model_path, map_location=device)
    model = CircleClassifier().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model loaded ({device})  val_acc={ckpt.get('val_acc','?'):.1%}")
    return model, device


infer_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def classify_crop(model, device, crop_bgr):
    t = infer_tf(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)
        conf, pred = probs.max(dim=1)
    return int(pred.item()), float(conf.item())


# ════════════════════════════════════════════════════════════════════════════
# LID DETECTION
# ════════════════════════════════════════════════════════════════════════════

def find_lids(image):
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w  = gray.shape
    total = h * w

    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    binary    = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary    = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (total * MIN_LID_AREA_RATIO < area < total * MAX_LID_AREA_RATIO):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        asp = bw / float(bh)
        if not (MIN_LID_ASPECT < asp < MAX_LID_ASPECT):
            continue
        lids.append((x, y, bw, bh))

    if not lids:
        return lids

    # Sort into grid rows then columns
    avg_h      = np.mean([b[3] for b in lids])
    row_groups = {}
    for lid in lids:
        rk = round(lid[1] / (avg_h * 0.6))
        row_groups.setdefault(rk, []).append(lid)

    grid = []
    for rk in sorted(row_groups):
        grid.append(sorted(row_groups[rk], key=lambda b: b[0]))

    return [(lid, r, c)
            for r, row in enumerate(grid)
            for c, lid in enumerate(row)]


# ════════════════════════════════════════════════════════════════════════════
# MARKER REGION — right strip of the lid where circles always live
# ════════════════════════════════════════════════════════════════════════════

def extract_marker_region(lid_crop):
    """
    The 3 circles are always on the right side of the sticker,
    roughly in the middle-upper vertical band.
    Crop right 25% horizontally, middle 60% vertically.
    Then scale to NORMALIZED_W x NORMALIZED_H.
    Returns (scaled_region, scale_x, scale_y, offset_x, offset_y)
    so detections can be mapped back to lid coordinates.
    """
    h, w     = lid_crop.shape[:2]
    offset_x = int(w * 0.76)   # rightmost 24%
    offset_y = int(h * 0.20)   # skip top 20% (no circles there)
    end_y    = int(h * 0.85)   # skip bottom 15%
    region   = lid_crop[offset_y:end_y, offset_x:w]

    rh, rw   = region.shape[:2]
    if rw < 2 or rh < 2:
        return None, 1, 1, offset_x, offset_y
    scale_x  = NORMALIZED_W / rw
    scale_y  = NORMALIZED_H / rh
    scaled   = cv2.resize(region, (NORMALIZED_W, NORMALIZED_H),
                          interpolation=cv2.INTER_LINEAR)

    return scaled, scale_x, scale_y, offset_x, offset_y


# ════════════════════════════════════════════════════════════════════════════
# BLOB DETECTION — uses Hough circles to detect black outlines directly
# ════════════════════════════════════════════════════════════════════════════

def find_blobs(frame):
    """
    Uses HoughCircles to detect the black circular outlines.
    Works regardless of fill color or background — only needs the edge.
    frame is expected to be the 100x300 normalized strip.
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)

    # Hough parameters calibrated for 100x300 strip with r~10-15px circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=15,          # min distance between circle centers
        param1=40,           # Canny high threshold
        param2=12,           # accumulator threshold — lower = more detections
        minRadius=MIN_BLOB_RADIUS,
        maxRadius=MAX_BLOB_RADIUS,
    )

    blobs = []
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for (cx, cy, r) in circles:
            blobs.append({
                "center":      (int(cx), int(cy)),
                "bbox":        (cx - r, cy - r, r * 2, r * 2),
                "radius":      int(r),
                "area":        int(np.pi * r * r),
                "circularity": 1.0,   # Hough already guarantees circular
            })

    return blobs


# ════════════════════════════════════════════════════════════════════════════
# GROUP DETECTION
# ════════════════════════════════════════════════════════════════════════════

def find_best_vertical_three(blobs):
    if len(blobs) < 3:
        return None

    best_group = None
    best_score = -1

    blobs_sorted = sorted(blobs, key=lambda b: b["center"][1])

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
                    continue

                x_spread = max(xs) - min(xs)
                if x_spread > avg_r * MAX_X_SPREAD_FACTOR:
                    continue

                mean_x     = float(np.mean(xs))
                line_error = float(np.mean([abs(x - mean_x) for x in xs]))
                if line_error > avg_r * MIN_LINE_ERROR_SCALE:
                    continue

                gap1 = ys[1] - ys[0]
                gap2 = ys[2] - ys[1]
                if gap1 <= 0 or gap2 <= 0:
                    continue

                g1r = gap1 / avg_r
                g2r = gap2 / avg_r
                if not (MIN_GAP_RATIO <= g1r <= MAX_GAP_RATIO):
                    continue
                if not (MIN_GAP_RATIO <= g2r <= MAX_GAP_RATIO):
                    continue

                gap_sim  = min(gap1, gap2) / max(gap1, gap2)
                if gap_sim < MIN_GAP_SIMILARITY:
                    continue

                area_sim = min(areas) / max(areas)
                if area_sim < MIN_AREA_SIMILARITY:
                    continue

                marker_h = (ys[2] + avg_r) - (ys[0] - avg_r)
                marker_w = (max(xs) + avg_r) - (min(xs) - avg_r)
                if marker_w <= 0:
                    continue
                marker_aspect = marker_h / marker_w
                if not (MIN_MARKER_ASPECT <= marker_aspect <= MAX_MARKER_ASPECT):
                    continue

                x_align = 1.0 - min(x_spread / max(avg_r * 1.5, 1), 1.0)
                l_score = 1.0 - min(line_error / max(avg_r, 1), 1.0)
                gconf   = (0.25 * radius_sim + 0.25 * gap_sim +
                           0.20 * x_align    + 0.15 * l_score +
                           0.10 * area_sim   + 0.05 * min(avg_r/22., 1.))

                if gconf < MIN_GEOMETRY_CONF:
                    continue

                score = 10000 * gconf + 300 * marker_aspect - 50 * x_spread - 40 * line_error
                if score > best_score:
                    best_score = score
                    best_group = group

    return best_group


# ════════════════════════════════════════════════════════════════════════════
# PROCESS ONE LID
# ════════════════════════════════════════════════════════════════════════════

def process_lid(image, lid_box, model, device, save_debug=False, debug_tag=""):
    x, y, w, h   = lid_box
    lid_crop     = image[y:y+h, x:x+w]

    # Extract and scale marker region to fixed size
    scaled, sx, sy, ox, oy = extract_marker_region(lid_crop)

    if scaled is None:
        return None, []

    if save_debug:
        cv2.imwrite(f"debug_{debug_tag}_scaled.png", scaled)

    blobs = find_blobs(scaled)
    group = find_best_vertical_three(blobs)

    if group is None:
        return None, []

    group_sorted = sorted(group, key=lambda b: b["center"][1])
    detections   = []

    for b in group_sorted:
        cx, cy = b["center"]
        r      = b["radius"]
        margin = int(r * 0.25)

        x1 = max(cx - r - margin, 0)
        y1 = max(cy - r - margin, 0)
        x2 = min(cx + r + margin, scaled.shape[1])
        y2 = min(cy + r + margin, scaled.shape[0])

        circle_crop = scaled[y1:y2, x1:x2]
        if circle_crop.size == 0:
            continue

        class_id, confidence = classify_crop(model, device, circle_crop)

        # Map center back to full image coords
        full_cx = x + ox + int(cx / sx)
        full_cy = y + oy + int(cy / sy)
        full_r  = int(r / min(sx, sy))

        detections.append({
            "class_id":   class_id,
            "confidence": confidence,
            "center":     (full_cx, full_cy),
            "radius":     full_r,
        })

    if len(detections) == 3:
        return [d["class_id"] for d in detections], detections
    return None, detections


# ════════════════════════════════════════════════════════════════════════════
# ANNOTATE
# ════════════════════════════════════════════════════════════════════════════

def annotate(image, flat_lids, all_results):
    out = image.copy()

    for (lid_box, row, col), result in zip(flat_lids, all_results):
        x, y, w, h = lid_box
        code       = result["code"]
        detections = result["detections"]
        color      = (0, 220, 0) if code is not None else (0, 60, 220)

        cv2.rectangle(out, (x, y), (x+w, y+h), color, 3)

        for d in detections:
            cx, cy = d["center"]
            r      = d["radius"]
            label  = CLASS_NAMES.get(d["class_id"], "?")
            cv2.circle(out, (cx, cy), r, color, 2)
            cv2.putText(out, label, (cx - r, cy - r - 4),
                        FONT, 0.45, color, 1)

        pos_txt  = f"R{row+1}C{col+1}"
        code_txt = str(code) if code is not None else "?"

        (tw, th), _ = cv2.getTextSize(code_txt, FONT, 0.8, 2)
        pad = 5
        cv2.rectangle(out, (x+4, y+4), (x+4+tw+pad*2, y+4+th*2+pad*3), (0,0,0), -1)
        cv2.putText(out, pos_txt,  (x+4+pad, y+4+pad+th//2+4), FONT, 0.45, (180,180,180), 1)
        cv2.putText(out, code_txt, (x+4+pad, y+4+pad*2+th+th//2+4), FONT, 0.8, color, 2)

    return out


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image",         type=str)
    parser.add_argument("--model",       type=str, default=MODEL_PATH)
    parser.add_argument("--out",         type=str, default=None)
    parser.add_argument("--save-debug",  action="store_true",
                        help="Save scaled marker region for each lid")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Not found: {image_path}"); sys.exit(1)

    out_path = args.out or str(image_path.parent / (image_path.stem + "_annotated.png"))

    image = cv2.imread(str(image_path))
    if image is None:
        print("Could not read image"); sys.exit(1)

    print(f"Image: {image.shape[1]}x{image.shape[0]}")
    model, device = load_model(args.model)

    print("\nDetecting lids...")
    flat_lids = find_lids(image)
    print(f"Found {len(flat_lids)} lids")
    if not flat_lids:
        print("No lids found"); sys.exit(1)

    print(f"\n{'Position':<10} {'Code':<22} {'Confidence'}")
    print("-" * 55)

    all_results = []
    for lid_box, row, col in flat_lids:
        code, detections = process_lid(
            image, lid_box, model, device,
            save_debug=args.save_debug,
            debug_tag=f"R{row+1}C{col+1}",
        )
        code_str = str(code) if code is not None else "NOT FOUND"
        conf_str = ""
        if detections:
            conf_str = "[" + ", ".join(f"{d['confidence']:.2f}" for d in detections) + "]"
        print(f"R{row+1}C{col+1:<7} {code_str:<22} {conf_str}")
        all_results.append({"code": code, "detections": detections})

    found = sum(1 for r in all_results if r["code"] is not None)
    print(f"\nDecoded: {found}/{len(all_results)}")

    annotated = annotate(image, flat_lids, all_results)
    cv2.imwrite(out_path, annotated)
    print(f"Saved: {out_path}")

    cv2.imshow("Grid Scan", annotated)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()