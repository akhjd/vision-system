"""
scan_grid.py  -  Smoothie Bar sticker circle decoder

Strategy:
  - No sticker detection needed
  - Search the whole image (with edge margin) for dark circular blobs
  - Group blobs into vertical columns of 3 (the circle triplets)
  - Classify each circle with the CNN
  - Works for single cup or grid of cups

Usage:
    python3 scan_grid.py image.png
    python3 scan_grid.py image.png --model model_best.pth
    python3 scan_grid.py image.png --out result.png --save-debug --verbose
"""

import argparse, os, sys, io
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageCms

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH  = "model_best.pth"
IMAGE_SIZE  = 64

# ---------- Image edge margin ----------
# Ignore this fraction of the image at each edge — kills background noise
EDGE_MARGIN = 0.05

def _auto_percentiles(eq):
    """
    Find the right threshold percentile(s) automatically by locating
    the valley in the brightness histogram after the dark peak.

    The circle outlines/fills form a clear dark cluster in the histogram.
    The valley after that cluster is the natural boundary.
    We return a small range around that valley to catch all circles.
    """
    from scipy.ndimage import gaussian_filter1d

    hist   = cv2.calcHist([eq], [0], None, [256], [0, 256]).flatten()
    smooth = gaussian_filter1d(hist, sigma=3)

    # Find the dark peak (within first 80 brightness values)
    dark   = smooth[:80]
    peak   = int(np.argmax(dark))

    # Find the valley after the peak
    valley = peak
    for i in range(peak + 1, 80):
        if smooth[i] > smooth[valley]:
            break
        valley = i

    # Convert valley brightness to percentile
    valley_pct = float((eq < valley).mean() * 100)

    # Return a small range: peak percentile, valley percentile, midpoint
    peak_pct = float((eq < peak).mean() * 100)
    mid_pct  = (peak_pct + valley_pct) / 2

    # Clamp to sane range
    pcts = sorted(set([
        max(3,  round(peak_pct)),
        max(3,  round(mid_pct)),
        min(20, round(valley_pct)),
    ]))
    return pcts

# ---------- Blob size limits (fraction of image SHORT side) ----------
OUTLINE_CLOSE_K = 7
# Using short side (min of w,h) so it works in portrait and landscape.
# Circles at typical scanning distance are 0.5-4% of the short side.
MIN_BLOB_H_FRAC = 0.010   # min circle diameter / short side
MAX_BLOB_H_FRAC = 0.060   # max circle diameter / short side

# ---------- Blob shape filters ----------
MIN_CIRCULARITY = 0.45   # 1.0 = perfect circle; lower = more permissive
MIN_BLOB_ASPECT = 0.50   # w/h lower bound
MAX_BLOB_ASPECT = 2.00   # w/h upper bound

# ---------- Pass 1 (strict) ----------
P1_MIN_CIRCULARITY = 0.75  # only obvious circles
P1_CLOSE_K         = OUTLINE_CLOSE_K
P1_DILATE_K        = 2     # thicken outlines before finding contours

# ---------- Pass 2 (lenient — anchored to pass 1 median radius) ----------
P2_MIN_CIRCULARITY = 0.45  # lower = catches more broken arcs
P2_SIZE_MIN_FRAC   = 0.50  # tight_min = med_r * this
P2_SIZE_MAX_FRAC   = 1.50  # tight_max = med_r * this
P2_CLOSE_K_FRAC    = 0.35  # adaptive_k = med_r * this (odd, min 3)
P2_DILATE_K        = 3     # thicken outlines more aggressively in pass 2
P2_EDGE_CANNY_LO   = 80    # raise = only strong edges
P2_EDGE_CANNY_HI   = 160   # raise = only strong edges (was 80)
P2_EDGE_DILATE     = 1     # iterations to dilate edges before OR-ing in

# ---------- Triplet column geometry ----------
MIN_RADIUS_SIM    = 0.70  # min(r)/max(r) across 3 circles — must be similar size
MIN_GAP_SIM       = 0.65  # min(gap)/max(gap) — must be equally spaced
MAX_X_SPREAD_FRAC = 1.50  # max horizontal drift / avg_radius — must be aligned

# ---------- Classify ----------
CROP_MARGIN = 0.40   # padding around circle crop sent to CNN (fraction of r)
MIN_CONF    = 0.50   # confidence below this → code reported as None

DEBUG_DIR   = "debug_scan"
FONT        = cv2.FONT_HERSHEY_SIMPLEX
CLASS_NAMES = {0:"W", 1:"D", 2:"L", 3:"R", 4:"T", 5:"B"}


# ============================================================
# ICC-AWARE IMAGE LOAD
# ============================================================

def imread_icc(path):
    """Load image respecting embedded ICC profile (iPhone P3 → sRGB)."""
    try:
        pil = Image.open(path)
        icc = pil.info.get("icc_profile")
        if icc:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            dst = ImageCms.createProfile("sRGB")
            pil = ImageCms.profileToProfile(pil, src, dst)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"  [load] ICC failed ({e}), using cv2.imread")
        return cv2.imread(str(path))


# ============================================================
# CNN CLASSIFIER
# ============================================================

class CircleClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.block1     = blk(3,  32)
        self.block2     = blk(32, 64)
        self.block3     = blk(64, 128)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128*8*8, 256),
            nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.block3(self.block2(self.block1(x))))


def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


def load_model(path):
    dev  = get_device()
    ckpt = torch.load(path, map_location=dev)
    m    = CircleClassifier().to(dev)
    m.load_state_dict(ckpt.get("model_state", ckpt))
    m.eval()
    va = ckpt.get("val_acc")
    print(f"CNN loaded  device={dev}" + (f"  val_acc={va:.1%}" if va else ""))
    return m, dev


_infer_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def classify_crop(model, device, bgr):
    if bgr is None or bgr.size == 0:
        return -1, 0.0
    t = _infer_tf(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(model(t), dim=1)
        conf, pred = p.max(dim=1)
    return int(pred.item()), float(conf.item())


# ============================================================
# DARK BLOB DETECTION
# ============================================================

def _find_blobs(image, verbose=False):
    ih, iw = image.shape[:2]
    short  = min(ih, iw)

    mx  = int(iw * EDGE_MARGIN)
    my  = int(ih * EDGE_MARGIN)
    roi = image[my:ih-my, mx:iw-mx]
    rh, rw = roi.shape[:2]

    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Bilateral filter — smooths print texture/noise while keeping
    # circle outline edges sharp (strong edges are preserved)
    gray  = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)

    percentiles = _auto_percentiles(eq)
    if verbose:
        print(f"  [blobs] auto percentiles: {percentiles}")

    def _extract_blobs(mask, min_h, max_h, min_circ):
        """Extract circular blobs from a binary mask."""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 10: continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if not (min_h <= bh <= max_h and min_h <= bw <= max_h): continue
            if bh >= rh * 0.85: continue
            asp = bw / float(bh)
            if not (MIN_BLOB_ASPECT <= asp <= MAX_BLOB_ASPECT): continue
            peri = cv2.arcLength(cnt, True)
            if peri <= 0: continue
            circ = 4 * np.pi * area / (peri * peri)
            if circ < min_circ: continue
            found.append({
                "cx":   bx + bw//2 + mx,
                "cy":   by + bh//2 + my,
                "r":    (bw + bh) // 4,
                "circ": circ,
                "area": area,
            })
        return found

    def _dedup(blobs):
        blobs.sort(key=lambda b: b["circ"], reverse=True)
        kept = []
        for b in blobs:
            if not any(abs(b["cx"]-k["cx"]) < k["r"]*1.2 and
                       abs(b["cy"]-k["cy"]) < k["r"]*1.2 for k in kept):
                kept.append(b)
        return kept

    def _run(min_h, max_h, min_circ, close_k, use_edges=False):
        """
        Run blob detection, stopping at the earliest step that already
        produces enough blobs to form a triplet. Steps in order:
          1. threshold only
          2. + open (remove specks)
          3. + dilate (thicken outlines)
          4. + close (bridge gaps)
          5. + fill (complete circles)
        """
        all_found  = []
        last_mask  = None

        for p in percentiles:
            thresh = max(float(np.percentile(eq, p)), 15)
            mask   = (eq < thresh).astype(np.uint8) * 255

            if use_edges:
                edges = cv2.Canny(eq, P2_EDGE_CANNY_LO, P2_EDGE_CANNY_HI)
                ek    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                edges = cv2.dilate(edges, ek, iterations=P2_EDGE_DILATE)
                mask  = cv2.bitwise_or(mask, edges)

            # Collect blobs at every morphology step — each step finds
            # different circles. Threshold finds clean ones, dilate finds
            # slightly broken outlines, close+fill finds heavily broken arcs.
            # Merge all and let dedup + size clustering clean up.
            all_found.extend(_extract_blobs(mask,  min_h, max_h, min_circ))

            dilate_k = P2_DILATE_K if use_edges else P1_DILATE_K
            k_d      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                           (dilate_k*2+1, dilate_k*2+1))
            mask2    = cv2.dilate(mask, k_d, iterations=1)
            all_found.extend(_extract_blobs(mask2, min_h, max_h, min_circ))

            k_c      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
            mask3    = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, k_c)
            all_found.extend(_extract_blobs(mask3, min_h, max_h, min_circ))

            k_o      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask4    = cv2.morphologyEx(mask3, cv2.MORPH_OPEN, k_o)
            all_found.extend(_extract_blobs(mask4, min_h, max_h, min_circ))

            flood    = mask4.copy()
            bm       = np.zeros((rh+2, rw+2), np.uint8)
            cv2.floodFill(flood, bm, (0, 0), 255)
            mask5    = cv2.bitwise_or(mask4, cv2.bitwise_not(flood))
            all_found.extend(_extract_blobs(mask5, min_h, max_h, min_circ))
            last_mask = mask5

        return _dedup(all_found), last_mask

    # ── Pass 1: strict ────────────────────────────────────────────────────
    # High circularity, broad size range — finds only the obvious circles.
    p1_min_h = short * MIN_BLOB_H_FRAC
    p1_max_h = short * MAX_BLOB_H_FRAC
    pass1, last_mask = _run(p1_min_h, p1_max_h, min_circ=P1_MIN_CIRCULARITY,
                             close_k=P1_CLOSE_K, use_edges=False)

    if verbose:
        print(f"  [pass1] {len(pass1)} strict blobs")

    if len(pass1) < 3:
        # Not enough confident blobs — just return what we have
        return pass1, last_mask

    # Derive stats from pass 1
    radii    = [b["r"] for b in pass1]
    med_r    = float(np.median(radii))

    p2_min_h   = max(p1_min_h, med_r * P2_SIZE_MIN_FRAC)
    p2_max_h   = min(p1_max_h, med_r * P2_SIZE_MAX_FRAC)
    adaptive_k = max(3, int(med_r * P2_CLOSE_K_FRAC) | 1)

    if verbose:
        print(f"  [pass1] median_r={med_r:.1f}  "
              f"pass2 size={p2_min_h:.0f}-{p2_max_h:.0f}  "
              f"close_k={adaptive_k}")

    pass2, last_mask = _run(p2_min_h, p2_max_h, min_circ=P2_MIN_CIRCULARITY,
                             close_k=adaptive_k, use_edges=False)

    if verbose:
        print(f"  [pass2] {len(pass2)} lenient blobs")

    # Merge pass1 + pass2
    # For duplicates, keep whichever is closer to the confirmed median radius
    merged = list(pass1)
    for b in pass2:
        overlap = [k for k in merged
                   if abs(b["cx"]-k["cx"]) < k["r"]*1.2 and
                      abs(b["cy"]-k["cy"]) < k["r"]*1.2]
        if not overlap:
            # New blob not seen in pass1 — add it
            merged.append(b)
        else:
            # Duplicate — replace existing if this one is closer to med_r
            existing = overlap[0]
            if abs(b["r"] - med_r) < abs(existing["r"] - med_r):
                merged = [x for x in merged if x is not existing]
                merged.append(b)

    if verbose:
        print(f"  [blobs] {len(merged)} final blobs after merge"
              f"  (med_r={med_r:.1f})")

    # ── Pass 3: targeted search for missing third circle ─────────────────
    # If pass 1+2 found 2 blobs with a gap matching the expected spacing,
    # we know exactly where the 3rd circle should be. Search that specific
    # region with a very lenient threshold rather than hoping global detection
    # finds it.
    merged = _find_missing_third(merged, eq, med_r, mx, my,
                                 p2_min_h, p2_max_h, adaptive_k, verbose)

    return merged, last_mask


def _find_missing_third(blobs, eq, med_r, mx, my,
                        min_h, max_h, close_k, verbose=False):
    """
    For every pair of blobs with the right vertical gap and alignment,
    check whether a third blob exists at the expected position above or
    below. If not found in blobs, search that region directly in eq.
    """
    rh, rw  = eq.shape[:2]
    expected_gap = med_r * 3.0   # typical centre-to-centre gap
    gap_tol      = med_r * 1.2   # ± tolerance on the gap
    search_r     = int(med_r * 1.8)  # search radius around expected centre

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))

    def _search_region(cx, cy):
        """Try to find a circle blob centred near (cx, cy) in eq coords."""
        # cx, cy are in full-image coords — convert to ROI coords
        rx = cx - mx
        ry = cy - my
        x1 = max(rx - search_r, 0)
        y1 = max(ry - search_r, 0)
        x2 = min(rx + search_r, rw)
        y2 = min(ry + search_r, rh)
        if x2 <= x1 or y2 <= y1:
            return None

        patch  = eq[y1:y2, x1:x2]
        # Use a very low local threshold — darkest 20% of this small patch
        t      = max(float(np.percentile(patch, 20)), 15)
        m      = (patch < t).astype(np.uint8) * 255
        k_o    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        m      = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k_o)
        m      = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best    = None
        best_dist = float('inf')
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 10: continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if not (min_h <= bh <= max_h and min_h <= bw <= max_h): continue
            asp = bw / float(bh)
            if not (MIN_BLOB_ASPECT <= asp <= MAX_BLOB_ASPECT): continue
            peri = cv2.arcLength(cnt, True)
            if peri <= 0: continue
            circ = 4 * np.pi * area / (peri * peri)
            if circ < 0.30: continue   # very lenient here
            # Centre in full-image coords
            full_cx = bx + bw//2 + x1 + mx
            full_cy = by + bh//2 + y1 + my
            dist = abs(full_cx - cx) + abs(full_cy - cy)
            if dist < best_dist:
                best_dist = dist
                best = {"cx": full_cx, "cy": full_cy,
                        "r": (bw+bh)//4, "circ": circ, "area": area}
        return best

    def _already_found(cx, cy):
        return any(abs(b["cx"]-cx) < med_r and abs(b["cy"]-cy) < med_r
                   for b in blobs)

    extra = []
    n = len(blobs)
    for i in range(n):
        for j in range(i+1, n):
            a, b_ = blobs[i], blobs[j]
            # Must be roughly vertically aligned
            if abs(a["cx"] - b_["cx"]) > med_r * MAX_X_SPREAD_FRAC:
                continue
            gap = abs(a["cy"] - b_["cy"])
            if not (expected_gap - gap_tol <= gap <= expected_gap + gap_tol):
                continue

            # Similar radius
            if min(a["r"], b_["r"]) / max(a["r"], b_["r"]) < 0.5:
                continue

            mid_cx = (a["cx"] + b_["cx"]) // 2
            top_cy = min(a["cy"], b_["cy"]) - int(gap)   # above top
            bot_cy = max(a["cy"], b_["cy"]) + int(gap)   # below bottom

            for target_cy in [top_cy, bot_cy]:
                if _already_found(mid_cx, target_cy):
                    continue
                found = _search_region(mid_cx, target_cy)
                if found:
                    if not _already_found(found["cx"], found["cy"]):
                        extra.append(found)
                        if verbose:
                            print(f"  [pass3] found missing circle at "
                                  f"({found['cx']},{found['cy']}) r={found['r']}")

    if extra:
        blobs = blobs + extra
        if verbose:
            print(f"  [pass3] added {len(extra)} missing circle(s)")

    return blobs


# ============================================================
# TRIPLET FINDER
# ============================================================

def _find_triplets(blobs, image_h, verbose=False):
    """
    Two-stage triplet finding:

    Stage 1 — find all candidate triplets with loose constraints.
    Stage 2 — measure the most common gap across all candidates,
               then reject any triplet whose gaps are > 3x that median.
               This prevents blobs from different stickers accidentally
               forming a false triplet with a huge gap.
    """
    if len(blobs) < 3:
        return []

    blobs  = sorted(blobs, key=lambda b: b["cy"])
    n      = len(blobs)
    scored = []

    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                grp   = [blobs[i], blobs[j], blobs[k]]
                ys    = [b["cy"] for b in grp]
                xs    = [b["cx"] for b in grp]
                rs    = [b["r"]  for b in grp]
                avg_r = float(np.mean(rs))
                if avg_r < 2: continue

                r_sim = min(rs) / max(rs)
                if r_sim < MIN_RADIUS_SIM: continue

                x_spread = max(xs) - min(xs)
                if x_spread > avg_r * MAX_X_SPREAD_FRAC: continue

                span = ys[2] - ys[0]
                if span > image_h * 0.25: continue

                g1, g2 = ys[1]-ys[0], ys[2]-ys[1]
                if g1 <= 0 or g2 <= 0: continue
                g_sim = min(g1, g2) / max(g1, g2)
                if g_sim < MIN_GAP_SIM: continue

                if g1 < avg_r * 0.8 or g2 < avg_r * 0.8: continue

                score = (r_sim * 0.35 + g_sim * 0.35
                         + (1 - min(x_spread / max(avg_r*2, 1), 1)) * 0.20
                         + float(np.mean([b["circ"] for b in grp])) * 0.10)

                scored.append((score, grp, g1, g2))

    if not scored:
        if verbose: print("  [triplets] none found in stage 1")
        return []

    # ── Stage 2: gap filter ───────────────────────────────────────────────
    # Collect all gaps from stage 1 candidates, find the median (most common
    # real gap), reject triplets with gaps > 3x that median.
    all_gaps   = [g for _, _, g1, g2 in scored for g in (g1, g2)]
    median_gap = float(np.median(all_gaps))
    max_gap    = median_gap * 3.0

    if verbose:
        print(f"  [triplets] stage1={len(scored)}  "
              f"median_gap={median_gap:.1f}  max_allowed={max_gap:.1f}")

    filtered = [(score, grp) for score, grp, g1, g2 in scored
                if g1 <= max_gap and g2 <= max_gap]

    if verbose:
        rejected = len(scored) - len(filtered)
        print(f"  [triplets] stage2 rejected {rejected} large-gap triplets  "
              f"remaining={len(filtered)}")

    if not filtered:
        if verbose: print("  [triplets] none survived gap filter")
        return []

    # Greedily pick non-overlapping triplets by score
    filtered.sort(key=lambda x: x[0], reverse=True)
    used    = set()
    results = []

    for score, grp in filtered:
        ids = tuple(id(b) for b in grp)
        if any(i in used for i in ids): continue
        for i in ids: used.add(i)
        results.append(grp)

    if verbose:
        print(f"  [triplets] {len(results)} final triplets")

    return results


def _sort_triplets(triplets):
    """Sort triplets into a grid (top→bottom, left→right)."""
    if not triplets: return []

    # Each triplet's position = centre of its 3 circles
    def centre(t):
        return (int(np.mean([b["cx"] for b in t])),
                int(np.mean([b["cy"] for b in t])))

    centres = [centre(t) for t in triplets]
    avg_h   = float(np.mean([t[2]["cy"] - t[0]["cy"] for t in triplets]))

    rows = []
    order = sorted(range(len(triplets)), key=lambda i: centres[i][1])
    for idx in order:
        cy = centres[idx][1]
        placed = False
        for row in rows:
            row_cy = np.mean([centres[i][1] for i in row])
            if abs(cy - row_cy) < avg_h * 0.55:
                row.append(idx); placed = True; break
        if not placed:
            rows.append([idx])

    rows = sorted(rows, key=lambda r: np.mean([centres[i][1] for i in r]))
    result = []
    for ri, row in enumerate(rows):
        for ci, idx in enumerate(sorted(row, key=lambda i: centres[i][0])):
            result.append((triplets[idx], ri, ci))
    return result


# ============================================================
# CLASSIFY ONE TRIPLET
# ============================================================

def process_triplet(image, triplet, model, device,
                    save_debug=False, debug_tag="", verbose=False):
    """Classify 3 circles. Returns (code, detections)."""
    detections = []
    code       = []

    for i, c in enumerate(triplet):   # already sorted top→bottom
        cx, cy, r = c["cx"], c["cy"], c["r"]
        mg  = max(2, int(r * CROP_MARGIN))
        x1  = max(cx - r - mg, 0)
        y1  = max(cy - r - mg, 0)
        x2  = min(cx + r + mg, image.shape[1])
        y2  = min(cy + r + mg, image.shape[0])
        crop = image[y1:y2, x1:x2]

        if save_debug and crop.size > 0:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(DEBUG_DIR, f"{debug_tag}_c{i}.png"), crop)

        cid, conf = classify_crop(model, device, crop)
        detections.append({
            "class_id":   cid,
            "confidence": conf,
            "center":     (cx, cy),
            "radius":     r,
        })
        code.append(cid)

    if any(d["confidence"] < MIN_CONF for d in detections):
        if verbose:
            print(f"  [process] {debug_tag}: low conf "
                  f"{[round(d['confidence'],2) for d in detections]}")
        return None, detections

    return code, detections


# ============================================================
# ANNOTATION
# ============================================================

def annotate(image, sorted_triplets, all_results):
    out = image.copy()

    for (triplet, row, col), result in zip(sorted_triplets, all_results):
        code, dets = result["code"], result["detections"]
        color = (0, 220, 0) if code else (0, 60, 220)

        # Bounding box around the triplet
        xs = [b["cx"] for b in triplet]
        ys = [b["cy"] for b in triplet]
        rs = [b["r"]  for b in triplet]
        pad = max(rs) + 8
        x1 = max(min(xs) - pad, 0)
        y1 = max(min(ys) - pad, 0)
        x2 = min(max(xs) + pad, image.shape[1])
        y2 = min(max(ys) + pad, image.shape[0])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Each circle
        for d in dets:
            cx, cy = d["center"]
            r      = d["radius"]
            conf   = d["confidence"]
            label  = CLASS_NAMES.get(d["class_id"], "?")
            dc     = (0, 220, 0) if conf >= MIN_CONF else (0, 100, 255)
            cv2.circle(out, (cx, cy), max(r, 4), dc, 2)
            cv2.putText(out, f"{label}:{conf:.2f}",
                        (cx - r, cy - r - 4), FONT, 0.32, dc, 1)

        # Code label
        code_txt    = str(code) if code else "?"
        (tw, _), _  = cv2.getTextSize(code_txt, FONT, 0.55, 2)
        cv2.rectangle(out, (x1, y1), (x1+max(tw+10,60), y1+34), (0,0,0), -1)
        cv2.putText(out, f"R{row+1}C{col+1}", (x1+4, y1+13), FONT, 0.30, (160,160,160), 1)
        cv2.putText(out, code_txt,             (x1+4, y1+29), FONT, 0.48, color, 2)

    return out


# ============================================================
# MAIN
# ============================================================

def process_image(image, model, device, verbose=False):
    """Run full pipeline on a single frame. Returns (annotated, results)."""
    blobs, mask = _find_blobs(image, verbose)
    if verbose:
        print(f"  {len(blobs)} blobs")

    triplets = _find_triplets(blobs, image.shape[0], verbose)
    if not triplets:
        return image, []

    sorted_triplets = _sort_triplets(triplets)
    all_results = []
    for triplet, row, col in sorted_triplets:
        code, dets = process_triplet(image, triplet, model, device,
                                     verbose=verbose)
        all_results.append({"code": code, "detections": dets})

    annotated = annotate(image, sorted_triplets, all_results)
    return annotated, all_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default=MODEL_PATH)
    ap.add_argument("--camera",  type=int, default=0)
    ap.add_argument("--image",   default=None, help="Process a single image instead")
    ap.add_argument("--rotate",  type=int, default=0, choices=[0,90,180,270],
                    help="Rotate image before processing (0/90/180/270)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    model, device = load_model(args.model)

    # ── Single image mode ──────────────────────────────────────────────────
    if args.image:
        img_path = Path(args.image)
        image    = imread_icc(str(img_path))
        if args.rotate == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif args.rotate == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif args.rotate == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        out_path = str(img_path.parent / f"{img_path.stem}_annotated.png")
        annotated, results = process_image(image, model, device, args.verbose)
        for i, r in enumerate(results):
            print(f"  [{i}] {r['code']}")
        cv2.imwrite(out_path, annotated)
        print(f"Saved: {out_path}")
        cv2.imshow("Result", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ── Live feed mode ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}"); sys.exit(1)

    print("\nLive feed started.")
    print("  SPACE  — snap and process")
    print("  R      — rotate 90° clockwise")
    print("  S      — save last result")
    print("  Q/ESC  — quit\n")

    last_annotated = None
    last_results   = None
    snap_count     = 0
    rotation       = 0   # 0, 90, 180, 270

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed"); break

        # Apply rotation
        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Show live feed with status overlay
        display = frame.copy()
        status  = f"SPACE=snap  R=rotate({rotation}deg)  S=save  Q=quit"
        if last_results is not None:
            decoded = sum(1 for r in last_results if r['code'])
            status += f"   Last: {decoded}/{len(last_results)} decoded"
        cv2.putText(display, status, (20, 40),
                    FONT, 0.7, (0, 220, 0), 2)
        cv2.imshow("Live Feed", display)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        elif key == ord('r'):
            rotation = (rotation + 90) % 360
            print(f"Rotation: {rotation}°")

        elif key == ord(' '):
            print("Processing...")
            annotated, results = process_image(frame, model, device, args.verbose)
            last_annotated = annotated
            last_results   = results
            print(f"  Found {len(results)} triplet(s)")
            for i, r in enumerate(results):
                print(f"  [{i}] code={r['code']}")
            cv2.imshow("Snap Result", annotated)

        elif key == ord('s') and last_annotated is not None:
            snap_count += 1
            fname = f"snap_{snap_count:03d}.png"
            cv2.imwrite(fname, last_annotated)
            print(f"Saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()