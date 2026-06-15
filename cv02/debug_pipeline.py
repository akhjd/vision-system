"""
debug_pipeline.py  -  Visual debugger for scan_grid pipeline

Shows every stage side by side in one window so you can see exactly
where detection breaks down.

Usage:
    python3 debug_pipeline.py image.png
    python3 debug_pipeline.py image.png --model model_best.pth
"""

import sys, io, argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms
from scipy.ndimage import gaussian_filter1d

# Import everything from scan_grid
sys.path.insert(0, str(Path(__file__).parent))
from scan_grid import (
    imread_icc, load_model, _auto_percentiles,
    _find_blobs, _find_triplets, _sort_triplets,
    process_triplet, annotate,
    OUTLINE_CLOSE_K, MIN_BLOB_H_FRAC, MAX_BLOB_H_FRAC,
    MIN_BLOB_ASPECT, MAX_BLOB_ASPECT, MIN_CIRCULARITY,
    EDGE_MARGIN, FONT, CLASS_NAMES, MODEL_PATH,
    P1_MIN_CIRCULARITY, P1_CLOSE_K, P1_DILATE_K,
    P2_MIN_CIRCULARITY, P2_SIZE_MIN_FRAC, P2_SIZE_MAX_FRAC,
    P2_CLOSE_K_FRAC, P2_DILATE_K,
    P2_EDGE_CANNY_LO, P2_EDGE_CANNY_HI, P2_EDGE_DILATE,
    MIN_RADIUS_SIM, MIN_GAP_SIM, MAX_X_SPREAD_FRAC,
    _find_missing_third,
)

WINDOW  = "Pipeline Debug"


def native(img):
    """Return image at native resolution — no resize."""
    return img


def resize_fit(img, max_w=3840, max_h=2160):
    """Resize only if larger than screen, preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = min(max_w/w, max_h/h, 1.0)
    if scale < 1.0:
        return cv2.resize(img, (int(w*scale), int(h*scale)))
    return img


def resize_to_width(img, w):
    h = int(img.shape[0] * w / img.shape[1])
    return cv2.resize(img, (w, h))


def label_panel(img, title, notes=""):
    """Add a title bar and optional notes to a panel."""
    out  = img.copy()
    # Dark banner at top
    cv2.rectangle(out, (0,0), (out.shape[1], 50), (20,20,20), -1)
    cv2.putText(out, title, (10, 32), FONT, 0.9, (0,220,0), 2)
    if notes:
        cv2.putText(out, notes, (10, out.shape[0]-12), FONT, 0.5, (200,200,200), 1)
    return out


def draw_blobs(image, blobs, color=(0,220,0)):
    out = image.copy()
    for b in blobs:
        cv2.circle(out, (b["cx"], b["cy"]), b["r"], color, 2)
        cv2.putText(out, f"r={b['r']} c={b['circ']:.2f}",
                    (b["cx"]-b["r"], b["cy"]-b["r"]-4),
                    FONT, 0.35, color, 1)
    return out


def draw_triplets(image, triplets):
    out = image.copy()
    colors = [(0,220,0),(0,180,255),(255,100,0),(180,0,255),
              (0,255,180),(255,220,0),(0,100,255)]
    for ti, triplet in enumerate(triplets):
        col = colors[ti % len(colors)]
        for b in triplet:
            cv2.circle(out, (b["cx"], b["cy"]), b["r"], col, 3)
        # Line connecting them
        pts = [(b["cx"], b["cy"]) for b in triplet]
        for a, b in zip(pts, pts[1:]):
            cv2.line(out, a, b, col, 2)
        # Label
        cx = int(np.mean([b["cx"] for b in triplet]))
        cy = triplet[0]["cy"] - triplet[0]["r"] - 10
        cv2.putText(out, f"T{ti+1}", (cx, cy), FONT, 0.7, col, 2)
    return out


def hist_panel(eq, percentiles, w=900):
    """Draw brightness histogram with threshold lines."""
    h_panel = 300
    panel   = np.zeros((h_panel, w, 3), np.uint8)
    panel[:] = (30, 30, 30)

    hist    = cv2.calcHist([eq], [0], None, [256], [0,256]).flatten()
    smooth  = gaussian_filter1d(hist, sigma=3)
    max_val = smooth.max()

    bar_w = w // 256
    for i, val in enumerate(smooth):
        bar_h = int((val / max_val) * (h_panel - 60))
        x     = i * bar_w
        cv2.rectangle(panel, (x, h_panel-30-bar_h), (x+bar_w, h_panel-30),
                      (100, 100, 180), -1)

    # Draw percentile threshold lines
    colors_p = [(0,220,0),(0,180,255),(255,180,0)]
    for pi, pct in enumerate(percentiles):
        thresh = float(np.percentile(eq, pct))
        x      = int(thresh * bar_w)
        cv2.line(panel, (x, 0), (x, h_panel-30), colors_p[pi % 3], 2)
        cv2.putText(panel, f"p{pct}={thresh:.0f}",
                    (x+3, 20 + pi*20), FONT, 0.45, colors_p[pi % 3], 1)

    cv2.putText(panel, "Brightness histogram  (green lines = thresholds)",
                (10, h_panel-10), FONT, 0.45, (180,180,180), 1)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model",  default=MODEL_PATH)
    ap.add_argument("--rotate", type=int, default=0, choices=[0,90,180,270])
    args = ap.parse_args()

    img_path = Path(args.image)
    image    = imread_icc(str(img_path))
    if args.rotate == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif args.rotate == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif args.rotate == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ih, iw   = image.shape[:2]
    model, device = load_model(args.model)

    print(f"Image: {iw}x{ih}")
    print("Keys: N=next stage  P=prev stage  Q=quit  S=save current panel")

    # ── Stage 0: Original ─────────────────────────────────────────────────
    stage0 = label_panel(image.copy(), "0. Original", f"{iw}x{ih}  {img_path.name}")

    # ── Stage 1: CLAHE equalised ──────────────────────────────────────────
    mx   = int(iw * EDGE_MARGIN)
    my   = int(ih * EDGE_MARGIN)
    roi  = image[my:ih-my, mx:iw-mx]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe_obj = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    eq   = clahe_obj.apply(gray)
    eq_bgr = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    # Pad eq_bgr back to full image size for display
    eq_full = np.zeros_like(image)
    eq_full[my:ih-my, mx:iw-mx] = eq_bgr
    stage1 = label_panel(eq_full, "1. Bilateral filter + CLAHE",
                         f"edge margin={EDGE_MARGIN}  clipLimit=3.0")

    # ── Stage 2: Histogram ────────────────────────────────────────────────
    percentiles = _auto_percentiles(eq)
    stage2 = label_panel(hist_panel(eq, percentiles),
                         "2. Histogram + auto thresholds",
                         f"percentiles={percentiles}")

    # ── Stages 3a-3j: Each mask step individually ─────────────────────────
    med_pct  = percentiles[len(percentiles)//2]
    thresh_v = max(float(np.percentile(eq, med_pct)), 15)

    blobs_for_k = _find_blobs(image, verbose=False)[0]
    radii_p1    = [b["r"] for b in blobs_for_k] if blobs_for_k else [10]
    med_r_p1    = float(np.median(radii_p1))
    adapt_k     = max(3, int(med_r_p1 * P2_CLOSE_K_FRAC) | 1)

    # Pass 1 steps: threshold → dilate → close → open → fill
    s_thresh    = (eq < thresh_v).astype(np.uint8) * 255

    k_dil_p1    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                      (P1_DILATE_K*2+1, P1_DILATE_K*2+1))
    s_dil_p1    = cv2.dilate(s_thresh, k_dil_p1, iterations=1)

    k_cls_p1    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                      (OUTLINE_CLOSE_K, OUTLINE_CLOSE_K))
    s_close_p1  = cv2.morphologyEx(s_dil_p1, cv2.MORPH_CLOSE, k_cls_p1)

    k_open      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    s_open_p1   = cv2.morphologyEx(s_close_p1, cv2.MORPH_OPEN, k_open)

    flood = s_open_p1.copy()
    bm    = np.zeros((flood.shape[0]+2, flood.shape[1]+2), np.uint8)
    cv2.floodFill(flood, bm, (0,0), 255)
    s_fill_p1   = cv2.bitwise_or(s_open_p1, cv2.bitwise_not(flood))

    # Pass 2 steps: same order, larger dilate + adaptive close
    k_dil_p2    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                      (P2_DILATE_K*2+1, P2_DILATE_K*2+1))
    s_dil_p2    = cv2.dilate(s_thresh, k_dil_p2, iterations=1)

    k_cls_p2    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (adapt_k, adapt_k))
    s_close_p2  = cv2.morphologyEx(s_dil_p2, cv2.MORPH_CLOSE, k_cls_p2)

    s_open_p2   = cv2.morphologyEx(s_close_p2, cv2.MORPH_OPEN, k_open)

    flood2 = s_open_p2.copy()
    bm2    = np.zeros((flood2.shape[0]+2, flood2.shape[1]+2), np.uint8)
    cv2.floodFill(flood2, bm2, (0,0), 255)
    s_fill_p2   = cv2.bitwise_or(s_open_p2, cv2.bitwise_not(flood2))

    def mask_to_full(m):
        """Embed ROI mask back into full image size for display."""
        full = np.zeros((ih, iw), np.uint8)
        full[my:ih-my, mx:iw-mx] = m
        return cv2.cvtColor(full, cv2.COLOR_GRAY2BGR)

    def lp(img, title, notes=""):
        return label_panel(img, title, notes)

    stage3a = lp(mask_to_full(s_thresh),    "3a. Threshold",    f"p{med_pct}={thresh_v:.0f}")
    stage3b = lp(mask_to_full(s_dil_p1),    "3b. Dilate P1",    f"k={P1_DILATE_K} — raw outline thickening")
    stage3c = lp(mask_to_full(s_close_p1),  "3c. Close P1",     f"k={OUTLINE_CLOSE_K} — bridges arc gaps")
    stage3d = lp(mask_to_full(s_open_p1),   "3d. Open P1",      "removes noise after close")
    stage3e = lp(mask_to_full(s_fill_p1),   "3e. Fill P1",      "flood fill — completes circles")
    stage3f = lp(mask_to_full(s_dil_p2),    "3f. Dilate P2",    f"k={P2_DILATE_K} — more aggressive")
    stage3g = lp(mask_to_full(s_close_p2),  "3g. Close P2",     f"k={adapt_k} adaptive")
    stage3h = lp(mask_to_full(s_fill_p2),   "3h. Fill P2",      f"circ>={P2_MIN_CIRCULARITY}  med_r={med_r_p1:.1f}")


    # ── Stage 4: Blobs — pass1 / pass2 / pass3 breakdown ─────────────────
    blobs, _ = _find_blobs(image, verbose=True)

    # Colour-code by which pass found each blob
    # We can approximate: pass1 = circ >= P1_MIN_CIRCULARITY
    # pass3 = blobs added by targeted search (hard to separate here,
    # but we show them all with radius labels so you can see)
    def draw_blobs_labelled(img, blob_list):
        out = img.copy()
        for b in blob_list:
            # Green = high circ (pass1), Yellow = medium (pass2), Red = low (pass3)
            if b["circ"] >= P1_MIN_CIRCULARITY:
                col = (0, 220, 0)
            elif b["circ"] >= P2_MIN_CIRCULARITY:
                col = (0, 200, 220)
            else:
                col = (0, 100, 255)
            cv2.circle(out, (b["cx"], b["cy"]), b["r"], col, 2)
            cv2.putText(out, f"r={b['r']} {b['circ']:.2f}",
                        (b["cx"]-b["r"], b["cy"]-b["r"]-4),
                        FONT, 0.32, col, 1)
        return out

    blob_vis = draw_blobs_labelled(image, blobs)
    p1_count = sum(1 for b in blobs if b["circ"] >= P1_MIN_CIRCULARITY)
    p2_count = sum(1 for b in blobs if P2_MIN_CIRCULARITY <= b["circ"] < P1_MIN_CIRCULARITY)
    p3_count = sum(1 for b in blobs if b["circ"] < P2_MIN_CIRCULARITY)
    stage4   = label_panel(blob_vis,
                           "4. Blobs detected (green=P1 cyan=P2 red=P3)",
                           f"total={len(blobs)}  P1={p1_count}  P2={p2_count}  P3(targeted)={p3_count}")

    # ── Stage 5: Triplets found + rejection reasons ───────────────────────
    triplets     = _find_triplets(blobs, ih, verbose=True)
    triplet_vis  = draw_triplets(image.copy(), triplets)

    # Show why blobs didn't form triplets — check all pairs of 3
    rejection_counts = {"r_sim": 0, "x_spread": 0, "span": 0,
                        "gap_sim": 0, "overlap": 0}
    if len(blobs) >= 3:
        bs = sorted(blobs, key=lambda b: b["cy"])
        n  = len(bs)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    grp = [bs[i], bs[j], bs[k]]
                    ys  = [b["cy"] for b in grp]
                    xs  = [b["cx"] for b in grp]
                    rs  = [b["r"]  for b in grp]
                    avg_r = float(np.mean(rs))
                    if avg_r < 2: continue
                    r_sim = min(rs)/max(rs)
                    if r_sim < MIN_RADIUS_SIM:
                        rejection_counts["r_sim"] += 1; continue
                    x_spread = max(xs)-min(xs)
                    if x_spread > avg_r * MAX_X_SPREAD_FRAC:
                        rejection_counts["x_spread"] += 1; continue
                    span = ys[2]-ys[0]
                    if span > ih * 0.25:
                        rejection_counts["span"] += 1; continue
                    g1,g2 = ys[1]-ys[0], ys[2]-ys[1]
                    if g1<=0 or g2<=0: continue
                    g_sim = min(g1,g2)/max(g1,g2)
                    if g_sim < MIN_GAP_SIM:
                        rejection_counts["gap_sim"] += 1; continue
                    if g1 < avg_r*0.8 or g2 < avg_r*0.8:
                        rejection_counts["overlap"] += 1; continue

    rej_str = "  ".join(f"{k}:{v}" for k,v in rejection_counts.items() if v>0)
    notes5  = f"{len(triplets)} triplet(s)   rejections: {rej_str or 'none'}"
    stage5  = label_panel(triplet_vis, "5. Triplets grouped", notes5)
    print(f"Triplet rejections: {rejection_counts}")

    # ── Stage 6: Final annotated result ───────────────────────────────────
    sorted_trips = _sort_triplets(triplets)
    all_results  = []
    for triplet, row, col in sorted_trips:
        code, dets = process_triplet(image, triplet, model, device)
        all_results.append({"code": code, "detections": dets})

    annotated = annotate(image, sorted_trips, all_results)
    found     = sum(1 for r in all_results if r["code"])
    stage6    = label_panel(annotated, "6. Final result",
                            f"Decoded {found}/{len(all_results)}")

    stages = [stage0, stage1, stage2,
              stage3a, stage3b, stage3c, stage3d, stage3e,
              stage3f, stage3g, stage3h,
              stage4, stage5, stage6]
    names  = ["Original", "CLAHE", "Histogram",
              "3a-Thresh", "3b-Open", "3c-DilP1", "3d-CloseP1", "3e-FillP1",
              "3f-DilP2", "3g-CloseP2", "3h-FillP2",
              "Blobs", "Triplets", "Result"]
    idx    = 0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    while True:
        panel   = stages[idx]
        display = resize_fit(panel)
        cv2.setWindowTitle(WINDOW,
            f"[{idx+1}/{len(stages)}] {names[idx]}   N=next  P=prev  S=save  Q=quit")
        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('n'):
            idx = min(idx+1, len(stages)-1)
        elif key == ord('p'):
            idx = max(idx-1, 0)
        elif key == ord('s'):
            fname = f"debug_{idx:02d}_{names[idx].lower()}.png"
            cv2.imwrite(fname, panel)   # save at FULL resolution
            print(f"Saved full-res: {fname}  ({panel.shape[1]}x{panel.shape[0]})")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()