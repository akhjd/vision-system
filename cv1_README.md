# Cup Lid Detection — v1 (ORB Mascot Matcher)

A lightweight, single-file computer-vision tool that detects the **Smoothie Bar chef
mascot** on a cup lid using classic ORB feature matching. It runs live on a webcam,
draws a green quadrilateral around the mascot when found, and reports a rough
confidence score.

This is the **simpler** of the two versions. It answers one question — *"is the mascot
in frame?"* — and does **not** read the recipe code. For recipe decoding (the three
circles on the sticker), see version 2 (`cv2`).

---

## What it does

1. Loads one or more reference images from `refs/` (currently just `front.png`, the
   chef mascot line drawing) and extracts ORB keypoints/descriptors from each.
2. Opens the webcam and, on every Nth frame, matches the live frame against the
   reference template(s).
3. If enough good matches survive the ratio test **and** a valid homography can be
   fit, it transforms the template corners into the frame and draws the detection box.
4. Overlays a status line: template name, a heuristic confidence %, the raw score,
   match count, and inlier count.

It is purely geometric/feature-based — **no neural network, no training, no GPU.**

---

## Files

```
cv/
├── cv.py              # the entire application (≈500 lines)
└── refs/
    └── front.png      # reference image: the chef mascot (273×320 RGBA)
```

To add more things to detect, drop additional `.png` / `.jpg` / `.jpeg` files into
`refs/`. Every image becomes a template and is matched independently each frame.
`front.png` is always checked first (the loader sorts it to the front).

---

## Requirements

- Python 3
- `opencv-python` (or `opencv-contrib-python`)
- `numpy`

```bash
pip install opencv-python numpy
```

> **Note on the camera backend:** `cv.py` opens the camera with
> `cv2.CAP_AVFOUNDATION`, which is **macOS-specific**. On Linux/Windows, change the
> `cv2.VideoCapture(...)` call to drop that flag (use
> `cv2.VideoCapture(CAMERA_INDEX)` or `cv2.CAP_V4L2` / `cv2.CAP_DSHOW`).

---

## Running

```bash
cd cv
python3 cv.py
```

A window titled **"ORB Multi-Reference Detection"** opens with the live feed.

### Keyboard controls

| Key | Action |
|-----|--------|
| `s` | Select a Region of Interest (ROI). Detection then runs **only inside that box**, which is faster and reduces false matches. |
| `r` | Reset the ROI back to the full frame. |
| `q` | Quit. |

When an ROI is active it's drawn in blue and labelled `ROI`. The mascot detection box
is green.

---

## How it works (pipeline)

```
webcam frame
   │
   ├─ (optional) crop to ROI
   │
   ▼
grayscale → CLAHE contrast equalisation
   │
   ▼
resize to PROCESS_WIDTH (1280px) for consistent feature scale
   │
   ▼
ORB.detectAndCompute  ──►  BFMatcher.knnMatch(k=2) against each template
   │
   ▼
Lowe ratio test (RATIO_TEST = 0.85)  → "good matches"
   │
   ▼
findHomography (RANSAC) → count inliers
   │
   ▼
perspectiveTransform template corners → detection quad
   │
   ▼
geometry sanity checks (is_reasonable_rectangle, area bounds)
   │
   ▼
score = inliers*2 + good_matches   → keep best template, early-exit if strong
```

**Key design choices**

- **ORB instead of SIFT/SURF** — free, fast, and good enough for a printed line-art
  logo. Tuned for small features: `nfetures=1500`, low `edgeThreshold` and
  `fastThreshold` so it picks up detail on a small mascot.
- **CLAHE preprocessing** on both template and frame so matching survives uneven
  lighting.
- **Homography + geometry gate.** A raw match count is easy to fool, so a detection
  is only accepted if the matched points fit a homography producing a *plausible
  quadrilateral* — `is_reasonable_rectangle()` rejects degenerate, non-convex, or
  wildly skewed boxes.
- **Frame skipping** (`DETECT_EVERY_N_FRAMES = 2`) — full detection runs every other
  frame and the last result is reused in between, keeping the display smooth. The
  comment notes bumping this to 3–4 on a Raspberry Pi.
- **Confidence is heuristic, not ML.** `calculate_confidence()` blends the inlier
  ratio (65%) and score strength (35%) into a 0–100 number. Treat it as a relative
  signal, not a calibrated probability.

---

## Tuning

All knobs live in the `CONFIG` block at the top of `cv.py`:

| Constant | Meaning | Raise it to… |
|----------|---------|--------------|
| `MIN_GOOD_MATCHES` | min ratio-test matches before trying homography | be stricter |
| `MIN_INLIERS` | min RANSAC inliers to accept | be stricter |
| `RATIO_TEST` | Lowe ratio threshold (0.85) | *lower* = stricter matches |
| `PROCESS_WIDTH` | frame is resized to this width | trade speed vs detail |
| `EARLY_EXIT_SCORE` | stop checking templates once a score hits this | speed up |
| `HOMOGRAPHY_GATE` | once detected, skip weaker candidate templates | speed up |
| `DETECT_EVERY_N_FRAMES` | run detection every N frames | save CPU (Pi) |
| `CAMERA_INDEX` | which camera | — |

If detection is **too jumpy / false-positive prone**, lower `RATIO_TEST` and raise
`MIN_GOOD_MATCHES` / `MIN_INLIERS`. If it **misses** the mascot, do the opposite.

---

## Status line reference

```
DETECTED | front.png | conf=72.4% | score=21 | m=14 | i=9
```

- `conf` — heuristic confidence (see above)
- `score` — `inliers*2 + good_matches`
- `m` — good matches after ratio test
- `i` — RANSAC inliers

`NO DETECTION | conf=0.0%` means no template passed all gates this frame.

---

## Limitations & relationship to v2

- Detects **presence of the mascot only** — it cannot read the 3-circle recipe code.
- Feature matching struggles with very small, blurry, or heavily rotated logos and
  with backgrounds that share texture with the line art.
- For the full recipe-decoding system (blob detection + CNN classifier that reads the
  three circles into a numeric code), use **version 2 (`cv2`)**. This version is best
  kept as a fast presence check or a fallback.
