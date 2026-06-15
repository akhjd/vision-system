# Cup Lid Detection — v2 (3-Circle Recipe Code Decoder)

The full Smoothie Bar lid system. Instead of just detecting the mascot, this version
**reads the recipe code** printed on each sticker: a vertical strip of **three
circles**, where each circle is one of six symbols. Three circles × six states gives
up to **216** distinct recipe codes.

It finds the circles by geometry alone (no sticker/logo matching needed), classifies
each one with a small CNN, and reports the code as a 3-number sequence such as
`[0, 2, 1]`.

```
Circle symbols (class IDs):
  0  white          (empty / light)
  1  dark           (solid teal #0E4446)
  2  half dark LEFT
  3  half dark RIGHT
  4  half dark TOP
  5  half dark BOTTOM
```

---

## The three entry points

This repo contains three different runnable detectors that share the same idea but
target different situations. Pick based on what you're doing:

| Script | Input | Classifier | Use it for |
|--------|-------|------------|-----------|
| **`run.py`** | live camera | **CNN** (`model_best.pth`) | The production live reader. Locks onto a marker, votes across frames for a stable code. |
| **`scan_grid.py`** | a still image **or** live camera (snap mode) | **CNN** | Reading **multiple cups at once** from a photo of a tray/grid. Multi-pass blob search, robust to broken outlines. |
| **`cv.py`** | live camera | **none** (pixel-ratio heuristic) | CNN-free fallback. Classifies circles by black-pixel ratio instead of a model. Good for debugging / no-PyTorch environments. |

All three share the same geometry core (find dark blobs → group into a vertical
triplet → classify each circle → temporal voting), so the README treats that as the
common pipeline and notes where each script differs.

---

## Files

```
cv2/
├── run.py                # ★ live CNN reader (main production script)
├── scan_grid.py          # ★ still-image / multi-cup grid reader (CNN)
├── cv.py                 # CNN-free live reader (heuristic classifier)
├── debug_pipeline.py     # visual debugger — shows every scan_grid stage side by side
│
├── test.py               # CNN training script (despite the name, this trains the model)
├── generate_data.py      # synthetic training-data generator for the 6 circle classes
│
├── model_best.pth        # ★ trained CNN checkpoint (best val acc) — used at inference
├── model_final.pth       # CNN checkpoint after final epoch
│
├── crop_dataset.py       # extract sticker crops from photos → for YOLO labelling
├── autolabel.py          # auto-label whole-image sticker crops → YOLO dataset
├── train_yolo.py         # train YOLOv8n sticker detector (optional/experimental)
├── best.pt               # trained YOLOv8 sticker-detector weights
├── yolov8n.pt            # YOLOv8 nano base weights (download cache)
│
├── lid_crops/            # 44 cropped sticker images (RxCy_lid.png), per-cup
├── lid_crops.zip         # zipped copy of the above
└── Screenshot*.png       # test images + *_annotated.png results
```

> ★ = what you need at runtime. The YOLO files (`train_yolo.py`, `crop_dataset.py`,
> `autolabel.py`, `best.pt`, `yolov8n.pt`) are an **optional sticker-localisation
> path** that isn't required by the main pipeline — see "Optional: YOLO sticker
> detection" below.

---

## Requirements

```bash
pip install opencv-python numpy torch torchvision pillow scipy
# only if you use the YOLO scripts:
pip install ultralytics
```

- **PyTorch** for the CNN (`run.py`, `scan_grid.py`, training). The code auto-selects
  device: Apple **MPS** → CUDA → CPU.
- **scipy** is used by `scan_grid.py` (`gaussian_filter1d` for histogram smoothing).
- **Pillow** (`ImageCms`) handles ICC colour profiles so iPhone P3 photos convert to
  sRGB correctly before processing.

> The camera scripts use a plain `cv2.VideoCapture(index)` (unlike v1, which was
> macOS-locked). `run.py` defaults to camera `0`, `cv.py` to camera `1` — override
> with `--camera`.

---

## Quick start

### Read a single photo (or a tray of cups)
```bash
python3 scan_grid.py --image Screenshot10.png --verbose
# writes Screenshot10_annotated.png and prints each triplet's code
```
Rotate first if the photo is sideways: `--rotate 90` (or 180 / 270).

Live snap mode (no `--image`):
```bash
python3 scan_grid.py            # SPACE = snap & decode, R = rotate, S = save, Q = quit
```

### Live continuous reader
```bash
python3 run.py                  # uses model_best.pth, camera 0
python3 run.py --camera 1 --verbose
# q = quit, s = save debug images
# --verbose prints WHY blob groups get rejected (great for tuning)
```

### CNN-free fallback
```bash
python3 cv.py                   # heuristic classifier, no model needed
python3 cv.py --no-debug        # terminal output only
```

### Visual pipeline debugger
```bash
python3 debug_pipeline.py Screenshot10.png
# shows threshold → blobs → triplets → classification as labelled panels
```

---

## The shared pipeline

```
image / frame
   │
   ▼
1. BLOB DETECTION
   grayscale → CLAHE → adaptive (or percentile) threshold → morphology
   find contours → keep circular ones (area, aspect, circularity, fill-ratio)
   │
   ▼
2. GROUP INTO A VERTICAL TRIPLET   (find_best_vertical_three / _find_triplets)
   try every 3-blob combination, keep the one that looks like a marker:
     • similar radii        • similar areas
     • vertically aligned   • equal gaps between centres
     • tall-strip aspect    • combined "geometry_conf" ≥ threshold
   │
   ▼
3. CLASSIFY EACH CIRCLE
   crop each circle (top→bottom) → CNN → (class_id, confidence)
   (cv.py instead normalises the crop and uses black-pixel ratios)
   │
   ▼
4. CODE = [top, middle, bottom]   e.g. [0, 2, 1]
   rejected if any circle's confidence < threshold
   │
   ▼
5. TEMPORAL VOTING  (run.py / cv.py only)
   keep a history of recent codes; only report one once it appears
   ≥ MIN_STABLE_COUNT times out of HISTORY_LEN frames
   + ROI lock: once found, search only near the last marker for speed
```

**Why geometry-first instead of detecting the sticker?** The circles have a very
specific, hard-to-fake spatial signature (three equal dots, equally spaced, vertically
stacked). Searching for that directly is more robust than first finding the sticker
logo and is what lets `scan_grid.py` read many cups in one shot.

---

## Script-specific notes

### `run.py` — live CNN reader (production)
- Loosened blob filters vs `cv.py` (real printed circles aren't perfect) plus a
  `MIN_FILL_RATIO` and `MAX_BLOB_RADIUS` to reject half-circle blobs and giant
  sticker-outline blobs.
- A code is only locked in when **all three** circles clear
  `MIN_MODEL_CONFIDENCE` (0.65).
- ROI lock (`ROI_EXPAND`, `MAX_MISSING_FRAMES`) + temporal voting
  (`HISTORY_LEN=18`, `MIN_STABLE_COUNT=11`) make the output stable.
- `--verbose` prints a tally of rejection reasons (`radius_sim`, `gap1_ratio`,
  `geometry_conf`, …) — the fastest way to see why a marker isn't being read.

### `scan_grid.py` — still-image / multi-cup reader
The most sophisticated blob finder. Built for photos where outlines may be broken or
lighting uneven:
- **ICC-aware load** (`imread_icc`) for correct iPhone colours.
- **Auto-threshold** (`_auto_percentiles`) finds the histogram valley after the dark
  peak instead of using a fixed cutoff.
- **Two-pass detection:** Pass 1 strict (only obvious circles) measures a median
  radius; Pass 2 lenient uses that radius to find broken/faint circles at the right
  scale. Each morphology step (threshold→dilate→close→open→fill) contributes blobs.
- **Pass 3** (`_find_missing_third`): if two good circles sit at the right spacing, it
  searches the exact spot where the third *should* be — recovers markers where one
  circle is faint.
- **Grid sorting** (`_sort_triplets`) orders detected markers into rows/columns and
  labels them `R1C1`, `R1C2`, … in the annotated output (matching the `lid_crops/`
  filenames).

### `cv.py` — CNN-free fallback
Same geometry core, but `classify_circle()` decides each symbol from black-pixel
ratios: <18% black → white, >82% → dark, otherwise it compares left/right/top/bottom
halves to pick a half-split class. No model file required. Handy for quick tests or a
PyTorch-less device, at some accuracy cost on borderline circles.

### `debug_pipeline.py`
Imports `scan_grid`'s internals and renders each stage as a labelled panel in one
window — use it when a particular image fails and you need to see whether it broke at
threshold, blob, triplet, or classification.

---

## The CNN classifier

A small 3-block conv net (`CircleClassifier`), defined identically in `run.py`,
`scan_grid.py`, and `test.py`:

```
Conv(3→32)  → BN → ReLU → MaxPool        (block1)
Conv(32→64) → BN → ReLU → MaxPool        (block2)
Conv(64→128)→ BN → ReLU → MaxPool        (block3)
Flatten → Linear(128*8*8 → 256) → ReLU → Dropout(0.4) → Linear(256 → 6)
```

> **Keep the layer names `block1`/`block2`/`block3`/`classifier` exactly.** The
> checkpoints store `state_dict` keys under those names; renaming the layers breaks
> `load_state_dict`.

- Input: 64×64 RGB, normalised mean/std = 0.5.
- Checkpoint dict: `{epoch, model_state, val_acc, class_names}`.
- Shipped `model_best.pth` / `model_final.pth` report **val_acc = 1.0** on the
  validation split.

> ⚠️ That 100% is on **synthetic, augmented data** generated by `generate_data.py`,
> not on real photos. It tells you the model learned the synthetic classes cleanly; it
> does **not** guarantee real-world accuracy. Treat real-cup performance as the thing
> to actually measure, and expand training data with real crops (e.g. from
> `lid_crops/`) if borderline symbols misclassify.

### Retraining
```bash
# 1. generate synthetic data (6 class folders under data/)
python3 generate_data.py --samples 1500 --size 64
#    → also writes data/_preview/preview_grid.png; eyeball it before training

# 2. train (test.py is the trainer; writes model_best.pth + model_final.pth)
python3 test.py --data data --epochs 30 --batch 64
```
`generate_data.py` renders each clean symbol then applies random backgrounds (biased
toward the real sticker colours), ±15° tilt, scale, brightness/contrast, blur, noise,
perspective warp, and JPEG artifacts. Note it tilts only ±15° on purpose — full
rotation would turn a "half-left" into a "half-top" and corrupt the label.

---

## Optional: YOLO sticker detection

A separate, experimental path to *localise the whole sticker* before decoding (not
required by the main pipeline, which works directly on circles):

```bash
python3 crop_dataset.py Screenshot*.png   # cut sticker crops out of photos
python3 autolabel.py --crops lid_crops/   # label each crop as one full-frame sticker
python3 train_yolo.py                      # fine-tune yolov8n → runs/.../best.pt
```
`best.pt` is a trained result of this. You'd use it if global circle search ever
proves too noisy and you want to constrain decoding to detected sticker regions first.

---

## Tuning cheat-sheet

| Symptom | Try |
|---------|-----|
| Misses faint / broken circles | lower `MIN_CIRCULARITY`, `P2_MIN_CIRCULARITY`; raise dilate/close kernels |
| False triplets across cups | raise `MIN_GAP_SIM`, `MIN_RADIUS_SIM`; lower `MAX_X_SPREAD_FRAC` |
| Code flickers / won't lock | raise `MIN_STABLE_COUNT`, `HISTORY_LEN` (run.py / cv.py) |
| Circles found but code = None | lower `MIN_CONF` / `MIN_MODEL_CONFIDENCE`, or retrain on real crops |
| Sideways phone photo | `scan_grid.py --rotate 90/180/270` |
| Need to see what's failing | `run.py --verbose` or `debug_pipeline.py <image>` |

Most constants sit in the `CONFIG` block at the top of each script. `run.py` and
`cv.py` keep separate copies — tune the one you're running.

---

## Relationship to v1

Version 1 (`cv/cv.py`, ORB) only answers *"is the mascot present?"*. This version adds
the part that matters operationally: **which recipe** the lid encodes, read straight
from the three circles. v1 is a fast presence check; v2 is the actual decoder.
