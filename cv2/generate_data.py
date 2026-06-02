"""
Synthetic data generator for the 6-class circle symbol classifier.

Classes:
    0 - full white
    1 - full dark (teal #0E4446)
    2 - half dark left,  half white right
    3 - half dark right, half white left
    4 - half dark top,   half white bottom
    5 - half dark bottom, half white top

Generates augmented circle images and saves them to:
    data/
        0_white/
        1_dark/
        2_half_left/
        3_half_right/
        4_half_top/
        5_half_bottom/

Usage:
    python3 generate_data.py
    python3 generate_data.py --samples 2000   # images per class (default 1500)
    python3 generate_data.py --size 64        # output image size (default 64)
    python3 generate_data.py --out my_data    # output directory (default: data)
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


# ── Brand colors ────────────────────────────────────────────────────────────
DARK_BGR  = (0x44, 0x44, 0x0E)   # #0E4446 in BGR  (teal)
WHITE_BGR = (255, 255, 255)


# ── Class definitions ────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "0_white",
    1: "1_dark",
    2: "2_half_left",
    3: "3_half_right",
    4: "4_half_top",
    5: "5_half_bottom",
}


# ════════════════════════════════════════════════════════════════════════════
# CIRCLE RENDERER
# ════════════════════════════════════════════════════════════════════════════

def render_clean_circle(size, class_id):
    """
    Renders a clean, ideal circle image for the given class.
    Returns a (size x size) BGR image with a grey background.
    The circle fills ~60% of the image.
    """
    img = np.full((size, size, 3), 180, dtype=np.uint8)   # grey background

    cx, cy = size // 2, size // 2
    r = int(size * 0.38)

    if class_id == 0:
        # Full white
        cv2.circle(img, (cx, cy), r, WHITE_BGR, -1)

    elif class_id == 1:
        # Full dark teal
        cv2.circle(img, (cx, cy), r, DARK_BGR, -1)

    elif class_id == 2:
        # Half dark LEFT, half white RIGHT
        cv2.circle(img, (cx, cy), r, WHITE_BGR, -1)
        # Draw left half dark
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        left_half = np.zeros((size, size), dtype=np.uint8)
        left_half[:, :cx] = 255
        combined = cv2.bitwise_and(mask, left_half)
        img[combined == 255] = DARK_BGR

    elif class_id == 3:
        # Half dark RIGHT, half white LEFT
        cv2.circle(img, (cx, cy), r, WHITE_BGR, -1)
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        right_half = np.zeros((size, size), dtype=np.uint8)
        right_half[:, cx:] = 255
        combined = cv2.bitwise_and(mask, right_half)
        img[combined == 255] = DARK_BGR

    elif class_id == 4:
        # Half dark TOP, half white BOTTOM
        cv2.circle(img, (cx, cy), r, WHITE_BGR, -1)
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        top_half = np.zeros((size, size), dtype=np.uint8)
        top_half[:cy, :] = 255
        combined = cv2.bitwise_and(mask, top_half)
        img[combined == 255] = DARK_BGR

    elif class_id == 5:
        # Half dark BOTTOM, half white TOP
        cv2.circle(img, (cx, cy), r, WHITE_BGR, -1)
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        bottom_half = np.zeros((size, size), dtype=np.uint8)
        bottom_half[cy:, :] = 255
        combined = cv2.bitwise_and(mask, bottom_half)
        img[combined == 255] = DARK_BGR

    return img


# ════════════════════════════════════════════════════════════════════════════
# AUGMENTATIONS
# ════════════════════════════════════════════════════════════════════════════

def random_background(size):
    """
    Generates a random background — solid color, gradient, or noisy patch.
    Simulates varying sticker background colors (pink, yellow, green, teal).
    """
    choice = np.random.randint(0, 4)

    if choice == 0:
        # Solid random color (bias toward the sticker background colors)
        palette = [
            (180, 160, 210),   # pink-ish (BGR)
            (100, 200, 230),   # yellow-ish
            (120, 200, 140),   # green-ish
            (180, 180, 100),   # teal-ish
            (200, 200, 200),   # neutral grey
            (240, 240, 240),   # near white
        ]
        color = palette[np.random.randint(len(palette))]
        bg = np.full((size, size, 3), color, dtype=np.uint8)

    elif choice == 1:
        # Random solid color anywhere in RGB space
        color = tuple(int(x) for x in np.random.randint(80, 240, 3))
        bg = np.full((size, size, 3), color, dtype=np.uint8)

    elif choice == 2:
        # Gaussian noise background
        bg = np.random.randint(100, 220, (size, size, 3), dtype=np.uint8)

    else:
        # Gradient
        bg = np.zeros((size, size, 3), dtype=np.uint8)
        start = np.random.randint(80, 200, 3)
        end   = np.random.randint(80, 200, 3)
        for i in range(size):
            t = i / size
            bg[i, :] = (start * (1 - t) + end * t).astype(np.uint8)

    return bg


def augment(img, size, class_id):
    """
    Applies a random chain of augmentations to a clean circle image.
    Returns an augmented (size x size) BGR image.
    """
    # ── 1. Random background ─────────────────────────────────────────────
    bg = random_background(size)

    # Build circle mask to composite circle onto background
    cx, cy = size // 2, size // 2
    r = int(size * 0.38)
    circle_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(circle_mask, (cx, cy), r, 255, -1)

    out = bg.copy()
    out[circle_mask == 255] = img[circle_mask == 255]

    # ── 2. Random rotation (full 360° for class invariance) ──────────────
    # NOTE: half-circle classes ARE rotation-sensitive, so we only allow
    # small tilts (±15°) to simulate camera not being perfectly level,
    # NOT full rotation (that would change which class it is).
    tilt = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((cx, cy), tilt, 1.0)
    out = cv2.warpAffine(out, M, (size, size),
                         borderMode=cv2.BORDER_REFLECT)

    # ── 3. Random scale (simulate distance) ──────────────────────────────
    scale = np.random.uniform(0.70, 1.15)
    new_r = int(r * scale)
    scaled = np.full((size, size, 3), 180, dtype=np.uint8)

    # Re-render at new scale and composite
    clean = render_clean_circle(size, class_id)
    new_size = int(size * scale)
    if new_size > 4:
        resized = cv2.resize(clean, (new_size, new_size))
        # Center crop or pad
        if new_size <= size:
            pad = (size - new_size) // 2
            scaled[pad:pad+new_size, pad:pad+new_size] = resized
        else:
            crop = (new_size - size) // 2
            scaled = resized[crop:crop+size, crop:crop+size]

        # Composite onto background
        new_mask = np.zeros((size, size), dtype=np.uint8)
        new_cx = size // 2
        new_cy = size // 2
        new_r2 = int(size * 0.38 * min(scale, 1.0))
        cv2.circle(new_mask, (new_cx, new_cy), new_r2, 255, -1)

        out2 = bg.copy()
        out2[new_mask == 255] = scaled[new_mask == 255]
        out = out2

    # ── 4. Random brightness & contrast ──────────────────────────────────
    alpha = np.random.uniform(0.65, 1.45)   # contrast
    beta  = np.random.randint(-40, 40)      # brightness
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # ── 5. Random blur (simulate distance / focus) ────────────────────────
    if np.random.random() < 0.6:
        ksize = np.random.choice([3, 5, 7])
        out = cv2.GaussianBlur(out, (ksize, ksize), 0)

    # ── 6. Random Gaussian noise ──────────────────────────────────────────
    if np.random.random() < 0.5:
        noise = np.random.normal(0, np.random.uniform(3, 18), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # ── 7. Random perspective warp (camera angle) ─────────────────────────
    if np.random.random() < 0.5:
        margin = int(size * 0.08)
        src = np.float32([
            [0, 0], [size-1, 0],
            [size-1, size-1], [0, size-1]
        ])
        dst = np.float32([
            [np.random.randint(0, margin),        np.random.randint(0, margin)],
            [np.random.randint(size-margin, size), np.random.randint(0, margin)],
            [np.random.randint(size-margin, size), np.random.randint(size-margin, size)],
            [np.random.randint(0, margin),        np.random.randint(size-margin, size)],
        ])
        M_persp = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, M_persp, (size, size),
                                  borderMode=cv2.BORDER_REFLECT)

    # ── 8. Random JPEG compression artifacts ─────────────────────────────
    if np.random.random() < 0.3:
        quality = np.random.randint(40, 90)
        _, enc = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, quality])
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return out


# ════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ════════════════════════════════════════════════════════════════════════════

def generate(out_dir, samples_per_class, size):
    out_path = Path(out_dir)

    print(f"Generating {samples_per_class} images per class ({6 * samples_per_class} total)")
    print(f"Image size: {size}x{size}")
    print(f"Output:     {out_path.resolve()}\n")

    for class_id, class_name in CLASS_NAMES.items():
        class_dir = out_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        clean = render_clean_circle(size, class_id)

        for i in range(samples_per_class):
            aug = augment(clean, size, class_id)
            fname = class_dir / f"{class_id}_{i:05d}.png"
            cv2.imwrite(str(fname), aug)

            if (i + 1) % 250 == 0 or (i + 1) == samples_per_class:
                print(f"  class {class_id} ({class_name}): {i+1}/{samples_per_class}")

    total = 6 * samples_per_class
    print(f"\nDone. {total} images saved to {out_path.resolve()}/")
    print("\nClass distribution:")
    for class_id, class_name in CLASS_NAMES.items():
        count = len(list((out_path / class_name).glob("*.png")))
        print(f"  {class_id} {class_name}: {count} images")


# ════════════════════════════════════════════════════════════════════════════
# PREVIEW — save one clean example of each class
# ════════════════════════════════════════════════════════════════════════════

def save_preview(out_dir, size):
    preview_dir = Path(out_dir) / "_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    previews = []
    for class_id, class_name in CLASS_NAMES.items():
        clean = render_clean_circle(size, class_id)
        cv2.imwrite(str(preview_dir / f"clean_{class_name}.png"), clean)

        aug_row = [render_clean_circle(size, class_id)]
        for _ in range(7):
            aug_row.append(augment(clean, size, class_id))

        row = np.hstack(aug_row)
        previews.append(row)

    grid = np.vstack(previews)
    cv2.imwrite(str(preview_dir / "preview_grid.png"), grid)
    print(f"\nPreview grid saved to {preview_dir}/preview_grid.png")
    print("Open it to check the augmentations look realistic before training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1500,
                        help="Images per class (default 1500)")
    parser.add_argument("--size",    type=int, default=64,
                        help="Image size in pixels (default 64)")
    parser.add_argument("--out",     type=str, default="data",
                        help="Output directory (default: data)")
    args = parser.parse_args()

    save_preview(args.out, args.size)
    generate(args.out, args.samples, args.size)