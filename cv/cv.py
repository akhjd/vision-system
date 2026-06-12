import cv2
import numpy as np
import glob
import os

# ================= CONFIG =================

REFERENCE_FOLDER = "refs"
CAMERA_INDEX = 0

TEMPLATE_SCALES = [1.0]

MIN_GOOD_MATCHES = 5
MIN_INLIERS = 4
RATIO_TEST = 0.85

PROCESS_WIDTH = 1280
EARLY_EXIT_SCORE = 18
HOMOGRAPHY_GATE = 8

# 2 for testing, 3-4 later for Raspberry Pi
DETECT_EVERY_N_FRAMES = 2

roi = None
frame_count = 0
last_result = None

# ================= ORB =================

orb = cv2.ORB_create(
    nfeatures=1500,
    scaleFactor=1.08,
    nlevels=16,
    edgeThreshold=5,
    patchSize=31,
    fastThreshold=3
)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

# ================= FUNCTIONS =================

def empty_result():
    return {
        "detected": False,
        "score": 0,
        "matches": 0,
        "inliers": 0,
        "confidence": 0.0,
        "corners": None,
        "template_scale": None,
        "template_name": None
    }


def calculate_confidence(score, good_matches, inliers):
    """
    ORB does not give real ML confidence.
    This creates a useful testing confidence from:
    - inlier ratio
    - score strength
    """
    if good_matches <= 0:
        return 0.0

    inlier_ratio = inliers / good_matches
    score_conf = min(score / EARLY_EXIT_SCORE, 1.0)

    confidence = (
        0.65 * inlier_ratio +
        0.35 * score_conf
    ) * 100.0

    return round(confidence, 1)


def preprocess(gray):
    return clahe.apply(gray)


def resize_to_width(img, width):
    h, w = img.shape[:2]

    if w <= width:
        return img, 1.0

    scale = width / w

    resized = cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale
    )

    return resized, scale


def is_reasonable_rectangle(corners, min_area=80):
    pts = corners.reshape(4, 2).astype(np.float32)

    d01 = np.linalg.norm(pts[0] - pts[1])
    d12 = np.linalg.norm(pts[1] - pts[2])
    d23 = np.linalg.norm(pts[2] - pts[3])
    d30 = np.linalg.norm(pts[3] - pts[0])

    if min(d01, d12, d23, d30) < 6:
        return False

    area = cv2.contourArea(pts)

    if area < min_area:
        return False

    # Opposite sides should not be wildly different
    if d01 / max(d23, 1) > 2.5 or d23 / max(d01, 1) > 2.5:
        return False

    if d12 / max(d30, 1) > 2.5 or d30 / max(d12, 1) > 2.5:
        return False

    # Convexity check, faster than calculating corner angles
    signs = []

    for i in range(4):
        v1 = pts[(i + 1) % 4] - pts[i]
        v2 = pts[(i + 2) % 4] - pts[(i + 1) % 4]

        cross = v1[0] * v2[1] - v1[1] * v2[0]
        signs.append(cross)

    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def load_reference_templates(folder):
    templates = []
    image_paths = []

    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(
            glob.glob(os.path.join(folder, ext))
        )

    if not image_paths:
        raise Exception(f"No reference images found in {folder}")

    # front.png checked first
    image_paths = sorted(
        image_paths,
        key=lambda p: 0 if os.path.basename(p) == "front.png" else 1
    )

    for path in image_paths:
        original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if original is None:
            continue

        print(f"\nLoading reference: {path}")

        for s in TEMPLATE_SCALES:
            scaled = cv2.resize(
                original,
                None,
                fx=s,
                fy=s
            )

            scaled = preprocess(scaled)

            kp, des = orb.detectAndCompute(
                scaled,
                None
            )

            if des is None or len(kp) < MIN_GOOD_MATCHES:
                print(f"  skipped scale={s}, not enough keypoints")
                continue

            h, w = scaled.shape[:2]

            corners = np.float32([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ]).reshape(-1, 1, 2)

            templates.append({
                "name": os.path.basename(path),
                "scale": s,
                "kp": kp,
                "des": des,
                "w": w,
                "h": h,
                "corners": corners
            })

            print(f"  scale={s} size={w}x{h} kp={len(kp)}")

    if not templates:
        raise Exception("No usable templates loaded")

    print(f"\nLoaded {len(templates)} usable template entries")

    return templates


def match_against_templates(gray, templates):
    original_h, original_w = gray.shape[:2]
    roi_area = original_w * original_h

    gray_small, frame_scale = resize_to_width(
        gray,
        PROCESS_WIDTH
    )

    kp_frame, des_frame = orb.detectAndCompute(
        gray_small,
        None
    )

    best = empty_result()

    if des_frame is None or len(kp_frame) < MIN_GOOD_MATCHES:
        return best

    for temp in templates:
        matches = bf.knnMatch(
            temp["des"],
            des_frame,
            k=2
        )

        good_matches = []

        for pair in matches:
            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < RATIO_TEST * n.distance:
                good_matches.append(m)

        if len(good_matches) < MIN_GOOD_MATCHES:
            continue

        # If we already have a detection, skip weak candidates
        if best["detected"] and len(good_matches) < HOMOGRAPHY_GATE:
            continue

        src_pts = np.float32([
            temp["kp"][m.queryIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        dst_pts = np.float32([
            kp_frame[m.trainIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        # Convert frame points back to original ROI scale
        dst_pts /= frame_scale

        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            4.0
        )

        if H is None or mask is None:
            continue

        inliers = int(mask.sum())

        if inliers < MIN_INLIERS:
            continue

        transformed_corners = cv2.perspectiveTransform(
            temp["corners"],
            H
        )

        xs = transformed_corners[:, 0, 0]
        ys = transformed_corners[:, 0, 1]

        box_w = xs.max() - xs.min()
        box_h = ys.max() - ys.min()
        box_area = box_w * box_h

        valid_box = (
            80 < box_area < roi_area * 0.75 and
            box_w > 8 and
            box_h > 8 and
            is_reasonable_rectangle(
                transformed_corners,
                min_area=80
            )
        )

        if not valid_box:
            continue

        score = inliers * 2 + len(good_matches)

        confidence = calculate_confidence(
            score,
            len(good_matches),
            inliers
        )

        if score > best["score"]:
            best = {
                "detected": True,
                "score": score,
                "matches": len(good_matches),
                "inliers": inliers,
                "confidence": confidence,
                "corners": transformed_corners,
                "template_scale": temp["scale"],
                "template_name": temp["name"]
            }

            # Strong enough, no need to check more templates
            if score >= EARLY_EXIT_SCORE:
                return best

    return best


# ================= LOAD REFERENCES =================

templates = load_reference_templates(REFERENCE_FOLDER)

# ================= CAMERA =================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_AVFOUNDATION
)

if not cap.isOpened():
    raise Exception("Could not open webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Optional: helps reduce webcam buffering on some setups
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ================= MAIN LOOP =================

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    display = frame.copy()
    frame_count += 1

    if roi is not None:
        x, y, w, h = roi

        search_frame = frame[y:y+h, x:x+w]
        offset_x, offset_y = x, y

        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            display,
            "ROI",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    else:
        search_frame = frame
        offset_x, offset_y = 0, 0

    gray = cv2.cvtColor(
        search_frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = preprocess(gray)

    # Run full detection only every N frames
    if frame_count % DETECT_EVERY_N_FRAMES == 0:
        last_result = match_against_templates(
            gray,
            templates
        )

    # On skipped frames, reuse last result so the display does not flicker
    result = last_result if last_result is not None else empty_result()

    if result["detected"]:
        corners = result["corners"].copy()

        corners[:, 0, 0] += offset_x
        corners[:, 0, 1] += offset_y

        pts = np.int32(corners)

        cv2.polylines(
            display,
            [pts],
            True,
            (0, 255, 0),
            3
        )

        status = (
            f"DETECTED | {result['template_name']} | "
            f"conf={result['confidence']}% | "
            f"score={result['score']} | "
            f"m={result['matches']} | "
            f"i={result['inliers']}"
        )

        color = (0, 255, 0)

    else:
        status = "NO DETECTION | conf=0.0%"
        color = (0, 0, 255)

    cv2.putText(
        display,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    cv2.putText(
        display,
        f"S: ROI | R: reset | Q: quit | detect every {DETECT_EVERY_N_FRAMES} frames",
        (20, display.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "ORB Multi-Reference Detection",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        selected = cv2.selectROI(
            "ORB Multi-Reference Detection",
            display,
            fromCenter=False,
            showCrosshair=True
        )

        sx, sy, sw, sh = selected

        if sw > 10 and sh > 10:
            roi = (sx, sy, sw, sh)
            last_result = None
            frame_count = 0
            print("ROI selected:", roi)

    elif key == ord("r"):
        roi = None
        last_result = None
        frame_count = 0
        print("ROI reset")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()