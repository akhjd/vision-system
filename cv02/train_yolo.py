"""
train_yolo.py  -  Train YOLOv8n to detect stickers.

Prerequisites:
    pip install ultralytics

Usage:
    1. Export your Roboflow dataset as "YOLOv8" format, unzip it here.
       You should have:  dataset/data.yaml
                         dataset/train/images/  + labels/
                         dataset/valid/images/  + labels/

    2. python3 train_yolo.py
    3. Best weights saved to: runs/detect/sticker/weights/best.pt
"""

from ultralytics import YOLO

DATA_YAML  = "dataset/data.yaml"   # path to your Roboflow export
EPOCHS     = 60
IMG_SIZE   = 640
MODEL_BASE = "yolov8n.pt"          # nano — fast, good enough for this task
PROJECT    = "runs/detect"
NAME       = "sticker"


def main():
    model = YOLO(MODEL_BASE)

    results = model.train(
        data    = DATA_YAML,
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = 16,
        project = PROJECT,
        name    = NAME,
        # Augmentations — helps a lot with limited data
        hsv_h   = 0.015,   # hue shift
        hsv_s   = 0.5,     # saturation shift
        hsv_v   = 0.4,     # brightness shift
        flipud  = 0.1,
        fliplr  = 0.5,
        degrees = 15,      # rotation
        scale   = 0.3,     # zoom
        mosaic  = 0.5,
    )

    print(f"\nTraining done.")
    print(f"Best weights: {PROJECT}/{NAME}/weights/best.pt")
    print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")


if __name__ == "__main__":
    main()