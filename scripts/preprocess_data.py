import cv2
import numpy as np
import os
import glob
import random


def color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.2):
    """
    Apply Color Jittering to simulate complex cabin lighting.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Saturation
    if saturation > 0:
        sat_factor = 1.0 + random.uniform(-saturation, saturation)
        hsv[:, :, 1] = hsv[:, :, 1] * sat_factor

    # Brightness (Value)
    if brightness > 0:
        val_factor = 1.0 + random.uniform(-brightness, brightness)
        hsv[:, :, 2] = hsv[:, :, 2] * val_factor

    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

    hsv = hsv.astype(np.uint8)
    jittered = cv2.cvtColor(hsv, cv2.HSV2BGR)

    # Contrast
    if contrast > 0:
        alpha = 1.0 + random.uniform(-contrast, contrast)
        jittered = cv2.convertScaleAbs(jittered, alpha=alpha, beta=0)

    return jittered


def create_yolo_directories(base_dir):
    """
    Create standard YOLO dataset directories.
    """
    for split in ["train", "val", "test"]:
        for sub in ["images", "labels"]:
            os.makedirs(os.path.join(base_dir, split, sub), exist_ok=True)


if __name__ == "__main__":
    PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    create_yolo_directories(PROCESSED_DIR)

    print(f"Created YOLO standard dataset structure in {PROCESSED_DIR}")
    print(
        "YOLOv8 inherently applies Mosaic during training; we rely on ultralytics dataloader for online Mosaic."
    )
    print("Use color_jitter function for offline data augmentation if needed.")
