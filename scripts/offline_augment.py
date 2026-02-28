import os
import cv2
import numpy as np
from glob import glob
import shutil


def augment_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return

    basename = os.path.basename(img_path)
    name, ext = os.path.splitext(basename)
    dir_name = os.path.dirname(img_path)

    # 1. Darkness Augmentation (Simulate Night/Tunnels)
    # Reduce brightness by 50%
    matrix_dark = np.ones(img.shape, dtype="uint8") * 50
    img_dark = cv2.subtract(img, matrix_dark)
    cv2.imwrite(os.path.join(dir_name, f"{name}_aug_dark{ext}"), img_dark)

    # 2. Gaussian Noise & Slight Blur (Simulate Cheap Camera/Motion Blur)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    noise = np.random.normal(0, 15, img_blur.shape).astype(np.uint8)
    img_noise = cv2.add(img_blur, noise)
    cv2.imwrite(os.path.join(dir_name, f"{name}_aug_noise{ext}"), img_noise)

    # 3. Random Slight Rotation (Simulate camera mounting angle)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    angle = np.random.uniform(-15, 15)
    scale = np.random.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    img_rotate = cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))
    cv2.imwrite(os.path.join(dir_name, f"{name}_aug_rot{ext}"), img_rotate)


if __name__ == "__main__":
    # Point this to your training directory
    dataset_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "statefarm_cls", "train")
    )

    if not os.path.exists(dataset_dir):
        print(f"Directory not found: {dataset_dir}")
        print("Please ensure your dataset is placed here.")
        exit(1)

    print(f"Starting Offline Data Augmentation for: {dataset_dir}")

    # Go through every subdirectory (c0, c1, ..., c9)
    classes = [
        d
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]

    total_augmented = 0
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        # Find all original images (ignore already augmented ones)
        images = [f for f in glob(os.path.join(cls_dir, "*.jpg")) if "_aug_" not in f]

        print(f"Processing Category [{cls}] - Found {len(images)} original images...")

        for img_path in images:
            augment_image(img_path)
            total_augmented += 3  # We generate 3 new images per original image

    print(f"\n🎉 Offline Augmentation Complete!")
    print(f"Successfully generated {total_augmented} new augmented images!")
    print(
        "You can now re-run `python scripts/train_yolo_cls.py` to train on this massively expanded dataset."
    )
