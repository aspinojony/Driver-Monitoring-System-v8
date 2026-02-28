import os
import shutil
import random
from collections import defaultdict
from tqdm import tqdm


def convert_to_classification_dataset(source_dir, dest_dir, val_ratio=0.2):
    """
    Converts the State Farm dataset (which is just a folder of classes)
    into a YOLOv8 classification format (train/val splits).
    """
    train_dir = os.path.join(dest_dir, "train")
    val_dir = os.path.join(dest_dir, "val")

    # Create root dirs
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Class mapping for better folder names
    class_map = {
        "c0": "Normal_Driving",
        "c1": "Texting_Right",
        "c2": "Talking_on_Phone_Right",
        "c3": "Texting_Left",
        "c4": "Talking_on_Phone_Left",
        "c5": "Operating_Radio",
        "c6": "Drinking",
        "c7": "Reaching_Behind",
        "c8": "Hair_and_Makeup",
        "c9": "Talking_to_Passenger",
    }

    print(f"Reading source images from {source_dir}...")

    source_train_dir = os.path.join(source_dir, "imgs", "train")
    if not os.path.exists(source_train_dir):
        # If user extracted directly
        source_train_dir = source_dir

    classes = [
        d
        for d in os.listdir(source_train_dir)
        if d.startswith("c") and os.path.isdir(os.path.join(source_train_dir, d))
    ]

    if not classes:
        print(f"Error: Could not find class folders (c0-c9) in {source_train_dir}")
        return

    for c in classes:
        target_class_name = class_map.get(c, c)

        # Create class folders in train and val directory
        os.makedirs(os.path.join(train_dir, target_class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, target_class_name), exist_ok=True)

        class_dir = os.path.join(source_train_dir, c)
        images = [
            f
            for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # Shuffle for random split
        random.shuffle(images)
        val_count = int(len(images) * val_ratio)

        val_images = images[:val_count]
        train_images = images[val_count:]

        print(
            f"Copying class {c} ({target_class_name}) - {len(train_images)} train, {len(val_images)} val"
        )

        # Copy train images
        for img in tqdm(train_images, desc=f"Train {target_class_name}", leave=False):
            shutil.copy2(
                os.path.join(class_dir, img),
                os.path.join(train_dir, target_class_name, img),
            )

        # Copy val images
        for img in tqdm(val_images, desc=f"Val {target_class_name}", leave=False):
            shutil.copy2(
                os.path.join(class_dir, img),
                os.path.join(val_dir, target_class_name, img),
            )

    print(f"\nSuccessfully created YOLO classification dataset at: {dest_dir}")
    print("Ready for YOLOv8-cls training!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert State Farm Dataset for YOLO Classification"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="state-farm-distracted-driver-detection",
        help="Path to original state farm data",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="data/statefarm_cls",
        help="Output directory for classification dataset",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="Validation set ratio"
    )

    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_path = os.path.join(root_dir, args.source)
    dest_path = os.path.join(root_dir, args.dest)

    convert_to_classification_dataset(source_path, dest_path, args.val_ratio)
