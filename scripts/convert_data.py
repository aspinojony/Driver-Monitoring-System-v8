import os
import shutil
import random
from glob import glob
from tqdm import tqdm
import cv2


def convert_state_farm_to_yolo(source_dir, dest_dir, train_ratio=0.8):
    """
    Convert State Farm dataset (classification: folders c0-c9)
    into YOLOv8 format (detection: images and labels).

    Since State Farm provides no bounding boxes, we will create a dummy bounding
    box covering the entire image (or central region) purely to allow the YOLO
    object detection pipeline to train on classification features.
    """
    classes_map = {
        "c0": "safe_driving",
        "c1": "texting_right",
        "c2": "talking_on_phone_right",
        "c3": "texting_left",
        "c4": "talking_on_phone_left",
        "c5": "operating_radio",
        "c6": "drinking",
        "c7": "reaching_behind",
        "c8": "hair_and_makeup",
        "c9": "talking_to_passenger",
    }

    # Target structure
    for split in ["train", "val"]:
        for sub in ["images", "labels"]:
            os.makedirs(os.path.join(dest_dir, split, sub), exist_ok=True)

    train_dir = os.path.join(source_dir, "imgs", "train")

    if not os.path.exists(train_dir):
        print(f"Error: Could not find train dir at {train_dir}")
        return

    print("Converting State Farm format to YOLO format...")

    # Process each class
    for class_folder in sorted(os.listdir(train_dir)):
        if not class_folder.startswith("c"):
            continue

        class_idx = int(class_folder[1])  # c0 -> 0, c1 -> 1
        class_name = classes_map[class_folder]

        folder_path = os.path.join(train_dir, class_folder)
        images = glob(os.path.join(folder_path, "*.jpg"))

        # We only take a subset to speed up the graduation project training
        # User has limited time. 400 per class is enough for a demo model.
        subset_size = min(len(images), 400)
        selected_images = random.sample(images, subset_size)

        print(f"Processing Class {class_folder} ({class_name}): {subset_size} images")

        for img_path in tqdm(selected_images):
            # 80/20 split
            split = "train" if random.random() < train_ratio else "val"

            img_name = os.path.basename(img_path)
            new_img_name = f"{class_folder}_{img_name}"

            # Destination paths
            dest_img_path = os.path.join(dest_dir, split, "images", new_img_name)
            dest_txt_path = os.path.join(
                dest_dir, split, "labels", new_img_name.replace(".jpg", ".txt")
            )

            # 1. Copy image
            shutil.copy(img_path, dest_img_path)

            # 2. To use standard YOLO object detection, it needs a bounding box.
            # We place a broad bounding box covering the central driver region.
            # State farm images are 640x480
            # format: class_id center_x center_y width height (normalized 0-1)
            # Center right side of image where driver usually is: cx=0.7, cy=0.5, w=0.5, h=0.8
            with open(dest_txt_path, "w") as f:
                f.write(f"{class_idx} 0.5 0.5 0.8 0.8\n")

    # Create YAML file
    yaml_content = f"path: {os.path.abspath(dest_dir)}\n"
    yaml_content += "train: train/images\n"
    yaml_content += "val: val/images\n\n"
    yaml_content += "names:\n"
    for i in range(10):
        yaml_content += f"  {i}: {classes_map[f'c{i}']}\n"

    yaml_path = os.path.join(os.path.dirname(dest_dir), "statefarm_dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n✅ Conversion Complete!")
    print(f"✅ Data stored in {dest_dir}")
    print(f"✅ YAML config saved to {yaml_path}")


if __name__ == "__main__":
    SOURCE_DIR = "/Users/a0000/Desktop/Projects_项目/毕业设计/state-farm-distracted-driver-detection"
    DEST_DIR = "/Users/a0000/Desktop/Projects_项目/毕业设计/data/processed_statefarm"
    convert_state_farm_to_yolo(SOURCE_DIR, DEST_DIR)
