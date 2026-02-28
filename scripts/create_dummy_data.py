import os
import cv2
import numpy as np


def create_dummy_dataset(base_dir, num_samples=100):
    """
    Creates a dummy dataset with solid colors to act as placeholders for 'smoking' and 'phone'
    so that the YOLO training pipeline can be tested end-to-end without real data.
    """
    classes = ["normal", "smoking", "phone"]

    for split in ["train", "val"]:
        for sub in ["images", "labels"]:
            os.makedirs(os.path.join(base_dir, split, sub), exist_ok=True)

        for i in range(num_samples if split == "train" else num_samples // 5):
            # Select random class
            cls_idx = np.random.randint(0, len(classes))
            img_name = f"dummy_{split}_{i:04d}.jpg"
            img_path = os.path.join(base_dir, split, "images", img_name)
            txt_path = os.path.join(
                base_dir, split, "labels", img_name.replace(".jpg", ".txt")
            )

            # Create a simple colored image
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            if cls_idx == 0:
                img[:] = (200, 200, 200)  # Gray for normal
            elif cls_idx == 1:
                img[:] = (0, 0, 255)  # Red for smoking
            else:
                img[:] = (255, 0, 0)  # Blue for phone

            # Draw a bounding box for the object to simulate detection
            # Format: class x_center y_center width height
            cv2.rectangle(img, (200, 200), (440, 440), (0, 255, 0), 2)
            cv2.imwrite(img_path, img)

            with open(txt_path, "w") as f:
                f.write(f"{cls_idx} 0.5 0.5 0.375 0.375\n")

    # Create dataset.yaml
    yaml_content = f"""
path: {base_dir}
train: train/images
val: val/images

names:
  0: normal
  1: smoking
  2: phone
"""
    yaml_path = os.path.join(os.path.dirname(base_dir), "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Dummy dataset created at {base_dir}")
    print(f"Dataset YAML configured at {yaml_path}")


if __name__ == "__main__":
    PROCESSED_DATA_DIR = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed"
    )
    create_dummy_dataset(PROCESSED_DATA_DIR)
