import os
import cv2
from ultralytics import YOLO

# load model
model_path = "data/weights/yolov8n_driver_cls2/weights/best.pt"
print(f"Loading '{model_path}'...")
model = YOLO(model_path)

# pick an image from val dataset where the driver is clearly using a phone
# Texting_Left
val_dir = "data/statefarm_cls/val/Talking_on_Phone_Right"
sample_img = os.path.join(val_dir, os.listdir(val_dir)[0])
print(f"Testing image from domain: {sample_img}")

img = cv2.imread(sample_img)
results = model(img)

probs = results[0].probs
class_id = int(probs.top1)
class_name = model.names[class_id]
conf = float(probs.top1conf)

print(f"\n✅ PREDICTION: {class_name} (Confidence: {conf:.4f})")
