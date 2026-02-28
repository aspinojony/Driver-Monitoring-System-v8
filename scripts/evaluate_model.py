import time
import cv2
from ultralytics import YOLO


def evaluate_fps(model_path, video_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    frames = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        frames += 1

    end_time = time.time()
    total_time = end_time - start_time
    fps = frames / total_time

    print(f"Processed {frames} frames in {total_time:.2f}s")
    print(f"Average FPS: {fps:.2f}")
    cap.release()


if __name__ == "__main__":
    print("Evaluating model FPS performance...")
    print(
        "Placeholder: Provide a valid model path (.pt) and video path (.mp4) to test."
    )
    # evaluate_fps('yolov8n.pt', 'test_video.mp4')
