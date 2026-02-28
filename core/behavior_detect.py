import os
from ultralytics import YOLO


class BehaviorDetector:
    def __init__(self, model_path=None, confidence_threshold=0.45, smoothing_window=15):
        """
        Load YOLOv8 model for behavior detection.
        In a real scenario, this would load weights trained to detect Normal,
        Smoking, and Using Phone.
        """
        self.confidence_threshold = confidence_threshold
        self.behavior_history = []  # Empty initially to support single-image analysis
        self.smoothing_window = smoothing_window
        if model_path is None:
            # Point to the newly trained YOLO classification model by default
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "weights",
                "yolov8n_driver_cls",
                "weights",
                "best.pt",
            )

            # Fallback for old detection model if cls doesn't exist
            if not os.path.exists(model_path):
                model_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data",
                    "weights",
                    "yolov8n_driver_behavior2",
                    "weights",
                    "best.pt",
                )

        # We load a small model purely for demonstration
        self.model = YOLO(model_path)

    def process_frame(self, frame):
        """
        Detect bounding boxes and return annotated frame & behavior label.
        """
        # Run inference (强制压缩到 320 图片大小，极速出图，解决滞后感)
        results = self.model(frame, verbose=False, imgsz=320)
        annotated_frame = results[0].plot()

        # Determine behavior. For placeholder purposes, we say Normal unless
        # specific custom classes are found.
        behavior = "正常驾驶"

        # Check if the model is a classification model (has probs) or detection model (has boxes)
        if hasattr(results[0], "probs") and results[0].probs is not None:
            # Classification inference
            class_id = int(results[0].probs.top1)
            class_name = self.model.names[class_id]
            confidence = float(results[0].probs.top1conf)

            # Only accept predictions with high confidence, otherwise default to Normal
            if confidence >= self.confidence_threshold:
                cn_lower = class_name.lower()
                if "phone" in cn_lower or "texting" in cn_lower:
                    behavior = "使用手机"
                elif "drinking" in cn_lower:
                    behavior = "喝水"
                elif "radio" in cn_lower:
                    behavior = "操作中控/电台"
                elif "normal" not in cn_lower:
                    # Translate common classes for better UI
                    if "hair_and_makeup" in cn_lower:
                        behavior = "整理仪容"
                    elif "reaching_behind" in cn_lower:
                        behavior = "向后拿东西"
                    elif "talking_to_passenger" in cn_lower:
                        behavior = "与乘客交谈"
                    else:
                        behavior = f"分心: {class_name}"
        else:
            # For detection mode, confidence is per box
            for box in results[0].boxes:
                if float(box.conf[0]) >= self.confidence_threshold:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    if class_id in [1, 2, 3, 4]:
                        behavior = "使用手机"
                    elif class_id == 0:  # StateFarm drinking class is 0
                        behavior = "喝水"
                    elif class_id == 3:  # StateFarm radio class is 3
                        behavior = "操作中控/电台"
                    elif class_id != 2:  # Normal Driving is 2
                        behavior = f"分心: {class_name}"

        # Basic Smoothing/Debounce Logic
        self.behavior_history.append(behavior)
        if len(self.behavior_history) > self.smoothing_window:
            self.behavior_history.pop(0)

        # 核心防误报逻辑（针对视频流的跳变优化）：必须在滑动窗口期内，[所有危险行为]占比加起来超过 50%
        # 如果模型在一秒内（15帧）分别预测出：玩手机*5帧、整理仪容*4帧、拿东西*1帧、正常*5帧
        # 原来系统会认为单一动作没达到60%而放过。现在会把危险动作叠加(10/15 > 50%)，精准逮捕！
        normal_count = self.behavior_history.count("正常驾驶")
        history_len = len(self.behavior_history)

        # 安全除零保护
        if history_len == 0:
            return annotated_frame, behavior

        abnormal_ratio = 1.0 - (float(normal_count) / history_len)

        if abnormal_ratio >= 0.50:
            # 过滤掉正常驾驶，统计具体是哪个危险动作占主导
            from collections import Counter

            abnormal_acts = [b for b in self.behavior_history if "正常" not in b]
            if abnormal_acts:
                final_behavior = Counter(abnormal_acts).most_common(1)[0][0]
            else:
                final_behavior = "正常驾驶"
        else:
            final_behavior = "正常驾驶"

        return annotated_frame, final_behavior
