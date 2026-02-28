import os
from ultralytics import YOLO


class BehaviorDetector:
    def __init__(self, model_path=None, confidence_threshold=0.30, smoothing_window=15):
        """
        Load YOLOv8 model for behavior detection.
        In a real scenario, this would load weights trained to detect Normal,
        Smoking, and Using Phone.
        """
        self.confidence_threshold = confidence_threshold
        self.behavior_history = []  # Empty initially to support single-image analysis
        self.smoothing_window = smoothing_window

        # EMA (Exponential Moving Average) & 滞回滤波参数
        self.ema_danger_score = 0.0
        self.alpha = 0.3  # EMA学习率
        self.instant_trigger_threshold = (
            0.70  # 确凿证据阈值 (0延迟)，针对桌面测试大幅下调
        )
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
        frame_behavior = "正常驾驶"
        frame_confidence = 0.0

        # Check if the model is a classification model (has probs) or detection model (has boxes)
        if hasattr(results[0], "probs") and results[0].probs is not None:
            # Classification inference
            class_id = int(results[0].probs.top1)
            class_name = self.model.names[class_id]
            frame_confidence = float(results[0].probs.top1conf)

            # Only accept predictions with high confidence, otherwise default to Normal
            if frame_confidence >= self.confidence_threshold:
                cn_lower = class_name.lower()
                if "phone" in cn_lower or "texting" in cn_lower:
                    frame_behavior = "使用手机"
                elif "drinking" in cn_lower:
                    frame_behavior = "喝水"
                elif "radio" in cn_lower:
                    frame_behavior = "操作中控/电台"
                elif "normal" not in cn_lower:
                    # Translate common classes for better UI
                    if "hair_and_makeup" in cn_lower:
                        frame_behavior = "整理仪容"
                    elif "reaching_behind" in cn_lower:
                        frame_behavior = "向后拿东西"
                    elif "talking_to_passenger" in cn_lower:
                        frame_behavior = "与乘客交谈"
                    else:
                        frame_behavior = f"分心: {class_name}"
        else:
            # For detection mode, find max confidence
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf >= self.confidence_threshold and conf > frame_confidence:
                    frame_confidence = conf
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    if class_id in [1, 2, 3, 4]:
                        frame_behavior = "使用手机"
                    elif class_id == 0:  # StateFarm drinking class is 0
                        frame_behavior = "喝水"
                    elif class_id == 3:  # StateFarm radio class is 3
                        frame_behavior = "操作中控/电台"
                    elif class_id != 2:  # Normal Driving is 2
                        frame_behavior = f"分心: {class_name}"

        # ---------------------------------------------------------
        # 核心升级：双重滤波与 EMA 指数移动平均 (Hysteresis Logic)
        # ---------------------------------------------------------

        # 1. 计算当前的瞬时危险指数 (0.0 为正常, 1.0 为极度危险)
        instant_danger = 0.0 if "正常" in frame_behavior else frame_confidence

        # 2. EMA 平滑计算 (融合过去的状态与现在的状态)
        self.ema_danger_score = (self.alpha * instant_danger) + (
            (1 - self.alpha) * self.ema_danger_score
        )

        # 3. 滑动窗口入栈 (记录历史行为类别)
        self.behavior_history.append(frame_behavior)
        if len(self.behavior_history) > self.smoothing_window:
            self.behavior_history.pop(0)

        history_len = len(self.behavior_history)
        normal_count = self.behavior_history.count("正常驾驶")
        abnormal_ratio = (
            1.0 - (float(normal_count) / history_len) if history_len > 0 else 0.0
        )

        # 4. 最终裁决：触发器逻辑
        final_behavior = "正常驾驶"

        # 【触发机制 A: 零延迟确凿证据】如果模型瞬间爆出极高的置信度 (>0.85)，无视所有历史缓冲，立刻报警！
        if instant_danger >= self.instant_trigger_threshold:
            final_behavior = frame_behavior

        # 【触发机制 B: 模糊积攒防抖】即使某一帧只有 0.45 的置信度，但如果在 15 帧内或者 EMA 均值累积到警戒水位，则报警！
        elif abnormal_ratio >= 0.50 or self.ema_danger_score > 0.40:
            from collections import Counter

            abnormal_acts = [b for b in self.behavior_history if "正常" not in b]
            if abnormal_acts:
                final_behavior = Counter(abnormal_acts).most_common(1)[0][0]

        return annotated_frame, final_behavior
