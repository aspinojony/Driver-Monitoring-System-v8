import os
from collections import Counter, deque

import cv2
import numpy as np
from ultralytics import YOLO

from core.config import (
    BEHAVIOR_MODEL_PATH,
    BEHAVIOR_MODEL_PATH_V2,
    BEHAVIOR_INFER_IMGSZ,
    BEHAVIOR_SMOOTHING_ALPHA,
    BEHAVIOR_TRUST_THRESHOLD,
    BEHAVIOR_ABNORMAL_RATIO,
    BEHAVIOR_EMA_DANGER_THRESHOLD,
    BEHAVIOR_USE_TTA,
    BEHAVIOR_TEMPERATURE,
    BEHAVIOR_NORMAL_PRIOR_BOOST,
    BEHAVIOR_VOTE_WINDOW,
)


class BehaviorDetector:
    def __init__(self, confidence_threshold=0.50, smoothing_window=15):
        """YOLOv8-cls 驾驶行为分类器。

        4 重推理增强：
          1) TTA：原图 + 水平翻转 softmax 平均
          2) 温度缩放：softmax 平滑过度自信
          3) 类别先验：Normal_Driving 加权
          4) 多帧 top-1 投票：与 EMA 互补
        """
        self.confidence_threshold = confidence_threshold
        self.behavior_history = []
        self.smoothing_window = smoothing_window

        # EMA 平滑参数
        self.ema_danger_score = 0.0
        self.alpha = BEHAVIOR_SMOOTHING_ALPHA
        self.instant_trigger_threshold = BEHAVIOR_TRUST_THRESHOLD
        self.abnormal_ratio_threshold = BEHAVIOR_ABNORMAL_RATIO
        self.ema_danger_threshold = BEHAVIOR_EMA_DANGER_THRESHOLD

        # 推理增强配置（运行时可调）
        self.use_tta = BEHAVIOR_USE_TTA
        self.temperature = BEHAVIOR_TEMPERATURE
        self.normal_prior_boost = BEHAVIOR_NORMAL_PRIOR_BOOST
        self.vote_window_size = BEHAVIOR_VOTE_WINDOW
        self._topk_history = deque(maxlen=max(1, BEHAVIOR_VOTE_WINDOW))

        # 暴露最近一次的类别概率分布（供 UI 实时显示）
        self.last_probs = None
        self.last_class_names = None
        self.last_class_name = "Normal_Driving"
        self.last_confidence = 0.0

        # 模型加载：v2 → 旧 domain_adapted → 公开 yolov8n-cls.pt
        for path in (BEHAVIOR_MODEL_PATH_V2, BEHAVIOR_MODEL_PATH):
            if os.path.exists(path):
                model_path = path
                break
        else:
            print(
                f"[BehaviorDetector] 未找到训练权重，回退到 yolov8n-cls.pt（仅用于本地启动调试）"
            )
            model_path = "yolov8n-cls.pt"

        self.model = YOLO(model_path)
        # 缓存 normal 类索引，避免每帧字符串匹配
        self._normal_idx = self._find_normal_idx()

    def _find_normal_idx(self):
        names = self.model.names
        if not names:
            return None
        for idx, name in names.items():
            if "normal" in str(name).lower() or "safe" in str(name).lower():
                return idx
        return None

    def reset_tracker(self):
        self.behavior_history.clear()
        self.ema_danger_score = 0.0
        self.last_probs = None
        self._topk_history.clear()

    def _run_once(self, frame):
        """单次模型 forward，返回 probs (np.ndarray) 或 None。"""
        results = self.model(frame, verbose=False, imgsz=BEHAVIOR_INFER_IMGSZ)
        if not hasattr(results[0], "probs") or results[0].probs is None:
            return results, None
        probs = results[0].probs.data.cpu().numpy()
        return results, probs

    def _enhance_probs(self, probs: np.ndarray) -> np.ndarray:
        """温度缩放 + 类别先验加权后重新归一化。

        温度缩放对已是 softmax 输出的概率近似为：p ← p^(1/T) / sum
        """
        if probs is None:
            return probs
        p = probs.astype(np.float64)

        # 温度缩放（>1 软化分布；<1 锐化）
        if abs(self.temperature - 1.0) > 1e-3:
            p = np.power(np.clip(p, 1e-9, 1.0), 1.0 / max(0.1, self.temperature))

        # 类别先验加权（Normal_Driving 在真实场景中频次最高）
        if (
            self._normal_idx is not None
            and 0 <= self._normal_idx < len(p)
            and abs(self.normal_prior_boost - 1.0) > 1e-3
        ):
            p[self._normal_idx] *= self.normal_prior_boost

        # 重新归一化
        s = p.sum()
        if s > 0:
            p = p / s
        return p.astype(np.float32)

    def _vote_topk(self, top1_idx: int) -> int:
        """最近 N 帧 top-1 多数投票，输出投票后的 top-1。"""
        self._topk_history.append(int(top1_idx))
        if len(self._topk_history) < 2:
            return int(top1_idx)
        cnt = Counter(self._topk_history)
        # 当 top1 在窗口内不是多数时，用多数票替换
        return cnt.most_common(1)[0][0]

    def _label_to_chinese(self, class_name: str) -> str:
        cn_lower = str(class_name).lower()
        if "phone" in cn_lower or "talking_on_phone" in cn_lower:
            return "打电话"
        if "texting" in cn_lower:
            return "发短信"
        if "drinking" in cn_lower:
            return "喝水"
        if "radio" in cn_lower or "operating" in cn_lower:
            return "操作中控"
        if "hair_and_makeup" in cn_lower or "makeup" in cn_lower:
            return "整理仪容"
        if "reaching_behind" in cn_lower:
            return "向后取物"
        if "talking_to_passenger" in cn_lower:
            return "与乘客交谈"
        if "normal" in cn_lower or "safe" in cn_lower:
            return "正常驾驶"
        return f"分心: {class_name}"

    def process_frame(self, frame):
        """单帧推理 + 4 重增强 + 双重防抖。返回（标注图, 最终行为标签）。"""
        # ===== 1. 模型推理（含 TTA）=====
        results, probs = self._run_once(frame)
        annotated_frame = results[0].plot()

        frame_behavior = "正常驾驶"
        frame_confidence = 0.0

        if probs is not None:
            # TTA：水平翻转后再推理一次，softmax 取平均
            if self.use_tta:
                flipped = cv2.flip(frame, 1)
                _, probs_flip = self._run_once(flipped)
                if probs_flip is not None and probs_flip.shape == probs.shape:
                    probs = (probs + probs_flip) / 2.0

            # 温度缩放 + 类别先验
            probs_enh = self._enhance_probs(probs)
            top1 = int(np.argmax(probs_enh))

            # 多帧投票（与 EMA 互补，主要解决单帧抖动）
            top1 = self._vote_topk(top1)

            class_name = self.model.names[top1]
            frame_confidence = float(probs_enh[top1])

            self.last_probs = probs_enh.tolist()
            self.last_class_names = self.model.names
            self.last_class_name = class_name
            self.last_confidence = frame_confidence

            if frame_confidence >= self.confidence_threshold:
                frame_behavior = self._label_to_chinese(class_name)
        else:
            # 检测模型分支（向后兼容）
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf >= self.confidence_threshold and conf > frame_confidence:
                    frame_confidence = conf
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    if class_id in [1, 2, 3, 4]:
                        frame_behavior = "打电话/发短信"
                    elif class_id == 0:
                        frame_behavior = "喝水"
                    elif class_id != 2:
                        frame_behavior = f"分心: {class_name}"

        # ===== 2. 双重防抖：EMA + 滑动窗口 =====
        instant_danger = 0.0 if "正常" in frame_behavior else frame_confidence
        self.ema_danger_score = (self.alpha * instant_danger) + (
            (1 - self.alpha) * self.ema_danger_score
        )

        self.behavior_history.append(frame_behavior)
        if len(self.behavior_history) > self.smoothing_window:
            self.behavior_history.pop(0)

        history_len = len(self.behavior_history)
        normal_count = self.behavior_history.count("正常驾驶")
        abnormal_ratio = (
            1.0 - (float(normal_count) / history_len) if history_len > 0 else 0.0
        )

        # 触发逻辑
        final_behavior = "正常驾驶"
        if instant_danger >= self.instant_trigger_threshold:
            final_behavior = frame_behavior
        elif (
            abnormal_ratio >= self.abnormal_ratio_threshold
            or self.ema_danger_score > self.ema_danger_threshold
        ):
            abnormal_acts = [b for b in self.behavior_history if "正常" not in b]
            if abnormal_acts:
                final_behavior = Counter(abnormal_acts).most_common(1)[0][0]

        return annotated_frame, final_behavior
