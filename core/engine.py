import cv2
import numpy as np

from core.behavior_detect import BehaviorDetector
from core.fatigue_detect import FatigueDetector
from core.session_logger import SessionLogger
from core.cross_validator import CrossValidator, RISK_SAFE, RISK_WARN, RISK_CRIT
from core.config import (
    SMART_NIGHT_VISION_THRESHOLD,
    BEHAVIOR_FRAME_SKIP,
    POSE_FRAME_SKIP,
)


class MonitoringEngine:
    """统一调度行为分类、疲劳检测、(可选) Pose 二级判定与多模态交叉验证。

    调用方只需 process_frame(frame) 拿到（标注图, 结构化结果字典）。
    """

    def __init__(self, enable_pose=False):
        self.behavior_detector = BehaviorDetector()
        self.fatigue_detector = FatigueDetector()
        self.cross_validator = CrossValidator()
        self.logger = SessionLogger()

        # CLAHE 暗光增强器（一次实例化复用）
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # Pose 二级判定（默认关闭，Phase 1.2 启用）
        self.enable_pose = enable_pose
        self.pose_validator = None
        self._pose_frame_skip = POSE_FRAME_SKIP
        self._behavior_frame_skip = BEHAVIOR_FRAME_SKIP
        self._frame_counter = 0
        if self.enable_pose:
            self._init_pose()

        # 跳帧期间复用的缓存
        self._last_pose_phone_confirmed = False
        self._last_behavior = "正常驾驶"
        self._last_annotated = None

    def _init_pose(self):
        """延迟加载 Pose 模型，避免不需要时白白占内存。"""
        try:
            from core.pose_detect import PoseConstrainedPhoneDetector
            from core.config import POSE_MODEL_PATH
            import os

            model_path = POSE_MODEL_PATH if os.path.exists(POSE_MODEL_PATH) else "yolov8n-pose.pt"
            self.pose_validator = PoseConstrainedPhoneDetector(model_path=model_path)
        except Exception as e:
            print(f"[engine] Pose 初始化失败，跳过：{e}")
            self.enable_pose = False
            self.pose_validator = None

    def reset(self):
        """画面源切换或中断时重置所有内部状态。"""
        if hasattr(self.fatigue_detector, "reset_tracker"):
            self.fatigue_detector.reset_tracker()
        if hasattr(self.behavior_detector, "reset_tracker"):
            self.behavior_detector.reset_tracker()
        self.cross_validator.reset()
        self._last_pose_phone_confirmed = False
        self._last_behavior = "正常驾驶"
        self._last_annotated = None
        self._frame_counter = 0
        self.logger.reset()

    def process_frame(self, frame, use_clahe=False):
        """单帧 AI 推理流水线。返回（标注图, 结构化结果）。"""
        if frame is None:
            return None, {}

        self._frame_counter += 1

        # 0. 暗光自动增强
        if use_clahe:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            if avg_brightness < SMART_NIGHT_VISION_THRESHOLD:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                cl = self.clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # 1. 行为分类（跳帧：每 N 帧推理 1 次，期间复用上次的标注图与标签）
        run_behavior = (self._frame_counter % self._behavior_frame_skip) == 1 or self._last_annotated is None
        if run_behavior:
            frame, behavior = self.behavior_detector.process_frame(frame)
            self._last_annotated = frame
            self._last_behavior = behavior
        else:
            # 复用上次推理结果，但保持帧不冻结：用当前 frame 而非缓存的 annotated（避免画面卡住）
            behavior = self._last_behavior
            # 不绘制 YOLO 注解，保留原始画面 → MediaPipe 仍会画脸网格

        # 修复 macOS Metal 后端内存只读问题
        frame = np.ascontiguousarray(frame.copy())

        # 2. Pose 二级判定（跳帧执行，跳帧期间复用上次结果）
        if self.enable_pose and self.pose_validator is not None:
            if self._frame_counter % self._pose_frame_skip == 0:
                _, pose_confirmed, _ = self.pose_validator.verify_phone_usage(frame)
                self._last_pose_phone_confirmed = pose_confirmed
        pose_phone_confirmed = self._last_pose_phone_confirmed

        # 3. 疲劳检测（EAR/MAR/PERCLOS）
        ear, mar, fatigue_level = self.fatigue_detector.process_frame(frame)

        # 4. 多模态交叉验证融合
        yolo_conf = getattr(self.behavior_detector, "last_confidence", 0.0)
        fused = self.cross_validator.fuse(
            yolo_behavior=behavior,
            yolo_confidence=yolo_conf,
            ear=ear,
            mar=mar,
            fatigue_state=fatigue_level,
            pose_phone_confirmed=pose_phone_confirmed,
        )

        # 5. 结构化结果（向后兼容旧字段 + 融合新字段）
        results = {
            # 旧字段（保持 UI 兼容）
            "behavior_state": fused["final_behavior"],
            "fatigue_state": fused["final_fatigue"],
            "ear": ear,
            "mar": mar,
            "is_warning": fused["risk_level"] in (RISK_WARN, RISK_CRIT),
            "is_critical": fused["risk_level"] == RISK_CRIT,
            # 融合后新字段
            "risk_level": fused["risk_level"],
            "fused_confidence": fused["fused_confidence"],
            "fusion_notes": fused["notes"],
            "yolo_raw_confidence": yolo_conf,
            "pose_phone_confirmed": pose_phone_confirmed,
            # 8 类完整概率分布（前端可视化用）
            "class_probs": getattr(self.behavior_detector, "last_probs", None),
            "class_names": getattr(self.behavior_detector, "last_class_names", None),
        }

        # 6. 报警边框 + 事件入库
        if results["is_warning"]:
            h, w = frame.shape[:2]
            border_thickness = 35 if results["is_critical"] else 15
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), border_thickness)
            self.logger.log_event(
                results["behavior_state"], results["fatigue_state"], results["is_critical"]
            )

        return frame, results
