import cv2
import numpy as np
import time

from core.behavior_detect import BehaviorDetector
from core.fatigue_detect import FatigueDetector
from core.session_logger import SessionLogger
from core.config import SMART_NIGHT_VISION_THRESHOLD


class MonitoringEngine:
    """
    核心监控引擎 (AI Pipeline Engine)
    统一调度所有底层深度学习分类器、姿态检测器、算法节点
    解耦 UI 层与算法层
    """

    def __init__(self):
        # 初始化各大核心组件
        self.behavior_detector = BehaviorDetector()
        self.fatigue_detector = FatigueDetector()
        self.logger = SessionLogger()

        # 预加载 CLAHE 夜视增强器（针对运算密集型，做一次化实例化）
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def reset(self):
        """当画面源切换或中断时，重置所有内部追迹器以防止历史幽灵数据残留"""
        if hasattr(self.fatigue_detector, "reset_tracker"):
            self.fatigue_detector.reset_tracker()
        if hasattr(self.behavior_detector, "reset_tracker"):
            self.behavior_detector.reset_tracker()
        self.logger.reset()

    def process_frame(self, frame, use_clahe=False):
        """
        AI流水线核心执行节点：
        输入图像 -> 返回处理后的图像以及结构字典化的报警状态
        """
        if frame is None:
            return None, {}

        # -------------------------------------------------------------------
        # 0. 底层预处理引擎（Smart Auto-Night-Vision）
        # -------------------------------------------------------------------
        if use_clahe:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            if avg_brightness < SMART_NIGHT_VISION_THRESHOLD:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                cl = self.clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # -------------------------------------------------------------------
        # 1. 行为分类检测树（YOLOv8-cls）
        # -------------------------------------------------------------------
        frame, behavior = self.behavior_detector.process_frame(frame)

        # 杜绝 macOS Metal 底层带来的只读内存报错锁死
        frame = np.ascontiguousarray(frame.copy())

        # -------------------------------------------------------------------
        # 2. 面部疲劳几何矩阵检测（MediaPipe/FaceMesh）
        # -------------------------------------------------------------------
        ear, mar, fatigue_level = self.fatigue_detector.process_frame(frame)

        # -------------------------------------------------------------------
        # 3. 状态路由与警报分发器（State Resolution）
        # -------------------------------------------------------------------
        # 打包核心输出字典，便于 UI 层进行不同维度的报警分发
        results = {
            "behavior_state": behavior,
            "fatigue_state": fatigue_level,
            "ear": ear,
            "mar": mar,
            "is_warning": ("正常" not in fatigue_level) or ("正常" not in behavior),
            "is_critical": "极度疲劳" in fatigue_level,
        }

        # -------------------------------------------------------------------
        # 4. 图形渲染挂载点与统计记录器
        # -------------------------------------------------------------------
        if results["is_warning"]:
            h, w = frame.shape[:2]
            border_thickness = 35 if results["is_critical"] else 15
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), border_thickness)

            # 将违规数据直接入库留存！(用于出具行车报表)
            self.logger.log_event(behavior, fatigue_level, results["is_critical"])

        return frame, results
