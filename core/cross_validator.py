"""多模态交叉验证：YOLO 行为分类 + MediaPipe EAR/MAR + (可选) Pose 空间约束。

将三路独立判别合成一个最终风险等级，降低 YOLO 单点误判带来的不稳定。

设计原则：
1) 互证：同时被多个模态命中的事件可信度更高
2) 互斥：互相矛盾的判定降级为"疑似"
3) 叠加：行为 + 疲劳同时违规，风险等级一律升至 critical
"""

from collections import deque


# 风险等级枚举
RISK_SAFE = "safe"
RISK_WARN = "warn"
RISK_CRIT = "critical"


class CrossValidator:
    """多模态融合判定器。无状态实现，每帧独立融合，但维护短期 EAR/MAR 历史辅助决策。"""

    def __init__(self, ear_history_size=30, mar_history_size=30):
        # 短期历史窗口（用于"嘴始终未张"等趋势判断）
        self._ear_window = deque(maxlen=ear_history_size)
        self._mar_window = deque(maxlen=mar_history_size)

    def reset(self):
        self._ear_window.clear()
        self._mar_window.clear()

    def fuse(
        self,
        yolo_behavior: str,
        yolo_confidence: float,
        ear: float,
        mar: float,
        fatigue_state: str,
        pose_phone_confirmed: bool = False,
    ):
        """对单帧输出做融合判定。

        参数
            yolo_behavior: YOLO 防抖后的中文行为标签（"打电话"/"喝水"/"正常驾驶"等）
            yolo_confidence: YOLO 当前帧 top1 置信度（0-1）
            ear: 当前帧 EAR
            mar: 当前帧 MAR
            fatigue_state: FatigueDetector 输出的疲劳标签
            pose_phone_confirmed: Pose 空间约束是否独立确认了"打电话"动作

        返回 dict：
            final_behavior, final_fatigue, risk_level, fused_confidence, notes(list[str])
        """
        # 维护短期历史
        self._ear_window.append(ear)
        self._mar_window.append(mar)

        notes = []
        fused_conf = yolo_confidence
        final_behavior = yolo_behavior
        final_fatigue = fatigue_state

        is_behavior_abnormal = "正常" not in yolo_behavior
        is_fatigue_abnormal = "正常" not in fatigue_state and "未对齐" not in fatigue_state

        # ----------- 规则 1：喝水 ↔ MAR 互证 -----------
        # 喝水时嘴必然张开（MAR 上升），如果 MAR 持续偏低则降级
        if "喝水" in yolo_behavior:
            recent_mar_max = max(self._mar_window) if self._mar_window else mar
            if recent_mar_max < 0.30:
                final_behavior = "正常驾驶"
                fused_conf *= 0.4
                notes.append("MAR 持续偏低，喝水判定降级")
            else:
                fused_conf = min(1.0, fused_conf * 1.15)
                notes.append("MAR 与喝水互证")

        # ----------- 规则 2：打电话 ↔ Pose 互证 -----------
        if "打电话" in yolo_behavior:
            if pose_phone_confirmed:
                fused_conf = min(1.0, fused_conf * 1.25)
                notes.append("Pose 手腕-耳部距离确认打电话")
            elif yolo_confidence < 0.70:
                # YOLO 不太自信，且 Pose 也未确认 → 降级为疑似
                final_behavior = f"疑似{yolo_behavior}"
                fused_conf *= 0.6
                notes.append("Pose 未确认，降级为疑似")

        # ----------- 规则 3：发短信时 EAR 应正常（人在低头看屏，但还在睁眼）-----------
        # 如果 YOLO 说发短信但 EAR 持续过低（眼睛闭着）→ 实际可能在打瞌睡
        if "发短信" in yolo_behavior:
            if len(self._ear_window) >= 15:
                avg_ear = sum(self._ear_window) / len(self._ear_window)
                if avg_ear < 0.18:
                    notes.append("发短信判定 + EAR 过低，警惕实际为闭眼")

        # ----------- 规则 4：行为 + 疲劳同时违规 → 升级 critical -----------
        # 注意：这里要用「最终行为」（可能已被降级）而非原始 yolo_behavior
        is_behavior_abnormal_final = (
            "正常" not in final_behavior or "疑似" in final_behavior
        )
        if is_behavior_abnormal_final and is_fatigue_abnormal:
            risk = RISK_CRIT
            notes.append("行为与疲劳同时异常，叠加升级")
        elif "极度疲劳" in fatigue_state:
            risk = RISK_CRIT
        elif is_behavior_abnormal_final or is_fatigue_abnormal:
            risk = RISK_WARN
        else:
            risk = RISK_SAFE

        # 融合置信度做兜底裁剪
        fused_conf = max(0.0, min(1.0, fused_conf))

        return {
            "final_behavior": final_behavior,
            "final_fatigue": final_fatigue,
            "risk_level": risk,
            "fused_confidence": fused_conf,
            "notes": notes,
        }
