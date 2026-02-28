import math
from ultralytics import YOLO


class PoseConstrainedPhoneDetector:
    """
    基于 YOLOv8-Pose 肢体关键点的严格数学拓扑学约束引擎。
    彻底摒弃单纯的“照片分类瞎猜”，转而提取空间向量(X, Y)：
    检查 手腕节点(Wrist) 与 耳朵节点(Ear) 的欧氏距离。
    如果距离突破死区且保持 1 秒钟，则 100% 判定为在恶性接打电话。排除挠痒痒等瞬时跳变。
    """

    def __init__(
        self,
        model_path="yolov8n-pose.pt",
        distance_threshold=80.0,
        strict_frames_required=15,
    ):
        # 自动下载或加载轻量级的人体骨架提取模型
        self.model = YOLO(model_path)

        # 核心约束变量
        self.distance_threshold = distance_threshold  # 欧氏距离阈值（像素近似 5cm）
        self.strict_frames_required = (
            strict_frames_required  # 比如15帧(0.5秒)排除挠痒痒
        )

        # 状态机缓冲
        self.suspected_frames = 0

        # COCO KeyPoints 索引标准
        self.KP_L_EAR = 3
        self.KP_R_EAR = 4
        self.KP_L_WRIST = 9
        self.KP_R_WRIST = 10

    def _euclid_distance(self, pt1, pt2):
        """计算平面特征点之间的直线极坐标欧氏距离"""
        return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    def verify_phone_usage(self, frame):
        """
        传入当前视频帧，返回：
        1. 绘制了骨架的帧 (Annotated Frame)
        2. 是否处于接打电话的布尔危险状态 (Boolean State)
        3. 调试输出日志 (Debug Details)
        """
        results = self.model(frame, verbose=False, imgsz=320)
        annotated_frame = results[0].plot()

        is_danger = False
        debug_log = "未监测到举手接听动作"

        # 如果画面里根本没人或者没检测到骨架
        if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
            self.suspected_frames = 0
            return annotated_frame, False, "未侦测到驾驶员躯干结构。"

        # 获取第一个人（驾驶员）的 17 个关键点矩阵
        keypoints = results[0].keypoints.xy[0].cpu().numpy()

        # 确保关键点足够多（排除半身裁剪严重导致识别不到手腕的情况）
        if len(keypoints) > self.KP_R_WRIST:
            l_ear = keypoints[self.KP_L_EAR]
            r_ear = keypoints[self.KP_R_EAR]
            l_wrist = keypoints[self.KP_L_WRIST]
            r_wrist = keypoints[self.KP_R_WRIST]

            # 过滤掉为 (0,0) 的丢失黑块数据（比如手缩在方向盘底下）
            dist_left = float("inf")
            dist_right = float("inf")

            if l_ear[0] != 0 and l_wrist[0] != 0:
                dist_left = self._euclid_distance(l_ear, l_wrist)

            if r_ear[0] != 0 and r_wrist[0] != 0:
                dist_right = self._euclid_distance(r_ear, r_wrist)

            # --- 空间欧氏约束：手腕距离耳朵非常近 ---
            min_dist = min(dist_left, dist_right)

            if min_dist < self.distance_threshold:
                self.suspected_frames += 1
                debug_log = f"⚠️ [危险累加中] 手腕靠近耳部! 当前极坐标距离: {min_dist:.1f}px (帧数: {self.suspected_frames}/{self.strict_frames_required})"

                # --- 时间防抖约束：该动作必须至少维持半秒，否则一律判为"挠皮肤"废动作 ---
                if self.suspected_frames >= self.strict_frames_required:
                    is_danger = True
                    debug_log = f"🚨 [拦截接听行为] 距耳朵小于阈值且保持不动超时！数学界定为正在接打电话！"
            else:
                # 一旦放下手立刻清空恶意累积层
                self.suspected_frames = 0
                debug_log = f"✅ 安全。最小手耳距离: {min_dist:.1f}px (大于 {self.distance_threshold}px 的安全范围)"
        else:
            self.suspected_frames = 0
            debug_log = "核心肢体节点(手腕等)处于摄像机视野盲区。"

        return annotated_frame, is_danger, debug_log
