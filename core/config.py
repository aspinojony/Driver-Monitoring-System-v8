import os

# ---------------------------------------------------------
# 全局配置（Configuration）
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- 模型权重路径 ----
# 优先使用 v2 训练产物；不存在则回退到旧权重；再不存在则由 BehaviorDetector 回退到公开 yolov8n-cls.pt
BEHAVIOR_MODEL_PATH_V2 = os.path.join(
    PROJECT_ROOT, "runs", "classify", "dms_v2_final", "weights", "best.pt"
)
BEHAVIOR_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "runs", "classify", "domain_adapted_cls_final", "weights", "best.pt"
)
POSE_MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "weights", "yolov8n-pose.pt")

# ---- 推理参数（必须与训练对齐，否则准确率断崖式下跌）----
BEHAVIOR_INFER_IMGSZ = 224

# ---- 声音报警 ----
ALARM_FATIGUE_SOUND = "/System/Library/Sounds/Glass.aiff"
ALARM_DISTRACT_SOUND = "/System/Library/Sounds/Ping.aiff"

# ---- 疲劳检测（MediaPipe EAR/MAR）----
PERCLOS_WINDOW_FRAMES = 450  # 统计窗口（约 15 秒，按 30fps 估算）
PERCLOS_DANGER_THRESHOLD = 0.50  # 窗口内闭眼比例阈值
CONTINUOUS_BLINK_FRAMES = 45  # 连续闭眼帧数（1.5 秒）
CONTINUOUS_YAWN_FRAMES = 60  # 连续张嘴帧数（2 秒）

# ---- 行为识别防抖 ----
BEHAVIOR_SMOOTHING_ALPHA = 0.30  # EMA 平滑系数（越大越敏感）
BEHAVIOR_TRUST_THRESHOLD = 0.75  # 即时触发阈值（置信度高于此值无视历史立刻报警）
BEHAVIOR_ABNORMAL_RATIO = 0.50  # 滑动窗口异常比例阈值
BEHAVIOR_EMA_DANGER_THRESHOLD = 0.45  # EMA 危险分数阈值

# ---- 摄像头处理 ----
# 是否对摄像头帧做水平镜像。
# True：训练数据采集时已镜像（自录数据集），推理也需镜像保持一致；
# False：训练数据未镜像（如 v2 模型走两阶段，统一不镜像）。
MIRROR_CAMERA_FRAME = True

# ---- 推理跳帧（性能优化）----
BEHAVIOR_FRAME_SKIP = 2  # 行为分类每 N 帧推理 1 次（跳帧期间复用上次结果）
POSE_FRAME_SKIP = 5  # Pose 二级判定每 N 帧推理 1 次

# ---- 推理增强（无需重训即可生效）----
BEHAVIOR_USE_TTA = True              # 测试时增强：原图 + 水平翻转 softmax 平均
BEHAVIOR_TEMPERATURE = 1.5           # 温度缩放（>1 软化分布，降低过度自信）
BEHAVIOR_NORMAL_PRIOR_BOOST = 1.20   # Normal_Driving 类先验加权（真实驾驶场景下正常状态占多数）
BEHAVIOR_VOTE_WINDOW = 5             # 多帧 top-1 多数投票窗口（与 EMA 互补）

# ---- UI ----
UI_FPS_LIMIT = 30
SMART_NIGHT_VISION_THRESHOLD = 80  # 平均亮度低于此值启用 CLAHE 增强
