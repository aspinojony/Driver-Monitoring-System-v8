import os

# ---------------------------------------------------------
# 全局配置文件 (Configuration Management)
# ---------------------------------------------------------

# 项目根目录获取
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- 模型权重路径配置 ----
# YOLOv8 行为检测分类模型权重 (指向目前最新提纯精调的桌面专属版本)
BEHAVIOR_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "runs", "classify", "domain_adapted_cls_final", "weights", "best.pt"
)

# YOLOv8 姿态检测模型权重 (备用/融合检测)
POSE_MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "weights", "yolov8n-pose.pt")

# ---- 声音报警器配置 ----
ALARM_FATIGUE_SOUND = "/System/Library/Sounds/Glass.aiff"
ALARM_DISTRACT_SOUND = "/System/Library/Sounds/Ping.aiff"

# ---- 核心业务逻辑阈值 ----
# 疲劳检测 (MediaPipe 耳目张合度设置)
PERCLOS_WINDOW_FRAMES = 450  # 统计区间帧数 (大幅延长到15秒的判定周期)
PERCLOS_DANGER_THRESHOLD = 0.50  # 15秒内有 >50% 的时间闭眼才报警
CONTINUOUS_BLINK_FRAMES = 45  # 连续闭眼帧数 (45帧 = 1.5秒，允许一般的缓慢眨目)
CONTINUOUS_YAWN_FRAMES = 60  # 连续张嘴帧数 (60帧 = 2秒，防止讲话张嘴被误抓)

# 行为识别 (YOLO 行为敏感度配置)
BEHAVIOR_SMOOTHING_ALPHA = (
    0.05  # EMA 融合系数极度下调 (0.05表示行为必须持续数秒才能冲破报警线)
)
BEHAVIOR_TRUST_THRESHOLD = 0.96  # 极度置信度 (必须有超强的确凿证据才允许零延迟触发)

# ---- 系统显示/UI 配置 ----
UI_FPS_LIMIT = 30  # 界面更新最大帧率
SMART_NIGHT_VISION_THRESHOLD = 80  # 平均像素亮度低于此值，自动启动 CLAHE 夜视增强算法
