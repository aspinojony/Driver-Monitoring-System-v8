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
PERCLOS_WINDOW_FRAMES = 300  # 统计区间帧数 (以30FPS计，约10秒)
PERCLOS_DANGER_THRESHOLD = 0.40  # 10秒内超过 40% 时间闭眼则极度危险报警
CONTINUOUS_BLINK_FRAMES = 15  # 连续闭眼多少次算极度疲劳/睡着 (15帧即0.5秒)
CONTINUOUS_YAWN_FRAMES = 15  # 连续张大嘴多少次算哈欠

# 行为识别 (YOLO 行为敏感度配置)
BEHAVIOR_SMOOTHING_ALPHA = 0.20  # EMA 指数平滑过滤器的融合系数 (越小越稳定，越大越灵敏)
BEHAVIOR_TRUST_THRESHOLD = 0.85  # 置信度积累到多少才确信违规 (避免闪烁)

# ---- 系统显示/UI 配置 ----
UI_FPS_LIMIT = 30  # 界面更新最大帧率
SMART_NIGHT_VISION_THRESHOLD = 80  # 平均像素亮度低于此值，自动启动 CLAHE 夜视增强算法
