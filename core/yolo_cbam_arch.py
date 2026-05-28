"""将 CBAM 注意力模块集成进 Ultralytics YOLOv8-cls。

Ultralytics 通过 yaml 描述模型结构，并用 nn.tasks.parse_model 把字符串模块名解析为类。
本模块负责：
1) 把 core.cbam.CBAM 注入到 nn.tasks 的全局命名空间（让 yaml 能识别）
2) 提供自定义 yolov8s-cls-cbam.yaml（在 backbone 最后一个 C2f 之后插入 CBAM）
3) 提供 build_cbam_cls_model() 工厂函数
"""

import os

YAML_FILENAME = "yolov8s_cls_cbam.yaml"

# YOLOv8s-cls + CBAM 的模型描述
# Ultralytics 解析 yaml 时，Conv/C2f 的 args[0] 会被 width_multiple 缩放，
# 但自定义模块（CBAM）不会自动缩放——所以 CBAM 这里必须写「缩放后」的真实通道。
# yolov8s: width=0.50, max_channels=1024 → 最后 C2f 输出实际是 min(1024, 1024) * 0.5 = 512
CBAM_CLS_YAML = """# YOLOv8s-cls with CBAM (Convolutional Block Attention Module)
# 在最后一个 C2f 之后插入 CBAM，通道+空间双注意力，强化关键特征
nc: 8  # 8 classes (DMS v2)
scales:
  s: [0.33, 0.50, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]   # 0  → 32 ch
  - [-1, 1, Conv, [128, 3, 2]]  # 1  → 64 ch
  - [-1, 3, C2f, [128, True]]   # 2  → 64 ch
  - [-1, 1, Conv, [256, 3, 2]]  # 3  → 128 ch
  - [-1, 6, C2f, [256, True]]   # 4  → 128 ch
  - [-1, 1, Conv, [512, 3, 2]]  # 5  → 256 ch
  - [-1, 6, C2f, [512, True]]   # 6  → 256 ch
  - [-1, 1, Conv, [1024, 3, 2]] # 7  → 512 ch
  - [-1, 3, C2f, [1024, True]]  # 8  → 512 ch
  - [-1, 1, CBAM, [512]]        # 9  ← CBAM 通道写「实际值 512」，因为它不参与 yaml 缩放

head:
  - [-1, 1, Classify, [nc]]     # 10
"""


def register_cbam_module():
    """把 core.cbam.CBAM 注入到 ultralytics 的解析空间，让 yaml 能识别 'CBAM' 字符串。"""
    from core.cbam import CBAM
    import ultralytics.nn.tasks as tasks_mod

    if not hasattr(tasks_mod, "CBAM"):
        tasks_mod.CBAM = CBAM


def get_yaml_path():
    """返回 yaml 文件的本地路径；不存在则写入当前模块目录。"""
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), YAML_FILENAME)
    if not os.path.exists(yaml_path):
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(CBAM_CLS_YAML)
    return yaml_path


def build_cbam_cls_model(pretrained_weights=None):
    """构造 YOLOv8s-cls + CBAM 的模型。

    Args:
        pretrained_weights: 可选的 .pt 文件路径，用于初始化权重（迁移学习）。
                            CBAM 层不在原版权重里，会被随机初始化，其余层加载预训练权重。

    Returns:
        ultralytics.YOLO 实例
    """
    register_cbam_module()
    from ultralytics import YOLO

    yaml_path = get_yaml_path()
    model = YOLO(yaml_path, task="classify")

    if pretrained_weights and os.path.exists(pretrained_weights):
        # 用 pretrained 初始化共享层；不匹配的 CBAM 层走默认初始化
        try:
            model.load(pretrained_weights)
            print(f"[cbam] 已从 {pretrained_weights} 加载共享层权重")
        except Exception as e:
            print(f"[cbam] 加载预训练权重失败（保留随机初始化）：{e}")

    return model
