"""DMS v2 两阶段训练脚本（YOLOv8s-cls + CBAM）

阶段 A（pretrain）：全量 dms_v2_cls 训 30 epoch，让 backbone 学到驾驶域通用特征
阶段 B（finetune）：加载 Stage A 的 best.pt，冻结 backbone（freeze=10），仅微调分类头

用法：
    # 准备数据
    python scripts/build_v2_dataset.py

    # 阶段 A
    python scripts/train_dms_v2.py --stage a

    # 阶段 B（自动找 stage A 的 best.pt）
    python scripts/train_dms_v2.py --stage b

    # 一键跑两阶段
    python scripts/train_dms_v2.py --stage all
"""

import argparse
import os
import sys
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "dms_v2_cls")
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs", "classify")
STAGE_A_NAME = "dms_v2_stage_a"
STAGE_B_NAME = "dms_v2_stage_b"
FINAL_NAME = "dms_v2_final"


def get_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def assert_dataset():
    if not os.path.isdir(DATASET_DIR):
        print(f"❌ 数据集不存在：{DATASET_DIR}")
        print("   请先运行：python scripts/build_v2_dataset.py")
        sys.exit(1)
    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "val")
    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        print(f"❌ 数据集结构不完整（缺 train/ 或 val/）：{DATASET_DIR}")
        sys.exit(1)


def train_stage_a(epochs: int, batch: int, imgsz: int, device: str):
    """从头训练 YOLOv8s-cls + CBAM（CBAM 层随机初始化，其余层用官方 yolov8s-cls.pt 预训权重）。"""
    from core.yolo_cbam_arch import build_cbam_cls_model

    print("=" * 60)
    print(f"Stage A: pretrain on dms_v2_cls  |  device={device}  imgsz={imgsz}")
    print("=" * 60)

    # 用官方 yolov8s-cls.pt 做共享层初始化
    pretrained = "yolov8s-cls.pt"
    model = build_cbam_cls_model(pretrained_weights=pretrained)

    model.train(
        data=DATASET_DIR,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=RUNS_ROOT,
        name=STAGE_A_NAME,
        exist_ok=True,
        patience=15,
        # 数据增强（保证泛化到桌面 domain）
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.2,
        scale=0.6,
        fliplr=0.5,
        erasing=0.4,
        mixup=0.2,  # 新增 mixup 提升边界泛化
        # 优化器
        cos_lr=True,
        close_mosaic=10,
    )

    best = os.path.join(RUNS_ROOT, STAGE_A_NAME, "weights", "best.pt")
    if os.path.exists(best):
        print(f"✓ Stage A 完成。best.pt → {best}")
        return best
    print("⚠ Stage A 未产出 best.pt")
    return None


def train_stage_b(stage_a_best: str, epochs: int, batch: int, imgsz: int, device: str):
    """加载 Stage A 权重，冻结 backbone，仅微调分类头。"""
    from core.yolo_cbam_arch import register_cbam_module
    from ultralytics import YOLO

    if not stage_a_best or not os.path.exists(stage_a_best):
        candidate = os.path.join(RUNS_ROOT, STAGE_A_NAME, "weights", "best.pt")
        if os.path.exists(candidate):
            stage_a_best = candidate
        else:
            print(f"❌ 未找到 Stage A 权重：{candidate}")
            sys.exit(1)

    print("=" * 60)
    print(f"Stage B: finetune（freeze backbone）  |  init={stage_a_best}")
    print("=" * 60)

    register_cbam_module()
    model = YOLO(stage_a_best)

    model.train(
        data=DATASET_DIR,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=RUNS_ROOT,
        name=STAGE_B_NAME,
        exist_ok=True,
        patience=10,
        # backbone 共 10 层（含 CBAM），冻结前 10 层只训分类头
        freeze=10,
        # Stage B 用更温和的增广，避免破坏已学到的表征
        hsv_v=0.2,
        degrees=8.0,
        translate=0.1,
        scale=0.4,
        fliplr=0.5,
        erasing=0.2,
        cos_lr=True,
        lr0=0.001,  # 微调用更小学习率
    )

    best = os.path.join(RUNS_ROOT, STAGE_B_NAME, "weights", "best.pt")
    if os.path.exists(best):
        print(f"✓ Stage B 完成。best.pt → {best}")
        return best
    print("⚠ Stage B 未产出 best.pt")
    return None


def publish_final(stage_b_best: str):
    """把 Stage B 的产物发布到 dms_v2_final/，由 BehaviorDetector 自动加载。"""
    if not stage_b_best or not os.path.exists(stage_b_best):
        return
    final_dir = os.path.join(RUNS_ROOT, FINAL_NAME, "weights")
    os.makedirs(final_dir, exist_ok=True)
    dst = os.path.join(final_dir, "best.pt")
    shutil.copy2(stage_b_best, dst)
    print(f"✓ 已发布最终权重 → {dst}")
    print("  系统启动后 BehaviorDetector 会自动加载该路径")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["a", "b", "all"], default="all")
    parser.add_argument("--epochs-a", type=int, default=30)
    parser.add_argument("--epochs-b", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", type=str, default="", help="留空自动选 mps/cuda/cpu")
    args = parser.parse_args()

    assert_dataset()
    device = args.device or get_device()
    print(f"[device] {device}")

    stage_a_best = None
    if args.stage in ("a", "all"):
        stage_a_best = train_stage_a(args.epochs_a, args.batch, args.imgsz, device)

    stage_b_best = None
    if args.stage in ("b", "all"):
        stage_b_best = train_stage_b(stage_a_best, args.epochs_b, args.batch, args.imgsz, device)

    if stage_b_best:
        publish_final(stage_b_best)

    print("\n训练流程结束。可查看：")
    print(f"  - {os.path.join(RUNS_ROOT, STAGE_A_NAME, 'results.png')}")
    print(f"  - {os.path.join(RUNS_ROOT, STAGE_B_NAME, 'results.png')}")
    print(f"  - {os.path.join(RUNS_ROOT, STAGE_B_NAME, 'confusion_matrix.png')}")


if __name__ == "__main__":
    main()
