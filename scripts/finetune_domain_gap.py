import os
import torch
from ultralytics import YOLO


def main():
    print("=" * 60)
    print("🎓 毕业设计论文大杀器：领域自适应小样本微调引擎 🎓")
    print("=" * 60)
    print("正在搜寻您昨天花了4个小时炼成的 99.7% 满级权重模型...")

    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    best_weights_path = os.path.join(
        project_root, "data", "weights", "yolov8n_driver_cls2", "weights", "best.pt"
    )

    if not os.path.exists(best_weights_path):
        print(f"❌ 找不到满级权重，请检查路径: {best_weights_path}")
        return

    print(f"✅ 满级权重装载完毕！准备将您刚拍的 200 张照片强行印入模型脑内...")

    # 加载已有的牛逼预训练权重
    model = YOLO(best_weights_path)

    # 制定极短微调计划 (只需 10 轮即可完成领域自制)
    epochs_to_run = 10
    batch_size = 32
    img_size = 224

    # 自动识别 Mac Apple Silicon mps 加速引擎
    device_type = "mps" if torch.backends.mps.is_available() else "cpu"
    print(device_type)

    print("\n[开始炼丹]...预计 3 分钟内完成您电脑书桌专属车厢的构建！")

    dataset_yaml = os.path.join(project_root, "data", "statefarm_cls")

    # 开始微调训练
    # 冻结部分底层特征层（让它别把认出车子的经验给忘了），只训练顶部分类器
    # freeze=10 意味着锁死前 10 层的金刚坏体
    results = model.train(
        data=dataset_yaml,
        epochs=epochs_to_run,
        imgsz=img_size,
        batch=batch_size,
        device=device_type,
        name="domain_adapted_cls_final",
        freeze=10,
        patience=5,  # 如果没有长进，5个周期就停止
        exist_ok=True,  # 允许覆盖
    )

    print("\n🎉 微调彻底杀青！")
    print(
        f"您的专属跨界桌面无敌版权重已经保存在: {os.path.join(project_root, 'runs', 'classify', 'domain_adapted_cls_final', 'weights', 'best.pt')}"
    )
    print(
        "系统下一步将自动把它替换到主程序 (main.py) 里面，届时您再开摄像头测试必将惊为天人！"
    )


if __name__ == "__main__":
    main()
