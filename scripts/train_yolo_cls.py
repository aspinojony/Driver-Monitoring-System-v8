import os
from ultralytics import YOLO
from save_training_results import archive_training_run


def main():
    # Load YOLOv8 Image Classification Model (Nano version for speed)
    print("Loading YOLOv8-cls nano model...")
    model = YOLO("yolov8n-cls.pt")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "statefarm_cls")

    if not os.path.exists(data_dir):
        print(f"Error: Classification dataset not found at {data_dir}")
        print("Please run convert_data_cls.py first.")
        return

    print("Starting YOLOv8 Classification Training for Driver Behavior...")
    train_project = os.path.join(project_root, "data", "weights")
    train_name = "yolov8n_driver_cls"
    # Train the model. It automatically detects train/val folders in data_dir
    results = model.train(
        data=data_dir,
        epochs=30,  # Classification converges much faster than detection
        imgsz=224,  # Standard image size for classification
        device="",  # Auto-detect MPS (Mac) or CPU
        project=train_project,
        name=train_name,
        # ======== 极其重要：高级数据增强参数 (分类任务专用版) ========
        hsv_h=0.015,  # 随机色彩变异 (突破车厢内复杂霓虹灯反光)
        hsv_s=0.7,  # 随机颜色的浓淡饱和度
        hsv_v=0.4,  # !重点 明度巨变 (模拟白天大太阳暴晒 与 夜黑过隧道)
        degrees=15.0,  # !重点 视角旋转15度 (完美模拟不同摄像头的倾斜差异)
        translate=0.2,  # 左右平移20% (司机偏离画面中心也可以稳稳识别)
        scale=0.6,  # !重点 缩放60% (模拟司机身体靠前、往后躺导致的距离差异)
        fliplr=0.5,  # 50% 镜像翻转 (左撇子/右撇子司机都能识别)
        erasing=0.4,  # 40%概率在图片上随机遮挡一块黑斑（完美模拟戴口罩、手遮挡脸）
        # =======================================================
    )

    # You can also use model.val() to evaluate its performance after training
    print(
        "Training Complete! The best weights are saved in: data/weights/yolov8n_driver_cls/weights/best.pt"
    )

    # 📦 自动归档训练结果
    run_dir = os.path.join(train_project, train_name)
    archive_training_run(run_dir, script_name="train_yolo_cls")


if __name__ == "__main__":
    main()
