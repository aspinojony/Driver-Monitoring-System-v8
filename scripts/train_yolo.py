import os
from ultralytics import YOLO


def main():
    # Load YOLOv8 model (Nano version for lightweight inference)
    # Note: To officially use CBAM inside Ultralytics, you need to add the CBAM
    # class to ultralytics/nn/modules.py and create a custom yolov8-cbam.yaml
    model = YOLO("yolov8n.pt")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_yaml_path = os.path.join(project_root, "data", "statefarm_dataset.yaml")

    if not os.path.exists(data_yaml_path):
        print(f"Warning: {data_yaml_path} does not exist.")
        print("Please configure your dataset YAML explicitly before training.")
        return

    print("Starting YOLOv8 training for Driver Anomaly Detection...")
    results = model.train(
        data=data_yaml_path,
        epochs=100,
        imgsz=640,
        device="",  # set to 'mps' for Apple Silicon, or '0' for CUDA
        project=os.path.join(project_root, "data", "weights"),
        name="yolov8n_driver_behavior",
        # ======== 极其重要：在此处实现『高级数据增强参数』 ========
        hsv_h=0.015,  # 随机色彩变异 (突破车厢内复杂霓虹灯、路灯反光)
        hsv_s=0.7,  # 随机颜色的浓淡饱和度
        hsv_v=0.4,  # !重点 明度巨变 (模拟白天大太阳暴晒 与 纯黑夜过隧道)
        degrees=15.0,  # !重点 视角旋转15度 (完美模拟大卡车/越野车/小轿车摄像头的倾斜差异)
        translate=0.2,  # 左右平移20% (司机偏离画面中心也可以稳稳识别)
        scale=0.6,  # !重点 缩放60% (模拟司机身体靠前、往后躺导致的距离差异)
        shear=2.0,  # 画面拉伸畸变 (广角镜头的边缘防畸变训练)
        fliplr=0.5,  # 50% 镜像翻转 (左撇子/右撇子司机都能识别)
        mosaic=1.0,  # !重点 100%强制开启将4张图切碎拼一起，彻底打碎车内背景，避免死记硬背！
        erasing=0.4,  # 40%概率在图片上随机构造一块“黑斑”（完美模拟人员戴口罩、手挡住了一半脸）
        # =======================================================
    )


if __name__ == "__main__":
    main()
