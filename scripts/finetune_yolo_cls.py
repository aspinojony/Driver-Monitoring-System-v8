import os
from ultralytics import YOLO


def main():
    # 1. 找到你之前已经辛辛苦苦练好的 "老大" 模型
    old_best_model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "weights",
        "yolov8n_driver_cls",
        "weights",
        "best.pt",
    )

    # 2. 找到你准备用来给老大哥 "进修" 的新数据文件夹
    # 它的结构必须和老数据一样：
    # new_night_data/
    #   ├── train/
    #   │   ├── c0/
    #   │   ├── c1/
    #   │   └── ...
    #   └── val/
    new_dataset_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "new_night_data",
    )

    if not os.path.exists(old_best_model_path):
        print(
            "❌ 错误：找不到原来的老模型文件 best.pt，请确认之前的一阶段训练已经结束！"
        )
        return

    if not os.path.exists(new_dataset_dir):
        print(f"❌ 错误：找不到新数据集文件夹：{new_dataset_dir}")
        print(
            "请在 data/ 目录下新建 'new_night_data' 文件夹，并按照 train/c0, train/c1 摆放好您的新照片。"
        )
        return

    print("✅ 找到老模型，准备开始将其送往夜校进修 (Transfer Learning)...")

    # 🌟 核心魔法：这里不再使用官方空白的 yolov8n-cls.pt，而是直接读取我们自己懂规矩的老模型！
    model = YOLO(old_best_model_path)

    # 3. 开始微调（Fine-Tuning）
    # 设置存入的文件夹名字为 finetuned_driver_cls
    results = model.train(
        data=new_dataset_dir,  # 指向新的文件夹
        epochs=10,  # 因为老大哥已经很聪明了，这里只需要稍微用 10-20 轮学新视角的特征即可
        imgsz=64,  # 依然保持和原来一样的图片大小要求
        batch=32,  # 苹果 M 芯片无压力
        device="mps",  # 开启苹果芯片专门的高速 GPU 加速
        project="data/weights",  # 新练出来的加强版模型保存路径
        name="finetuned_yolov8_cls",  # 新文件夹名字
        val=True,  # 训练完验证一下看看加强效果
        # ======== 极其重要：高级数据增强进修版 ========
        hsv_v=0.4,  # !重点 明度巨变 (特别是进修夜间数据，强迫模型忽略低光环境)
        degrees=15.0,  # 视角旋转15度 (完美模拟不同摄像头的倾斜差异)
        translate=0.2,  # 左右平移20% (司机偏离画面中心也可以稳稳识别)
        scale=0.6,  # 缩放60% (模拟司机身体靠前、往后躺导致的距离差异)
        fliplr=0.5,  # 50% 镜像翻转
        erasing=0.4,  # 40%概率在图片上随机遮挡一块黑斑（完美模拟戴口罩、手遮挡脸）
        # =======================================================
    )

    print("\n🎉 进修完成！更强大的新版模型已出炉！")
    print(
        "您可以去修改 core/behavior_detect.py 里的 model_path，让系统享受这个拥有夜晚/正面超能力的新模型了！"
    )


if __name__ == "__main__":
    main()
