import cv2
import os
import time

# 定义要补充的三大核心桌面类别的保存库路径
BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "statefarm_cls",
    "train",
)

# 确保文件夹存在 (由于 StateFarm 数据集已被脚本处理过，正常情况下这些文件夹是有的)
classes_map = {
    "n": ("Normal_Driving", "正常驾驶 (双手放好)"),
    "p": ("Talking_on_Phone_Right", "右手接打电话 (假装拿手机)"),
    "d": ("Drinking", "喝水 (拿起水杯靠近嘴部)"),
}

for k, (folder, desc) in classes_map.items():
    os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)


def record_images(cap, class_key, num_images=50):
    folder_name, desc = classes_map[class_key]
    save_dir = os.path.join(BASE_DIR, folder_name)

    # 统计这个文件夹目前拍了多少张 domain_gap 照片，防止覆盖
    existing = len([f for f in os.listdir(save_dir) if f.startswith("domain_gap_")])

    print(f"\n[启动快门] 准备拍摄 50 张【{desc}】...")
    print("👉 请保持该动作并在 3 秒内稍微变换一下头部的角度，让模型学习更多方位。")
    time.sleep(2)  # 留给用户摆好动作的准备时间

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        # Mac 摄像头惯例镜像翻转，使得收集的数据符合第一视角
        frame = cv2.flip(frame, 1)

        # 裁剪成接近车内行车记录仪的正方形视口比例 (避免长方形干扰)
        # MacOS 默认摄像头通常是 1280x720 或 1920x1080，裁掉左右两端
        h, w = frame.shape[:2]
        center = w // 2
        crop_w = h  # 以高度为基准截取正方形 (比如 720x720)
        start_x = center - crop_w // 2

        if start_x >= 0:
            frame_cropped = frame[:, start_x : start_x + crop_w]
        else:
            frame_cropped = frame  # 兜底逻辑

        img_name = f"domain_gap_{existing + count}.jpg"
        save_path = os.path.join(save_dir, img_name)

        # 画面的屏幕反馈
        overlay = frame_cropped.copy()
        cv2.putText(
            overlay,
            f"Recording: {count}/{num_images} [{folder_name}]",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.imshow("Gathering Domain Data", overlay)
        cv2.waitKey(50)  # 每张照片间隔 50 毫秒，大概花 2.5 秒钟拍完

        # 保存无遮挡纯净图片
        cv2.imwrite(save_path, frame_cropped)
        count += 1

    print(
        f"✅ {num_images} 张专属桌面图片已成功混杂进千万张汽车训练图片中 ({save_dir})"
    )


def main():
    print("=" * 50)
    print("🚀 论文神器：【领域自适应数据采集器】启动！")
    print("=" * 50)
    print("操作指南:")
    for k, (folder, desc) in classes_map.items():
        print(f"  👉 按下键盘上字母 '{k.upper()}' : 连拍 50 张【{desc}】")
    print("  👉 按下键盘上字母 'Q' : 退出拍摄并关闭摄像头")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法调用 MacBook 摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        cv2.putText(
            frame,
            "Press N(Normal) P(Phone) D(Drinking) | Q to Quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Gathering Domain Data", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("退出采集器。")
            break
        elif chr(key) in classes_map.keys():
            record_images(cap, chr(key))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
