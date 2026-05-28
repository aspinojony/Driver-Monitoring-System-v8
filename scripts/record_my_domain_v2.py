"""桌面领域数据采集器 v2

升级要点：
- 类别合并为 8 类（Texting / Talking_on_Phone 不再区分左右手）
- 单次按键录制 100 张，约 5 秒
- 每 25 张自动切换头部姿态引导（中→左→右→上→下），增强角度多样性
- 录制 metadata 写入 manifest.csv，按 session_id 分组，便于后续按 session 切 train/val
- 文件命名 {session_id}_{class}_{idx}.jpg，避免覆盖

使用：
    python scripts/record_my_domain_v2.py
    > 输入 session 标签（如 night_hoodie）
    > 按 0-7 录制对应类别，q 退出
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "desk_domain_v2")
MANIFEST_PATH = os.path.join(DATA_ROOT, "manifest.csv")

# 8 类设计（合并 StateFarm 的左右手类）
CLASSES = {
    "0": ("Normal_Driving", "正常驾驶 - 双手放好或轻扶方向盘"),
    "1": ("Texting", "发短信 - 任意手在胸前打字"),
    "2": ("Talking_on_Phone", "打电话 - 任意手举手机贴耳"),
    "3": ("Operating_Radio", "操作中控 - 单手伸出按屏幕"),
    "4": ("Drinking", "喝水 - 手拿水杯贴近嘴唇"),
    "5": ("Reaching_Behind", "向后取物 - 转身朝后排"),
    "6": ("Hair_and_Makeup", "整理仪容 - 手摸头发或脸"),
    "7": ("Talking_to_Passenger", "与乘客交谈 - 头转向右侧"),
}

# 头部姿态引导提示（每 25 张换一个方向）
POSE_GUIDES = [
    ("正面看向摄像头", (0, 255, 0)),
    ("头部稍微左转", (0, 200, 255)),
    ("头部稍微右转", (255, 200, 0)),
    ("头部稍微下俯", (255, 100, 200)),
]


def ensure_dirs():
    os.makedirs(DATA_ROOT, exist_ok=True)
    for _, (folder, _) in CLASSES.items():
        os.makedirs(os.path.join(DATA_ROOT, folder), exist_ok=True)


def append_manifest(rows):
    new_file = not os.path.exists(MANIFEST_PATH)
    with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                ["filename", "class", "session_id", "light", "outfit", "timestamp"]
            )
        writer.writerows(rows)


def square_crop(frame):
    """从中心裁出正方形，模拟车内固定摄像头视野。"""
    h, w = frame.shape[:2]
    side = min(h, w)
    start_x = (w - side) // 2
    start_y = (h - side) // 2
    return frame[start_y : start_y + side, start_x : start_x + side]


def record_class(cap, class_key, session_id, light, outfit, num_images=100):
    folder, desc = CLASSES[class_key]
    save_dir = os.path.join(DATA_ROOT, folder)

    print(f"\n[录制] {desc}（共 {num_images} 张）")
    print("3 秒后开始，请保持动作并跟随屏幕引导调整头部角度")
    for i in range(3, 0, -1):
        print(f"  {i}...", end=" ", flush=True)
        time.sleep(1)
    print("开始")

    rows = []
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        # 统一镜像，与推理时一致（详见 core/config.py 的 MIRROR_CAMERA_FRAME）
        frame = cv2.flip(frame, 1)
        frame = square_crop(frame)

        # 25 张换一个姿态引导
        guide_idx = min(count // (num_images // len(POSE_GUIDES)), len(POSE_GUIDES) - 1)
        guide_text, guide_color = POSE_GUIDES[guide_idx]

        # 屏幕反馈层（不写入图片本体）
        overlay = frame.copy()
        progress = int((count / num_images) * overlay.shape[1])
        cv2.rectangle(overlay, (0, 0), (progress, 6), guide_color, -1)
        cv2.putText(
            overlay,
            f"{folder}  {count}/{num_images}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            guide_color,
            2,
        )
        cv2.putText(
            overlay,
            guide_text,
            (10, overlay.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            guide_color,
            2,
        )
        cv2.imshow("Recorder v2", overlay)
        cv2.waitKey(50)

        # 保存原始帧（无 overlay）
        ts = int(time.time() * 1000)
        filename = f"{session_id}_{folder}_{count:04d}_{ts}.jpg"
        cv2.imwrite(os.path.join(save_dir, filename), frame)
        rows.append(
            [
                filename,
                folder,
                session_id,
                light,
                outfit,
                datetime.now().isoformat(timespec="seconds"),
            ]
        )
        count += 1

    append_manifest(rows)
    print(f"完成。已保存 {count} 张到 {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引")
    parser.add_argument("--per-class", type=int, default=100, help="每次按键录制张数")
    parser.add_argument("--session", type=str, default="", help="本次 session 标签（不填则用时间戳）")
    parser.add_argument("--light", type=str, default="bright", help="光照标签：bright/dim/night")
    parser.add_argument("--outfit", type=str, default="default", help="着装标签")
    args = parser.parse_args()

    ensure_dirs()

    session_id = args.session or datetime.now().strftime("sess%Y%m%d_%H%M%S")

    print("=" * 60)
    print(f"桌面数据采集 v2  |  session={session_id}  light={args.light}  outfit={args.outfit}")
    print("=" * 60)
    for k, (folder, desc) in CLASSES.items():
        print(f"  按 {k} : {folder:<22}  {desc}")
    print("  按 q : 退出")
    print("=" * 60)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"错误：无法打开摄像头 {args.camera}")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = square_crop(frame)

        cv2.putText(
            frame,
            "Press 0-7 to record, q to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"session={session_id}",
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )
        cv2.imshow("Recorder v2", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        ch = chr(key) if 0 <= key < 128 else ""
        if ch in CLASSES:
            record_class(cap, ch, session_id, args.light, args.outfit, args.per_class)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n已退出。manifest 路径：{MANIFEST_PATH}")


if __name__ == "__main__":
    main()
