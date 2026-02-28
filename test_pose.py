import cv2
import time
from core.pose_detect import PoseConstrainedPhoneDetector


def main():
    print("🚀 正在加载 YOLOv8-pose 模型，如果是第一次运行将自动下载预训练小模型...")
    detector = PoseConstrainedPhoneDetector(
        distance_threshold=80.0, strict_frames_required=15
    )

    # 打开 Mac 默认的摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ 无法打开摄像头！")
        return

    print("✅ 摄像头已打开，请尝试：1. 正常挥手 2. 将手贴紧耳朵停留半秒 观察系统输出。")
    print("按 'q' 键退出测试程序。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 翻转画面(镜像)更符合日常直觉
        frame = cv2.flip(frame, 1)

        # 传入算法核心
        start_time = time.time()
        annotated_frame, is_danger, debug_log = detector.verify_phone_usage(frame)
        fps = int(1.0 / (time.time() - start_time))

        # 绘制诊断日志在画面上
        color = (0, 0, 255) if is_danger else (0, 255, 0)  # 危险标红，安全标绿

        cv2.putText(
            annotated_frame,
            f"FPS: {fps}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            annotated_frame,
            f"Status: {'DANGER (Phoning)' if is_danger else 'SAFE'}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        # 规避含有中文无法显示，英文输出 Log 的大致意思
        # CV2 默认不支持全角中文写入所以只写在控制台。

        # 终端输出核心判定逻辑日志（极其震撼的数值监控）
        print(f"[{'🚨 高危' if is_danger else '🟢 正常'}] {debug_log}")

        cv2.imshow("Advanced Mathematical Pose Constraints", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
