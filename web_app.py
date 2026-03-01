import os
import cv2
import time
import json
import threading
from flask import Flask, render_template, Response, jsonify, request

from core.engine import MonitoringEngine

app = Flask(__name__)

# 全局共享 AI 引擎
print("🔄 Web App 正在预热 AI 引擎，请稍候...")
engine = MonitoringEngine()
print("✅ Web App: AI 引擎预热完毕！")

current_frame = None
current_stats = {
    "behavior_state": "正在初始化...",
    "fatigue_state": "正在初始化...",
    "ear": 0.0,
    "mar": 0.0,
    "is_warning": False,
    "is_critical": False,
}
camera_stream = None
camera_id = 0
thread_lock = threading.Lock()


def init_camera():
    global camera_stream, camera_id
    if camera_stream is not None:
        camera_stream.release()

    # 智能遍历寻找能用的摄像头
    candidates = [camera_id, 0, 1, 2]
    for i in list(dict.fromkeys(candidates)):  # 去重保持寻找顺序
        cap = cv2.VideoCapture(i)
        # 强制 Mac 底层不要乱丢警告
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                camera_stream = cap
                camera_id = i
                print(f"✅ 成功开启并锁定了摄像头 ID -> {i}")
                with thread_lock:
                    current_stats["behavior_state"] = "摄像头准备就绪"
                    current_stats["fatigue_state"] = "等待进行人脸检测..."
                return
        cap.release()
    print("❌ 灾难性错误: 未能找到任何可用的物理或虚拟摄像头输入源。")


def process_video_loop():
    global current_frame, current_stats, camera_stream
    init_camera()

    while True:
        if camera_stream is None or not camera_stream.isOpened():
            time.sleep(1)
            continue

        ret, frame = camera_stream.read()
        if not ret:
            print("警告: 视频流异常终断。尝试重启...")
            init_camera()
            continue

        # Mac 前置摄像头都是物理镜像的，必须翻转才能对齐训练集
        if isinstance(camera_id, int):
            frame = cv2.flip(frame, 1)

        # 挂载底层最强大的引擎加工节点
        out_frame, results = engine.process_frame(frame, use_clahe=False)

        with thread_lock:
            current_stats = results
            # 推流压缩到 JPEG
            ret, buffer = cv2.imencode(
                ".jpg", out_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            if ret:
                current_frame = buffer.tobytes()

        # 让出一点点 CPU 呼吸时间给 Web Server
        time.sleep(0.01)


# 暴力启动后台多线程处理核心
threading.Thread(target=process_video_loop, daemon=True).start()


@app.route("/")
def index():
    return render_template("index.html")


def generate_mjpeg():
    while True:
        with thread_lock:
            frame = current_frame
        if frame is None:
            time.sleep(0.1)
            continue

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)  # MJPEG推流极限帧率控制


@app.route("/video_feed")
def video_feed():
    # 使用 Multipart 流技术暴力渲染视频帧
    return Response(
        generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/stats")
def get_stats():
    # 给 Web 前端提供 AJAX / Fetch 拉取分析报表
    with thread_lock:
        return jsonify(current_stats)


@app.route("/export")
def export_report():
    import os
    import subprocess

    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    file_name = f"驾驶员会话监控报告_{int(time.time())}.txt"
    save_path = os.path.join(desktop, file_name)
    engine.logger.export_report(save_path)
    # 直接越权呼叫 MacOS 底层去强行用文本编辑器弹窗
    subprocess.Popen(["open", save_path])
    return jsonify({"status": "success", "path": save_path})


@app.route("/change_camera/<int:cam_id>")
def change_camera(cam_id):
    global camera_id
    camera_id = cam_id
    # 重启引擎
    engine.reset()
    init_camera()
    return jsonify({"status": "success", "camera_id": camera_id})


if __name__ == "__main__":
    # 不要用 debug=True, 那会拉起两个进程，直接把模型和显存撑爆！
    app.run(host="0.0.0.0", port=5050, debug=False)
