"""DMS Web Dashboard 后端（Flask + SocketIO）

特性：
- WebSocket 推送 frame + 推理结果（替代 MJPEG + 200ms AJAX 轮询）
- 推理循环常驻；新客户端连入立即开始接收最新帧
- 提供历史指标查询接口（最近 60 秒）
- 一键导出会话报告
"""

import base64
import collections
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

import cv2
from flask import Flask, jsonify, render_template, send_from_directory

try:
    from flask_socketio import SocketIO
except ImportError:
    print(
        "缺少依赖 flask-socketio，请运行：pip install flask-socketio simple-websocket",
        file=sys.stderr,
    )
    sys.exit(1)

from core.engine import MonitoringEngine
from core.config import MIRROR_CAMERA_FRAME

# --------------------------------------------------------------------------
# 全局状态
# --------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = "dms-dashboard-2026"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

print("[web] 正在加载 AI 引擎...")
engine = MonitoringEngine(enable_pose=False)  # Pose 默认关闭，UI 可在线打开
print("[web] AI 引擎就绪")


def _get_model_tag():
    """返回当前权重的简洁标识，用于前端展示。"""
    try:
        from core.config import BEHAVIOR_MODEL_PATH_V2, BEHAVIOR_MODEL_PATH

        if os.path.exists(BEHAVIOR_MODEL_PATH_V2):
            return "DMS v2 (s+CBAM)"
        if os.path.exists(BEHAVIOR_MODEL_PATH):
            return "domain_adapted v1"
        return "yolov8n-cls (fallback)"
    except Exception:
        return "unknown"


MODEL_TAG = _get_model_tag()
print(f"[web] 模型标签: {MODEL_TAG}")

camera_stream = None
camera_id = 0
camera_lock = threading.Lock()

# 历史窗口（用于前端折线图初始化）
HISTORY_SIZE = 60 * 5  # 假设 5fps 推送给前端，约 60 秒
metrics_history = collections.deque(maxlen=HISTORY_SIZE)
metrics_lock = threading.Lock()

# FPS 统计
_fps_window = collections.deque(maxlen=30)


def init_camera(prefer_id: int = None):
    global camera_stream, camera_id
    with camera_lock:
        if camera_stream is not None:
            try:
                camera_stream.release()
            except Exception:
                pass
            camera_stream = None

        candidates = []
        if prefer_id is not None:
            candidates.append(prefer_id)
        candidates.extend([camera_id, 0, 1, 2])
        seen = set()
        for cid in candidates:
            if cid in seen:
                continue
            seen.add(cid)
            cap = cv2.VideoCapture(cid)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    camera_stream = cap
                    camera_id = cid
                    print(f"[camera] 已锁定 ID={cid}")
                    return True
            cap.release()
        print("[camera] 未找到可用摄像头")
        return False


def encode_jpeg_b64(frame, quality=70):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def inference_loop():
    """常驻推理线程：抓帧→推理→广播。"""
    init_camera()
    last_emit_t = 0.0
    EMIT_INTERVAL = 1 / 25  # 限制广播 25fps，避免冲爆 socket buffer

    while True:
        with camera_lock:
            cap = camera_stream
        if cap is None or not cap.isOpened():
            time.sleep(0.5)
            init_camera()
            continue

        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[camera] 视频流中断，正在重启")
            init_camera()
            continue

        if isinstance(camera_id, int) and MIRROR_CAMERA_FRAME:
            frame = cv2.flip(frame, 1)

        out_frame, results = engine.process_frame(frame, use_clahe=False)

        # FPS
        dt = time.time() - t0
        _fps_window.append(dt)
        avg_dt = sum(_fps_window) / len(_fps_window) if _fps_window else 0.0
        fps = 1.0 / avg_dt if avg_dt > 0 else 0.0

        # 拼装载荷
        payload = {
            "ts": time.time() * 1000,
            "fps": round(fps, 1),
            "model_tag": MODEL_TAG,
            "behavior": results.get("behavior_state", "—"),
            "fatigue": results.get("fatigue_state", "—"),
            "ear": round(float(results.get("ear", 0.0)), 3),
            "mar": round(float(results.get("mar", 0.0)), 3),
            "risk_level": results.get("risk_level", "safe"),
            "fused_confidence": round(float(results.get("fused_confidence", 0.0)), 3),
            "yolo_raw_confidence": round(float(results.get("yolo_raw_confidence", 0.0)), 3),
            "fusion_notes": results.get("fusion_notes", []),
            "is_warning": bool(results.get("is_warning", False)),
            "is_critical": bool(results.get("is_critical", False)),
        }

        # 8 类置信度（如有）
        probs = results.get("class_probs")
        names = results.get("class_names")
        if probs is not None and names is not None:
            payload["class_probs"] = [
                {"name": names[i], "prob": round(float(probs[i]), 3)}
                for i in range(len(probs))
            ]

        # 写历史
        with metrics_lock:
            metrics_history.append(
                {
                    "ts": payload["ts"],
                    "ear": payload["ear"],
                    "mar": payload["mar"],
                    "risk_level": payload["risk_level"],
                }
            )

        # 控制广播频率
        now = time.time()
        if now - last_emit_t >= EMIT_INTERVAL:
            payload["frame"] = encode_jpeg_b64(out_frame, quality=72)
            socketio.emit("frame", payload)
            last_emit_t = now


# 启动后台推理线程
threading.Thread(target=inference_loop, daemon=True).start()


# --------------------------------------------------------------------------
# 路由
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/static/<path:fname>")
def static_files(fname):
    return send_from_directory(app.static_folder, fname)


@app.route("/api/history")
def api_history():
    with metrics_lock:
        return jsonify(list(metrics_history))


@app.route("/api/export")
def api_export():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    file_name = f"DMS_Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    save_path = os.path.join(desktop, file_name)
    engine.logger.export_report(save_path)
    try:
        subprocess.Popen(["open", save_path])
    except Exception:
        pass
    return jsonify({"status": "ok", "path": save_path})


@app.route("/api/camera/<int:cam_id>")
def api_change_camera(cam_id):
    engine.reset()
    success = init_camera(prefer_id=cam_id)
    return jsonify({"status": "ok" if success else "failed", "camera_id": camera_id})


@app.route("/api/pose/<int:enabled>")
def api_pose(enabled):
    engine.enable_pose = bool(enabled)
    if engine.enable_pose and engine.pose_validator is None:
        engine._init_pose()
    return jsonify({"status": "ok", "pose_enabled": engine.enable_pose})


@app.route("/api/enhance", methods=["GET"])
def api_enhance_get():
    """读取当前推理 4 件套设置。"""
    bd = engine.behavior_detector
    return jsonify(
        {
            "use_tta": bool(bd.use_tta),
            "temperature": float(bd.temperature),
            "normal_prior_boost": float(bd.normal_prior_boost),
            "vote_window": int(bd.vote_window_size),
            "confidence_threshold": float(bd.confidence_threshold),
        }
    )


@app.route("/api/enhance", methods=["POST"])
def api_enhance_set():
    """运行时调整推理 4 件套。POST JSON 任意子集即可。"""
    from flask import request

    bd = engine.behavior_detector
    payload = request.get_json(silent=True) or {}
    if "use_tta" in payload:
        bd.use_tta = bool(payload["use_tta"])
    if "temperature" in payload:
        bd.temperature = max(0.5, min(3.0, float(payload["temperature"])))
    if "normal_prior_boost" in payload:
        bd.normal_prior_boost = max(0.5, min(2.0, float(payload["normal_prior_boost"])))
    if "vote_window" in payload:
        n = max(1, min(15, int(payload["vote_window"])))
        bd.vote_window_size = n
        from collections import deque
        # 重建队列
        bd._topk_history = deque(list(bd._topk_history)[-n:], maxlen=n)
    if "confidence_threshold" in payload:
        bd.confidence_threshold = max(0.1, min(0.95, float(payload["confidence_threshold"])))
    return api_enhance_get()


@socketio.on("connect")
def on_connect():
    print("[ws] 客户端已连接")


@socketio.on("disconnect")
def on_disconnect():
    print("[ws] 客户端已断开")


if __name__ == "__main__":
    print("[web] http://127.0.0.1:5050")
    socketio.run(app, host="0.0.0.0", port=5050, debug=False, allow_unsafe_werkzeug=True)
