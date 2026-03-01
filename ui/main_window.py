import cv2
import numpy as np
import time
import subprocess
import threading
import queue
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QCheckBox,
    QSlider,
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QUrl
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtMultimedia import QSoundEffect

from core.engine import MonitoringEngine
from core.config import ALARM_FATIGUE_SOUND, ALARM_DISTRACT_SOUND


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_stats_signal = pyqtSignal(float, float, str, str)
    log_signal = pyqtSignal(str)

    def __init__(self, engine, source=0, use_clahe=False):
        super().__init__()
        self._run_flag = True
        self.source = source
        self.use_clahe = use_clahe
        self.engine = engine
        self.engine.reset()
        self.last_beep_time = 0  # Cooldown for audio alarm

        # Throttles
        self.last_yolo_time = 0
        self.last_yolo_behavior = "正常"
        self.last_log_time = 0
        self.last_tts_time = 0

        # 从全局 config 加载跨平台音频文件位置
        self.alarm_sound = QSoundEffect()
        self.alarm_sound.setSource(QUrl.fromLocalFile(ALARM_DISTRACT_SOUND))
        self.alarm_sound.setVolume(1.0)

        self.critical_alarm_sound = QSoundEffect()
        self.critical_alarm_sound.setSource(QUrl.fromLocalFile(ALARM_FATIGUE_SOUND))
        self.critical_alarm_sound.setVolume(1.0)

        # 独立的帧队列分离视频采集和模型推理
        self.frame_queue = queue.Queue(maxsize=1)
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.source)
        # 尝试设置适当的缓冲大小以减少延迟
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.log_signal.emit(f"Opened video source: {self.source}")

        # 获取视频原生帧率以控制播放速度 (如果是一段视频文件)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 100:
            fps = 30  # 默认30帧
        frame_delay = 1.0 / fps

        is_video_file = isinstance(self.source, str) and not self.source.isdigit()

        while self._run_flag:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                self.log_signal.emit("End of video stream.")
                break

            # 如果队列满了，丢弃最旧的帧以保证低延迟(实时性)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

            # 对于视频文件，硬性限制读取速度，防止视频1秒钟像快进一样闪播完
            if is_video_file:
                elapsed = time.time() - start_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)

        cap.release()
        # 通知推理循环结束
        try:
            self.frame_queue.put(None)
        except Exception:
            pass

    def run(self):
        # 启动视频采集子线程
        self.capture_thread.start()

        while self._run_flag:
            try:
                # 使用带有 timeout 的 get 以便能够响应 _run_flag 的关闭信号
                cv_img = self.frame_queue.get(timeout=0.1)

                # 收到 None 表示视频流结束
                if cv_img is None:
                    break

                # 对摄像头图像进行镜像反转，满足 Mac 用户习惯并与之前的录制数据几何对齐
                if str(self.source).isdigit() or self.source == 0:
                    cv_img = cv2.flip(cv_img, 1)

                # 核心升级：一键调用 AI 引擎，屏蔽底层所有算法的脏活累活
                cv_img, results = self.engine.process_frame(cv_img, self.use_clahe)

                behavior = results["behavior_state"]
                fatigue_level = results["fatigue_state"]
                ear, mar = results["ear"], results["mar"]
                is_warning = results["is_warning"]
                is_critical = results["is_critical"]

                current_time = time.time()

                if is_warning:

                    if is_critical:
                        # 极度危险：取消 3 秒冷却，改为 0.5 秒夺命连环 Call，并播放刺耳音效
                        if current_time - self.last_beep_time > 0.5:
                            self.last_beep_time = current_time
                            self.critical_alarm_sound.play()

                        # 极度疲劳专属语音
                        if current_time - self.last_tts_time > 10.0:
                            self.last_tts_time = current_time
                            subprocess.Popen(
                                [
                                    "say",
                                    "-r",
                                    "180",
                                    "警告！警告！检测到极度疲劳，请立即停车休息！",
                                ]
                            )

                    else:
                        # 普通警告：保留 3 秒防打扰冷却期
                        if current_time - self.last_beep_time > 3.0:
                            self.last_beep_time = current_time
                            self.alarm_sound.play()

                        # 普通TTS语音（针对不同行为进行个性化播报）
                        if current_time - self.last_tts_time > 15.0:
                            self.last_tts_time = current_time
                            if "正常" not in behavior:
                                subprocess.Popen(
                                    [
                                        "say",
                                        f"请注意，检测到{behavior}的危险行为，请专心驾驶。",
                                    ]
                                )
                            elif "正常" not in fatigue_level:
                                subprocess.Popen(
                                    [
                                        "say",
                                        f"请注意，检测到{fatigue_level}状态，请提高警惕。",
                                    ]
                                )

                # Signal Emit
                self.change_pixmap_signal.emit(cv_img)
                self.update_stats_signal.emit(ear, mar, fatigue_level, behavior)

                if is_warning:
                    # 日志防刷屏冷却 (每2秒最多弹出一条同样的日志)
                    if current_time - self.last_log_time > 2.0:
                        self.last_log_time = current_time
                        self.log_signal.emit(
                            f"⚠️ 警报触发 | 行为: {behavior} | 疲劳: {fatigue_level}"
                        )

            except queue.Empty:
                # 队列为空则继续循环等待
                continue
            except Exception as e:
                import traceback

                error_details = traceback.format_exc()
                print(f"--- Inference Loop Exception ---\n{error_details}")
                self.log_signal.emit(
                    f"❌ 运行异常已拦截: {e}\n(系统不会崩溃，丢弃损坏帧并尝试继续)"
                )

    def stop(self):
        self._run_flag = False
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("驾驶员监控系统 (Driver Monitoring System)")
        self.resize(1024, 768)
        self.thread = None

        # 🟢 在全局启动时单例加载引擎并常驻内存（彻底消灭卡顿）
        print("正在全局初始化 AI 核心引擎，请稍候...")
        self.global_engine = MonitoringEngine()
        print("引擎及其背后相关模型加载完毕！")

        self._init_ui()

    def _init_ui(self):
        # Configure global styling for Dark Theme
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #1a1a1a;
            }
            QLabel {
                color: #e0e0e0;
                font-family: 'Segoe UI', 'SF Pro', 'Helvetica Neue', Arial, sans-serif;
            }
            QGroupBox {
                border: 1px solid #444444;
                border-radius: 2px;
                margin-top: 1.5ex;
                background-color: #242424;
                color: #ffffff;
                font-size: 13px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #0078d7;
                color: #ffffff;
                border-radius: 2px;
                padding: 8px;
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #005a9e;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QComboBox {
                background-color: #333333;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 2px;
                padding: 4px;
            }
            QTextEdit {
                background-color: #000000;
                color: #00ff00;
                border: 1px solid #444444;
                border-radius: 2px;
                padding: 5px;
                font-family: Consolas, monospace;
            }
        """
        )

        central_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Left Panel: Video Feed Group
        video_group = QGroupBox("实时监控视频源")
        video_layout = QVBoxLayout()
        self.image_label = QLabel(self)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet(
            "background-color: #0d0d0d; border: 1px solid #333333; border-radius: 2px;"
        )
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("正在等待视频输入...")
        self.image_label.setFont(QFont("Arial", 16))
        video_layout.addWidget(self.image_label)
        video_group.setLayout(video_layout)
        main_layout.addWidget(video_group, stretch=5)

        # Right Panel: Controls and Stats
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)

        # Control Group
        control_group = QGroupBox("系统控制设置")
        control_layout = QVBoxLayout()
        control_layout.setSpacing(10)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["摄像头 (0)", "本地视频 / 图片", "RTSP 网络流"])
        control_layout.addWidget(QLabel("选择数据源:"))
        control_layout.addWidget(self.source_combo)

        # CLAHE Checkbox
        self.chk_clahe = QCheckBox("启用 CLAHE 直方图环境光自适应计算")
        self.chk_clahe.setStyleSheet("color: #cdd6f4; font-weight: bold;")
        self.chk_clahe.setChecked(False)
        self.chk_clahe.toggled.connect(self.toggle_clahe)
        control_layout.addWidget(self.chk_clahe)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("开始检测 ▶")
        self.btn_start.clicked.connect(self.start_detection)
        self.btn_stop = QPushButton("停止检测 ⏹")
        self.btn_stop.setStyleSheet(
            "background-color: #d13438; border: 1px solid #a80000; color: white;"
        )
        self.btn_stop.clicked.connect(self.stop_detection)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        control_layout.addLayout(btn_layout)
        control_group.setLayout(control_layout)
        right_panel.addWidget(control_group)

        # Parameters Control Group
        param_group = QGroupBox("算法参数控制面板")
        param_layout = QVBoxLayout()
        param_layout.setSpacing(8)

        # 1. auto tune fatigue
        self.chk_auto_tune = QCheckBox("启用 EAR/MAR 动态自适应阈值回归")
        self.chk_auto_tune.setStyleSheet("color: #cdd6f4; font-weight: bold;")
        self.chk_auto_tune.setChecked(True)
        self.chk_auto_tune.toggled.connect(self.toggle_auto_tune)
        param_layout.addWidget(self.chk_auto_tune)

        # 2. Behavior Confidence Threshold
        self.lbl_conf = QLabel("算法推理置信度 (Confidence): 0.45")
        self.slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_conf.setRange(10, 90)
        self.slider_conf.setValue(45)
        self.slider_conf.valueChanged.connect(self.update_conf)
        param_layout.addWidget(self.lbl_conf)
        param_layout.addWidget(self.slider_conf)

        # 3. Behavior Smoothing
        self.lbl_smooth = QLabel("特征信号滑动窗口延迟 (Smooth): 15帧 (0.5秒)")
        self.slider_smooth = QSlider(Qt.Orientation.Horizontal)
        self.slider_smooth.setRange(5, 90)
        self.slider_smooth.setValue(15)
        self.slider_smooth.valueChanged.connect(self.update_smooth)
        param_layout.addWidget(self.lbl_smooth)
        param_layout.addWidget(self.slider_smooth)

        param_group.setLayout(param_layout)
        right_panel.addWidget(param_group)

        # Stats Display Group (Premium Grid Layout Redesign)
        stats_group = QGroupBox("驾驶状态追踪与分析视图")
        stats_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")

        from PyQt6.QtWidgets import QGridLayout

        stats_layout = QGridLayout()
        stats_layout.setSpacing(10)

        # Style template for premium cards
        card_style_normal = """
            background-color: #2b2b2b;
            border-radius: 2px;
            border: 1px solid #444444;
            padding: 10px;
            font-size: 13px;
            color: #d0d0d0;
        """

        self.lbl_ear = QLabel("结构特征 EAR<br><br>--")
        self.lbl_mar = QLabel("结构特征 MAR<br><br>--")
        self.lbl_fatigue = QLabel("视觉疲劳指数<br><br>正常")
        self.lbl_behavior = QLabel("驾驶行为预测<br><br>正常")

        # Setup alignment and initial style
        for lbl in [self.lbl_ear, self.lbl_mar, self.lbl_fatigue, self.lbl_behavior]:
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(card_style_normal)
            lbl.setMinimumHeight(100)  # Force a stable height so UI never deforms

        # Add to Grid (2x2)
        stats_layout.addWidget(self.lbl_ear, 0, 0)
        stats_layout.addWidget(self.lbl_mar, 0, 1)
        stats_layout.addWidget(self.lbl_fatigue, 1, 0)
        stats_layout.addWidget(self.lbl_behavior, 1, 1)

        stats_group.setLayout(stats_layout)
        right_panel.addWidget(stats_group)

        # Log Output Group
        log_group = QGroupBox("后台事件日志采集")
        log_layout = QVBoxLayout()
        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.append("-> 系统初始化完成.")

        self.btn_export = QPushButton("📊 导出本次全息驾车报告 (EXCEL/PDF记录器)")
        self.btn_export.clicked.connect(self.export_session_report)
        self.btn_export.setStyleSheet(
            "background-color: #107c10; border: 1px solid #0b5a0b; color: white;"
        )  # Industrial Green highlight

        log_layout.addWidget(self.log_window)
        log_layout.addWidget(self.btn_export)
        log_group.setLayout(log_layout)

        right_panel.addWidget(log_group, stretch=1)

        main_layout.addLayout(right_panel, stretch=2)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def start_detection(self):
        source_idx = self.source_combo.currentIndex()
        source = 0
        if source_idx == 1:
            file_name, _ = QFileDialog.getOpenFileName(
                self, "Open Video File", "", "Video Files (*.mp4 *.avi *.jpg *.png)"
            )
            if not file_name:
                return
            source = file_name
        elif source_idx == 2:
            # For demonstration, prompt RTSP url could be added here
            source = "rtsp://example_stream"

        self.thread = VideoThread(
            engine=self.global_engine,
            source=source,
            use_clahe=self.chk_clahe.isChecked(),
        )
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_stats_signal.connect(self.update_stats)
        self.thread.log_signal.connect(self.append_log)
        self.thread.start()

    def export_session_report(self):
        """调用全局 AI 引擎中的记录仪，一键生成报表到桌面"""
        if not hasattr(self, "global_engine"):
            return

        import os, time, subprocess

        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        file_name = f"驾驶员会话监控报告_{int(time.time())}.txt"
        save_path = os.path.join(desktop, file_name)

        # 引擎内嵌的记录仪提供计算与组装逻辑
        self.global_engine.logger.export_report(save_path)

        self.append_log(f"✅ 完美！系统已出具临床级体检报告！\n📁 存放于: {save_path}")

        # 专为 MacOS 打造的物理召唤神迹：写完直接弹窗打开
        subprocess.Popen(["open", save_path])

    def toggle_clahe(self, checked):
        if self.thread:
            self.thread.use_clahe = checked
            state_msg = "开启" if checked else "关闭"
            self.log_window.append(f"-> 🌙 已{state_msg}环境自适应夜视模式")

    def stop_detection(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
            self.image_label.clear()
            self.image_label.setText("监控已停止。")
            self.image_label.setStyleSheet(
                "background-color: #0d0d0d; border: 1px solid #333333; border-radius: 2px;"
            )

    def update_image(self, cv_img):
        # 拦截残留信号：如果你已经点击了停止（self.thread变成了None），就扔掉残影
        if self.thread is None:
            return
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def toggle_auto_tune(self, checked):
        self.global_fatigue_detector.auto_tune_thresholds = checked
        if checked:
            self.append_log("系统已开启自适应疲劳计算 (千人千面动态阈值生效中)")
        else:
            self.append_log("已固定疲劳阈值 (手动安全底线 EAR:0.25, MAR:0.50)")
            # Reset
            self.global_fatigue_detector.dynamic_ear_threshold = 0.25
            self.global_fatigue_detector.dynamic_mar_threshold = 0.50

    def update_conf(self, val):
        conf = val / 100.0
        self.global_behavior_detector.confidence_threshold = conf
        self.lbl_conf.setText(f"行为判定置信度/灵敏度: {conf:.2f}")

    def update_smooth(self, val):
        self.global_behavior_detector.smoothing_window = val
        if len(self.global_behavior_detector.behavior_history) > val:
            self.global_behavior_detector.behavior_history = (
                self.global_behavior_detector.behavior_history[-val:]
            )
        sec = val / 30.0
        self.lbl_smooth.setText(f"信号防抖过滤延迟: {val}帧 ({sec:.1f}秒)")

    def update_stats(self, ear, mar, fatigue, behavior):
        # 拦截残留信号
        if self.thread is None:
            return

        # Premium Card Text Update
        self.lbl_ear.setText(
            f"结构特征 EAR<br><br><span style='font-size: 26px; color: #ffffff;'>{ear:.2f}</span>"
        )
        self.lbl_mar.setText(
            f"结构特征 MAR<br><br><span style='font-size: 26px; color: #ffffff;'>{mar:.2f}</span>"
        )

        # Color palettes for dynamic states (Industrial)
        COLOR_SAFE = "#107c10"  # Microsoft/Industrial Green
        COLOR_WARN = "#ffb900"  # Amber/Yellow
        COLOR_CRIT = "#d13438"  # Danger Red

        # Determine Fatigue Style
        if "正常" in fatigue:
            f_color = COLOR_SAFE
            b_color = "#444444"
            bg_color = "#2b2b2b"
        elif "极度" in fatigue:
            f_color = "#ffffff"
            b_color = COLOR_CRIT
            bg_color = COLOR_CRIT
        else:
            f_color = COLOR_WARN
            b_color = COLOR_WARN
            bg_color = "#2b2b2b"

        self.lbl_fatigue.setText(
            f"视觉疲劳指数<br><br><span style='font-size: 24px; color: {f_color}; font-weight: 900;'>{fatigue}</span>"
        )
        self.lbl_fatigue.setStyleSheet(
            f"background-color: {bg_color}; border-radius: 2px; border: 1px solid {b_color}; padding: 10px; font-size: 13px;"
        )

        # Determine Behavior Style
        if "正常" in behavior:
            b_text_col = COLOR_SAFE
            b_border_col = "#444444"
            b_bg_col = "#2b2b2b"
        else:
            b_text_col = "#ffffff"
            b_border_col = COLOR_CRIT
            b_bg_col = COLOR_CRIT

        self.lbl_behavior.setText(
            f"驾驶行为预测<br><br><span style='font-size: 24px; color: {b_text_col}; font-weight: 900;'>{behavior}</span>"
        )
        self.lbl_behavior.setStyleSheet(
            f"background-color: {b_bg_col}; border-radius: 2px; border: 1px solid {b_border_col}; padding: 10px; font-size: 13px;"
        )

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        p = convert_to_Qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def append_log(self, text):
        self.log_window.append(text)

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()
