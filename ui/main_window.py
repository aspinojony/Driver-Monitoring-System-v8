"""DMS 桌面端主窗口 (v2)

特斯拉驾舱风设计：
- 深色背景 + 霓虹蓝主色 + 状态色（safe/warn/critical）
- pyqtgraph 实时 EAR/MAR 折线图
- 自定义环形风险仪表（QPainter）
- 8 类行为置信度条形列
- 事件时间轴

修复要点（相对 v1）：
- 修复 self.global_fatigue_detector / self.global_behavior_detector 不存在的崩溃
- 镜像统一走 core.config.MIRROR_CAMERA_FRAME
- closeEvent 安全释放线程
- 摄像头索引解析与 select 实际编号匹配
"""

import collections
import math
import subprocess
import time
from datetime import datetime

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    import pyqtgraph as pg

    HAS_PG = True
except ImportError:
    HAS_PG = False

from core.config import ALARM_DISTRACT_SOUND, ALARM_FATIGUE_SOUND, MIRROR_CAMERA_FRAME
from core.engine import MonitoringEngine

# ==========================================================================
# 配色（与 Web 端保持一致）
# ==========================================================================
COLOR_BG_0 = "#050810"
COLOR_BG_1 = "#0a0e1a"
COLOR_BG_2 = "#121829"
COLOR_LINE = "#1f2942"
COLOR_TEXT_1 = "#e6ebff"
COLOR_TEXT_2 = "#97a3c4"
COLOR_TEXT_3 = "#5b6789"
COLOR_ACCENT = "#00d4ff"
COLOR_SAFE = "#34c759"
COLOR_WARN = "#ffb300"
COLOR_CRIT = "#ff3b30"


GLOBAL_QSS = f"""
QMainWindow, QWidget {{
    background-color: {COLOR_BG_0};
    color: {COLOR_TEXT_1};
    font-family: "SF Pro Display", "Inter", "PingFang SC", "Microsoft YaHei", sans-serif;
}}
QFrame#card {{
    background-color: {COLOR_BG_1};
    border: 1px solid {COLOR_LINE};
    border-radius: 10px;
}}
QLabel#cardTitle {{
    color: {COLOR_TEXT_2};
    font-size: 11px;
    letter-spacing: 1.5px;
    font-weight: 600;
    text-transform: uppercase;
}}
QLabel#cardValue {{
    color: {COLOR_TEXT_1};
    font-size: 28px;
    font-weight: 300;
}}
QLabel#cardValueAccent {{
    color: {COLOR_ACCENT};
    font-size: 28px;
    font-weight: 300;
}}
QLabel#brand {{
    color: {COLOR_ACCENT};
    font-size: 22px;
    font-weight: 200;
    letter-spacing: 5px;
}}
QPushButton {{
    background-color: {COLOR_BG_2};
    color: {COLOR_TEXT_1};
    border: 1px solid {COLOR_LINE};
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 12px;
    letter-spacing: 0.5px;
}}
QPushButton:hover {{
    border-color: {COLOR_ACCENT};
    color: {COLOR_ACCENT};
}}
QPushButton#primary {{
    background-color: {COLOR_ACCENT};
    color: #001220;
    font-weight: 600;
}}
QPushButton#primary:hover {{
    background-color: #00f0ff;
}}
QPushButton#danger {{
    background-color: transparent;
    color: {COLOR_CRIT};
    border-color: {COLOR_CRIT};
}}
QComboBox {{
    background-color: {COLOR_BG_2};
    color: {COLOR_TEXT_1};
    border: 1px solid {COLOR_LINE};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}}
QComboBox QAbstractItemView {{
    background-color: {COLOR_BG_2};
    color: {COLOR_TEXT_1};
    selection-background-color: {COLOR_ACCENT};
    selection-color: #001220;
}}
QCheckBox {{
    color: {COLOR_TEXT_2};
    font-size: 12px;
    spacing: 8px;
}}
QSlider::groove:horizontal {{
    border: 1px solid {COLOR_LINE};
    height: 4px;
    border-radius: 2px;
    background: {COLOR_BG_2};
}}
QSlider::handle:horizontal {{
    background: {COLOR_ACCENT};
    border: none;
    width: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}
QTextEdit {{
    background-color: {COLOR_BG_0};
    color: {COLOR_TEXT_2};
    border: 1px solid {COLOR_LINE};
    border-radius: 8px;
    padding: 8px;
    font-family: "SF Mono", "Menlo", monospace;
    font-size: 11px;
}}
QProgressBar {{
    background-color: {COLOR_BG_2};
    border: 1px solid {COLOR_LINE};
    border-radius: 4px;
    text-align: right;
    color: {COLOR_TEXT_1};
    font-size: 10px;
    height: 14px;
}}
QProgressBar::chunk {{
    background-color: {COLOR_ACCENT};
    border-radius: 3px;
}}
"""


# ==========================================================================
# 自定义环形风险仪表
# ==========================================================================
class RiskGauge(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0  # 0-100
        self._risk = "safe"
        self.setMinimumSize(180, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_value(self, value: float, risk: str):
        self._value = max(0, min(100, int(value * 100)))
        self._risk = risk
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        side = min(rect.width(), rect.height()) - 24
        x = (rect.width() - side) // 2
        y = (rect.height() - side) // 2 + 8

        # 背景圆环
        pen_bg = QPen(QColor(COLOR_LINE))
        pen_bg.setWidth(12)
        pen_bg.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen_bg)
        # Qt 角度单位 1/16 度
        start_angle = 220 * 16
        span_full = -260 * 16
        p.drawArc(x, y, side, side, start_angle, span_full)

        # 进度色环
        color = COLOR_SAFE
        if self._risk == "warn":
            color = COLOR_WARN
        elif self._risk == "critical":
            color = COLOR_CRIT
        pen_fg = QPen(QColor(color))
        pen_fg.setWidth(12)
        pen_fg.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen_fg)
        span_done = int(span_full * (self._value / 100))
        p.drawArc(x, y, side, side, start_angle, span_done)

        # 中心数字
        p.setPen(QColor(color))
        f = QFont("SF Pro Display", 32, QFont.Weight.Light)
        p.setFont(f)
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{self._value}")

        # 风险文字
        p.setPen(QColor(COLOR_TEXT_2))
        f2 = QFont("SF Pro Display", 9)
        f2.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 130)
        p.setFont(f2)
        text_rect = rect.adjusted(0, side // 2 + 30, 0, 0)
        p.drawText(text_rect, Qt.AlignmentFlag.AlignHCenter, self._risk.upper())


# ==========================================================================
# 视频采集 + 推理线程
# ==========================================================================
class VideoThread(QThread):
    new_frame = pyqtSignal(np.ndarray, dict)
    log_msg = pyqtSignal(str)

    def __init__(self, engine: MonitoringEngine, source, use_clahe=False):
        super().__init__()
        self._running = True
        self.engine = engine
        self.source = source
        self.use_clahe = use_clahe

        self.alarm_distract = QSoundEffect()
        self.alarm_distract.setSource(QUrl.fromLocalFile(ALARM_DISTRACT_SOUND))
        self.alarm_distract.setVolume(1.0)
        self.alarm_critical = QSoundEffect()
        self.alarm_critical.setSource(QUrl.fromLocalFile(ALARM_FATIGUE_SOUND))
        self.alarm_critical.setVolume(1.0)

        self._last_beep = 0.0
        self._last_tts = 0.0

        self.engine.reset()

    def run(self):
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.log_msg.emit(f"[error] 无法打开视频源 {self.source}")
            return

        self.log_msg.emit(f"[info] 已打开视频源 {self.source}")

        is_video_file = isinstance(self.source, str) and not str(self.source).isdigit()
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not (0 < fps < 100):
            fps = 30
        delay = 1.0 / fps

        while self._running:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                self.log_msg.emit("[info] 视频流结束")
                break

            if (str(self.source).isdigit() or self.source == 0) and MIRROR_CAMERA_FRAME:
                frame = cv2.flip(frame, 1)

            try:
                out_frame, results = self.engine.process_frame(frame, self.use_clahe)
            except Exception as e:
                self.log_msg.emit(f"[warn] 推理异常: {e}")
                continue

            self.new_frame.emit(out_frame, results)

            # 报警声音
            now = time.time()
            if results.get("is_critical"):
                if now - self._last_beep > 0.5:
                    self._last_beep = now
                    self.alarm_critical.play()
                if now - self._last_tts > 10:
                    self._last_tts = now
                    try:
                        subprocess.Popen(["say", "-r", "180", "警告，检测到极度疲劳，请立即停车休息"])
                    except Exception:
                        pass
            elif results.get("is_warning"):
                if now - self._last_beep > 3:
                    self._last_beep = now
                    self.alarm_distract.play()

            # 视频文件需限速
            if is_video_file:
                used = time.time() - t0
                if used < delay:
                    time.sleep(delay - used)

        cap.release()

    def stop(self):
        self._running = False
        self.wait(2000)


# ==========================================================================
# 主窗口
# ==========================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DMS · Driver Monitoring System")
        self.resize(1440, 900)
        self.setStyleSheet(GLOBAL_QSS)

        # 8 类标签（与 BehaviorDetector 输出对齐）
        self.class_labels = [
            "Normal_Driving",
            "Texting",
            "Talking_on_Phone",
            "Operating_Radio",
            "Drinking",
            "Reaching_Behind",
            "Hair_and_Makeup",
            "Talking_to_Passenger",
        ]

        print("[ui] 正在加载 AI 引擎...")
        self.engine = MonitoringEngine(enable_pose=False)
        print("[ui] 引擎就绪")

        self.thread = None

        # 趋势数据
        self.ear_buf = collections.deque(maxlen=200)
        self.mar_buf = collections.deque(maxlen=200)

        # FPS 统计
        self._frame_times = collections.deque(maxlen=30)

        self._build_ui()

        # 时钟定时器
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._tick_clock)
        self.clock_timer.start(500)
        self._tick_clock()

    # ----------------------------------------------------------------------
    # UI 构造
    # ----------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # 顶部 KPI 栏
        root.addWidget(self._build_top_bar())

        # 主体（3 列）
        body = QHBoxLayout()
        body.setSpacing(10)
        body.addWidget(self._build_video_panel(), stretch=6)
        body.addLayout(self._build_right_column(), stretch=4)
        root.addLayout(body, stretch=1)

        # 底部日志/控制
        root.addWidget(self._build_bottom_bar())

        self.setCentralWidget(central)

    def _make_card(self):
        card = QFrame()
        card.setObjectName("card")
        return card

    def _build_top_bar(self):
        bar = self._make_card()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 12, 20, 12)

        brand = QLabel("DMS")
        brand.setObjectName("brand")
        sub = QLabel("DRIVER MONITORING SYSTEM")
        sub.setStyleSheet(f"color: {COLOR_TEXT_3}; font-size: 10px; letter-spacing: 3px; padding-left: 10px;")
        layout.addWidget(brand)
        layout.addWidget(sub)
        layout.addStretch()

        # KPI 块
        for title, attr in [
            ("FPS", "lbl_fps"),
            ("YOLO 置信度", "lbl_yolo_conf"),
            ("融合置信度", "lbl_fused_conf"),
            ("系统时间", "lbl_clock"),
        ]:
            block = QVBoxLayout()
            t = QLabel(title)
            t.setObjectName("cardTitle")
            t.setAlignment(Qt.AlignmentFlag.AlignRight)
            v = QLabel("--")
            v.setObjectName("cardValueAccent" if attr in ("lbl_fps", "lbl_fused_conf") else "cardValue")
            v.setAlignment(Qt.AlignmentFlag.AlignRight)
            setattr(self, attr, v)
            block.addWidget(t)
            block.addWidget(v)
            wrap = QWidget()
            wrap.setLayout(block)
            wrap.setMinimumWidth(110)
            layout.addWidget(wrap)

        return bar

    def _build_video_panel(self):
        card = self._make_card()
        v = QVBoxLayout(card)
        v.setContentsMargins(14, 14, 14, 14)

        title_row = QHBoxLayout()
        t = QLabel("实时视频源 · LIVE")
        t.setObjectName("cardTitle")
        title_row.addWidget(t)
        title_row.addStretch()

        self.cmb_source = QComboBox()
        self.cmb_source.addItems(
            ["摄像头 ID 0", "摄像头 ID 1", "摄像头 ID 2", "选择本地视频…"]
        )
        title_row.addWidget(self.cmb_source)

        self.btn_start = QPushButton("▶ 开始")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self.start_detection)
        self.btn_stop = QPushButton("■ 停止")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(self.stop_detection)
        title_row.addWidget(self.btn_start)
        title_row.addWidget(self.btn_stop)
        v.addLayout(title_row)

        # 视频画面
        self.video_label = QLabel("等待视频输入…")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            f"background: #000; border: 1px solid {COLOR_LINE}; border-radius: 8px; color: {COLOR_TEXT_3};"
        )
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        v.addWidget(self.video_label, stretch=1)

        # 视频下方状态
        status = QHBoxLayout()
        # EAR
        ear_box = self._labeled_value("EAR · 眼部张合度", "0.00", "lbl_ear", accent=True)
        # MAR
        mar_box = self._labeled_value("MAR · 嘴部张合度", "0.00", "lbl_mar", accent=True)
        # behavior
        beh_box = self._labeled_value("行为识别", "—", "lbl_behavior")
        # fatigue
        fat_box = self._labeled_value("疲劳判定", "—", "lbl_fatigue")
        for w in (ear_box, mar_box, beh_box, fat_box):
            status.addWidget(w, stretch=1)
        v.addLayout(status)

        return card

    def _labeled_value(self, title, value, attr, accent=False):
        wrap = QFrame()
        wrap.setObjectName("card")
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(12, 10, 12, 10)
        t = QLabel(title)
        t.setObjectName("cardTitle")
        v = QLabel(value)
        v.setObjectName("cardValueAccent" if accent else "cardValue")
        setattr(self, attr, v)
        lay.addWidget(t)
        lay.addWidget(v)
        return wrap

    def _build_right_column(self):
        col = QVBoxLayout()
        col.setSpacing(10)

        # 风险仪表
        risk_card = self._make_card()
        risk_lay = QVBoxLayout(risk_card)
        risk_lay.setContentsMargins(14, 14, 14, 14)
        rt = QLabel("综合风险等级")
        rt.setObjectName("cardTitle")
        risk_lay.addWidget(rt)
        self.risk_gauge = RiskGauge()
        risk_lay.addWidget(self.risk_gauge, stretch=1)
        self.lbl_fusion_notes = QLabel("等待数据…")
        self.lbl_fusion_notes.setStyleSheet(
            f"color: {COLOR_TEXT_3}; font-size: 11px;"
        )
        self.lbl_fusion_notes.setWordWrap(True)
        risk_lay.addWidget(self.lbl_fusion_notes)
        col.addWidget(risk_card, stretch=2)

        # 8 类置信度
        prob_card = self._make_card()
        prob_lay = QVBoxLayout(prob_card)
        prob_lay.setContentsMargins(14, 14, 14, 14)
        pt = QLabel("8 类行为置信度")
        pt.setObjectName("cardTitle")
        prob_lay.addWidget(pt)

        self.prob_bars = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        for i, name in enumerate(self.class_labels):
            label = QLabel(name)
            label.setStyleSheet(f"color: {COLOR_TEXT_2}; font-size: 11px;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("%v%")
            self.prob_bars[name] = bar
            grid.addWidget(label, i, 0)
            grid.addWidget(bar, i, 1)
        prob_lay.addLayout(grid)
        col.addWidget(prob_card, stretch=3)

        # EAR/MAR 趋势图
        trend_card = self._make_card()
        trend_lay = QVBoxLayout(trend_card)
        trend_lay.setContentsMargins(14, 14, 14, 14)
        tt = QLabel("EAR / MAR 趋势")
        tt.setObjectName("cardTitle")
        trend_lay.addWidget(tt)

        if HAS_PG:
            pg.setConfigOptions(antialias=True)
            self.trend_plot = pg.PlotWidget(background=COLOR_BG_1)
            self.trend_plot.setMouseEnabled(x=False, y=False)
            self.trend_plot.hideButtons()
            self.trend_plot.showGrid(x=True, y=True, alpha=0.15)
            axis_pen = pg.mkPen(color=COLOR_LINE, width=1)
            self.trend_plot.getAxis("bottom").setPen(axis_pen)
            self.trend_plot.getAxis("left").setPen(axis_pen)
            self.trend_plot.getAxis("bottom").setTextPen(QColor(COLOR_TEXT_3))
            self.trend_plot.getAxis("left").setTextPen(QColor(COLOR_TEXT_3))
            self.trend_plot.addLegend(offset=(-10, 10))
            self.curve_ear = self.trend_plot.plot(
                pen=pg.mkPen(COLOR_ACCENT, width=2), name="EAR"
            )
            self.curve_mar = self.trend_plot.plot(
                pen=pg.mkPen(COLOR_WARN, width=2), name="MAR"
            )
            trend_lay.addWidget(self.trend_plot)
        else:
            placeholder = QLabel("缺少 pyqtgraph，请 pip install pyqtgraph")
            placeholder.setStyleSheet(f"color: {COLOR_TEXT_3}; padding: 24px;")
            trend_lay.addWidget(placeholder)
        col.addWidget(trend_card, stretch=3)

        return col

    def _build_bottom_bar(self):
        card = self._make_card()
        lay = QHBoxLayout(card)
        lay.setContentsMargins(14, 10, 14, 10)

        # 参数控制
        controls = QVBoxLayout()
        controls.setSpacing(6)

        self.chk_clahe = QCheckBox("启用 CLAHE 暗光增强")
        self.chk_clahe.setChecked(False)
        self.chk_clahe.toggled.connect(self.toggle_clahe)
        controls.addWidget(self.chk_clahe)

        self.chk_pose = QCheckBox("启用 Pose 空间约束（二级判定）")
        self.chk_pose.setChecked(False)
        self.chk_pose.toggled.connect(self.toggle_pose)
        controls.addWidget(self.chk_pose)

        self.chk_tta = QCheckBox("启用 TTA（测试时增强：原图+翻转）")
        self.chk_tta.setChecked(True)
        self.chk_tta.toggled.connect(self.toggle_tta)
        controls.addWidget(self.chk_tta)

        self.chk_auto_tune = QCheckBox("启用 EAR/MAR 自适应阈值")
        self.chk_auto_tune.setChecked(True)
        self.chk_auto_tune.toggled.connect(self.toggle_auto_tune)
        controls.addWidget(self.chk_auto_tune)

        slider_row = QHBoxLayout()
        self.lbl_conf = QLabel("行为置信度: 0.50")
        self.lbl_conf.setStyleSheet(f"color: {COLOR_TEXT_2}; font-size: 11px;")
        self.slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_conf.setRange(20, 90)
        self.slider_conf.setValue(50)
        self.slider_conf.valueChanged.connect(self.update_conf)
        slider_row.addWidget(self.lbl_conf)
        slider_row.addWidget(self.slider_conf, stretch=1)
        controls.addLayout(slider_row)

        lay.addLayout(controls, stretch=1)

        # 日志
        log_block = QVBoxLayout()
        lt = QLabel("事件日志")
        lt.setObjectName("cardTitle")
        log_block.addWidget(lt)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        log_block.addWidget(self.log)
        lay.addLayout(log_block, stretch=2)

        # 报告导出
        export_block = QVBoxLayout()
        export_block.addStretch()
        self.btn_export = QPushButton("📊 导出会话报告")
        self.btn_export.setObjectName("primary")
        self.btn_export.clicked.connect(self.export_report)
        export_block.addWidget(self.btn_export)
        export_block.addStretch()
        lay.addLayout(export_block)

        return card

    # ----------------------------------------------------------------------
    # 行为
    # ----------------------------------------------------------------------
    def _tick_clock(self):
        self.lbl_clock.setText(datetime.now().strftime("%H:%M:%S"))

    def start_detection(self):
        if self.thread is not None:
            return
        idx = self.cmb_source.currentIndex()
        if idx == 3:
            file_name, _ = QFileDialog.getOpenFileName(
                self, "选择视频", "", "Video Files (*.mp4 *.avi *.mov)"
            )
            if not file_name:
                return
            source = file_name
        else:
            source = idx

        self.thread = VideoThread(self.engine, source, use_clahe=self.chk_clahe.isChecked())
        self.thread.new_frame.connect(self._on_new_frame)
        self.thread.log_msg.connect(self._append_log)
        self.thread.start()
        self._append_log("▶ 监测已启动")

    def stop_detection(self):
        if self.thread is None:
            return
        self.thread.stop()
        self.thread = None
        self.video_label.setText("监测已停止")
        self._append_log("■ 监测已停止")

    def toggle_clahe(self, checked):
        if self.thread is not None:
            self.thread.use_clahe = checked
        self._append_log(f"CLAHE {'开启' if checked else '关闭'}")

    def toggle_pose(self, checked):
        self.engine.enable_pose = checked
        if checked and self.engine.pose_validator is None:
            self.engine._init_pose()
        self._append_log(f"Pose 二级判定 {'开启' if checked else '关闭'}")

    def toggle_tta(self, checked):
        self.engine.behavior_detector.use_tta = checked
        self._append_log(f"TTA 测试时增强 {'开启' if checked else '关闭'}")

    def toggle_auto_tune(self, checked):
        fd = self.engine.fatigue_detector
        fd.auto_tune_thresholds = checked
        if not checked:
            fd.dynamic_ear_threshold = 0.25
            fd.dynamic_mar_threshold = 0.50
        self._append_log(f"动态阈值 {'开启' if checked else '关闭'}")

    def update_conf(self, val):
        c = val / 100.0
        self.engine.behavior_detector.confidence_threshold = c
        self.lbl_conf.setText(f"行为置信度: {c:.2f}")

    def export_report(self):
        import os

        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        path = os.path.join(desktop, f"DMS_Session_{int(time.time())}.txt")
        self.engine.logger.export_report(path)
        self._append_log(f"已导出报告 → {path}")
        try:
            subprocess.Popen(["open", path])
        except Exception:
            pass

    def _append_log(self, text):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {text}")

    # ----------------------------------------------------------------------
    # 帧回调
    # ----------------------------------------------------------------------
    def _on_new_frame(self, cv_img, results):
        if self.thread is None:
            return

        # FPS 统计
        now = time.time()
        self._frame_times.append(now)
        if len(self._frame_times) >= 2:
            dt = self._frame_times[-1] - self._frame_times[0]
            fps = (len(self._frame_times) - 1) / dt if dt > 0 else 0
            self.lbl_fps.setText(f"{fps:.0f}")

        # 视频图像
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)

        # 数值
        ear = float(results.get("ear", 0.0))
        mar = float(results.get("mar", 0.0))
        self.lbl_ear.setText(f"{ear:.2f}")
        self.lbl_mar.setText(f"{mar:.2f}")
        self.lbl_behavior.setText(str(results.get("behavior_state", "—")))
        self.lbl_fatigue.setText(str(results.get("fatigue_state", "—")))
        self.lbl_yolo_conf.setText(f"{float(results.get('yolo_raw_confidence', 0.0)):.2f}")
        self.lbl_fused_conf.setText(f"{float(results.get('fused_confidence', 0.0)):.2f}")

        # 风险仪表
        risk = results.get("risk_level", "safe")
        self.risk_gauge.set_value(float(results.get("fused_confidence", 0.0)), risk)

        # 染色行为/疲劳/EAR/MAR
        if "正常" in results.get("behavior_state", ""):
            self.lbl_behavior.setStyleSheet(f"color: {COLOR_SAFE}; font-size: 22px;")
        else:
            self.lbl_behavior.setStyleSheet(f"color: {COLOR_CRIT}; font-size: 22px; font-weight: 600;")

        if "正常" in results.get("fatigue_state", ""):
            self.lbl_fatigue.setStyleSheet(f"color: {COLOR_SAFE}; font-size: 22px;")
        elif "极度" in results.get("fatigue_state", ""):
            self.lbl_fatigue.setStyleSheet(f"color: {COLOR_CRIT}; font-size: 22px; font-weight: 600;")
        else:
            self.lbl_fatigue.setStyleSheet(f"color: {COLOR_WARN}; font-size: 22px;")

        # 融合 notes
        notes = results.get("fusion_notes") or []
        if notes:
            self.lbl_fusion_notes.setText("· " + "\n· ".join(notes))
        else:
            self.lbl_fusion_notes.setText("单模态判定")

        # 8 类概率
        probs = results.get("class_probs")
        names = results.get("class_names")
        if probs and names:
            for i, name in enumerate(names):
                bar = self.prob_bars.get(name)
                if bar:
                    bar.setValue(int(probs[i] * 100))

        # 趋势
        if HAS_PG:
            self.ear_buf.append(ear)
            self.mar_buf.append(mar)
            xs = list(range(len(self.ear_buf)))
            self.curve_ear.setData(xs, list(self.ear_buf))
            self.curve_mar.setData(xs, list(self.mar_buf))

        # 重要事件入日志
        if results.get("is_warning"):
            self._append_log(
                f"{'[CRIT]' if results.get('is_critical') else '[WARN]'} "
                f"行为={results.get('behavior_state')} | 疲劳={results.get('fatigue_state')}"
            )

    # ----------------------------------------------------------------------
    def closeEvent(self, event):
        self.stop_detection()
        event.accept()


def main():
    import sys

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
