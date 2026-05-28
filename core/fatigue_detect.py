import math
import mediapipe as mp
import cv2
import threading


from core.config import (
    PERCLOS_WINDOW_FRAMES,
    PERCLOS_DANGER_THRESHOLD,
    CONTINUOUS_BLINK_FRAMES,
    CONTINUOUS_YAWN_FRAMES,
)


class FatigueDetector:
    def __init__(self, perclos_time_window=PERCLOS_WINDOW_FRAMES):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.lock = threading.Lock()

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.2,  # Extremely aggressive face search
            min_tracking_confidence=0.2,  # Keep tracking even if face is blurred
        )

        # PERCLOS variables (from config)
        self.perclos_time_window = perclos_time_window
        self.history = []

        # Dynamic baseline queues
        self.ear_history = []
        self.mar_history = []
        self.baseline_frames = 60  # Require at least 60 frames to stabilize
        self.auto_tune_thresholds = True  # Can be toggled by UI

        # Initial or manual fallback thresholds before dynamics kick in
        self.dynamic_ear_threshold = 0.25
        self.dynamic_mar_threshold = 0.50

        # Frame counters for continuous behavior (prevents false positives)
        self.blink_frames = 0
        self.yawn_frames = 0

    def reset_tracker(self):
        """Safely close and recreate the MediaPipe graph to prevent timestamp mismatch crash."""
        with self.lock:
            if hasattr(self, "face_mesh"):
                try:
                    self.face_mesh.close()
                except Exception:
                    pass
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.2,  # Extremely aggressive face search
                min_tracking_confidence=0.2,
            )
        self.history.clear()
        self.ear_history.clear()
        self.mar_history.clear()
        self.blink_frames = 0
        self.yawn_frames = 0

    def _euclidean_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def _calculate_ear(self, landmarks, eye_indices):
        v1 = self._euclidean_distance(
            landmarks[eye_indices[1]], landmarks[eye_indices[5]]
        )
        v2 = self._euclidean_distance(
            landmarks[eye_indices[2]], landmarks[eye_indices[4]]
        )
        h = self._euclidean_distance(
            landmarks[eye_indices[0]], landmarks[eye_indices[3]]
        )
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def _calculate_mar(self, landmarks, mouth_indices):
        v1 = self._euclidean_distance(
            landmarks[mouth_indices[1]], landmarks[mouth_indices[7]]
        )
        v2 = self._euclidean_distance(
            landmarks[mouth_indices[2]], landmarks[mouth_indices[6]]
        )
        v3 = self._euclidean_distance(
            landmarks[mouth_indices[3]], landmarks[mouth_indices[5]]
        )
        h = self._euclidean_distance(
            landmarks[mouth_indices[0]], landmarks[mouth_indices[4]]
        )
        mar = (v1 + v2 + v3) / (3.0 * h)
        return mar

    def process_frame(self, frame):
        """
        Process a single frame to detect 468 face landmarks,
        compute EAR and MAR, and evaluate fatigue state.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with self.lock:
            try:
                results = self.face_mesh.process(rgb_frame)

                # Auto-Fallback: If no face found, image might be too dark.
                # Enhance contrast aggressively and try one more time!
                if not results.multi_face_landmarks:
                    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    enhanced_gray = clahe.apply(gray)
                    enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
                    results = self.face_mesh.process(enhanced_rgb)

            except Exception as e:
                # Catch MediaPipe underlying graph race conditions when thread is restarting
                return 1.0, 0.0, "初始化中"

        ear = 0.0
        mar = 0.0
        fatigue_level = "人员特征未对齐/光线过暗"

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark

            # Eye / Mouth landmark 索引
            left_eye = [33, 160, 158, 133, 153, 144]
            right_eye = [362, 385, 387, 263, 373, 380]
            mouth = [78, 81, 13, 311, 308, 402, 14, 178]

            # 仅绘制眼睛与嘴的关键点 + 连线（去掉 FACEMESH_CONTOURS 全网格，节省 ~30% 单帧开销）
            h, w = frame.shape[:2]
            eye_mouth_color = (0, 212, 255)  # 与 UI 主色 #00d4ff 对齐（BGR）
            for idx_group in (left_eye, right_eye, mouth):
                pts = [
                    (int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in idx_group
                ]
                # 闭合多边形（眼睛/嘴轮廓）
                for k in range(len(pts)):
                    cv2.line(frame, pts[k], pts[(k + 1) % len(pts)], eye_mouth_color, 1)
                for p in pts:
                    cv2.circle(frame, p, 2, (0, 255, 200), -1)

            left_ear = self._calculate_ear(landmarks, left_eye)
            right_ear = self._calculate_ear(landmarks, right_eye)
            ear = (left_ear + right_ear) / 2.0

            mar = self._calculate_mar(landmarks, mouth)

            # Update dynamic queues (maintain up to last 300 frames ~10s)
            if len(self.ear_history) >= 300:
                self.ear_history.pop(0)
            self.ear_history.append(ear)

            if len(self.mar_history) >= 300:
                self.mar_history.pop(0)
            self.mar_history.append(mar)

            # Recalculate dynamic thresholds once we have enough frames and if auto-tune is ON
            if (
                self.auto_tune_thresholds
                and len(self.ear_history) >= self.baseline_frames
            ):
                # Top 90% EAR represents "open eyes state" for this specific driver/camera
                sorted_ear = sorted(self.ear_history)
                open_eye_baseline = sorted_ear[int(len(sorted_ear) * 0.90)]
                # Safe clamp: usually closed eyes are ~60% of open eyes, but never below 0.15
                self.dynamic_ear_threshold = max(
                    0.15, min(0.3, open_eye_baseline * 0.65)
                )

                # Bottom 10% MAR represents "closed mouth state"
                sorted_mar = sorted(self.mar_history)
                closed_mouth_baseline = sorted_mar[int(len(sorted_mar) * 0.10)]
                # Safe clamp: yawn is usually baseline + 0.3
                self.dynamic_mar_threshold = max(0.40, closed_mouth_baseline + 0.25)

            # PERCLOS logic using dynamically adapted thresholds
            is_closed = 1 if ear < self.dynamic_ear_threshold else 0
            self.history.append(is_closed)
            if len(self.history) > self.perclos_time_window:
                self.history.pop(0)

            # Calculate the percentage of time eyes are closed (protect against short history at startup)
            if (
                len(self.history) < 150
            ):  # Require at least 5 seconds of data before PERCLOS is valid
                perclos = 0
            else:
                perclos = sum(self.history) / float(len(self.history))

            # Continuous Yawn Tracking (require mouth open for ~15+ contiguous frames)
            if mar > self.dynamic_mar_threshold:
                self.yawn_frames += 1
            else:
                self.yawn_frames = 0

            # Continuous Blink Tracking (require eyes closed for ~15+ contiguous frames)
            if is_closed:
                self.blink_frames += 1
            else:
                self.blink_frames = 0

            # Determine State based on Continuous Counters
            if perclos > PERCLOS_DANGER_THRESHOLD:
                fatigue_level = "极度疲劳 (PERCLOS超标)"
            elif self.blink_frames > CONTINUOUS_BLINK_FRAMES:
                fatigue_level = (
                    "极度疲劳"
                    if self.blink_frames > CONTINUOUS_BLINK_FRAMES * 2
                    else "闭眼"
                )
            elif self.yawn_frames > CONTINUOUS_YAWN_FRAMES:
                fatigue_level = "打哈欠"
            else:
                fatigue_level = "正常"

        return ear, mar, fatigue_level
