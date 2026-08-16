import sys
import os
import cv2
import time
import numpy as np
from typing import Optional, List, Callable, Dict, Tuple

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen, QBrush, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit,
    QSlider, QCheckBox, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QScrollArea, QFrame, QSplitter, QFileDialog, QMessageBox, QProgressBar,
    QStyleOption, QStyle, QSpinBox, QComboBox
)

from config import AppConfig, QualityWeights, QualityThresholds
from analyzer import AnalysisResult
from stream_processor import StreamProcessor, CapturedScreenshot
from overlay import draw_overlay

DARK_STYLESHEET = """
QMainWindow {
    background-color: #0f172a;
}
QWidget {
    color: #f8fafc;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #334155;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 14px;
    background-color: #1e293b;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #38bdf8;
}
QPushButton {
    background-color: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #1d4ed8;
}
QPushButton:pressed {
    background-color: #1e40af;
}
QPushButton#btn_preset {
    background-color: #0d9488;
}
QPushButton#btn_preset:hover {
    background-color: #0f766e;
}
QPushButton#btn_capture {
    background-color: #16a34a;
    font-size: 14px;
}
QPushButton#btn_capture:hover {
    background-color: #15803d;
}
QLineEdit {
    background-color: #0f172a;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 6px 10px;
    color: #f8fafc;
}
QLineEdit:focus {
    border: 1px solid #38bdf8;
}
QProgressBar {
    border: 1px solid #334155;
    border-radius: 6px;
    text-align: center;
    background-color: #0f172a;
    font-weight: bold;
}
QProgressBar::chunk {
    background-color: #3b82f6;
    border-radius: 5px;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #334155;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #38bdf8;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}
QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #475569;
    background-color: #0f172a;
}
QCheckBox::indicator:checked {
    background-color: #2563eb;
    border-color: #38bdf8;
}
QScrollArea {
    border: 1px solid #334155;
    border-radius: 8px;
    background-color: #0f172a;
}
"""

class VideoDisplayWidget(QLabel):
    person_toggle_signal = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #020617; border: 2px solid #1e293b; border-radius: 10px;")
        self.setText("Connecting to Sony SRG-A40 Camera Feed...")
        self.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        
        self.show_mesh = True
        self.show_hud = True
        self.latest_result: Optional[AnalysisResult] = None
        self._last_frame_size: Optional[Tuple[int, int]] = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.latest_result and hasattr(self.latest_result, 'faces') and self.latest_result.faces:
            pos = event.position()
            click_x, click_y = pos.x(), pos.y()
            
            if self._last_frame_size and self.pixmap() and not self.pixmap().isNull():
                pm_w, pm_h = self.pixmap().width(), self.pixmap().height()
                lbl_w, lbl_h = self.width(), self.height()
                offset_x = (lbl_w - pm_w) / 2.0
                offset_y = (lbl_h - pm_h) / 2.0
                
                vid_w, vid_h = self._last_frame_size
                frame_x = (click_x - offset_x) * (vid_w / float(pm_w))
                frame_y = (click_y - offset_y) * (vid_h / float(pm_h))
                
                for f_item in self.latest_result.faces:
                    bx, by, bw, bh = f_item.bbox
                    if bx <= frame_x <= bx + bw and by <= frame_y <= by + bh:
                        self.person_toggle_signal.emit(f_item.person_id)
                        return
        super().mousePressEvent(event)

    def update_frame(self, frame_bgr: np.ndarray, analysis: AnalysisResult):
        self.latest_result = analysis
        h, w, c = frame_bgr.shape
        self._last_frame_size = (w, h)
        display_frame = frame_bgr.copy()
        
        # Color palette for up to 6 tracked people
        PERSON_COLORS = [
            (0, 255, 0),      # P1: Green
            (255, 255, 0),    # P2: Cyan
            (255, 191, 0),    # P3: Sky Blue
            (255, 0, 255),    # P4: Magenta
            (0, 165, 255),    # P5: Orange
            (0, 255, 255)     # P6: Yellow
        ]
        
        # 1. Draw Hand Skeletal Overlay if hands detected
        if hasattr(analysis, 'hand_skeletons') and analysis.hand_skeletons:
            from analyzer import HAND_CONNECTIONS
            for hand_pts in analysis.hand_skeletons:
                for start_idx, end_idx in HAND_CONNECTIONS:
                    if start_idx < len(hand_pts) and end_idx < len(hand_pts):
                        pt1 = hand_pts[start_idx]
                        pt2 = hand_pts[end_idx]
                        cv2.line(display_frame, pt1, pt2, (255, 255, 0), 2)
                for px, py in hand_pts:
                    cv2.circle(display_frame, (px, py), 3, (0, 0, 255), -1)

        # 2. Draw Pose & Shoulder Level HUD Overlay
        if hasattr(analysis, 'pose_skeletons') and analysis.pose_skeletons and self.show_hud:
            for pose in analysis.pose_skeletons:
                if pose.get('visible') and pose.get('left_shoulder') and pose.get('right_shoulder'):
                    ls = pose['left_shoulder']
                    rs = pose['right_shoulder']
                    is_lvl = pose.get('is_level', True)
                    tilt = pose.get('tilt_deg', 0.0)
                    
                    sh_color = (74, 222, 128) if is_lvl else (0, 140, 255) # Green when level, amber/red when tilted
                    cv2.line(display_frame, ls, rs, sh_color, 3)
                    cv2.circle(display_frame, ls, 5, (56, 189, 248), -1)
                    cv2.circle(display_frame, rs, 5, (56, 189, 248), -1)
                    
                    # Arm skeleton connections if present
                    if pose.get('left_elbow'):
                        cv2.line(display_frame, ls, pose['left_elbow'], (56, 189, 248), 2)
                        cv2.circle(display_frame, pose['left_elbow'], 4, (0, 200, 255), -1)
                    if pose.get('right_elbow'):
                        cv2.line(display_frame, rs, pose['right_elbow'], (56, 189, 248), 2)
                        cv2.circle(display_frame, pose['right_elbow'], 4, (0, 200, 255), -1)

                    # Shoulder Level Badge near midpoint
                    mid_x = int((ls[0] + rs[0]) / 2)
                    mid_y = int((ls[1] + rs[1]) / 2) - 8
                    badge_txt = f"{tilt:.1f}deg {'[LEVEL]' if is_lvl else '[TILTED]'}"
                    (bw, bh), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                    cv2.rectangle(display_frame, (mid_x - bw // 2 - 4, mid_y - bh - 4), (mid_x + bw // 2 + 4, mid_y + 4), (15, 23, 42), -1)
                    cv2.rectangle(display_frame, (mid_x - bw // 2 - 4, mid_y - bh - 4), (mid_x + bw // 2 + 4, mid_y + 4), sh_color, 1)
                    cv2.putText(display_frame, badge_txt, (mid_x - bw // 2, mid_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, sh_color, 1)

        # 3. Draw Multi-Person HUD Overlays
        if analysis.face_detected and self.show_hud:
            faces_to_render = analysis.faces if hasattr(analysis, 'faces') and analysis.faces else []
            
            # Fallback if legacy single face
            if not faces_to_render and analysis.face_bbox:
                from analyzer import SingleFaceAnalysis
                f_single = SingleFaceAnalysis(
                    person_id=1, bbox=analysis.face_bbox, score=analysis.overall_score,
                    is_qualified=analysis.is_qualified, is_blinking=analysis.is_blinking,
                    is_smiling=analysis.is_smiling, is_awkward_expression=analysis.is_awkward_expression,
                    mesh_points=analysis.mesh_points
                )
                faces_to_render = [f_single]

            for idx, f_item in enumerate(faces_to_render):
                bx, by, bw, bh = f_item.bbox
                is_sel = getattr(f_item, 'is_selected', True)
                
                if is_sel:
                    color = PERSON_COLORS[idx % len(PERSON_COLORS)] if f_item.is_qualified else (0, 165, 255)
                    label = f"✓ P{f_item.person_id}: {int(f_item.score * 100)}%"
                    if f_item.is_smiling:
                        label += " [SMILE]"
                    elif f_item.is_blinking:
                        label += " [BLINK]"
                    elif f_item.is_looking_down:
                        label += " [LOOK DOWN]"
                    elif f_item.is_awkward_expression:
                        label += " [SPEECH]"
                    elif f_item.is_shoulder_tilted:
                        label += " [SHOULDER TILT]"
                    
                    cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), color, 2)
                    cv2.rectangle(display_frame, (bx, max(0, by - 26)), (bx + max(140, len(label)*11), by), color, -1)
                    cv2.putText(display_frame, label, (bx + 6, max(12, by - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
                else:
                    # Unselected / Ignored Person — Dimmed Grey
                    color = (120, 120, 120)
                    label = f"OFF P{f_item.person_id} (Ignored)"
                    cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), color, 1)
                    cv2.rectangle(display_frame, (bx, max(0, by - 22)), (bx + 140, by), color, -1)
                    cv2.putText(display_frame, label, (bx + 6, max(12, by - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                # Draw Facial Mesh / Keypoints
                if self.show_mesh and f_item.mesh_points is not None and is_sel:
                    for pt in f_item.mesh_points:
                        cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)

            # Dynamic Composite Score Meter on top-left of video
            cv2.rectangle(display_frame, (15, 15), (380, 135), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (15, 15), (380, 135), (51, 65, 85), 1)
            
            score_color = (74, 222, 128) if analysis.overall_score >= 0.65 else (96, 165, 250)
            found_cnt = getattr(analysis, 'people_detected_count', 1)
            
            cv2.putText(display_frame, f"Active Quality Score: {int(analysis.overall_score * 100)}%", (25, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, score_color, 2)
            
            people_txt = f"People: {found_cnt} Detected (Click Face to Toggle)"
            cv2.putText(display_frame, people_txt, (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (56, 189, 248), 1)
            
            pose_txt = f"Primary Yaw: {analysis.yaw_deg:+.1f}deg Pitch: {analysis.pitch_deg:+.1f}deg"
            cv2.putText(display_frame, pose_txt, (25, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (203, 213, 225), 1)

            if analysis.shoulder_detected:
                sh_txt = f"Shoulders: {analysis.shoulder_tilt_deg:.1f}deg ({'LEVEL' if analysis.is_shoulders_level else 'TILTED'})"
                sh_col = (74, 222, 128) if analysis.is_shoulders_level else (248, 113, 113)
            else:
                sh_txt = "Shoulders: Headshot (Out of Frame)"
                sh_col = (148, 163, 184)
            cv2.putText(display_frame, sh_txt, (25, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sh_col, 1)

        # Convert OpenCV BGR to Qt QImage
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_frame.data, w, h, w * c, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Scale to fit widget smoothly & fast
        scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        self.setPixmap(scaled_pixmap)

class ThumbnailCard(QFrame):
    def __init__(self, capture: CapturedScreenshot, rank: int,
                 on_click_callback: Callable[[CapturedScreenshot], None],
                 on_favorite_callback: Optional[Callable[[CapturedScreenshot], None]] = None,
                 on_reject_callback: Optional[Callable[[CapturedScreenshot], None]] = None):
        super().__init__()
        self.capture = capture
        self.on_click_callback = on_click_callback
        self.on_favorite_callback = on_favorite_callback
        self.on_reject_callback = on_reject_callback
        
        self.setFrameShape(QFrame.Shape.StyledPanel)
        rank_border = "#eab308" if rank == 1 else "#38bdf8"
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #1e293b;
                border: 2px solid {rank_border};
                border-radius: 8px;
                padding: 4px;
            }}
            QFrame:hover {{
                border: 2px solid #ffffff;
                background-color: #334155;
            }}
        """)
        self.setFixedWidth(145)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Image label
        img_label = QLabel()
        img_label.setFixedSize(135, 75)
        img_label.setStyleSheet("border-radius: 4px;")
        pixmap = QPixmap(capture.filepath)
        if not pixmap.isNull():
            img_label.setPixmap(pixmap.scaled(135, 75, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation))
        layout.addWidget(img_label)
        
        # Rank & Score label
        score_pct = int(capture.overall_score * 100)
        rank_badge = f"<span style='color: #f59e0b; font-weight: bold;'>🏆 Rank #{rank}</span>"
        info_label = QLabel(f"{rank_badge} | <b>{score_pct}%</b>")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)
        
        # ML Feedback Training Buttons
        btn_bar = QHBoxLayout()
        btn_bar.setContentsMargins(0, 0, 0, 0)
        btn_bar.setSpacing(2)
        
        btn_fav = QPushButton("⭐ Best")
        btn_fav.setToolTip("Train AI: Mark this screenshot as a Favorite/Winner")
        btn_fav.setStyleSheet("background-color: #15803d; color: white; padding: 2px 4px; font-size: 11px; font-weight: bold; border-radius: 3px;")
        btn_fav.clicked.connect(self._on_fav_click)
        
        btn_rej = QPushButton("👎 Bad")
        btn_rej.setToolTip("Train AI: Mark this shot as Bad / Inferior")
        btn_rej.setStyleSheet("background-color: #991b1b; color: white; padding: 2px 4px; font-size: 11px; border-radius: 3px;")
        btn_rej.clicked.connect(self._on_rej_click)
        
        btn_bar.addWidget(btn_fav)
        btn_bar.addWidget(btn_rej)
        layout.addLayout(btn_bar)

    def _on_fav_click(self):
        if self.on_favorite_callback:
            self.on_favorite_callback(self.capture)

    def _on_rej_click(self):
        if self.on_reject_callback:
            self.on_reject_callback(self.capture)

    def mousePressEvent(self, event):
        self.on_click_callback(self.capture)

class MainWindow(QMainWindow):
    frame_signal = pyqtSignal(np.ndarray, AnalysisResult)
    capture_signal = pyqtSignal(CapturedScreenshot)
    status_signal = pyqtSignal(str, int)  # connection status, retry attempt

    def __init__(self):
        super().__init__()
        self.config = AppConfig()
        self.processor: Optional[StreamProcessor] = None
        self.captures_list: List[CapturedScreenshot] = []
        self._is_capturing = False   # Track whether capture is active

        self.setWindowTitle("Sony SRG-A40 Smart Screenshot Grabber")
        self.resize(1280, 800)
        self.setStyleSheet(DARK_STYLESHEET)
        
        self._init_ui()
        
        # Connect signals
        self.frame_signal.connect(self._on_frame_received)
        self.capture_signal.connect(self._on_capture_received)
        self.status_signal.connect(self._on_status_changed)
        self.video_widget.person_toggle_signal.connect(self._on_toggle_person_id)
        
        # Always auto-connect the video feed (capture starts paused)
        self.start_processor()
        if self.processor:
            self.processor.auto_capture_enabled = False

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 1. Top Header & Camera Stream URL Control
        top_bar = QHBoxLayout()
        
        # Connection Status Indicator
        self.lbl_status_dot = QLabel("●")
        self.lbl_status_dot.setFixedWidth(20)
        self.lbl_status_dot.setStyleSheet("color: #ef4444; font-size: 18px;")  # Red = disconnected
        top_bar.addWidget(self.lbl_status_dot)
        
        self.lbl_status_text = QLabel("Disconnected")
        self.lbl_status_text.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.lbl_status_text.setFixedWidth(120)
        top_bar.addWidget(self.lbl_status_text)
        
        lbl_title = QLabel("Sony SRG-A40 AI Thumbnail Selector")
        lbl_title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        lbl_title.setStyleSheet("color: #38bdf8;")
        top_bar.addWidget(lbl_title)
        
        top_bar.addStretch()
        
        lbl_source = QLabel("Source:")
        self.txt_source = QLineEdit(self.config.stream_source)
        self.txt_source.setMinimumWidth(320)
        
        btn_apply_source = QPushButton("Connect Stream")
        btn_apply_source.clicked.connect(self._on_apply_source)
        
        btn_preset_sony = QPushButton("Sony SRG-A40 Preset")
        btn_preset_sony.setObjectName("btn_preset")
        btn_preset_sony.clicked.connect(self._on_preset_sony)
        
        # Start/Stop Capture Button
        self.btn_start_stop = QPushButton("▶  Start Capture")
        self.btn_start_stop.setFixedWidth(160)
        self.btn_start_stop.setStyleSheet("""
            QPushButton {
                background-color: #16a34a;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 700;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #15803d; }
        """)
        self.btn_start_stop.clicked.connect(self._on_start_stop)
        
        top_bar.addWidget(lbl_source)
        top_bar.addWidget(self.txt_source)
        top_bar.addWidget(btn_apply_source)
        top_bar.addWidget(btn_preset_sony)
        top_bar.addWidget(self.btn_start_stop)
        
        main_layout.addLayout(top_bar)

        # 2. Main Content Splitter (Left: Video Display, Right: Metrics & Controls)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left Container: Video Feed
        left_box = QWidget()
        left_layout = QVBoxLayout(left_box)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_widget = VideoDisplayWidget()
        left_layout.addWidget(self.video_widget, stretch=1)
        
        # Video HUD Overlay Toggles Toolbar
        hud_bar = QHBoxLayout()
        self.chk_mesh = QCheckBox("Show Face Mesh")
        self.chk_mesh.setChecked(True)
        self.chk_mesh.toggled.connect(lambda val: setattr(self.video_widget, 'show_mesh', val))
        
        self.chk_hud = QCheckBox("Show Quality HUD")
        self.chk_hud.setChecked(True)
        self.chk_hud.toggled.connect(lambda val: setattr(self.video_widget, 'show_hud', val))
        
        self.chk_auto_capture = QCheckBox("Enable Auto-Capture")
        self.chk_auto_capture.setChecked(True)
        self.chk_auto_capture.toggled.connect(self._on_toggle_auto_capture)
        
        self.chk_single_winner = QCheckBox("🏆 Keep 1 Best Shot Only")
        self.chk_single_winner.setChecked(True)
        self.chk_single_winner.toggled.connect(self._on_toggle_single_winner)
        
        btn_manual_capture = QPushButton("📸 Manual Screenshot")
        btn_manual_capture.setObjectName("btn_capture")
        btn_manual_capture.clicked.connect(self._on_manual_capture)
        
        hud_bar.addWidget(self.chk_mesh)
        hud_bar.addWidget(self.chk_hud)
        hud_bar.addWidget(self.chk_auto_capture)
        hud_bar.addWidget(self.chk_single_winner)
        hud_bar.addStretch()
        hud_bar.addWidget(btn_manual_capture)
        left_layout.addLayout(hud_bar)
        
        splitter.addWidget(left_box)

        # Right Container: Quality Telemetry & Controls Panel
        right_box = QWidget()
        right_layout = QVBoxLayout(right_box)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Group 1: Live Telemetry Metrics
        grp_telemetry = QGroupBox("Live Face & Expression Telemetry")
        telemetry_layout = QVBoxLayout(grp_telemetry)
        
        # Composite Score Progress Bar
        lbl_comp = QLabel("Overall Frame Quality Score")
        self.bar_composite = QProgressBar()
        self.bar_composite.setRange(0, 100)
        self.bar_composite.setValue(0)
        telemetry_layout.addWidget(lbl_comp)
        telemetry_layout.addWidget(self.bar_composite)
        
        # Metric Readouts Grid
        grid_metrics = QGridLayout()
        grid_metrics.setVerticalSpacing(8)
        
        self.lbl_head_pose = QLabel("Head Pose: --")
        self.lbl_eyes = QLabel("Eye Openness: --")
        self.lbl_smile = QLabel("Expression: --")
        self.lbl_shoulders = QLabel("Shoulders: --")
        self.lbl_sharpness = QLabel("Sharpness: --")
        self.lbl_hand = QLabel("Hand Position: --")
        self.lbl_exposure = QLabel("Exposure: --")
        self.lbl_centering = QLabel("Centering: --")
        self.lbl_stability = QLabel("Stability: --")
        self.lbl_ai_model = QLabel("AI Preference Model: Untrained (Default Rule Mode)")
        self.lbl_ai_model.setStyleSheet("color: #38bdf8; font-weight: bold;")
        
        grid_metrics.addWidget(self.lbl_head_pose, 0, 0)
        grid_metrics.addWidget(self.lbl_eyes, 0, 1)
        grid_metrics.addWidget(self.lbl_smile, 1, 0)
        grid_metrics.addWidget(self.lbl_shoulders, 1, 1)
        grid_metrics.addWidget(self.lbl_sharpness, 2, 0)
        grid_metrics.addWidget(self.lbl_exposure, 2, 1)
        grid_metrics.addWidget(self.lbl_centering, 3, 0)
        grid_metrics.addWidget(self.lbl_hand, 3, 1)
        grid_metrics.addWidget(self.lbl_stability, 4, 0, 1, 2)
        grid_metrics.addWidget(self.lbl_ai_model, 5, 0, 1, 2)
        
        telemetry_layout.addLayout(grid_metrics)
        right_layout.addWidget(grp_telemetry)

        # Group 2: Sensitivity Sliders & Threshold Controls
        grp_sliders = QGroupBox("Capture Threshold Controls")
        sliders_layout = QFormLayout_Grid()
        
        # Min Score Threshold Slider
        self.lbl_min_score_val = QLabel(f"{int(self.config.thresholds.min_overall_score * 100)}%")
        self.sld_min_score = QSlider(Qt.Orientation.Horizontal)
        self.sld_min_score.setRange(30, 95)
        self.sld_min_score.setValue(int(self.config.thresholds.min_overall_score * 100))
        self.sld_min_score.valueChanged.connect(self._on_min_score_changed)
        sliders_layout.addRow("Min Quality Threshold:", self.sld_min_score, self.lbl_min_score_val)
        
        # Target People Count Selector (1 to 6)
        self.spn_people = QSpinBox()
        self.spn_people.setRange(1, 6)
        self.spn_people.setValue(self.config.max_people_target)
        self.spn_people.setSuffix(" Max People")
        self.spn_people.valueChanged.connect(self._on_people_target_changed)
        self.lbl_people_val = QLabel("Track Up To 6")
        sliders_layout.addRow("Max Tracked Count:", self.spn_people, self.lbl_people_val)
        
        # Interactive Person Selection Toggles (P1 to P6)
        grp_targets = QGroupBox("Active Target People Filter (Click Video or Check)")
        targets_layout = QHBoxLayout(grp_targets)
        targets_layout.setContentsMargins(8, 8, 8, 8)
        
        self.person_checkboxes: Dict[int, QCheckBox] = {}
        for pid in range(1, 7):
            chk = QCheckBox(f"P{pid}")
            chk.setChecked(True)
            chk.toggled.connect(lambda checked, p=pid: self._on_person_checkbox_toggled(p, checked))
            self.person_checkboxes[pid] = chk
            targets_layout.addWidget(chk)
            
        right_layout.addWidget(grp_targets)
        
        # Max Yaw Angle Slider
        self.lbl_max_yaw_val = QLabel(f"{int(self.config.thresholds.max_yaw_deg)}°")
        self.sld_max_yaw = QSlider(Qt.Orientation.Horizontal)
        self.sld_max_yaw.setRange(5, 45)
        self.sld_max_yaw.setValue(int(self.config.thresholds.max_yaw_deg))
        self.sld_max_yaw.valueChanged.connect(self._on_max_yaw_changed)
        sliders_layout.addRow("Max Head Turn (Yaw):", self.sld_max_yaw, self.lbl_max_yaw_val)

        # Max Shoulder Tilt Slider
        self.lbl_max_shoulder_val = QLabel(f"{int(self.config.thresholds.max_shoulder_tilt_deg)}°")
        self.sld_max_shoulder = QSlider(Qt.Orientation.Horizontal)
        self.sld_max_shoulder.setRange(4, 25)
        self.sld_max_shoulder.setValue(int(self.config.thresholds.max_shoulder_tilt_deg))
        self.sld_max_shoulder.valueChanged.connect(self._on_max_shoulder_tilt_changed)
        sliders_layout.addRow("Max Shoulder Tilt:", self.sld_max_shoulder, self.lbl_max_shoulder_val)
        
        # Burst Window Slider
        self.lbl_burst_val = QLabel(f"{self.config.burst_window_seconds:.1f}s")
        self.sld_burst = QSlider(Qt.Orientation.Horizontal)
        self.sld_burst.setRange(10, 60) # 1.0s to 6.0s
        self.sld_burst.setValue(int(self.config.burst_window_seconds * 10))
        self.sld_burst.valueChanged.connect(self._on_burst_changed)
        sliders_layout.addRow("Burst Peak Window:", self.sld_burst, self.lbl_burst_val)
        
        grp_sliders.setLayout(sliders_layout)
        right_layout.addWidget(grp_sliders)

        # Output Folder Control
        grp_output = QGroupBox("Output Folder")
        output_layout = QVBoxLayout(grp_output)
        
        # Current path display
        self.txt_output_path = QLineEdit(self.config.output_dir)
        self.txt_output_path.setReadOnly(True)
        self.txt_output_path.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 4px; padding: 4px 8px; font-size: 11px;")
        output_layout.addWidget(self.txt_output_path)
        
        output_btns = QHBoxLayout()
        
        btn_browse_folder = QPushButton("📂 Browse...")
        btn_browse_folder.clicked.connect(self._on_browse_output)
        output_btns.addWidget(btn_browse_folder)
        
        btn_open_folder = QPushButton("📁 Open Folder")
        btn_open_folder.clicked.connect(self._on_open_folder)
        output_btns.addWidget(btn_open_folder)
        
        output_layout.addLayout(output_btns)
        right_layout.addWidget(grp_output)

        right_layout.addStretch()
        splitter.addWidget(right_box)
        splitter.setSizes([750, 450])
        
        main_layout.addWidget(splitter, stretch=1)

        # 3. Bottom Captured Thumbnails Bar
        grp_gallery = QGroupBox("Captured Best Screenshots Gallery")
        gallery_layout = QHBoxLayout(grp_gallery)
        gallery_layout.setContentsMargins(8, 8, 8, 8)
        
        self.scroll_gallery = QScrollArea()
        self.scroll_gallery.setWidgetResizable(True)
        self.scroll_gallery.setFixedHeight(140)
        
        self.gallery_widget = QWidget()
        self.gallery_layout = QHBoxLayout(self.gallery_widget)
        self.gallery_layout.setContentsMargins(4, 4, 4, 4)
        self.gallery_layout.setSpacing(10)
        self.gallery_layout.addStretch()
        
        self.scroll_gallery.setWidget(self.gallery_widget)
        gallery_layout.addWidget(self.scroll_gallery)
        
        main_layout.addWidget(grp_gallery)

    def start_processor(self):
        if self.processor:
            self.processor.stop()
            
        self.processor = StreamProcessor(
            config=self.config,
            on_frame_callback=lambda frame, analysis: self.frame_signal.emit(frame, analysis),
            on_capture_callback=lambda capture: self.capture_signal.emit(capture)
        )
        self.processor.on_status_callback = lambda status, attempt: self.status_signal.emit(status, attempt)
        self.processor.start()

    def _on_frame_received(self, frame: np.ndarray, analysis: AnalysisResult):
        # Update video widget display
        self.video_widget.update_frame(frame, analysis)
        
        # Update Telemetry Gauges
        score_pct = int(analysis.overall_score * 100)
        self.bar_composite.setValue(score_pct)
        
        # Color bar according to qualification
        if analysis.is_qualified:
            self.bar_composite.setStyleSheet("QProgressBar::chunk { background-color: #22c55e; }")
        else:
            self.bar_composite.setStyleSheet("QProgressBar::chunk { background-color: #3b82f6; }")
            
        if analysis.face_detected:
            self.lbl_head_pose.setText(f"Head Yaw: {analysis.yaw_deg:+.1f}° | Pitch: {analysis.pitch_deg:+.1f}° | Roll: {analysis.roll_deg:+.1f}°")
            
            eye_status = "CLOSED/BLINK" if analysis.is_blinking else "OPEN"
            self.lbl_eyes.setText(f"Eyes: {eye_status} (EAR: {analysis.avg_ear:.2f})")
            
            expr_status = "SMILE 😊" if analysis.is_smiling else ("AWKWARD 😮" if analysis.is_awkward_expression else "NEUTRAL")
            self.lbl_smile.setText(f"Expression: {expr_status} (Smile: {analysis.smile_ratio:.2f})")

            if getattr(analysis, 'shoulder_detected', False):
                sh_status = "LEVEL ✅" if analysis.is_shoulders_level else "TILTED ⚠️"
                self.lbl_shoulders.setText(f"Shoulders: {sh_status} ({analysis.shoulder_tilt_deg:.1f}°)")
                self.lbl_shoulders.setStyleSheet("color: #4ade80; font-weight: 500;" if analysis.is_shoulders_level else "color: #f87171; font-weight: bold;")
            else:
                self.lbl_shoulders.setText("Shoulders: Headshot (Out of Frame)")
                self.lbl_shoulders.setStyleSheet("color: #94a3b8;")
            
            self.lbl_sharpness.setText(f"Sharpness: {int(analysis.laplacian_variance)} | EMA: {int(analysis.smoothed_score * 100)}%")
            
            hand_status = "OBSTRUCTING ⚠️" if analysis.hands_covering_face else ("BLURRY 🔄" if analysis.is_hand_blurry else "CLEAR ✅")
            self.lbl_hand.setText(f"Hand: {hand_status}")
            
            # New metrics
            exp_pct = int(analysis.exposure_score * 100)
            exp_icon = "☀️" if analysis.face_brightness > 180 else ("🌑" if analysis.face_brightness < 60 else "✅")
            self.lbl_exposure.setText(f"Exposure: {exp_pct}% {exp_icon} ({analysis.face_brightness:.0f})")
            
            ctr_pct = int(analysis.centering_score * 100)
            self.lbl_centering.setText(f"Centering: {ctr_pct}%")
            
            stab_icon = "🟢 Stable" if analysis.is_stable else "🟡 Unstable"
            self.lbl_stability.setText(f"Score: {stab_icon}")
            
            # Live AI Model Telemetry Readout
            if self.processor and self.processor.analyzer.preference_model.is_trained:
                pm = self.processor.analyzer.preference_model
                self.lbl_ai_model.setText(f"AI Model: Trained ({pm.confidence*100:.0f}% fit | {len(pm.samples)} samples | ML Score: {int(analysis.ml_score*100)}%)")
                self.lbl_ai_model.setStyleSheet("color: #4ade80; font-weight: bold;")
            else:
                self.lbl_ai_model.setText("AI Model: Untrained (Click ⭐ Best in gallery to train AI)")
                self.lbl_ai_model.setStyleSheet("color: #38bdf8; font-weight: bold;")
        else:
            self.lbl_head_pose.setText("Head Pose: No Face Detected")
            self.lbl_eyes.setText("Eye Openness: --")
            self.lbl_smile.setText("Expression: --")
            self.lbl_shoulders.setText("Shoulders: --")
            self.lbl_shoulders.setStyleSheet("color: #f8fafc;")
            self.lbl_sharpness.setText("Sharpness: --")
            self.lbl_hand.setText("Hand Position: --")
            self.lbl_exposure.setText("Exposure: --")
            self.lbl_centering.setText("Centering: --")
            self.lbl_stability.setText("Stability: --")

    def _on_status_changed(self, status: str, attempt: int):
        """Update the connection status indicator in the GUI."""
        status_map = {
            "disconnected": ("●", "#ef4444", "Disconnected"),
            "connecting":   ("●", "#f59e0b", "Connecting..."),
            "connected":    ("●", "#22c55e", "Connected"),
            "reconnecting": ("●", "#f59e0b", f"Reconnecting ({attempt})..."),
            "failed":       ("●", "#ef4444", "Connection Failed"),
        }
        dot, color, text = status_map.get(status, ("●", "#94a3b8", "Unknown"))
        self.lbl_status_dot.setText(dot)
        self.lbl_status_dot.setStyleSheet(f"color: {color}; font-size: 18px;")
        self.lbl_status_text.setText(text)

    def _on_toggle_single_winner(self, checked: bool):
        self.config.single_winner_mode = checked
        if checked and self.processor:
            self.processor.prune_to_best_n(5)
            self._refresh_gallery_display()

    def _on_capture_received(self, capture: CapturedScreenshot):
        self._refresh_gallery_display()

    def _refresh_gallery_display(self):
        # Clear existing gallery widgets
        while self.gallery_layout.count() > 1:
            child = self.gallery_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if self.processor and self.processor.saved_captures:
            # Sort captures by score descending
            captures = sorted(self.processor.saved_captures, key=lambda c: c.overall_score, reverse=True)
            if self.config.single_winner_mode:
                captures = captures[:self.config.max_retained_captures]
                
            for idx, cap in enumerate(captures):
                card = ThumbnailCard(
                    cap, rank=idx+1, 
                    on_click_callback=self._on_thumbnail_clicked,
                    on_favorite_callback=self._on_card_favorite,
                    on_reject_callback=self._on_card_reject
                )
                self.gallery_layout.insertWidget(idx, card)
                
            QTimer.singleShot(100, lambda: self.scroll_gallery.horizontalScrollBar().setValue(0))

    def _on_thumbnail_clicked(self, capture: CapturedScreenshot):
        # Open full preview dialog
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Captured Best Screenshot Details")
        dlg.setText(f"<b>Screenshot Saved:</b><br>{capture.filepath}")
        
        sh_str = f"{capture.analysis.shoulder_tilt_deg:.1f}° ({'Level' if capture.analysis.is_shoulders_level else 'Tilted'})" if getattr(capture.analysis, 'shoulder_detected', False) else "Headshot / Out of Frame"
        info = (
            f"<b>Overall Quality Score:</b> {int(capture.overall_score * 100)}%<br>"
            f"<b>Head Yaw:</b> {capture.analysis.yaw_deg:.1f}°<br>"
            f"<b>Shoulder Tilt:</b> {sh_str}<br>"
            f"<b>Eye Aspect Ratio:</b> {capture.analysis.avg_ear:.2f}<br>"
            f"<b>Smile Ratio:</b> {capture.analysis.smile_ratio:.2f}<br>"
            f"<b>Sharpness Var:</b> {int(capture.analysis.laplacian_variance)}<br>"
        )
        dlg.setInformativeText(info)
        dlg.setIconPixmap(QPixmap(capture.filepath).scaled(320, 180, Qt.AspectRatioMode.KeepAspectRatio))
        dlg.exec()

    def _on_apply_source(self):
        new_src = self.txt_source.text().strip()
        if new_src and self.processor:
            self.processor.update_source(new_src)

    def _on_preset_sony(self):
        preset_url = self.config.sony_srg_a40_preset_url
        self.txt_source.setText(preset_url)
        self._on_apply_source()

    def _on_toggle_auto_capture(self, checked: bool):
        if self.processor:
            self.processor.auto_capture_enabled = checked

    def _on_manual_capture(self):
        if self.processor and self.video_widget.latest_result:
            frame = None
            with self.processor.frame_lock:
                if self.processor.latest_frame is not None:
                    frame = self.processor.latest_frame.copy()
            if frame is not None:
                self.processor.save_screenshot(frame, self.video_widget.latest_result)

    def _on_people_target_changed(self, val: int):
        self.config.max_people_target = val
        if self.processor:
            self.processor.config.max_people_target = val
            self.processor.analyzer.max_people_target = val
        self.lbl_people_val.setText(f"Track Up To {val}")

    def _on_toggle_person_id(self, person_id: int):
        """Called when user clicks directly on a person's face bounding box on the video stream."""
        if person_id in self.person_checkboxes:
            current_state = self.person_checkboxes[person_id].isChecked()
            self.person_checkboxes[person_id].setChecked(not current_state)

    def _on_person_checkbox_toggled(self, person_id: int, checked: bool):
        """Update active selected person IDs in analyzer."""
        if self.processor:
            if checked:
                self.processor.analyzer.selected_person_ids.add(person_id)
            else:
                self.processor.analyzer.selected_person_ids.discard(person_id)

    def _on_card_favorite(self, capture: CapturedScreenshot):
        """User clicked ⭐ Best on a thumbnail card -> train Scikit-Learn preference model."""
        if self.processor and self.processor.analyzer:
            self.processor.analyzer.preference_model.add_sample(capture.analysis, is_favorite=True)
            pm = self.processor.analyzer.preference_model
            QMessageBox.information(
                self, "AI Model Trained ⭐",
                f"Trained AI Preference Model!\n\n"
                f"• Learned that this shot is a Favorite Winner.\n"
                f"• Total Training Samples: {len(pm.samples)}\n"
                f"• Scikit-Learn Model Fit: {pm.confidence * 100:.0f}%\n\n"
                f"The AI will now prioritize shots with similar features!"
            )
            
    def _on_card_reject(self, capture: CapturedScreenshot):
        """User clicked 👎 Bad on a thumbnail card -> train Scikit-Learn preference model."""
        if self.processor and self.processor.analyzer:
            self.processor.analyzer.preference_model.add_sample(capture.analysis, is_favorite=False)
            pm = self.processor.analyzer.preference_model
            QMessageBox.information(
                self, "AI Model Trained 👎",
                f"Trained AI Preference Model!\n\n"
                f"• Learned to penalize shots with these features.\n"
                f"• Total Training Samples: {len(pm.samples)}\n"
                f"• Scikit-Learn Model Fit: {pm.confidence * 100:.0f}%\n"
            )

    def _on_min_score_changed(self, val: int):
        pct = val / 100.0
        self.config.thresholds.min_overall_score = pct
        self.lbl_min_score_val.setText(f"{val}%")

    def _on_max_yaw_changed(self, val: int):
        self.config.thresholds.max_yaw_deg = float(val)
        self.lbl_max_yaw_val.setText(f"{val}°")

    def _on_max_shoulder_tilt_changed(self, val: int):
        self.config.thresholds.max_shoulder_tilt_deg = float(val)
        self.lbl_max_shoulder_val.setText(f"{val}°")

    def _on_burst_changed(self, val: int):
        sec = val / 10.0
        self.config.burst_window_seconds = sec
        self.lbl_burst_val.setText(f"{sec:.1f}s")

    def _on_open_folder(self):
        folder = os.path.abspath(self.config.output_dir)
        os.makedirs(folder, exist_ok=True)
        os.startfile(folder)

    def _on_browse_output(self):
        """Open a folder picker to choose the output directory."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", self.config.output_dir
        )
        if folder:
            self.config.output_dir = folder
            self.txt_output_path.setText(folder)
            if self.processor:
                self.processor.config.output_dir = folder

    def _on_start_stop(self):
        """Toggle auto-capture on/off. The video feed stays connected."""
        if self._is_capturing:
            # STOP CAPTURING
            if self.processor:
                self.processor.auto_capture_enabled = False
            self._is_capturing = False
            self.btn_start_stop.setText("▶  Start Capture")
            self.btn_start_stop.setStyleSheet("""
                QPushButton {
                    background-color: #16a34a;
                    color: #ffffff;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-weight: 700;
                    font-size: 14px;
                }
                QPushButton:hover { background-color: #15803d; }
            """)
        else:
            # START CAPTURING
            if not self.processor:
                self.start_processor()
            if self.processor:
                self.processor.auto_capture_enabled = True
            self._is_capturing = True
            self.btn_start_stop.setText("⏹  Stop Capture")
            self.btn_start_stop.setStyleSheet("""
                QPushButton {
                    background-color: #dc2626;
                    color: #ffffff;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-weight: 700;
                    font-size: 14px;
                }
                QPushButton:hover { background-color: #b91c1c; }
            """)

    def closeEvent(self, event):
        if self.processor:
            self.processor.stop()
        event.accept()

# Helper grid for Form Layout styling
class QFormLayout_Grid(QGridLayout):
    def addRow(self, label_text: str, widget1: QWidget, widget2: QWidget):
        row = self.rowCount()
        lbl = QLabel(label_text)
        lbl.setStyleSheet("font-weight: 500;")
        self.addWidget(lbl, row, 0)
        self.addWidget(widget1, row, 1)
        self.addWidget(widget2, row, 2)
