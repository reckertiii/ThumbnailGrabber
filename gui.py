import sys
import os
import cv2
import time
import numpy as np
import threading
from typing import Optional, List, Callable, Dict, Tuple

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QUrl
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen, QBrush, QIcon, QDesktopServices
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit,
    QSlider, QCheckBox, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QScrollArea, QFrame, QSplitter, QFileDialog, QMessageBox, QProgressBar,
    QStyleOption, QStyle, QSpinBox, QComboBox, QDialog
)

from config import AppConfig, QualityWeights, QualityThresholds
from analyzer import AnalysisResult
from stream_processor import StreamProcessor, CapturedScreenshot
from overlay import draw_overlay
from video_scanner import VideoRecordingScanner, ScanProgress, VideoInfo

DARK_STYLESHEET = """
QMainWindow, QDialog {
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
QPushButton#btn_scan_video {
    background-color: #7c3aed;
    color: #ffffff;
    font-weight: bold;
}
QPushButton#btn_scan_video:hover {
    background-color: #6d28d9;
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
QComboBox, QSpinBox {
    background-color: #0f172a;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 6px 10px;
    color: #f8fafc;
}
QComboBox:focus, QSpinBox:focus {
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
        display_frame = draw_overlay(frame_bgr, analysis, self.show_mesh, self.show_hud)

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
                 on_reject_callback: Optional[Callable[[CapturedScreenshot], None]] = None,
                 on_pin_callback: Optional[Callable[[CapturedScreenshot], None]] = None):
        super().__init__()
        self.capture = capture
        self.on_click_callback = on_click_callback
        self.on_favorite_callback = on_favorite_callback
        self.on_reject_callback = on_reject_callback
        self.on_pin_callback = on_pin_callback

        is_pinned = bool(getattr(capture, "is_pinned", False))
        self.setFrameShape(QFrame.Shape.StyledPanel)
        rank_border = "#eab308" if rank == 1 else ("#0ea5e9" if is_pinned else "#38bdf8")
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
        self.setFixedWidth(165)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Image label
        img_label = QLabel()
        img_label.setFixedSize(155, 84)
        img_label.setStyleSheet("border-radius: 4px;")
        pixmap = QPixmap(capture.filepath)
        if not pixmap.isNull():
            img_label.setPixmap(pixmap.scaled(155, 84, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation))
        layout.addWidget(img_label)
        
        # Rank & Score label
        score_pct = int(capture.overall_score * 100)
        rank_badge = f"<span style='color: #f59e0b; font-weight: bold;'>🏆 #{rank}</span>"
        prop_tag = " | 🎭" if getattr(capture, "is_prop_moment", False) else ""
        pin_tag = " | <span style='color: #38bdf8; font-weight: bold;'>📌</span>" if is_pinned else ""
        info_label = QLabel(f"{rank_badge} | <b>{score_pct}%</b>{prop_tag}{pin_tag}")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)
        
        # Pin + ML Feedback Training Buttons
        btn_bar = QHBoxLayout()
        btn_bar.setContentsMargins(0, 0, 0, 0)
        btn_bar.setSpacing(2)

        btn_pin = QPushButton("📌 Saved" if is_pinned else "📌 Keep")
        btn_pin.setToolTip("Protect this screenshot from auto-deletion when new Top-N shots arrive")
        pin_bg = "#0369a1" if is_pinned else "#334155"
        btn_pin.setStyleSheet(f"background-color: {pin_bg}; color: white; padding: 2px 4px; font-size: 10px; font-weight: bold; border-radius: 3px;")
        btn_pin.clicked.connect(self._on_pin_click)
        
        btn_fav = QPushButton("⭐ Best")
        btn_fav.setToolTip("Train AI: Mark this screenshot as a Favorite/Winner & protect from deletion")
        btn_fav.setStyleSheet("background-color: #15803d; color: white; padding: 2px 4px; font-size: 10px; font-weight: bold; border-radius: 3px;")
        btn_fav.clicked.connect(self._on_fav_click)
        
        btn_rej = QPushButton("👎 Bad")
        btn_rej.setToolTip("Train AI: Mark this shot as Bad / Inferior")
        btn_rej.setStyleSheet("background-color: #991b1b; color: white; padding: 2px 4px; font-size: 10px; border-radius: 3px;")
        btn_rej.clicked.connect(self._on_rej_click)
        
        btn_bar.addWidget(btn_pin)
        btn_bar.addWidget(btn_fav)
        btn_bar.addWidget(btn_rej)
        layout.addLayout(btn_bar)

    def _on_pin_click(self):
        if self.on_pin_callback:
            self.on_pin_callback(self.capture)

    def _on_fav_click(self):
        if self.on_favorite_callback:
            self.on_favorite_callback(self.capture)

    def _on_rej_click(self):
        if self.on_reject_callback:
            self.on_reject_callback(self.capture)

    def mousePressEvent(self, event):
        self.on_click_callback(self.capture)

class VideoScanWorker(QThread):
    progress_signal = pyqtSignal(object)
    finished_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)

    def __init__(self, video_path: str, top_n: int, min_score: float, sample_fps: float,
                 min_interval_sec: float, output_dir: str, config: AppConfig):
        super().__init__()
        self.video_path = video_path
        self.top_n = top_n
        self.min_score = min_score
        self.sample_fps = sample_fps
        self.min_interval_sec = min_interval_sec
        self.output_dir = output_dir
        self.config = config
        self.cancel_event = threading.Event()

    def run(self):
        try:
            scanner = VideoRecordingScanner(self.config)
            results = scanner.scan_video(
                video_path=self.video_path,
                top_n=self.top_n,
                min_score=self.min_score,
                sample_fps=self.sample_fps,
                min_interval_sec=self.min_interval_sec,
                output_dir=self.output_dir,
                progress_callback=lambda p: self.progress_signal.emit(p),
                cancel_event=self.cancel_event
            )
            self.finished_signal.emit(results)
        except Exception as e:
            self.error_signal.emit(str(e))

    def cancel(self):
        self.cancel_event.set()

class VideoScanDialog(QDialog):
    scan_completed_signal = pyqtSignal(list)

    def __init__(self, parent=None, config: Optional[AppConfig] = None):
        super().__init__(parent)
        self.config = config or AppConfig()
        self.worker: Optional[VideoScanWorker] = None
        self.video_info: Optional[VideoInfo] = None
        self.scanned_results: List[CapturedScreenshot] = []

        self.setWindowTitle("Scan Video Recording — Best Thumbnail Picker")
        self.resize(800, 680)
        self.setStyleSheet(DARK_STYLESHEET)

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header
        lbl_header = QLabel("📼 Scan Video Recording for Best Thumbnails")
        lbl_header.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        lbl_header.setStyleSheet("color: #a78bfa;")
        layout.addWidget(lbl_header)

        lbl_sub = QLabel("Select any pre-recorded video (MP4, MOV, MKV, AVI) to analyze head pose, eye openness, expressions, and posture to extract the highest-quality shots.")
        lbl_sub.setStyleSheet("color: #94a3b8; font-size: 12px;")
        lbl_sub.setWordWrap(True)
        layout.addWidget(lbl_sub)

        # 1. Video File Selection Group
        grp_file = QGroupBox("Video File")
        grp_file_layout = QVBoxLayout(grp_file)

        file_row = QHBoxLayout()
        self.txt_video_path = QLineEdit()
        self.txt_video_path.setPlaceholderText("Select a video file to scan...")
        self.txt_video_path.textChanged.connect(self._on_video_path_changed)
        btn_browse = QPushButton("📂 Browse Video...")
        btn_browse.clicked.connect(self._on_browse_video)
        file_row.addWidget(self.txt_video_path)
        file_row.addWidget(btn_browse)
        grp_file_layout.addLayout(file_row)

        self.lbl_video_info = QLabel("No video selected. Select a file above.")
        self.lbl_video_info.setStyleSheet("color: #38bdf8; font-size: 12px; font-weight: 500;")
        grp_file_layout.addWidget(self.lbl_video_info)
        layout.addWidget(grp_file)

        # 2. Settings Grid Group
        grp_settings = QGroupBox("Scan Configuration")
        grid_settings = QGridLayout(grp_settings)
        grid_settings.setVerticalSpacing(8)
        grid_settings.setHorizontalSpacing(12)

        # Number of Top Thumbnails
        grid_settings.addWidget(QLabel("Top Best Thumbnails to Pick:"), 0, 0)
        self.spn_top_n = QSpinBox()
        self.spn_top_n.setRange(1, 50)
        self.spn_top_n.setValue(5)
        self.spn_top_n.setSuffix(" Thumbnails")
        grid_settings.addWidget(self.spn_top_n, 0, 1)

        # Sampling Speed
        grid_settings.addWidget(QLabel("Scanning Speed / Rate:"), 0, 2)
        self.cmb_speed = QComboBox()
        self.cmb_speed.addItem("Fast (5 FPS sampling — Recommended)", 5.0)
        self.cmb_speed.addItem("Ultra-Fast (2 FPS sampling)", 2.0)
        self.cmb_speed.addItem("High Precision (10 FPS sampling)", 10.0)
        self.cmb_speed.addItem("Exhaustive (Every Frame)", 0.0)
        grid_settings.addWidget(self.cmb_speed, 0, 3)

        # Quality Threshold Slider
        grid_settings.addWidget(QLabel("Min Quality Threshold:"), 1, 0)
        self.lbl_min_score = QLabel("55%")
        self.sld_min_score = QSlider(Qt.Orientation.Horizontal)
        self.sld_min_score.setRange(30, 95)
        self.sld_min_score.setValue(55)
        self.sld_min_score.valueChanged.connect(lambda v: self.lbl_min_score.setText(f"{v}%"))
        score_layout = QHBoxLayout()
        score_layout.addWidget(self.sld_min_score)
        score_layout.addWidget(self.lbl_min_score)
        grid_settings.addLayout(score_layout, 1, 1)

        # Time Interval / Diversity Separation
        grid_settings.addWidget(QLabel("Min Time Gap (Diversity):"), 1, 2)
        self.lbl_interval = QLabel("2.0s")
        self.sld_interval = QSlider(Qt.Orientation.Horizontal)
        self.sld_interval.setRange(5, 100) # 0.5s to 10.0s
        self.sld_interval.setValue(20)
        self.sld_interval.valueChanged.connect(lambda v: self.lbl_interval.setText(f"{v/10.0:.1f}s"))
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(self.sld_interval)
        interval_layout.addWidget(self.lbl_interval)
        grid_settings.addLayout(interval_layout, 1, 3)

        # Output folder
        grid_settings.addWidget(QLabel("Output Folder:"), 2, 0)
        self.txt_output_dir = QLineEdit(self.config.output_dir)
        btn_browse_out = QPushButton("Browse...")
        btn_browse_out.clicked.connect(self._on_browse_output)
        out_row = QHBoxLayout()
        out_row.addWidget(self.txt_output_dir)
        out_row.addWidget(btn_browse_out)
        grid_settings.addLayout(out_row, 2, 1, 1, 3)

        layout.addWidget(grp_settings)

        # 3. Live Progress & Preview Group
        grp_progress = QGroupBox("Scan Progress & Live Preview")
        v_prog_layout = QVBoxLayout(grp_progress)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        v_prog_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Ready to scan.")
        self.lbl_status.setStyleSheet("color: #cbd5e1; font-weight: 500; font-size: 12px;")
        v_prog_layout.addWidget(self.lbl_status)

        # Preview frames side-by-side
        h_previews = QHBoxLayout()
        
        # Left preview: Current Frame
        v_curr = QVBoxLayout()
        lbl_curr_title = QLabel("Current Frame Scanning")
        lbl_curr_title.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold;")
        lbl_curr_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview_curr = QLabel("Video Preview")
        self.lbl_preview_curr.setFixedSize(360, 202)
        self.lbl_preview_curr.setStyleSheet("background-color: #020617; border: 1px solid #334155; border-radius: 6px;")
        self.lbl_preview_curr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v_curr.addWidget(lbl_curr_title)
        v_curr.addWidget(self.lbl_preview_curr)
        h_previews.addLayout(v_curr)

        # Right preview: Champion Best Thumbnail
        v_champ = QVBoxLayout()
        lbl_champ_title = QLabel("⭐ Champion Best Shot Found")
        lbl_champ_title.setStyleSheet("color: #f59e0b; font-size: 11px; font-weight: bold;")
        lbl_champ_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview_champ = QLabel("Best Shot Preview")
        self.lbl_preview_champ.setFixedSize(360, 202)
        self.lbl_preview_champ.setStyleSheet("background-color: #020617; border: 2px solid #f59e0b; border-radius: 6px;")
        self.lbl_preview_champ.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v_champ.addWidget(lbl_champ_title)
        v_champ.addWidget(self.lbl_preview_champ)
        h_previews.addLayout(v_champ)

        v_prog_layout.addLayout(h_previews)
        layout.addWidget(grp_progress)

        # 4. Action Buttons Bar
        btn_bar = QHBoxLayout()
        self.btn_start = QPushButton("▶  Start Scan")
        self.btn_start.setStyleSheet("background-color: #7c3aed; font-size: 14px; font-weight: bold; padding: 10px 24px;")
        self.btn_start.clicked.connect(self._on_toggle_scan)
        btn_bar.addWidget(self.btn_start)

        self.btn_open_out = QPushButton("📁 Open Output Folder")
        self.btn_open_out.clicked.connect(self._on_open_output)
        btn_bar.addWidget(self.btn_open_out)

        btn_bar.addStretch()

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        btn_bar.addWidget(self.btn_close)

        layout.addLayout(btn_bar)

    def _on_browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File to Scan", "",
            "Video Files (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.ts *.wmv *.flv);;All Files (*.*)"
        )
        if file_path:
            self.txt_video_path.setText(file_path)

    def _on_browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.txt_output_dir.text())
        if folder:
            self.txt_output_dir.setText(folder)

    def _on_open_output(self):
        folder = os.path.abspath(self.txt_output_dir.text().strip())
        os.makedirs(folder, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _on_video_path_changed(self, text: str):
        path = text.strip()
        if os.path.exists(path) and os.path.isfile(path):
            info = VideoRecordingScanner.get_video_info(path)
            if info:
                self.video_info = info
                dur_str = VideoRecordingScanner.format_time_display(info.duration_sec)
                self.lbl_video_info.setText(
                    f"✓ <b>{info.filename}</b> — Duration: <b>{dur_str}</b> ({info.duration_sec:.1f}s) | "
                    f"Resolution: <b>{info.width}x{info.height}</b> | FPS: <b>{info.fps:.1f}</b> | Total Frames: <b>{info.total_frames:,}</b>"
                )
                self.lbl_video_info.setStyleSheet("color: #4ade80; font-size: 12px;")
                return
        self.video_info = None
        self.lbl_video_info.setText("No valid video selected.")
        self.lbl_video_info.setStyleSheet("color: #94a3b8; font-size: 12px;")

    def _on_toggle_scan(self):
        if self.worker and self.worker.isRunning():
            # Stop scan
            self.btn_start.setEnabled(False)
            self.lbl_status.setText("Cancelling scan...")
            self.worker.cancel()
        else:
            # Start scan
            v_path = self.txt_video_path.text().strip()
            if not v_path or not os.path.exists(v_path):
                QMessageBox.warning(self, "Select Video", "Please select a valid video file first.")
                return

            out_dir = self.txt_output_dir.text().strip()
            if not out_dir:
                out_dir = self.config.output_dir

            top_n = self.spn_top_n.value()
            min_score = self.sld_min_score.value() / 100.0
            sample_fps = float(self.cmb_speed.currentData())
            min_interval_sec = self.sld_interval.value() / 10.0

            self.btn_start.setText("⏹  Stop Scan")
            self.btn_start.setStyleSheet("background-color: #dc2626; font-size: 14px; font-weight: bold; padding: 10px 24px;")
            self.progress_bar.setValue(0)
            self.lbl_status.setText("Initializing scanner...")
            self.lbl_preview_curr.setText("Scanning...")
            self.lbl_preview_champ.setText("Finding Best Shot...")

            self.worker = VideoScanWorker(
                video_path=v_path,
                top_n=top_n,
                min_score=min_score,
                sample_fps=sample_fps,
                min_interval_sec=min_interval_sec,
                output_dir=out_dir,
                config=self.config
            )
            self.worker.progress_signal.connect(self._on_scan_progress)
            self.worker.finished_signal.connect(self._on_scan_finished)
            self.worker.error_signal.connect(self._on_scan_error)
            self.worker.start()

    def _on_scan_progress(self, p: ScanProgress):
        self.progress_bar.setValue(int(p.progress_pct))

        curr_time = VideoRecordingScanner.format_time_display(p.current_time_sec)
        tot_time = VideoRecordingScanner.format_time_display(p.total_duration_sec)
        eta_str = f"{int(p.estimated_time_remaining_sec)}s" if p.estimated_time_remaining_sec > 0 else "0s"
        speed_str = f"{p.fps_speed:.1f} FPS"
        
        best_str = f"Best Score: {int(p.best_score_found * 100)}%" if p.best_score_found > 0 else "No qualified face yet"
        status_text = (
            f"Scanning: <b>{curr_time} / {tot_time}</b> ({p.progress_pct:.0f}%) • "
            f"Speed: <b>{speed_str}</b> • ETA: <b>{eta_str}</b> • {best_str}"
        )
        self.lbl_status.setText(status_text)

        # Update preview frames
        if p.current_frame is not None:
            self._set_label_pixmap(self.lbl_preview_curr, p.current_frame)
            
        if p.champion_frame is not None:
            self._set_label_pixmap(self.lbl_preview_champ, p.champion_frame)

    def _set_label_pixmap(self, label: QLabel, frame_bgr: np.ndarray):
        h, w, c = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * c, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        label.setPixmap(pix.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))

    def _on_scan_finished(self, results: List[CapturedScreenshot]):
        self.scanned_results = results
        self.btn_start.setText("▶  Start Scan")
        self.btn_start.setStyleSheet("background-color: #7c3aed; font-size: 14px; font-weight: bold; padding: 10px 24px;")
        self.btn_start.setEnabled(True)
        self.progress_bar.setValue(100)

        count = len(results)
        self.lbl_status.setText(f"✓ <b>Scan Complete!</b> Successfully extracted <b>{count}</b> best thumbnail{'s' if count != 1 else ''}.")
        self.lbl_status.setStyleSheet("color: #4ade80; font-weight: bold; font-size: 13px;")

        self.scan_completed_signal.emit(results)

        if count > 0:
            best = results[0]
            QMessageBox.information(
                self, "Video Scan Finished",
                f"Successfully extracted {count} best thumbnails from the video recording!\n\n"
                f"• Top Champion Score: {int(best.overall_score * 100)}%\n"
                f"• Saved To: {self.txt_output_dir.text()}\n\n"
                f"The thumbnails have been imported into the main gallery."
            )
        else:
            QMessageBox.warning(
                self, "Scan Complete",
                "Scan completed, but no frames met the minimum quality threshold.\n"
                "Try lowering the Min Quality Threshold or checking the video."
            )

    def _on_scan_error(self, err_msg: str):
        self.btn_start.setText("▶  Start Scan")
        self.btn_start.setStyleSheet("background-color: #7c3aed; font-size: 14px; font-weight: bold; padding: 10px 24px;")
        self.btn_start.setEnabled(True)
        self.lbl_status.setText(f"Scan failed: {err_msg}")
        self.lbl_status.setStyleSheet("color: #ef4444; font-weight: bold;")
        QMessageBox.critical(self, "Scan Error", f"Error scanning video:\n{err_msg}")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(1000)
        event.accept()

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

        btn_scan_video = QPushButton("📼 Scan Video File...")
        btn_scan_video.setObjectName("btn_scan_video")
        btn_scan_video.clicked.connect(self._on_open_video_scanner)
        
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
        top_bar.addWidget(btn_scan_video)
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
        self.chk_auto_capture.setChecked(False)
        self.chk_auto_capture.toggled.connect(self._on_toggle_auto_capture)
        
        self.chk_single_winner = QCheckBox("🏆 Keep Top")
        self.chk_single_winner.setChecked(self.config.single_winner_mode)
        self.chk_single_winner.toggled.connect(self._on_toggle_single_winner)

        self.spn_max_retained = QSpinBox()
        self.spn_max_retained.setRange(1, 200)
        self.spn_max_retained.setValue(int(self.config.max_retained_captures))
        self.spn_max_retained.setSuffix(" Shots")
        self.spn_max_retained.setToolTip("Number of top auto-captured shots to keep (pinned 📌 shots are always saved)")
        self.spn_max_retained.setFixedWidth(92)
        self.spn_max_retained.valueChanged.connect(self._on_max_retained_changed)

        self.chk_prop_moments = QCheckBox("🎭 Capture Stage Props")
        self.chk_prop_moments.setChecked(getattr(self.config, "capture_prop_moments", True))
        self.chk_prop_moments.toggled.connect(self._on_toggle_prop_moments)
        
        btn_manual_capture = QPushButton("📸 Manual Screenshot")
        btn_manual_capture.setObjectName("btn_capture")
        btn_manual_capture.clicked.connect(self._on_manual_capture)
        
        hud_bar.addWidget(self.chk_mesh)
        hud_bar.addWidget(self.chk_hud)
        hud_bar.addWidget(self.chk_auto_capture)
        hud_bar.addWidget(self.chk_single_winner)
        hud_bar.addWidget(self.spn_max_retained)
        hud_bar.addWidget(self.chk_prop_moments)
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
        self.lbl_centering = QLabel("Cutout: --")
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
            
            hand_status = (
                "OBSTRUCTING ⚠️" if analysis.hands_covering_face
                else ("BLURRY 🔄" if analysis.is_hand_blurry
                      else ("🎭 STAGE PROP / CANDID" if getattr(analysis, "is_prop_moment", False) else "CLEAR ✅"))
            )
            self.lbl_hand.setText(f"Hand: {hand_status}")
            
            # New metrics
            exp_pct = int(analysis.exposure_score * 100)
            exp_icon = "☀️" if analysis.face_brightness > 180 else ("🌑" if analysis.face_brightness < 60 else "✅")
            self.lbl_exposure.setText(f"Exposure: {exp_pct}% {exp_icon} ({analysis.face_brightness:.0f})")
            
            cut_pct = int(analysis.cutout_score * 100)
            cut_flags = []
            if analysis.is_subject_clipped:
                cut_flags.append("CLIPPED ⚠️")
            if analysis.hands_on_torso:
                cut_flags.append("hands on body")
            suffix = (" · " + " · ".join(cut_flags)) if cut_flags else ""
            self.lbl_centering.setText(
                f"Cutout: {cut_pct}% · subject {int(analysis.subject_height_ratio * 100)}% of frame"
                f" · bg {int(analysis.background_score * 100)}%{suffix}")
            
            stab_icon = "🟢 Stable" if analysis.is_stable else "🟡 Unstable"
            self.lbl_stability.setText(f"Score: {stab_icon}")
            
            # Live AI Model Telemetry Readout
            if self.processor and self.processor.analyzer.preference_model.is_trained:
                pm = self.processor.analyzer.preference_model
                self.lbl_ai_model.setText(
                    f"AI Model: {len(pm.samples)} ratings | {pm.confidence*100:.0f}% balanced accuracy"
                    f" | influence {getattr(analysis, 'ml_weight_applied', 0.0)*100:.0f}%"
                    + (f" | ML Score: {int(analysis.ml_score*100)}%" if pm.trust > 0
                       else " | no better than guessing, ignored"))
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
            self.lbl_centering.setText("Cutout: --")
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
        if self.processor:
            self.processor.config.single_winner_mode = checked
            if checked:
                self.processor.prune_to_best_n(self.config.max_retained_captures)
            self._refresh_gallery_display()

    def _on_max_retained_changed(self, val: int):
        n = max(1, min(200, int(val)))
        self.config.max_retained_captures = n
        if self.processor:
            self.processor.config.max_retained_captures = n
            if self.config.single_winner_mode:
                self.processor.prune_to_best_n(n)
            self._refresh_gallery_display()

    def _on_toggle_prop_moments(self, checked: bool):
        self.config.capture_prop_moments = checked
        if self.processor:
            self.processor.config.capture_prop_moments = checked
            self.processor.analyzer.capture_prop_moments = checked

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
                pinned_or_scan = [c for c in captures if getattr(c, "is_pinned", False) or getattr(c, "from_video_scan", False)]
                unpinned_live = [c for c in captures if not getattr(c, "is_pinned", False) and not getattr(c, "from_video_scan", False)]
                captures = sorted(
                    pinned_or_scan + unpinned_live[:self.config.max_retained_captures],
                    key=lambda c: c.overall_score,
                    reverse=True,
                )
                
            for idx, cap in enumerate(captures):
                card = ThumbnailCard(
                    cap, rank=idx+1, 
                    on_click_callback=self._on_thumbnail_clicked,
                    on_favorite_callback=self._on_card_favorite,
                    on_reject_callback=self._on_card_reject,
                    on_pin_callback=self._on_card_pin
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
            + (f"<b>AI judge:</b> {int(capture.judge_score * 100)}% "
               f"(analyzer {int(capture.heuristic_score * 100)}%)<br><i>{capture.judge_summary}</i><br>"
               + (f"<b>Flagged:</b> {'; '.join(capture.judge_issues)}<br>" if capture.judge_issues else "")
               if getattr(capture, "judge_score", None) is not None else "")
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
        if self._is_capturing != checked:
            self._on_start_stop()

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

    def _on_card_pin(self, capture: CapturedScreenshot):
        """User clicked 📌 Keep / 📌 Saved on a thumbnail card -> toggle deletion protection."""
        if self.processor:
            self.processor.set_capture_pinned(capture.filepath, None)
        else:
            capture.is_pinned = not getattr(capture, "is_pinned", False)
        self._refresh_gallery_display()

    def _on_card_favorite(self, capture: CapturedScreenshot):
        """User clicked ⭐ Best on a thumbnail card -> train Scikit-Learn preference model and pin."""
        capture.is_pinned = True
        if self.processor and self.processor.analyzer:
            self.processor.analyzer.preference_model.add_sample(capture.analysis, is_favorite=True)
            pm = self.processor.analyzer.preference_model
            self._refresh_gallery_display()
            QMessageBox.information(
                self, "AI Model Trained ⭐",
                f"Trained AI Preference Model & Pinned Shot!\n\n"
                f"• Learned that this shot is a Favorite Winner.\n"
                f"• Flagged (📌 Saved) so it will not be auto-deleted.\n"
                f"• Total Training Samples: {len(pm.samples)}\n"
                f"• Scikit-Learn Model Fit: {pm.confidence * 100:.0f}%\n\n"
                f"The AI will now prioritize shots with similar features!"
            )
            
    def _on_card_reject(self, capture: CapturedScreenshot):
        """User clicked 👎 Bad on a thumbnail card -> prompt for reason/comment and train AI."""
        if not (self.processor and self.processor.analyzer):
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("👎 Teach AI: Why is this photo bad?")
        dlg.setMinimumWidth(460)
        dlg.setStyleSheet(DARK_STYLE)
        v_lay = QVBoxLayout(dlg)

        lbl_intro = QLabel("Select any issues and/or add a comment so the heuristic scorer, ML model, and AI Judge learn to reject shots like this:")
        lbl_intro.setWordWrap(True)
        lbl_intro.setStyleSheet("color: #cbd5e1; font-size: 12px;")
        v_lay.addWidget(lbl_intro)

        reason_options = [
            "Head bowed / looking down",
            "Eyes closed / hidden",
            "Mouth open / mid-word",
            "Mic / hand blocking face",
            "Not a real prop / cut off",
            "Blurry / motion blur",
        ]
        reason_checks = []
        for text in reason_options:
            cb = QCheckBox(text)
            if "prop" in text.lower() and getattr(capture, "is_prop_moment", False):
                cb.setChecked(True)
            v_lay.addWidget(cb)
            reason_checks.append(cb)

        txt_comment = QLineEdit()
        txt_comment.setPlaceholderText("Optional comment (e.g. 'Looking down into box, mic covering mouth')...")
        v_lay.addWidget(txt_comment)

        chk_remove = QCheckBox("Delete this photo from gallery")
        chk_remove.setChecked(True)
        chk_remove.setStyleSheet("color: #f87171;")
        v_lay.addWidget(chk_remove)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(dlg.reject)
        btn_submit = QPushButton("👎 Teach AI && Reject")
        btn_submit.setStyleSheet("background-color: #991b1b; color: #fecaca; font-weight: bold;")
        btn_submit.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_submit)
        v_lay.addLayout(btn_row)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        selected_reasons = [cb.text() for cb in reason_checks if cb.isChecked()]
        comment = txt_comment.text().strip()
        pm = self.processor.analyzer.preference_model
        pm.add_sample(capture.analysis, is_favorite=False, comment=comment, reasons=selected_reasons)

        if chk_remove.isChecked():
            self.processor.remove_capture(capture.filepath)
            self._refresh_gallery_display()

        QMessageBox.information(
            self, "AI Model Trained 👎",
            f"Trained AI Preference Model & Updated Rules!\n\n"
            f"• Reasons recorded: {', '.join(selected_reasons) if selected_reasons else 'General rejection'}\n"
            + (f"• Comment rule saved: \"{comment}\"\n" if comment else "")
            + f"• Active comment rules: {len(pm.feedback_notes)}\n"
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
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

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

    def _on_open_video_scanner(self):
        """Open the dedicated Video Recording Scanner dialog."""
        dlg = VideoScanDialog(self, config=self.config)
        dlg.scan_completed_signal.connect(self._on_video_scan_completed)
        dlg.exec()

    def _on_video_scan_completed(self, results: List[CapturedScreenshot]):
        """Called when video scan finishes and extracts screenshots."""
        if not results:
            return
        if self.processor:
            with self.processor.captures_lock:
                for item in results:
                    # Avoid duplicate filepath entries
                    if not any(c.filepath == item.filepath for c in self.processor.saved_captures):
                        self.processor.saved_captures.append(item)
            self._refresh_gallery_display()

    def _on_start_stop(self):
        """Toggle auto-capture on/off. The video feed stays connected."""
        if self._is_capturing:
            # STOP CAPTURING
            if self.processor:
                self.processor.auto_capture_enabled = False
            self._is_capturing = False
            self.chk_auto_capture.blockSignals(True)
            self.chk_auto_capture.setChecked(False)
            self.chk_auto_capture.blockSignals(False)
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
            self.chk_auto_capture.blockSignals(True)
            self.chk_auto_capture.setChecked(True)
            self.chk_auto_capture.blockSignals(False)
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
