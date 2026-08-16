import sys
import os

# Suppress MediaPipe C++ telemetry & warning messages
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from gui import MainWindow
from config import AppConfig

def main():
    parser = argparse.ArgumentParser(description="Sony SRG-A40 Intelligent Thumbnail & Best Screenshot Grabber")
    parser.add_argument("--source", type=str, help="Video source (RTSP URL, webcam index '0', or MP4 file path)")
    parser.add_argument("--output", type=str, help="Output directory for captured screenshots")
    args = parser.parse_args()

    # Enable High DPI scaling for crisp UI rendering
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Sony SRG-A40 Best Screenshot Grabber")

    window = MainWindow()

    # Apply command-line overrides if specified
    if args.source:
        window.txt_source.setText(args.source)
        window._on_apply_source()
    if args.output:
        window.config.output_dir = args.output

    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
