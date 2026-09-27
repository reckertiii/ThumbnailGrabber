import sys
import os

# Suppress MediaPipe C++ telemetry & warning messages
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from config import AppConfig
from video_scanner import VideoRecordingScanner, ScanProgress

def run_headless_scan(video_path: str, top_n: int, min_score: float, sample_fps: float,
                      min_interval_sec: float, output_dir: str):
    print(f"\n=======================================================")
    print(f"   Thumbnail Grabber - Video Recording Scanner (CLI)   ")
    print(f"=======================================================\n")
    print(f"Input Video: {video_path}")
    
    info = VideoRecordingScanner.get_video_info(video_path)
    if not info:
        print(f"Error: Could not open or read video file '{video_path}'")
        sys.exit(1)
        
    dur_str = VideoRecordingScanner.format_time_display(info.duration_sec)
    print(f"Video Info : {info.filename} | Duration: {dur_str} ({info.duration_sec:.1f}s) | "
          f"{info.width}x{info.height} @ {info.fps:.1f} FPS | Total: {info.total_frames:,} frames")
    print(f"Settings   : Top {top_n} thumbnails | Min Score: {min_score:.0%} | "
          f"Sample Rate: {sample_fps:.1f} FPS | Diversity Gap: {min_interval_sec:.1f}s")
    print(f"Output Dir : {output_dir}\n")

    scanner = VideoRecordingScanner()
    
    last_print = 0.0
    def print_progress(p: ScanProgress):
        nonlocal last_print
        now = time.time()
        if now - last_print >= 0.25 or p.status in ("completed", "cancelled", "failed"):
            last_print = now
            curr_str = VideoRecordingScanner.format_time_display(p.current_time_sec)
            tot_str = VideoRecordingScanner.format_time_display(p.total_duration_sec)
            bar_len = 30
            filled = int(bar_len * (p.progress_pct / 100.0))
            bar = "=" * filled + "-" * (bar_len - filled)
            best_str = f"{p.best_score_found:.0%}" if p.best_score_found > 0 else "--"
            sys.stdout.write(f"\r[{bar}] {p.progress_pct:5.1f}% | {curr_str}/{tot_str} | {p.fps_speed:5.1f} FPS | ETA: {int(p.estimated_time_remaining_sec)}s | Best: {best_str}   ")
            sys.stdout.flush()

    import time
    start_t = time.time()
    results = scanner.scan_video(
        video_path=video_path,
        top_n=top_n,
        min_score=min_score,
        sample_fps=sample_fps,
        min_interval_sec=min_interval_sec,
        output_dir=output_dir,
        progress_callback=print_progress
    )
    print("\n")
    elapsed = time.time() - start_t
    print(f"Scan finished in {elapsed:.1f}s! Extracted {len(results)} thumbnails:")
    for idx, r in enumerate(results, 1):
        print(f"  #{idx} [Score: {r.overall_score:.0%}] Timestamp: {r.timestamp_str} -> {r.filepath}")
    print()

def main():
    parser = argparse.ArgumentParser(description="Sony SRG-A40 Intelligent Thumbnail & Best Screenshot Grabber")
    parser.add_argument("--source", type=str, help="Video source (RTSP URL, webcam index '0', or MP4 file path)")
    parser.add_argument("--output", type=str, help="Output directory for captured screenshots")
    
    # Video file scanner arguments
    parser.add_argument("--scan-video", type=str, help="Scan a recorded video file and extract top thumbnails")
    parser.add_argument("--top", type=int, default=5, help="Number of best thumbnails to extract (default: 5)")
    parser.add_argument("--sample-fps", type=float, default=5.0, help="Frame sampling rate in FPS (default: 5.0, 0 for all)")
    parser.add_argument("--min-score", type=float, default=0.55, help="Minimum quality score threshold (default: 0.55)")
    parser.add_argument("--min-interval", type=float, default=2.0, help="Minimum time gap between thumbnails in seconds (default: 2.0)")
    parser.add_argument("--headless", action="store_true", help="Run in console/headless mode without launching GUI")
    
    args = parser.parse_args()

    # If headless scan is requested
    if args.scan_video and (args.headless or "DISPLAY" not in os.environ and sys.platform != "win32"):
        cfg = AppConfig()
        out = args.output or cfg.output_dir
        run_headless_scan(
            video_path=args.scan_video,
            top_n=args.top,
            min_score=args.min_score,
            sample_fps=args.sample_fps,
            min_interval_sec=args.min_interval,
            output_dir=out
        )
        return

    # Desktop GUI Mode
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from gui import MainWindow

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

    # If --scan-video was passed with GUI, open the scanner dialog pre-loaded with the video
    if args.scan_video:
        window.show()
        from gui import VideoScanDialog
        dlg = VideoScanDialog(window, config=window.config)
        dlg.txt_video_path.setText(args.scan_video)
        dlg.spn_top_n.setValue(args.top)
        dlg.sld_min_score.setValue(int(args.min_score * 100))
        dlg.scan_completed_signal.connect(window._on_video_scan_completed)
        dlg.exec()
    else:
        window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
