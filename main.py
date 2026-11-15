"""Entrypoint for Thumbnail Grabber (Web UI server & CLI video recording scanner)."""
import os
import sys
import time
import argparse

# Suppress MediaPipe C++ telemetry & warning messages before importing CV modules
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from config import AppConfig
from video_scanner import VideoRecordingScanner, ScanProgress


def run_headless_scan(
    video_path: str,
    top_n: int,
    min_score: float,
    sample_fps: float,
    min_interval_sec: float,
    output_dir: str,
    start_sec: float = 0.0,
    end_sec: float = 0.0,
    capture_props: bool = True,
    use_ai_judge: bool = True,
):
    print("\n=======================================================")
    print("   Thumbnail Grabber - Video Recording Scanner (CLI)   ")
    print("=======================================================\n")
    print(f"Input Video: {video_path}")

    info = VideoRecordingScanner.get_video_info(video_path)
    if not info:
        print(f"Error: Could not open or read video file '{video_path}'")
        sys.exit(1)

    dur_str = VideoRecordingScanner.format_time_display(info.duration_sec)
    print(f"Video Info : {info.filename} | Duration: {dur_str} ({info.duration_sec:.1f}s) | "
          f"{info.width}x{info.height} @ {info.fps:.1f} FPS | Total: {info.total_frames:,} frames")
    trim_desc = ""
    if start_sec > 0 or (end_sec and end_sec > 0):
        end_lbl = VideoRecordingScanner.format_time_display(end_sec) if end_sec > 0 else dur_str
        trim_desc = f" | Trim: {VideoRecordingScanner.format_time_display(start_sec)} -> {end_lbl}"
    print(f"Settings   : Top {top_n} thumbnails | Min Score: {min_score:.0%} | "
          f"Sample Rate: {sample_fps:.1f} FPS | Diversity Gap: {min_interval_sec:.1f}s{trim_desc}")
    print(f"Output Dir : {output_dir}\n")

    cfg = AppConfig()
    cfg.output_dir = output_dir
    cfg.capture_prop_moments = capture_props
    if not use_ai_judge:
        cfg.ai_judge_enabled = False
        cfg.ai_judge_backend = "off"

    scanner = VideoRecordingScanner(cfg)

    last_print = 0.0

    def print_progress(p: ScanProgress):
        nonlocal last_print
        now = time.time()
        if now - last_print >= 0.25 or p.status in ("completed", "cancelled", "failed", "reviewing"):
            last_print = now
            curr_str = VideoRecordingScanner.format_time_display(p.current_time_sec)
            tot_str = VideoRecordingScanner.format_time_display(p.total_duration_sec)
            bar_len = 30
            filled = int(bar_len * (p.progress_pct / 100.0))
            bar = "=" * filled + "-" * (bar_len - filled)
            best_str = f"{p.best_score_found:.0%}" if p.best_score_found > 0 else "--"
            status_note = f" | {p.error_message}" if p.status == "reviewing" and p.error_message else ""
            sys.stdout.write(
                f"\r[{bar}] {p.progress_pct:5.1f}% | {curr_str}/{tot_str} | "
                f"{p.fps_speed:5.1f} FPS | ETA: {int(p.estimated_time_remaining_sec)}s | "
                f"Best: {best_str}{status_note}   "
            )
            sys.stdout.flush()

    start_t = time.time()
    results = scanner.scan_video(
        video_path=video_path,
        top_n=top_n,
        min_score=min_score,
        sample_fps=sample_fps,
        min_interval_sec=min_interval_sec,
        output_dir=output_dir,
        progress_callback=print_progress,
        start_sec=start_sec,
        end_sec=(end_sec if end_sec and end_sec > 0 else None),
    )
    print("\n")
    elapsed = time.time() - start_t
    print(f"Scan finished in {elapsed:.1f}s! Extracted {len(results)} thumbnails:")
    for idx, r in enumerate(results, 1):
        prop_lbl = " [Prop]" if getattr(r, "is_prop_moment", False) else ""
        print(f"  #{idx} [Score: {r.overall_score:.0%}]{prop_lbl} Timestamp: {r.timestamp_str} -> {r.filepath}")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thumbnail Grabber — Web Server & Video Recording Scanner")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"), help="Web server bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")), help="Web server port (default: 8000)")
    parser.add_argument("--source", type=str, help="Override video source (RTSP URL, webcam index '0', or MP4 file path)")
    parser.add_argument("--output", type=str, help="Output directory for captured screenshots")

    # Video file scanner CLI arguments
    parser.add_argument("--scan-video", type=str, help="Scan a recorded video file in CLI mode and extract top thumbnails")
    parser.add_argument("--top", type=int, default=5, help="Number of best thumbnails to extract (default: 5)")
    parser.add_argument("--sample-fps", type=float, default=5.0, help="Frame sampling rate in FPS (default: 5.0, 0 for all)")
    parser.add_argument("--min-score", type=float, default=0.55, help="Minimum quality score threshold (default: 0.55)")
    parser.add_argument("--min-interval", type=float, default=2.0, help="Minimum time gap between thumbnails in seconds (default: 2.0)")
    parser.add_argument("--start-sec", type=str, default="", help="Start timestamp in seconds or MM:SS / HH:MM:SS (default: 0)")
    parser.add_argument("--end-sec", type=str, default="", help="End timestamp in seconds or MM:SS / HH:MM:SS (default: full length)")
    parser.add_argument("--no-props", action="store_true", help="Disable stage prop detection during scan")
    parser.add_argument("--no-ai-judge", action="store_true", help="Disable AI Judge vision review during scan")
    parser.add_argument("--headless", action="store_true", help="Kept for backwards CLI compatibility (always headless)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.scan_video:
        cfg = AppConfig()
        out = args.output or cfg.output_dir
        start_s = VideoRecordingScanner.parse_timestamp_sec(args.start_sec) or 0.0
        end_s = VideoRecordingScanner.parse_timestamp_sec(args.end_sec) or 0.0
        run_headless_scan(
            video_path=args.scan_video,
            top_n=args.top,
            min_score=args.min_score,
            sample_fps=args.sample_fps,
            min_interval_sec=args.min_interval,
            output_dir=out,
            start_sec=max(0.0, start_s),
            end_sec=max(0.0, end_s),
            capture_props=not args.no_props,
            use_ai_judge=not args.no_ai_judge,
        )
        return

    # Launch the FastAPI Web UI server
    import uvicorn
    from web.server import app, state

    if args.source is not None:
        state.config.stream_source = args.source
    if args.output:
        state.config.output_dir = args.output

    print(f"\n  Thumbnail Grabber Web UI:  http://localhost:{args.port}")
    print("  On your LAN, use this machine's IP address with the same port.\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
