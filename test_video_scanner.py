import unittest
import os
import tempfile
import cv2
import numpy as np

from config import AppConfig
from analyzer import AnalysisResult
from video_scanner import VideoRecordingScanner, ScannedCandidate, VideoInfo, ScanProgress

class TestVideoRecordingScanner(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig()
        self.scanner = VideoRecordingScanner(self.config)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temporary test files
        for f in os.listdir(self.temp_dir):
            try:
                os.remove(os.path.join(self.temp_dir, f))
            except Exception:
                pass
        try:
            os.rmdir(self.temp_dir)
        except Exception:
            pass

    def _create_synthetic_video(self, duration_sec: float = 3.0, fps: float = 10.0) -> str:
        """Helper to create a temporary dummy video file."""
        video_path = os.path.join(self.temp_dir, "test_recording.mp4")
        width, height = 320, 240
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        total_frames = int(duration_sec * fps)
        for i in range(total_frames):
            # Create a frame with some varying content
            frame = np.full((height, width, 3), (i * 5) % 255, dtype=np.uint8)
            cv2.putText(frame, f"Frame {i}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            out.write(frame)
            
        out.release()
        return video_path

    def test_timestamp_formatting(self):
        """Test timestamp formatting functions."""
        self.assertEqual(VideoRecordingScanner.format_timestamp(0.0), "00m00s000ms")
        self.assertEqual(VideoRecordingScanner.format_timestamp(74.35), "01m14s350ms")
        self.assertEqual(VideoRecordingScanner.format_timestamp(3665.123), "01h01m05s123ms")

        self.assertEqual(VideoRecordingScanner.format_time_display(0.0), "00:00")
        self.assertEqual(VideoRecordingScanner.format_time_display(74.0), "01:14")
        self.assertEqual(VideoRecordingScanner.format_time_display(3665.0), "01:01:05")

    def test_get_video_info(self):
        """Test reading technical metadata from a video file."""
        video_path = self._create_synthetic_video(duration_sec=2.0, fps=10.0)
        info = VideoRecordingScanner.get_video_info(video_path)
        
        self.assertIsNotNone(info)
        self.assertEqual(info.total_frames, 20)
        self.assertAlmostEqual(info.fps, 10.0, places=1)
        self.assertEqual(info.width, 320)
        self.assertEqual(info.height, 240)
        self.assertAlmostEqual(info.duration_sec, 2.0, places=1)

    def test_diverse_candidate_selection_nms(self):
        """Test that temporal non-maximum suppression spreads out thumbnail selections."""
        dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        
        candidates = [
            # Burst 1 at t=1.0s to 1.3s
            ScannedCandidate(frame_idx=10, timestamp_sec=1.0, timestamp_str="00m01s000ms", frame=dummy_frame, analysis=AnalysisResult(), overall_score=0.88),
            ScannedCandidate(frame_idx=11, timestamp_sec=1.1, timestamp_str="00m01s100ms", frame=dummy_frame, analysis=AnalysisResult(), overall_score=0.95), # Burst 1 peak
            ScannedCandidate(frame_idx=12, timestamp_sec=1.2, timestamp_str="00m01s200ms", frame=dummy_frame, analysis=AnalysisResult(), overall_score=0.90),
            
            # Burst 2 at t=5.0s to 5.2s
            ScannedCandidate(frame_idx=50, timestamp_sec=5.0, timestamp_str="00m05s000ms", frame=dummy_frame, analysis=AnalysisResult(), overall_score=0.82),
            ScannedCandidate(frame_idx=51, timestamp_sec=5.1, timestamp_str="00m05s100ms", frame=dummy_frame, analysis=AnalysisResult(), overall_score=0.91), # Burst 2 peak
            
            # Burst 3 at t=10.0s
            ScannedCandidate(frame_idx=100, timestamp_sec=10.0, timestamp_str="00m10s000ms", frame=dummy_frame, analysis=AnalysisResult(), overall_score=0.85),
        ]
        
        # Pick top 3 with min_interval_sec=2.0
        selected = self.scanner._select_diverse_top_candidates(candidates, top_n=3, min_interval_sec=2.0)
        
        self.assertEqual(len(selected), 3)
        # Should pick the highest from Burst 1 (0.95 at 1.1s), Burst 2 (0.91 at 5.1s), and Burst 3 (0.85 at 10.0s)
        scores = [c.overall_score for c in selected]
        self.assertEqual(scores, [0.95, 0.91, 0.85])
        timestamps = [c.timestamp_sec for c in selected]
        self.assertEqual(timestamps, [1.1, 5.1, 10.0])

    def test_scan_video_progress_and_execution(self):
        """Test running scan_video on a synthetic video file with progress tracking."""
        video_path = self._create_synthetic_video(duration_sec=2.0, fps=10.0)
        
        progress_updates = []
        def progress_cb(p: ScanProgress):
            progress_updates.append(p)
            
        results = self.scanner.scan_video(
            video_path=video_path,
            top_n=3,
            min_score=0.0,  # accept any score for test
            sample_fps=5.0,
            output_dir=self.temp_dir,
            progress_callback=progress_cb
        )
        
        self.assertGreater(len(progress_updates), 0)
        self.assertEqual(progress_updates[-1].status, "completed")
        self.assertEqual(progress_updates[-1].progress_pct, 100.0)

    def test_scan_video_cancellation(self):
        """Test that scanning can be stopped gracefully with cancel_event."""
        import threading
        video_path = self._create_synthetic_video(duration_sec=3.0, fps=10.0)
        cancel_event = threading.Event()
        
        progress_updates = []
        def progress_cb(p: ScanProgress):
            progress_updates.append(p)
            if p.current_frame_idx >= 5:
                cancel_event.set()
                
        results = self.scanner.scan_video(
            video_path=video_path,
            top_n=3,
            min_score=0.0,
            sample_fps=10.0,
            output_dir=self.temp_dir,
            progress_callback=progress_cb,
            cancel_event=cancel_event
        )
        
        self.assertGreater(len(progress_updates), 0)
        self.assertEqual(progress_updates[-1].status, "cancelled")

if __name__ == "__main__":
    unittest.main()
