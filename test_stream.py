import unittest
import os
import shutil
import tempfile
import time
import cv2
import numpy as np

from config import AppConfig
from analyzer import PersonQualityAnalyzer, AnalysisResult
from stream_processor import StreamProcessor, CapturedScreenshot

class TestStreamProcessor(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.config = AppConfig()
        self.config.output_dir = self.tmp_dir
        self.config.ai_judge_backend = "off"
        self.config.dvr_buffer_seconds = 0.0
        self.captured_items = []
        
        self.processor = StreamProcessor(
            config=self.config,
            on_capture_callback=lambda cap: self.captured_items.append(cap)
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_screenshot(self):
        # Create test frame (640x480)
        test_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        analysis = AnalysisResult(
            face_detected=True,
            overall_score=0.85,
            is_qualified=True,
            yaw_deg=2.5,
            avg_ear=0.35,
            smile_ratio=0.80,
            laplacian_variance=120.0
        )
        
        cap_item = self.processor.save_screenshot(test_frame, analysis)
        self.assertTrue(os.path.exists(cap_item.filepath), "Saved screenshot file should exist on disk")
        self.assertGreater(os.path.getsize(cap_item.filepath), 1000, "Screenshot file size should be > 1KB")

    def test_top_5_pruning(self):
        self.config.single_winner_mode = True
        self.config.max_retained_captures = 5
        
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        
        # Save 6 shots with scores 0.60, 0.65, 0.70, 0.75, 0.80, 0.85
        captures = []
        for s in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
            a = AnalysisResult(face_detected=True, overall_score=s, is_qualified=True)
            cap = self.processor.save_screenshot(frame, a)
            captures.append(cap)
            
        # Lowest score (0.60) should be pruned, top 5 (0.65 - 0.85) preserved
        self.assertFalse(os.path.exists(captures[0].filepath), "Lowest scoring 6th shot (0.60) should be pruned")
        self.assertTrue(os.path.exists(captures[-1].filepath), "Highest scoring shot (0.85) should be preserved")
        self.assertEqual(len(self.processor.saved_captures), 5, "Exactly Top 5 screenshots should remain in memory and on disk")

if __name__ == "__main__":
    unittest.main()
