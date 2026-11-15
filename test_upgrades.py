"""Comprehensive unit tests for Phase 1, Phase 2, and Phase 3 upgrades:
  - Desktop GUI / PyQt6 removal & headless main.py CLI
  - RTSP standby polling, blank stream source, and Zero-CPU Idle Standby
  - Multi-person ignored-P1 crop/refine/scoring fixes
  - Horizontal eye-gaze, temporal mouth velocity, RGB LED-wash/blowout exposure,
    fast_reject MediaPipe early exit, and Smile Diversity slot reservation
  - Compressed Q95 JPEG DVR buffer round-trip & memory savings
  - Persistent UI settings & folder-scoped gallery/pin manifests (settings_store)
  - Video scanner start/end timestamp trimming
  - Web API security (TG_VIDEO_ROOTS, TG_WEB_PASSWORD Basic Auth), healthcheck,
    auto-wake AI Judge + Max Capture Window timer, video browser, and ZIP export
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from fastapi.testclient import TestClient

import main as main_entry
import settings_store
import web.server as ws
from analyzer import AnalysisResult, PersonQualityAnalyzer, SingleFaceAnalysis
from config import AppConfig
from stream_processor import CapturedScreenshot, StreamProcessor, subject_crop_rect
from video_scanner import VideoRecordingScanner


def _make_analyzer_stub() -> PersonQualityAnalyzer:
    """Lightweight analyzer instance without loading heavy MediaPipe .task models."""
    a = PersonQualityAnalyzer.__new__(PersonQualityAnalyzer)
    cfg = AppConfig()
    a.weights = cfg.weights
    a.thresholds = cfg.thresholds
    a.max_people_target = 2
    a.selected_person_ids = {1, 2}
    a.capture_prop_moments = True
    a._prev_mouth_state = {}
    a._smooth_bboxes = {}
    a._bbox_ema_alpha = 0.5
    a.pose_hand_interval = 1
    a._detail_tick = 0
    a._cached_hand_result = None
    a._cached_pose_result = None
    a.preference_model = mock.MagicMock()
    a.preference_model.learned_penalties = {}
    a.preference_model.is_trained = False
    a.preference_model.predict_score.return_value = 0.5
    a.preference_model.effective_weight.return_value = 0.0
    return a


class TestDesktopRemovalAndCLI(unittest.TestCase):
    def test_gui_removed_and_requirements_clean(self):
        root = Path(__file__).resolve().parent
        self.assertFalse((root / "gui.py").exists(), "gui.py should be deleted")
        reqs = (root / "requirements.txt").read_text(encoding="utf-8").lower()
        self.assertNotIn("pyqt", reqs)

    def test_main_cli_parser_supports_trim_and_server_flags(self):
        parser = main_entry.build_parser()
        args = parser.parse_args([
            "--scan-video", "sermon.mp4",
            "--start-sec", "12:30",
            "--end-sec", "45:00",
            "--top", "6",
            "--no-props",
            "--no-ai-judge",
        ])
        self.assertEqual(args.scan_video, "sermon.mp4")
        self.assertEqual(args.start_sec, "12:30")
        self.assertEqual(args.end_sec, "45:00")
        self.assertEqual(args.top, 6)
        self.assertTrue(args.no_props)
        self.assertTrue(args.no_ai_judge)


class TestScoringAndMultiPersonFixes(unittest.TestCase):
    def test_subject_crop_rect_uses_selected_person_when_p1_ignored(self):
        p1_ignored = SingleFaceAnalysis(
            person_id=1, bbox=(50, 100, 120, 150), is_selected=False, is_qualified=True
        )
        p2_selected = SingleFaceAnalysis(
            person_id=2, bbox=(900, 120, 130, 160), is_selected=True, is_qualified=True
        )
        res = AnalysisResult(
            face_detected=True,
            face_bbox=(50, 100, 120, 150),
            faces=[p1_ignored, p2_selected],
        )
        cx, cy, cw, ch = subject_crop_rect(res, 1920, 1080, padding=0.08)
        # Crop center should sit near P2 (~965), not P1 (~110)
        crop_center_x = cx + cw / 2.0
        self.assertGreater(crop_center_x, 700)

    def test_refine_analyzer_syncs_selection_and_fast_reject(self):
        from collections import deque

        proc = StreamProcessor.__new__(StreamProcessor)
        proc.config = AppConfig(
            capture_prop_moments=False,
            max_people_target=3,
            dvr_refine_window_sec=1.0,
            dvr_refine_candidates=2,
        )
        proc.analyzer = _make_analyzer_stub()
        proc.analyzer.selected_person_ids = {2}
        proc.analyzer.max_people_target = 3
        proc.analyzer.require_exact_people_count = False
        proc.analyzer.capture_prop_moments = False
        proc._refine_analyzer = None
        proc._dvr_lock = threading.Lock()
        f1 = np.full((240, 320, 3), 120, dtype=np.uint8)
        f2 = np.full((240, 320, 3), 125, dtype=np.uint8)
        proc._dvr = deque([(10.0, f1), (10.1, f2)], maxlen=30)

        p1 = SingleFaceAnalysis(person_id=1, bbox=(10, 20, 60, 60), is_selected=False)
        p2 = SingleFaceAnalysis(person_id=2, bbox=(180, 30, 60, 60), is_selected=True)
        chosen = AnalysisResult(face_detected=True, face_bbox=(10, 20, 60, 60), faces=[p1, p2], is_qualified=True, overall_score=0.80)

        fake_refine = _make_analyzer_stub()
        fake_refine.analyze_frame = mock.MagicMock(return_value=AnalysisResult(face_detected=False))
        with mock.patch("stream_processor.PersonQualityAnalyzer", return_value=fake_refine) as MockPQA:
            proc._refine_from_buffer(10.05, f1, chosen)
            _, kwargs = MockPQA.call_args
            self.assertTrue(kwargs.get("fast_reject"))
            self.assertEqual(kwargs.get("max_people_target"), 3)
            self.assertEqual(set(kwargs.get("selected_person_ids", [])), {2})

        # Changing live analyzer settings syncs onto existing _refine_analyzer on subsequent bursts
        proc.analyzer.max_people_target = 4
        proc.analyzer.selected_person_ids = {1, 3}
        proc._refine_from_buffer(10.05, f1, chosen)
        self.assertIs(proc._refine_analyzer.preference_model, proc.analyzer.preference_model)
        self.assertEqual(proc._refine_analyzer.max_people_target, 4)
        self.assertEqual(proc._refine_analyzer.selected_person_ids, {1, 3})
        self.assertFalse(proc._refine_analyzer.capture_prop_moments)

    def test_horizontal_eye_gaze_and_mouth_velocity_penalties(self):
        a = _make_analyzer_stub()

        def _base_face() -> SingleFaceAnalysis:
            return SingleFaceAnalysis(
                person_id=1,
                bbox=(400, 200, 200, 240),
                head_pose_score=0.95,
                eye_openness_score=0.95,
                expression_score=0.90,
                sharpness_score=0.90,
                exposure_score=0.90,
                centering_score=0.90,
                cutout_score=0.90,
                shoulder_pose_score=0.95,
                avg_ear=0.35,
                laplacian_variance=120.0,
                face_brightness=125.0,
                subject_height_ratio=0.55,
            )

        engaged = _base_face()
        a._finalize_face_score(engaged)

        gaze_off = _base_face()
        gaze_off.eye_gaze_away_score = 0.65
        a._finalize_face_score(gaze_off)
        self.assertLess(gaze_off.score, engaged.score - 0.10)

        mid_syllable = _base_face()
        mid_syllable.mouth_velocity = 0.25
        a._finalize_face_score(mid_syllable)
        self.assertLess(mid_syllable.score, engaged.score - 0.05)

    def test_exposure_penalizes_single_channel_rgb_led_wash_and_blowout(self):
        a = _make_analyzer_stub()
        # Normal warm skin tone (BGR)
        normal = np.full((80, 80, 3), (110, 130, 155), dtype=np.uint8)
        normal_gray = cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY)
        normal_score, _ = a._compute_exposure_score(normal, normal_gray)

        # Heavy blue/magenta stage LED wash with clipped blue channel (255),
        # yet grayscale mean stays inside [80, 180]
        led_wash = np.full((80, 80, 3), (255, 45, 160), dtype=np.uint8)
        led_gray = cv2.cvtColor(led_wash, cv2.COLOR_BGR2GRAY)
        led_score, _ = a._compute_exposure_score(led_wash, led_gray)

        self.assertEqual(normal_score, 1.0)
        self.assertLess(led_score, 0.70)

    def test_smile_diversity_reserves_slot_in_top_n_and_scanner(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = StreamProcessor.__new__(StreamProcessor)
            proc.config = AppConfig(
                output_dir=tmp,
                single_winner_mode=True,
                max_retained_captures=3,
                min_capture_spacing_sec=10.0,
                capture_prop_moments=True,
                max_prop_captures=1,
            )
            proc.captures_lock = threading.RLock()
            proc.persist_manifests = False
            proc.saved_captures = []

            def _mk_cap(idx: int, score: float, smiling: bool, ts: float) -> CapturedScreenshot:
                fp = os.path.join(tmp, f"shot_{idx}.png")
                Path(fp).write_bytes(b"png")
                ar = AnalysisResult(face_detected=True, overall_score=score, is_smiling=smiling)
                return CapturedScreenshot(
                    filepath=fp,
                    timestamp_str=f"000{idx}",
                    overall_score=score,
                    analysis=ar,
                    captured_at=ts,
                )

            # 3 serious shots (0.90, 0.88, 0.86) and 1 genuine smile (0.81)
            proc.saved_captures = [
                _mk_cap(1, 0.90, False, 100.0),
                _mk_cap(2, 0.88, False, 130.0),
                _mk_cap(3, 0.86, False, 160.0),
                _mk_cap(4, 0.81, True, 190.0),
            ]
            proc.prune_to_best_n(3)
            kept_files = {os.path.basename(c.filepath) for c in proc.saved_captures}
            self.assertEqual(len(kept_files), 3)
            self.assertIn("shot_4.png", kept_files, "Genuine smile should claim the smile diversity slot")

            # Also test _would_improve_retained when current 3 are all serious
            proc.saved_captures = [
                _mk_cap(10, 0.90, False, 100.0),
                _mk_cap(11, 0.88, False, 130.0),
                _mk_cap(12, 0.86, False, 160.0),
            ]
            self.assertFalse(proc._would_improve_retained(0.82, 200.0, is_smiling=False))
            self.assertTrue(proc._would_improve_retained(0.82, 200.0, is_smiling=True))


class TestDvrCompressionStandbyAndPersistence(unittest.TestCase):
    def test_compressed_jpeg_dvr_buffer_saves_memory_and_decodes_cleanly(self):
        proc = StreamProcessor.__new__(StreamProcessor)
        proc.config = AppConfig(dvr_jpeg_compression=True)

        # Realistic gradient + shapes 1080p frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.circle(frame, (960, 540), 250, (120, 180, 220), -1)
        ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        self.assertTrue(ok)
        jpg_bytes = enc.tobytes()

        entry = (100.0, jpg_bytes)
        raw_bytes = frame.nbytes
        comp_bytes = StreamProcessor._dvr_entry_nbytes(entry)
        self.assertLess(comp_bytes, raw_bytes * 0.25, "Q95 JPEG should be >75% smaller than raw BGR")

        decoded = proc._decode_dvr_frame(entry)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape, frame.shape)

    def test_blank_stream_source_enters_unconfigured_status(self):
        with mock.patch.dict(os.environ, {"TG_STREAM_SOURCE": ""}, clear=False):
            with mock.patch("camera_store.load", return_value=None):
                cfg = AppConfig()
                self.assertEqual(cfg.stream_source, "")

    def test_folder_scoped_manifests_and_ui_settings_persistence(self):
        with tempfile.TemporaryDirectory() as tmp_root:
            data_dir = os.path.join(tmp_root, "data")
            dir_a = os.path.join(tmp_root, "campus_a")
            dir_b = os.path.join(tmp_root, "campus_b")
            os.makedirs(dir_a, exist_ok=True)
            os.makedirs(dir_b, exist_ok=True)

            with mock.patch.dict(os.environ, {"TG_DATA_DIR": data_dir}, clear=False):
                settings_store.save_ui_settings({"min_score": 68, "capture_max_minutes": 90})
                loaded_ui = settings_store.load_ui_settings()
                self.assertEqual(loaded_ui["min_score"], 68)
                self.assertEqual(loaded_ui["capture_max_minutes"], 90)

                proc = StreamProcessor.__new__(StreamProcessor)
                proc.config = AppConfig(output_dir=dir_a)
                proc.captures_lock = threading.RLock()
                proc.saved_captures = []
                proc.captures_count = 0
                proc.persist_manifests = False

                shot_a = os.path.join(dir_a, "best_shot_20261115_100000_score88.png")
                Path(shot_a).write_bytes(b"png_a")
                proc.enable_persistence(adopt_orphans=True)
                self.assertEqual(len(proc.saved_captures), 1)
                proc.set_capture_pinned(shot_a, True)

                # Switch to Campus B (empty) -> gallery becomes empty without deleting Campus A's file
                proc.switch_output_dir(dir_b, adopt_orphans=True)
                self.assertEqual(len(proc.saved_captures), 0)
                self.assertTrue(os.path.exists(shot_a))

                # Switch back to Campus A -> restores shot_a WITH its pinned state!
                proc.switch_output_dir(dir_a, adopt_orphans=True)
                self.assertEqual(len(proc.saved_captures), 1)
                self.assertTrue(proc.saved_captures[0].is_pinned)


class TestVideoScannerTrimming(unittest.TestCase):
    def test_parse_timestamp_sec_formats(self):
        parse = VideoRecordingScanner.parse_timestamp_sec
        self.assertIsNone(parse(""))
        self.assertIsNone(parse(None))
        self.assertAlmostEqual(parse("45"), 45.0)
        self.assertAlmostEqual(parse("02:30"), 150.0)
        self.assertAlmostEqual(parse("1:05:10"), 3910.0)
        with self.assertRaises(ValueError):
            parse("-5")

    def test_scan_video_respects_start_and_end_sec(self):
        with tempfile.TemporaryDirectory() as tmp:
            vpath = os.path.join(tmp, "synthetic.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(vpath, fourcc, 10.0, (320, 240))
            for i in range(60):  # 6.0 seconds @ 10 fps
                frm = np.full((240, 320, 3), i * 3, dtype=np.uint8)
                writer.write(frm)
            writer.release()

            cfg = AppConfig(output_dir=tmp, ai_judge_enabled=False)
            with mock.patch("video_scanner.PersonQualityAnalyzer") as MockAnalyzer:
                inst = MockAnalyzer.return_value
                inst.analyze_frame.return_value = AnalysisResult(face_detected=False)
                scanner = VideoRecordingScanner(cfg)
                scanner.scan_video(
                    video_path=vpath,
                    top_n=2,
                    sample_fps=5.0,
                    output_dir=tmp,
                    start_sec=2.0,
                    end_sec=4.0,
                )
                # Only 2.0s..4.0s @ 5 fps (~10-11 sampled frames) should be analyzed, not all 30!
                self.assertGreaterEqual(inst.analyze_frame.call_count, 8)
                self.assertLessEqual(inst.analyze_frame.call_count, 13)


class TestWebServerUpgrades(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.allowed_vids = self.root / "videos"
        self.outside_vids = self.root / "secret"
        self.out_dir = self.root / "captures"
        self.data_dir = self.root / "data"
        for d in (self.allowed_vids, self.outside_vids, self.out_dir, self.data_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Create dummy video files
        (self.allowed_vids / "sunday.mp4").write_bytes(b"fake_mp4")
        (self.outside_vids / "private.mp4").write_bytes(b"secret_mp4")

        # Save & isolate global ws.state attributes
        self._orig_cfg = ws.state.config
        self._orig_proc = ws.state.processor
        self._orig_confirmed = ws.state.output_confirmed
        self._orig_capturing = ws.state.is_capturing
        self._orig_started_at = ws.state.capture_started_at

        cfg = AppConfig(
            output_dir=str(self.out_dir),
            video_scan_roots=str(self.allowed_vids),
            output_browse_roots=str(self.root),
            web_password="",
            capture_max_minutes=120.0,
            idle_standby_enabled=True,
        )
        ws.state.config = cfg
        ws.state.output_confirmed = True
        ws.state.is_capturing = False
        ws.state.capture_started_at = None

        # Lightweight mock processor
        proc = StreamProcessor.__new__(StreamProcessor)
        proc.config = cfg
        proc.connection_status = StreamProcessor.STATUS_CONNECTED
        proc.reconnect_attempt = 0
        proc.idle_standby = False
        proc.current_fps = 8.0
        proc.frames_processed = 42
        proc.frames_grabbed = 100
        proc.auto_capture_enabled = False
        proc.frame_lock = threading.Lock()
        proc.latest_frame = None
        proc.latest_analysis = None
        proc.captures_lock = threading.RLock()
        proc.saved_captures = []
        proc.persist_manifests = False
        proc._dvr_lock = threading.Lock()
        proc._dvr = []
        proc.candidate_buffer = []
        proc.analyzer = _make_analyzer_stub()
        proc.judge = mock.MagicMock()
        proc.judge.enabled = True
        proc.judge.available = True
        proc.judge.seconds_remaining = 3600.0
        proc.judge._expiry_handled = False
        proc.judge.status_line.return_value = "AI judge active"
        proc.buffer_stats = lambda: {"enabled": True, "seconds": 5, "frames_held": 0, "frames_capacity": 150, "mb_held": 0, "mb_capacity": 20}
        proc.update_source = mock.MagicMock()
        ws.state.processor = proc

        self.env_patch = mock.patch.dict(os.environ, {"TG_DATA_DIR": str(self.data_dir)}, clear=False)
        self.env_patch.start()
        self.client = TestClient(ws.app, raise_server_exceptions=True)

    def tearDown(self):
        self.env_patch.stop()
        ws.state.config = self._orig_cfg
        ws.state.processor = self._orig_proc
        ws.state.output_confirmed = self._orig_confirmed
        ws.state.is_capturing = self._orig_capturing
        ws.state.capture_started_at = self._orig_started_at
        self._tmp.cleanup()

    def test_video_roots_enforced_on_source_and_camera_endpoints(self):
        outside_path = str(self.outside_vids / "private.mp4")
        inside_path = str(self.allowed_vids / "sunday.mp4")

        # /api/source blocks outside file with 403 and redacted URL with 400
        r_out = self.client.post("/api/source", json={"source": outside_path})
        self.assertEqual(r_out.status_code, 403)

        r_redacted = self.client.post("/api/source", json={"source": "rtsp://***@192.168.1.10:554/stream"})
        self.assertEqual(r_redacted.status_code, 400)

        r_in = self.client.post("/api/source", json={"source": inside_path})
        self.assertEqual(r_in.status_code, 200)

        # /api/camera/test & /api/camera/connect block outside file with 403
        r_cam_test = self.client.post("/api/camera/test", json={"kind": "file", "file_path": outside_path})
        self.assertEqual(r_cam_test.status_code, 403)

        r_cam_conn = self.client.post("/api/camera/connect", json={"kind": "file", "file_path": outside_path})
        self.assertEqual(r_cam_conn.status_code, 403)

    def test_browse_videos_lists_allowed_and_blocks_outside(self):
        r = self.client.get("/api/browse/videos")
        self.assertEqual(r.status_code, 200)
        names = [e["name"] for e in r.json()["entries"]]
        self.assertIn("sunday.mp4", names)

        r_bad = self.client.get("/api/browse/videos", params={"path": str(self.outside_vids)})
        self.assertEqual(r_bad.status_code, 403)

    def test_capture_toggle_wakes_judge_and_auto_stops_when_window_elapses(self):
        r = self.client.post("/api/capture/toggle", json={"enabled": True, "max_minutes": 60})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["is_capturing"])
        self.assertIsNotNone(data["capture_seconds_remaining"])
        ws.state.processor.judge.wake.assert_called_once()

        # Simulate elapsed capture window -> check_lifecycle_tick stops capture & sleeps judge
        ws.state.capture_started_at = time.time() - 3605.0
        actions = ws.state.check_lifecycle_tick(idle_grace_sec=20.0)
        self.assertTrue(actions["auto_stopped"])
        self.assertFalse(ws.state.is_capturing)
        self.assertFalse(ws.state.processor.auto_capture_enabled)
        ws.state.processor.judge.sleep_now.assert_called_once()

        # Simulate 25s of no browser viewers -> enters Zero-CPU Idle Standby
        ws.state.last_client_activity = time.time() - 25.0
        ws.state.active_ws_clients = 0
        ws.state.active_mjpeg_clients = 0
        actions2 = ws.state.check_lifecycle_tick(idle_grace_sec=20.0)
        self.assertTrue(actions2["idle_standby"])
        self.assertTrue(ws.state.processor.idle_standby)

        # Any client activity wakes standby immediately
        ws.state.touch_activity()
        self.assertFalse(ws.state.processor.idle_standby)

    def test_basic_auth_protects_ui_while_exempting_healthcheck(self):
        ws.state.config.web_password = "church-secret-password"

        # /api/health stays open for Docker HEALTHCHECK
        r_health = self.client.get("/api/health")
        self.assertEqual(r_health.status_code, 200)
        self.assertTrue(r_health.json()["ok"])

        # Unauthenticated UI/API returns 401
        r_unauth = self.client.get("/api/state")
        self.assertEqual(r_unauth.status_code, 401)

        # Wrong password returns 401
        bad_hdr = base64.b64encode(b"admin:wrong").decode("ascii")
        r_wrong = self.client.get("/api/state", headers={"Authorization": f"Basic {bad_hdr}"})
        self.assertEqual(r_wrong.status_code, 401)

        # Correct password returns 200
        ok_hdr = base64.b64encode(b"admin:church-secret-password").decode("ascii")
        r_ok = self.client.get("/api/state", headers={"Authorization": f"Basic {ok_hdr}"})
        self.assertEqual(r_ok.status_code, 200)

    def test_single_download_and_zip_export(self):
        img_path = self.out_dir / "best_shot_20261115_120000_score91.png"
        ok, buf = cv2.imencode(".png", np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertTrue(ok)
        img_path.write_bytes(buf.tobytes())

        ws.state.processor.saved_captures = [
            CapturedScreenshot(
                filepath=str(img_path),
                timestamp_str="20261115_120000",
                overall_score=0.91,
                analysis=AnalysisResult(face_detected=True, overall_score=0.91),
                is_pinned=True,
            )
        ]

        r_single = self.client.get(f"/api/captures/{img_path.name}/download")
        self.assertEqual(r_single.status_code, 200)
        self.assertIn("attachment", r_single.headers.get("content-disposition", ""))

        r_zip = self.client.get("/api/captures/download-zip?mode=all")
        self.assertEqual(r_zip.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(r_zip.content), "r") as zf:
            names = zf.namelist()
            self.assertEqual(len(names), 1)
            self.assertTrue(names[0].endswith(img_path.name))


if __name__ == "__main__":
    unittest.main()
