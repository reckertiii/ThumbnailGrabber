"""Comprehensive unit & integration tests for the 16 Next-Gen Thumbnail Grabber features."""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from ai_judge import FinalsVerdict, JudgeVerdict, build_finals_contact_sheet
from analyzer import AnalysisResult, PersonQualityAnalyzer, SingleFaceAnalysis
from audio_analyzer import LiveAudioMonitor, SermonNotesGenerator
from camera_store import CameraSettings
from config import AppConfig
from ndi_source import discover_ndi_sources, ndi_status
from overlay import draw_overlay
from postprocess import CutoutAndPolishEngine, cutout_path_for, save_cutout_png
from preference_model import UserPreferenceModel
from sony_ptz import SonyPTZController
from speaker_id import SpeakerBaseline, SpeakerIdentifier
from stage_zones import StageZoneConfig
from stream_processor import CapturedScreenshot, StreamProcessor
from video_scanner import VideoRecordingScanner
from visual_embedder import VisualEmbedder


def _make_portrait_bgr(
    w: int = 320,
    h: int = 240,
    face_color: tuple[int, int, int] = (145, 170, 210),
    bg_color: tuple[int, int, int] = (32, 38, 48),
    glare: bool = False,
) -> np.ndarray:
    """Creates a synthetic portrait BGR image with distinct background, torso, and face."""
    img = np.full((h, w, 3), bg_color, dtype=np.uint8)
    # Torso
    cv2.rectangle(img, (int(w * 0.28), int(h * 0.50)), (int(w * 0.72), h - 4), (70, 85, 110), -1)
    # Head / face oval
    cx, cy = w // 2, int(h * 0.30)
    rx, ry = int(w * 0.14), int(h * 0.17)
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, face_color, -1)
    # Eyes & mouth detail for Laplacian sharpness
    cv2.circle(img, (cx - rx // 3, cy - ry // 5), 4, (25, 25, 25), -1)
    cv2.circle(img, (cx + rx // 3, cy - ry // 5), 4, (25, 25, 25), -1)
    cv2.ellipse(img, (cx, cy + ry // 3), (rx // 3, 5), 0, 0, 180, (50, 50, 150), 2)
    if glare:
        cv2.rectangle(img, (cx - rx // 2, cy - ry // 3), (cx + rx // 2, cy), (252, 252, 255), -1)
    return img


def _make_analysis(
    score: float = 0.84,
    yaw: float = 3.0,
    pitch: float = 2.0,
    ear: float = 0.34,
    smile: float = 0.45,
    mar: float = 0.12,
    sharpness: float = 0.90,
    lap_var: float = 220.0,
    is_smiling: bool = True,
    is_prop: bool = False,
    mood_bucket: str = "warm_smile",
    face_bbox: tuple[int, int, int, int] = (115, 32, 90, 82),
    subject_bbox: tuple[int, int, int, int] = (75, 18, 170, 215),
) -> AnalysisResult:
    fa = SingleFaceAnalysis(
        person_id=0,
        bbox=face_bbox,
        raw_bbox=face_bbox,
        score=score,
        is_qualified=(score >= 0.55),
        is_selected=True,
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=1.0,
        head_pose_score=0.95,
        avg_ear=ear,
        is_blinking=(ear <= 0.20),
        eye_openness_score=0.95 if ear > 0.22 else 0.15,
        eye_gaze_away_score=0.05,
        mar=mar,
        mouth_velocity=0.02,
        smile_ratio=smile,
        expression_score=0.92 if smile >= 0.3 else 0.72,
        is_smiling=is_smiling,
        sharpness_score=sharpness,
        laplacian_variance=lap_var,
        exposure_score=0.92,
        color_cast_score=0.95,
        cutout_score=0.90,
        torso_clear_score=0.95,
        background_score=0.88,
        subject_bbox=subject_bbox,
        subject_height_ratio=0.72,
        is_prop_moment=is_prop,
        mood_bucket=mood_bucket,
    )
    res = AnalysisResult(
        face_detected=True,
        people_detected_count=1,
        faces=[fa],
        face_bbox=face_bbox,
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=1.0,
        head_pose_score=fa.head_pose_score,
        left_ear=ear,
        right_ear=ear,
        avg_ear=ear,
        is_blinking=fa.is_blinking,
        eye_openness_score=fa.eye_openness_score,
        eye_gaze_away_score=0.05,
        mar=mar,
        mouth_velocity=0.02,
        smile_ratio=smile,
        expression_score=fa.expression_score,
        is_smiling=is_smiling,
        sharpness_score=sharpness,
        laplacian_variance=lap_var,
        hand_laplacian_variance=lap_var,
        hand_pose_score=0.95,
        exposure_score=0.92,
        color_cast_score=0.95,
        cutout_score=0.90,
        torso_clear_score=0.95,
        background_score=0.88,
        subject_height_ratio=0.72,
        shoulder_pose_score=0.95,
        overall_score=score,
        is_qualified=(score >= 0.55),
        is_prop_moment=is_prop,
        mood_bucket=mood_bucket,
    )
    return res


class TestVisualEmbedder(unittest.TestCase):
    def test_perceptual_64d_is_unit_norm_and_discriminative(self):
        emb = VisualEmbedder(enable_onnx=False)
        st = emb.status()
        self.assertEqual(st["backend"], "perceptual")
        self.assertEqual(st["dim"], 64)

        clean_crop = _make_portrait_bgr(glare=False)
        glare_crop = _make_portrait_bgr(face_color=(210, 110, 90), bg_color=(200, 20, 180), glare=True)

        v1 = emb.embed(clean_crop)
        v1_repeat = emb.embed(clean_crop)
        v2 = emb.embed(glare_crop)

        self.assertEqual(v1.shape, (64,))
        self.assertAlmostEqual(float(np.linalg.norm(v1)), 1.0, places=4)
        np.testing.assert_allclose(v1, v1_repeat, atol=1e-6)
        cosine = float(np.dot(v1, v2))
        self.assertLess(cosine, 0.99)


class TestNextGenPreferenceModel(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env = patch.dict(os.environ, {"TG_DATA_DIR": self._tmp.name})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def test_v5_ensemble_pairwise_visual_and_inspector(self):
        pm = UserPreferenceModel()
        self.assertEqual(pm.FEATURE_VERSION, 5)
        self.assertEqual(pm.feature_count, 18)

        good_crop = _make_portrait_bgr(glare=False)
        bad_crop = _make_portrait_bgr(face_color=(90, 90, 90), glare=True)

        # Feed 14 positive and 14 negative samples (>= 24 total triggers HistGB ensemble)
        last_id = ""
        for i in range(14):
            good = _make_analysis(score=0.88, yaw=2.0 + (i % 3), ear=0.35, smile=0.55, mar=0.10, sharpness=0.92)
            bad = _make_analysis(score=0.42, yaw=28.0, pitch=18.0, ear=0.14, smile=0.05, mar=0.55, sharpness=0.25, is_smiling=False)
            pm.add_sample(good, is_favorite=True, crop_bgr=good_crop, filename=f"good_{i}.png")
            last_id = pm.add_sample(bad, is_favorite=False, reasons=["Eyes Closed"], crop_bgr=bad_crop, filename=f"bad_{i}.png")
            pm.add_pairwise_preference(good, bad)

        self.assertTrue(pm.is_trained)
        summary = pm.get_brain_summary()
        self.assertTrue(summary["ensemble_active"])
        self.assertTrue(summary["pairwise_active"])
        self.assertTrue(summary["visual_active"])
        self.assertGreaterEqual(len(summary["top_drivers"]), 5)
        self.assertGreater(summary["likes"], 0)
        self.assertGreater(summary["rejects"], 0)

        p_good = pm.predict(pm.extract_features(_make_analysis(score=0.90, ear=0.35, smile=0.55)), crop_bgr=good_crop)
        p_bad = pm.predict(pm.extract_features(_make_analysis(score=0.40, yaw=30.0, ear=0.12, smile=0.02, mar=0.60, sharpness=0.20)), crop_bgr=bad_crop)
        self.assertGreater(p_good, p_bad + 0.25)

        # Flip & Undo via ML-6 inspector APIs
        self.assertTrue(pm.flip_rating(last_id))
        self.assertTrue(pm.undo_rating(last_id))

        # Reset
        pm.reset_all()
        self.assertFalse(pm.is_trained)
        self.assertEqual(len(pm.samples), 0)

    def test_implicit_feedback_distillation_and_folder_bootstrap(self):
        pm = UserPreferenceModel()
        cap_win = CapturedScreenshot(
            filepath=os.path.join(self._tmp.name, "win.png"),
            timestamp_str="120000",
            overall_score=0.88,
            analysis=_make_analysis(score=0.88),
        )
        cap_peer = CapturedScreenshot(
            filepath=os.path.join(self._tmp.name, "peer.png"),
            timestamp_str="120005",
            overall_score=0.74,
            analysis=_make_analysis(score=0.74, smile=0.15),
        )
        sid_pin = pm.record_implicit_feedback(cap_win, peer_captures=[cap_win, cap_peer], action="pin")
        self.assertIsNotNone(sid_pin)
        self.assertEqual(len(pm.pairwise_diffs), 1)

        sid_del = pm.record_implicit_feedback(cap_peer, action="delete")
        self.assertIsNotNone(sid_del)

        # Teacher-Student distillation (ML-4)
        v_high = JudgeVerdict(usable=True, score=0.92, holding_prop=False, issues=[], strengths=["crisp"], summary="great")
        v_mid = JudgeVerdict(usable=True, score=0.65, holding_prop=False, issues=[], strengths=[], summary="ok")
        v_low = JudgeVerdict(usable=False, score=0.20, holding_prop=False, issues=["(serious) eyes closed"], strengths=[], summary="bad")

        self.assertIsNotNone(pm.record_judge_distillation(_make_analysis(score=0.90), v_high))
        self.assertIsNone(pm.record_judge_distillation(_make_analysis(score=0.65), v_mid))
        self.assertIsNotNone(pm.record_judge_distillation(_make_analysis(score=0.30, ear=0.12), v_low))

        # Bootstrap from folder (ML-3)
        folder = os.path.join(self._tmp.name, "past_winners")
        os.makedirs(folder, exist_ok=True)
        for idx in range(6):
            cv2.imwrite(os.path.join(folder, f"winner_{idx}.png"), _make_portrait_bgr())
        # Should ignore *_cutout.png files
        cv2.imwrite(os.path.join(folder, "winner_0_cutout.png"), _make_portrait_bgr())

        fake_analyzer = MagicMock()
        fake_analyzer.thresholds = None
        fake_analyzer.analyze_frame.side_effect = lambda _img: _make_analysis(score=0.89, smile=0.52)

        pm_fresh = UserPreferenceModel()
        boot = pm_fresh.bootstrap_from_folder(folder, fake_analyzer)
        self.assertTrue(boot["ok"])
        self.assertEqual(boot["added_positives"], 6)
        self.assertGreaterEqual(boot["synthetic_negatives"], 4)
        self.assertTrue(pm_fresh.is_trained)


class TestSpeakerLockAndBaselines(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env = patch.dict(os.environ, {"TG_DATA_DIR": self._tmp.name})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def test_lock_filter_persistence_and_welford_baseline(self):
        spk = SpeakerIdentifier(auto_download_sface=False)
        frame = np.full((240, 480, 3), 30, dtype=np.uint8)
        # Left speaker: warm skin + dark suit
        cv2.rectangle(frame, (40, 40), (140, 140), (130, 165, 215), -1)
        cv2.circle(frame, (75, 75), 8, (20, 20, 20), -1)
        cv2.circle(frame, (105, 75), 8, (20, 20, 20), -1)
        # Right bystander: completely different high-contrast cyan/yellow pattern
        cv2.rectangle(frame, (300, 40), (400, 140), (240, 220, 30), -1)
        cv2.line(frame, (300, 40), (400, 140), (10, 10, 240), 6)

        lock_res = spk.lock_from_frame(frame, (40, 40, 100, 100), save_as_name="Lead Pastor", download_sface=False)
        self.assertTrue(lock_res["ok"])
        self.assertTrue(spk.locked)
        prof_id = lock_res["active_profile_id"]
        self.assertTrue(prof_id)

        f_target = _make_analysis().faces[0]
        f_target.person_id = 0
        f_target.bbox = (40, 40, 100, 100)
        f_target.raw_bbox = (40, 40, 100, 100)

        f_other = _make_analysis().faces[0]
        f_other.person_id = 1
        f_other.bbox = (300, 40, 100, 100)
        f_other.raw_bbox = (300, 40, 100, 100)
        f_other.disqualified_reasons = []

        spk.evaluate_and_filter_faces(frame, [f_target, f_other])
        self.assertTrue(f_target.is_selected)
        self.assertEqual(getattr(f_target, "speaker_name", ""), "Lead Pastor")
        self.assertFalse(f_other.is_selected)
        self.assertIn("not locked speaker", f_other.disqualified_reasons)

        # Reload from disk
        spk2 = SpeakerIdentifier(auto_download_sface=False)
        self.assertIn(prof_id, spk2.profiles)
        self.assertTrue(spk2.locked)
        self.assertEqual(spk2.active_name, "Lead Pastor")

        spk2.unlock()
        self.assertFalse(spk2.locked)
        self.assertTrue(spk2.activate_profile(prof_id)["ok"])
        self.assertTrue(spk2.delete_profile(prof_id))
        self.assertFalse(spk2.locked)

        # Welford personal baseline (narrow-eyed speaker + big smile Z-score bonus)
        bl = SpeakerBaseline()
        for k in range(15):
            bl.update(ear=0.23 + (0.005 if k % 2 else -0.005), smile=0.18 + (0.02 if k % 2 else -0.02), mar=0.12)
        f_narrow = _make_analysis(ear=0.23, smile=0.45).faces[0]
        f_narrow.eye_openness_score = 0.70
        f_narrow.expression_score = 0.80
        spk._apply_baseline_adjustment(bl, f_narrow, persist_prof=None)
        self.assertGreater(f_narrow.eye_openness_score, 0.80)
        self.assertGreater(f_narrow.expression_score, 0.80)


class TestStageZonesAndOverlay(unittest.TestCase):
    def test_polygon_safe_zone_obstruction_and_overlay(self):
        raw = {
            "safe_zone": [[0.2, 0.1], [0.8, 0.1], [0.8, 0.9], [0.2, 0.9]],
            "obstruction_zones": [
                {"name": "Lectern", "points": [[0.4, 0.55], [0.6, 0.55], [0.6, 0.95], [0.4, 0.95]]}
            ],
        }
        zones = StageZoneConfig.from_dict(raw)
        self.assertTrue(zones.has_safe_zone)
        self.assertTrue(zones.has_obstructions)

        # Center face inside safe zone; far-left audience face outside
        self.assertTrue(zones.is_face_inside_safe_zone((280, 100, 80, 80), 640, 360))
        self.assertFalse(zones.is_face_inside_safe_zone((10, 100, 60, 60), 640, 360))

        overlap, name = zones.compute_torso_obstruction((250, 190, 140, 150), 640, 360)
        self.assertGreater(overlap, 0.25)
        self.assertEqual(name, "Lectern")

        roundtrip = StageZoneConfig.from_dict(zones.to_dict())
        self.assertEqual(len(roundtrip.safe_zone), 4)
        self.assertEqual(len(roundtrip.obstruction_zones), 1)

        frame = _make_portrait_bgr(640, 360)
        rendered = draw_overlay(frame, _make_analysis(), show_mesh=False, show_hud=True, stage_zones=zones)
        self.assertEqual(rendered.shape, frame.shape)


class TestMoodBucketsTimeMachineBurstAndFinals(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env = patch.dict(
            os.environ,
            {
                "TG_DATA_DIR": os.path.join(self._tmp.name, "data"),
                "TG_OUTPUT_DIR": os.path.join(self._tmp.name, "out"),
            },
        )
        self._env.start()
        os.makedirs(os.environ["TG_DATA_DIR"], exist_ok=True)
        os.makedirs(os.environ["TG_OUTPUT_DIR"], exist_ok=True)

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def test_mood_classification_and_color_cast_scoring(self):
        cfg = AppConfig()
        az = PersonQualityAnalyzer.__new__(PersonQualityAnalyzer)
        az.weights = cfg.weights
        az.thresholds = cfg.thresholds
        az.preference_model = SimpleNamespace(learned_penalties={})
        az.capture_prop_moments = True

        f_prop = _make_analysis(is_prop=True, is_smiling=True).faces[0]
        az._finalize_face_score(f_prop)
        self.assertEqual(f_prop.mood_bucket, "prop")

        f_smile = _make_analysis(is_prop=False, is_smiling=True, smile=0.48).faces[0]
        az._finalize_face_score(f_smile)
        self.assertEqual(f_smile.mood_bucket, "warm_smile")

        f_passion = _make_analysis(is_prop=False, is_smiling=False, smile=0.10).faces[0]
        f_passion.arms_clear = True
        az._finalize_face_score(f_passion)
        self.assertEqual(f_passion.mood_bucket, "passionate")

        f_calm = _make_analysis(is_prop=False, is_smiling=False, smile=0.10).faces[0]
        f_calm.arms_clear = False
        f_calm.jaw_open_score = 0.0
        az._finalize_face_score(f_calm)
        self.assertEqual(f_calm.mood_bucket, "calm_portrait")

        neutral_skin = np.full((80, 80, 3), (140, 165, 200), dtype=np.uint8)
        heavy_led = np.full((80, 80, 3), (245, 30, 235), dtype=np.uint8)
        self.assertGreater(
            PersonQualityAnalyzer._compute_color_cast_score(neutral_skin),
            PersonQualityAnalyzer._compute_color_cast_score(heavy_led),
        )

    def test_mood_buckets_preserved_during_prune_to_best_n(self):
        cfg = AppConfig()
        cfg.output_dir = os.environ["TG_OUTPUT_DIR"]
        cfg.single_winner_mode = True
        cfg.max_retained_captures = 5
        cfg.mood_buckets_enabled = True
        cfg.min_capture_spacing_sec = 0.0

        proc = StreamProcessor.__new__(StreamProcessor)
        proc.config = cfg
        proc.captures_lock = threading.Lock()
        proc.saved_captures = []
        proc.captures_count = 0
        proc.on_capture_callback = None
        proc._persist_manifest = lambda: None

        specs = [
            ("calm1.png", 0.95, False, False, "calm_portrait"),
            ("calm2.png", 0.94, False, False, "calm_portrait"),
            ("calm3.png", 0.93, False, False, "calm_portrait"),
            ("calm4.png", 0.92, False, False, "calm_portrait"),
            ("calm5.png", 0.91, False, False, "calm_portrait"),
            ("smile.png", 0.78, True, False, "warm_smile"),
            ("passion.png", 0.76, False, False, "passionate"),
            ("prop.png", 0.74, False, True, "prop"),
        ]
        for idx, (fname, sc, sm, pr, mb) in enumerate(specs):
            fp = os.path.join(cfg.output_dir, fname)
            cv2.imwrite(fp, _make_portrait_bgr())
            proc.saved_captures.append(
                CapturedScreenshot(
                    filepath=fp,
                    timestamp_str=f"10000{idx}",
                    overall_score=sc,
                    analysis=_make_analysis(score=sc, is_smiling=sm, is_prop=pr, mood_bucket=mb),
                    captured_at=1000.0 + idx * 10,
                    is_prop_moment=pr,
                    mood_bucket=mb,
                )
            )

        proc.prune_to_best_n(5)
        kept_moods = {c.mood_bucket for c in proc.saved_captures}
        self.assertEqual(len(proc.saved_captures), 5)
        self.assertIn("warm_smile", kept_moods)
        self.assertIn("passionate", kept_moods)
        self.assertIn("prop", kept_moods)
        self.assertIn("calm_portrait", kept_moods)

    def test_time_machine_buffer_burst_swap_and_ai_finals(self):
        cfg = AppConfig()
        cfg.output_dir = os.environ["TG_OUTPUT_DIR"]
        cfg.dvr_max_seconds = 180.0
        cfg.ai_judge_enabled = True
        cfg.single_winner_mode = False

        fake_analyzer = MagicMock()
        fake_analyzer.preference_model = UserPreferenceModel()
        fake_analyzer.analyze_frame.side_effect = lambda _f, fast_reject=False: _make_analysis(score=0.86)

        with patch("stream_processor.PersonQualityAnalyzer", return_value=fake_analyzer):
            proc = StreamProcessor(cfg)
        proc._refine_analyzer = fake_analyzer
        proc.set_buffer(120.0)
        self.assertAlmostEqual(proc.config.dvr_buffer_seconds, 120.0, places=1)

        # Populate DVR with 12 Q95 JPEG frames across 12 seconds
        base_t = time.time() - 15.0
        ok_j, enc = cv2.imencode(".jpg", _make_portrait_bgr(), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        self.assertTrue(ok_j)
        jb = enc.tobytes()
        with proc._dvr_lock:
            for i in range(12):
                proc._dvr.append((base_t + i * 1.0, jb))

        mined = proc.mine_dvr_window(window_seconds=30.0, max_picks=2, min_spacing_sec=4.0)
        self.assertEqual(len(mined), 2)

        first_fn = os.path.basename(mined[0].filepath)
        burst_info = proc.get_burst_frames(first_fn)
        self.assertTrue(burst_info["available"])
        self.assertGreaterEqual(burst_info["count"], 2)
        self.assertIsNotNone(proc.get_burst_frame_jpeg(first_fn, 0))

        swapped = proc.select_burst_frame(first_fn, 0)
        self.assertIsNotNone(swapped)

        # Contact sheet & Tournament Finals (#4a)
        sheet = build_finals_contact_sheet([_make_portrait_bgr(), _make_portrait_bgr()])
        self.assertEqual(sheet.ndim, 3)

        proc.judge.enabled = True
        proc.judge.judge_finals = MagicMock(
            return_value=FinalsVerdict(
                overall_winner=2,
                best_smile=1,
                best_gesture=2,
                ranking=[2, 1],
                rationale="Tile #2 has warmer eye contact and cleaner posture.",
            )
        )
        finals_res = proc.run_ai_finals(include_scan=True)
        self.assertTrue(finals_res["ok"])
        self.assertEqual(finals_res["winner_index"], 2)
        badges = [c.finals_badge for c in proc.saved_captures if c.finals_badge]
        self.assertTrue(any("#1 Overall" in b for b in badges))
        self.assertTrue(any("Best Smile" in b for b in badges))


class TestCutoutAndBroadcastPolish(unittest.TestCase):
    def test_cutout_rgba_and_broadcast_polish_4k(self):
        with tempfile.TemporaryDirectory() as td:
            img = _make_portrait_bgr(320, 240, face_color=(230, 80, 210))  # strong magenta LED cast
            analysis = _make_analysis(face_bbox=(115, 32, 90, 82), subject_bbox=(75, 18, 170, 215))

            engine = CutoutAndPolishEngine(auto_load_existing=False)
            rgba = engine.generate_cutout_rgba(img, analysis=analysis, crop_to_subject=False)
            self.assertEqual(rgba.shape, (240, 320, 4))
            self.assertEqual(rgba.dtype, np.uint8)
            # Outer top-left corner should be transparent; central face should be opaque
            self.assertEqual(int(rgba[2, 2, 3]), 0)
            self.assertGreater(int(rgba[72, 160, 3]), 200)

            dest = cutout_path_for(os.path.join(td, "shot.png"))
            saved = save_cutout_png(img, analysis, dest, crop_to_subject=False, polish_first=True)
            self.assertEqual(saved, dest)
            loaded = cv2.imread(dest, cv2.IMREAD_UNCHANGED)
            self.assertEqual(loaded.shape, (240, 320, 4))

            polished_4k = engine.polish_broadcast_frame(img, analysis=analysis, deblock=True, neutralize_led=True, upscale_2x=True)
            self.assertEqual(polished_4k.shape, (480, 640, 3))


class TestParallelVideoScanner(unittest.TestCase):
    def test_multi_worker_chunked_video_scan(self):
        with tempfile.TemporaryDirectory() as td:
            vid_path = os.path.join(td, "sermon.mp4")
            out_dir = os.path.join(td, "scan_out")
            os.makedirs(out_dir, exist_ok=True)

            # Write 260 frames (>= 240 triggers multi-worker chunked ThreadPoolExecutor)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(vid_path, fourcc, 30.0, (320, 240))
            frame = _make_portrait_bgr(320, 240)
            for _ in range(260):
                vw.write(frame)
            vw.release()

            cfg = AppConfig()
            cfg.scan_workers = 2
            cfg.ai_judge_enabled = False

            def _fast_fake_analyzer(*_a, **_kw):
                m = MagicMock()
                m.analyze_frame.side_effect = lambda _f, fast_reject=False: _make_analysis(score=0.85)
                return m

            burst_calls = []
            with patch("video_scanner.PersonQualityAnalyzer", side_effect=_fast_fake_analyzer):
                scanner = VideoRecordingScanner(
                    cfg,
                    burst_cache_callback=lambda fn, ts, fr, bframes: burst_calls.append((fn, len(bframes))),
                )
                results = scanner.scan_video(
                    vid_path,
                    top_n=2,
                    sample_fps=6.0,
                    min_interval_sec=2.0,
                    output_dir=out_dir,
                )

            self.assertGreaterEqual(len(results), 1)
            self.assertTrue(os.path.isfile(results[0].filepath))
            self.assertGreaterEqual(len(burst_calls), 1)


class TestAudioSermonNotesNdiAndSonyPtz(unittest.TestCase):
    def test_audio_pause_detector_and_wav_export(self):
        mon = LiveAudioMonitor(max_buffer_minutes=5.0)
        t0 = 1000.0
        quiet = np.zeros(800, dtype=np.int16)
        loud = (np.sin(np.linspace(0, 40.0, 800)) * 6000).astype(np.int16)

        for i in range(12):
            mon._process_chunk(quiet.tobytes(), quiet.astype(np.float32), t0 + i * 0.05)
        for i in range(6):
            mon._process_chunk(loud.tobytes(), loud.astype(np.float32), t0 + 0.60 + i * 0.05)
        self.assertTrue(mon.telemetry().is_speaking)

        # 0.30s post-speech breath pause -> in_speech_pause & positive capture_bonus
        for i in range(6):
            mon._process_chunk(quiet.tobytes(), quiet.astype(np.float32), t0 + 0.90 + (i + 1) * 0.05)
        tel = mon.telemetry()
        self.assertTrue(tel.in_speech_pause)
        self.assertGreater(tel.capture_bonus, 0.0)

        with tempfile.TemporaryDirectory() as td:
            wav_path = os.path.join(td, "clip.wav")
            self.assertTrue(mon.export_wav(wav_path))
            with wave.open(wav_path, "rb") as wf:
                self.assertEqual(wf.getframerate(), 16000)
                self.assertEqual(wf.getnchannels(), 1)

    def test_sermon_notes_generator_ollama_and_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            gen = SermonNotesGenerator()
            sample_text = (
                "Grace changes the way we walk through uncertainty every single week. "
                "Courage is not the absence of fear, it is trusting God in the middle of the storm. "
                "When we anchored our hope together, our community found peace and lasting purpose."
            )
            # 1. Offline extractive fallback (unreachable port)
            res_fb = gen.generate_and_save(
                output_dir=td,
                raw_transcript=sample_text,
                ollama_url="http://127.0.0.1:9",
            )
            self.assertTrue(res_fb["titles"])
            self.assertTrue(os.path.isfile(os.path.join(td, "sermon_notes.md")))

            # 2. Mocked Ollama structured response
            fake_payload = {
                "message": {
                    "content": json.dumps({
                        "titles": ["Anchored In The Storm", "Walking In Grace", "Peace Beyond Fear", "Hope Together", "Courage Today"],
                        "quotes": ["Courage is not the absence of fear, it is trusting God in the middle of the storm."],
                        "summary": "A message on finding peace and courage through grace.",
                    })
                }
            }
            fake_resp = MagicMock()
            fake_resp.read.return_value = json.dumps(fake_payload).encode("utf-8")
            fake_resp.__enter__.return_value = fake_resp
            with patch("urllib.request.urlopen", return_value=fake_resp):
                res_ol = gen.generate_and_save(
                    output_dir=td,
                    raw_transcript=sample_text,
                    ollama_url="http://127.0.0.1:11434",
                    ollama_model="qwen3-vl:30b",
                )
            self.assertEqual(res_ol["titles"][0], "Anchored In The Storm")
            self.assertIn("ollama:", res_ol["source"])

    def test_sony_ptz_cgi_and_ndi_settings(self):
        cam = CameraSettings(kind="ndi", ndi_source_name="STAGE-CAM-1")
        self.assertEqual(cam.url(), "ndi://STAGE-CAM-1")

        st = ndi_status()
        self.assertIn("available", st)
        with patch("ndi_source._mdns_probe_ndi", return_value=["SONY-SRG (192.168.32.50)"]):
            disc = discover_ndi_sources(timeout_ms=200)
            self.assertIn("sources", disc)

        ptz = SonyPTZController(host="192.168.32.50", username="admin", password="pw", http_port=80)
        calls = []
        ptz._request_cgi = lambda path: (calls.append(path) or {"ok": True, "status": 200, "body": "PtzAutoFraming=on"})
        self.assertTrue(ptz.move("left", speed=10, duration_ms=0)["ok"])
        self.assertTrue(ptz.zoom("in", speed=4, duration_ms=0)["ok"])
        self.assertTrue(ptz.home()["ok"])
        self.assertTrue(ptz.recall_preset(3)["ok"])
        self.assertTrue(ptz.set_auto_framing(True)["ok"])
        st_ptz = ptz.status()
        self.assertTrue(st_ptz["reachable"])
        self.assertTrue(st_ptz["auto_framing"])
        self.assertTrue(any("ptzf.cgi?Move=left,10" in c for c in calls))
        self.assertTrue(any("presetposition.cgi?PresetCall=3" in c for c in calls))
        self.assertTrue(any("ptzautoframing.cgi?PtzAutoFraming=on" in c for c in calls))


class TestNextGenWebEndpoints(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self._tmp.name, "data")
        self.out_dir = os.path.join(self._tmp.name, "captures")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)
        self._env = patch.dict(
            os.environ,
            {
                "TG_DATA_DIR": self.data_dir,
                "TG_OUTPUT_DIR": self.out_dir,
                "TG_OUTPUT_ROOTS": self._tmp.name,
                "TG_VIDEO_ROOTS": self._tmp.name,
                "TG_WEB_PASSWORD": "",
            },
        )
        self._env.start()

        import web.server as ws
        self.ws = ws
        self._orig_proc = ws.state.processor
        self._orig_out = ws.state.config.output_dir
        self._orig_conf = ws.state.output_confirmed
        self._orig_zones = ws.state.stage_zones

        ws.state.config.output_dir = self.out_dir
        ws.state.output_confirmed = True

        fake_analyzer = SimpleNamespace(
            preference_model=UserPreferenceModel(),
            speaker_id=SpeakerIdentifier(auto_download_sface=False),
            stage_zones=StageZoneConfig(),
            thresholds=ws.state.config.thresholds,
            analyze_frame=lambda _f, fast_reject=False: _make_analysis(score=0.87),
        )
        with patch("stream_processor.PersonQualityAnalyzer", return_value=fake_analyzer):
            proc = StreamProcessor(ws.state.config)
        proc._refine_analyzer = fake_analyzer
        ws.state.processor = proc
        self.proc = proc
        self.client = TestClient(ws.app)

    def tearDown(self):
        self.ws.state.processor = self._orig_proc
        self.ws.state.config.output_dir = self._orig_out
        self.ws.state.output_confirmed = self._orig_conf
        self.ws.state.stage_zones = self._orig_zones
        self._env.stop()
        self._tmp.cleanup()

    def test_ml_stage_zones_cutout_polish_burst_sermon_and_delete_routes(self):
        # 1. ML Brain routes
        pm = self.proc.analyzer.preference_model
        sid = pm.add_sample(_make_analysis(score=0.90), is_favorite=True, filename="sample.png")
        r_brain = self.client.get("/api/ml/brain")
        self.assertEqual(r_brain.status_code, 200)
        self.assertEqual(r_brain.json()["sample_count"], 1)

        r_flip = self.client.post("/api/ml/flip", json={"sample_id": sid})
        self.assertEqual(r_flip.status_code, 200)
        self.assertEqual(r_flip.json()["rejects"], 1)

        r_undo = self.client.post("/api/ml/undo", json={"sample_id": sid})
        self.assertEqual(r_undo.status_code, 200)
        self.assertEqual(r_undo.json()["sample_count"], 0)

        r_reset = self.client.post("/api/ml/reset")
        self.assertEqual(r_reset.status_code, 200)

        # 2. Stage Zones routes
        zone_payload = {
            "safe_zone": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            "obstruction_zones": [{"name": "Stand", "points": [[0.4, 0.6], [0.6, 0.6], [0.5, 0.9]]}],
        }
        rz_post = self.client.post("/api/stage-zones", json=zone_payload)
        self.assertEqual(rz_post.status_code, 200)
        rz_get = self.client.get("/api/stage-zones")
        self.assertEqual(len(rz_get.json()["safe_zone"]), 4)
        self.assertEqual(len(rz_get.json()["obstruction_zones"]), 1)

        # 3. Speaker status, NDI discover & PTZ status
        self.assertEqual(self.client.get("/api/speaker").status_code, 200)
        with patch("web.server.discover_ndi_sources", return_value={"available": False, "sources": []}):
            self.assertEqual(self.client.get("/api/ndi/discover").status_code, 200)
        self.assertEqual(self.client.get("/api/ptz/status").status_code, 200)

        # 4. Mine Time-Machine buffer, inspect burst, generate cutout + 4K polish, then delete
        ok_j, enc = cv2.imencode(".jpg", _make_portrait_bgr(320, 240), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        self.assertTrue(ok_j)
        with self.proc._dvr_lock:
            self.proc._dvr.append((time.time() - 1.0, enc.tobytes()))
        r_mine = self.client.post("/api/buffer/mine", json={"seconds": 30, "max_picks": 1})
        self.assertEqual(r_mine.status_code, 200)
        self.assertEqual(r_mine.json()["saved_count"], 1)

        frame = _make_portrait_bgr(320, 240)
        cap = self.proc.save_screenshot(frame, _make_analysis(score=0.88), captured_at=time.time())
        fname = os.path.basename(cap.filepath)

        r_burst = self.client.get(f"/api/captures/{fname}/burst")
        self.assertEqual(r_burst.status_code, 200)
        self.assertTrue(r_burst.json()["available"])
        self.assertEqual(self.client.get(f"/api/captures/{fname}/burst/0").status_code, 200)

        r_cut = self.client.get(f"/api/captures/{fname}/cutout")
        self.assertEqual(r_cut.status_code, 200)
        self.assertEqual(r_cut.headers["content-type"], "image/png")

        r_pol = self.client.get(f"/api/captures/{fname}/polish?upscale=1")
        self.assertEqual(r_pol.status_code, 200)

        r_del = self.client.delete(f"/api/captures/{fname}")
        self.assertEqual(r_del.status_code, 200)
        self.assertFalse(os.path.exists(cap.filepath))

        # 5. Sermon Notes from transcript
        r_notes = self.client.post(
            "/api/sermon-notes/generate",
            json={
                "source": "transcript",
                "transcript": "Faith gives us steady hope when everything around us feels uncertain and loud.",
            },
        )
        self.assertEqual(r_notes.status_code, 200)
        self.assertTrue(r_notes.json()["titles"])
        self.assertEqual(self.client.get("/api/sermon-notes/latest").status_code, 200)


if __name__ == "__main__":
    unittest.main()
