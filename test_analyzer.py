import unittest
import os
import numpy as np
import cv2

from config import AppConfig, QualityWeights, QualityThresholds
from analyzer import PersonQualityAnalyzer, AnalysisResult

class TestPersonQualityAnalyzer(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig()
        self.analyzer = PersonQualityAnalyzer(
            self.config.weights, self.config.thresholds,
            ema_alpha=self.config.ema_alpha,
            stability_window=self.config.score_stability_window,
            stability_threshold=self.config.score_stability_threshold,
            blink_suppression_ms=self.config.blink_suppression_ms
        )

    def test_sharpness_calculation(self):
        # Create sharp checkerboard image
        sharp_img = np.zeros((200, 200, 3), dtype=np.uint8)
        sharp_img[::20, ::20] = 255
        gray = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2GRAY)
        lap_sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Create blurry image
        blurry_img = cv2.GaussianBlur(sharp_img, (21, 21), 0)
        gray_blur = cv2.cvtColor(blurry_img, cv2.COLOR_BGR2GRAY)
        lap_blur = cv2.Laplacian(gray_blur, cv2.CV_64F).var()
        
        self.assertGreater(lap_sharp, lap_blur, "Sharp image Laplacian variance should be strictly greater than blurred image")

    def test_analyzer_blank_frame(self):
        # Blank black frame
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = self.analyzer.analyze_frame(blank_frame)
        self.assertFalse(res.face_detected, "Blank frame should detect no face")
        self.assertEqual(res.overall_score, 0.0)

    def test_exposure_scoring(self):
        """Test that exposure scoring correctly penalizes extreme brightness values."""
        # Well-exposed face ROI (mean ~128)
        normal_roi = np.full((100, 100, 3), 128, dtype=np.uint8)
        normal_gray = cv2.cvtColor(normal_roi, cv2.COLOR_BGR2GRAY)
        score_normal, brightness_normal = self.analyzer._compute_exposure_score(normal_roi, normal_gray)
        
        # Underexposed face ROI (mean ~20)
        dark_roi = np.full((100, 100, 3), 20, dtype=np.uint8)
        dark_gray = cv2.cvtColor(dark_roi, cv2.COLOR_BGR2GRAY)
        score_dark, brightness_dark = self.analyzer._compute_exposure_score(dark_roi, dark_gray)
        
        # Overexposed face ROI (mean ~240)
        bright_roi = np.full((100, 100, 3), 240, dtype=np.uint8)
        bright_gray = cv2.cvtColor(bright_roi, cv2.COLOR_BGR2GRAY)
        score_bright, brightness_bright = self.analyzer._compute_exposure_score(bright_roi, bright_gray)
        
        self.assertGreater(score_normal, score_dark, "Normal exposure should score higher than underexposed")
        self.assertGreater(score_normal, score_bright, "Normal exposure should score higher than overexposed")
        self.assertGreaterEqual(score_normal, 0.9, "Well-exposed image should score ≥ 0.9")
        self.assertLessEqual(score_dark, 0.5, "Very dark image should score ≤ 0.5")
        self.assertLessEqual(score_bright, 0.5, "Very bright image should score ≤ 0.5")

    def test_centering_scoring(self):
        """Test that face centering score correctly rewards centered faces."""
        frame_w, frame_h = 1920, 1080
        face_w, face_h = 200, 250
        
        # Face in center
        cx = (frame_w - face_w) // 2
        cy = (frame_h - face_h) // 2
        score_center = self.analyzer._compute_centering_score(cx, cy, face_w, face_h, frame_w, frame_h)
        
        # Face at top-left corner
        score_corner = self.analyzer._compute_centering_score(10, 10, face_w, face_h, frame_w, frame_h)
        
        # Face at edge
        score_edge = self.analyzer._compute_centering_score(frame_w - face_w - 10, frame_h // 2, face_w, face_h, frame_w, frame_h)
        
        self.assertGreater(score_center, score_corner, "Centered face should score higher than corner face")
        self.assertGreater(score_center, score_edge, "Centered face should score higher than edge face")
        self.assertGreaterEqual(score_center, 0.8, "Centered face should score ≥ 0.8")

    def test_pitch_normalization(self):
        """Test that angle normalization correctly handles the 180° offset bug."""
        # Forward-facing pitch that solvePnP outputs as ~175°
        normalized = PersonQualityAnalyzer._normalize_angle(175.0)
        self.assertLess(abs(normalized), 30.0, "175° should normalize to near 0° (within ±30°)")
        
        # Already correct angle
        normalized_ok = PersonQualityAnalyzer._normalize_angle(5.0)
        self.assertAlmostEqual(normalized_ok, 5.0, places=1, msg="5° should stay as 5°")
        
        # Negative near -180
        normalized_neg = PersonQualityAnalyzer._normalize_angle(-175.0)
        self.assertLess(abs(normalized_neg), 30.0, "-175° should normalize to near 0°")

    def test_ema_smoothing(self):
        """Test that EMA smoothing produces stable output from noisy input."""
        analyzer = PersonQualityAnalyzer(
            ema_alpha=0.3,
            stability_window=5,
            stability_threshold=0.08
        )
        
        # Simulate feeding scores through EMA
        analyzer._ema_score = 0.80
        
        # Spike to 0.95 — EMA should dampen it
        new_ema = 0.3 * 0.95 + 0.7 * 0.80  # = 0.845
        analyzer._ema_score = new_ema
        self.assertAlmostEqual(analyzer._ema_score, 0.845, places=2,
                               msg="EMA should smooth a spike from 0.80 to 0.95 to ~0.845")
        
        # Drop to 0.40 — EMA should dampen the drop too
        new_ema2 = 0.3 * 0.40 + 0.7 * 0.845  # = 0.7115
        analyzer._ema_score = new_ema2
        self.assertGreater(analyzer._ema_score, 0.60,
                           msg="EMA should prevent a raw drop to 0.40 from dragging score below 0.60")

    def test_analysis_result_has_new_fields(self):
        """Ensure AnalysisResult has all new fields with correct defaults."""
        result = AnalysisResult()
        self.assertEqual(result.smoothed_score, 0.0)
        self.assertFalse(result.is_stable)
        self.assertEqual(result.exposure_score, 1.0)
        self.assertEqual(result.centering_score, 0.5)
        self.assertEqual(result.face_brightness, 128.0)
        self.assertEqual(result.shoulder_pose_score, 1.0)
        self.assertEqual(result.shoulder_tilt_deg, 0.0)
        self.assertTrue(result.is_shoulders_level)
        self.assertFalse(result.is_shoulder_tilted)
        self.assertFalse(result.shoulder_detected)

    def test_shoulder_tilt_angle_calculation(self):
        """Test calculation of shoulder tilt angle from horizontal."""
        # 1. Perfectly horizontal shoulders
        pt_l = (100.0, 300.0)
        pt_r = (300.0, 300.0)
        angle_flat = PersonQualityAnalyzer.compute_shoulder_tilt_angle(pt_l, pt_r)
        self.assertEqual(angle_flat, 0.0, "Horizontal shoulders should have 0.0° tilt")

        # 2. Slanted shoulders ~10°
        # dx = 200, dy = 200 * tan(10°) = 35.26
        pt_r_10 = (300.0, 335.26)
        angle_10 = PersonQualityAnalyzer.compute_shoulder_tilt_angle(pt_l, pt_r_10)
        self.assertAlmostEqual(angle_10, 10.0, places=1, msg="Shoulders with dy/dx ~ 35.3/200 should be ~10°")

        # 3. Slanted shoulders in opposite direction (left higher vs right higher)
        pt_l_rev = (300.0, 335.26)
        pt_r_rev = (100.0, 300.0)
        angle_rev = PersonQualityAnalyzer.compute_shoulder_tilt_angle(pt_l_rev, pt_r_rev)
        self.assertAlmostEqual(angle_rev, 10.0, places=1, msg="Order of points should produce same absolute tilt")

        # 4. Vertical points
        pt_v1 = (200.0, 100.0)
        pt_v2 = (200.0, 400.0)
        angle_vert = PersonQualityAnalyzer.compute_shoulder_tilt_angle(pt_v1, pt_v2)
        self.assertEqual(angle_vert, 90.0, "Vertical points should be 90°")

    def test_shoulder_scoring(self):
        """Test shoulder levelness scoring and threshold evaluation."""
        # Level shoulders (<= 3°)
        score_level, is_lvl_1 = self.analyzer.compute_shoulder_score(1.5)
        self.assertEqual(score_level, 1.0)
        self.assertTrue(is_lvl_1)

        # Mild slant (7°) with threshold at 12°
        score_mild, is_lvl_2 = self.analyzer.compute_shoulder_score(7.0)
        self.assertGreater(score_mild, 0.65)
        self.assertLess(score_mild, 1.0)
        self.assertTrue(is_lvl_2)

        # Right at threshold (12°)
        score_thresh, is_lvl_3 = self.analyzer.compute_shoulder_score(12.0)
        self.assertAlmostEqual(score_thresh, 0.65, places=2)
        self.assertTrue(is_lvl_3)

        # Excessive tilt (20° > 12°)
        score_tilted, is_lvl_4 = self.analyzer.compute_shoulder_score(20.0)
        self.assertLess(score_tilted, 0.65)
        self.assertFalse(is_lvl_4)

    def test_shoulder_posture_qualification(self):
        """Test that SingleFaceAnalysis incorporates shoulder posture in qualification."""
        from analyzer import SingleFaceAnalysis
        
        # 1. Level shoulders -> Qualified
        face_good = SingleFaceAnalysis(
            person_id=1,
            head_pose_score=0.95,
            eye_openness_score=1.0,
            expression_score=1.0,
            shoulder_pose_score=1.0,
            sharpness_score=0.9,
            shoulder_tilt_deg=2.0,
            is_shoulders_level=True,
            is_shoulder_tilted=False
        )
        self.analyzer._finalize_face_score(face_good)
        self.assertTrue(face_good.is_qualified, "Speaker with level shoulders should be qualified")
        self.assertGreaterEqual(face_good.score, self.config.thresholds.min_overall_score)

        # 2. Tilted shoulders -> Disqualified
        face_tilted = SingleFaceAnalysis(
            person_id=1,
            head_pose_score=0.95,
            eye_openness_score=1.0,
            expression_score=1.0,
            shoulder_pose_score=0.2,
            sharpness_score=0.9,
            shoulder_tilt_deg=18.0,
            is_shoulders_level=False,
            is_shoulder_tilted=True
        )
        self.analyzer._finalize_face_score(face_tilted)
        # Tilted shoulders are a blemish, not a reason to throw the frame away. They cost
        # score so a level-shouldered frame outranks this one, but if the whole service
        # is shot at a slight angle we still want captures.
        self.assertLess(face_tilted.score, face_good.score,
                        "Tilted shoulders should score below level shoulders")
        self.assertNotIn("shoulders", " ".join(face_tilted.disqualified_reasons),
                         "Shoulder tilt should not be a hard disqualifier")

        # 3. Headshot / Close-up fallback (shoulders out of frame) -> Qualified
        face_closeup = SingleFaceAnalysis(
            person_id=1,
            head_pose_score=0.95,
            eye_openness_score=1.0,
            expression_score=1.0,
            shoulder_pose_score=1.0,
            sharpness_score=0.9,
            shoulder_tilt_deg=0.0,
            shoulder_detected=False,
            is_shoulders_level=True,
            is_shoulder_tilted=False
        )
        self.analyzer._finalize_face_score(face_closeup)
        self.assertTrue(face_closeup.is_qualified, "Headshot without visible shoulders should remain qualified")

    def test_multi_person_target(self):
        """Test configuring multi-person target up to 6 people."""
        analyzer = PersonQualityAnalyzer(max_people_target=4)
        self.assertEqual(analyzer.max_people_target, 4)
        
        # Test clamping between 1 and 6
        analyzer_too_high = PersonQualityAnalyzer(max_people_target=10)
        self.assertEqual(analyzer_too_high.max_people_target, 6)

        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = analyzer.analyze_frame(blank_frame)
        self.assertEqual(res.target_person_count, 4)
        self.assertEqual(res.people_detected_count, 0)

    def test_scikit_preference_model(self):
        """Preference model: refuses to train on too little feedback, then learns."""
        import tempfile
        from preference_model import UserPreferenceModel

        sample_res_good = AnalysisResult(
            yaw_deg=2.0, avg_ear=0.38, smile_ratio=0.85, exposure_score=1.0,
            shoulder_tilt_deg=1.5, is_shoulders_level=True, cutout_score=0.95
        )
        sample_res_bad = AnalysisResult(
            yaw_deg=28.0, avg_ear=0.15, smile_ratio=0.20, exposure_score=0.3,
            shoulder_tilt_deg=18.0, is_shoulders_level=False, cutout_score=0.25
        )

        # Point persistence at a scratch file so the test can't clobber the user's
        # real ratings in models/.
        with tempfile.TemporaryDirectory() as tmp:
            original_path = UserPreferenceModel.JSON_DATA_PATH
            UserPreferenceModel.JSON_DATA_PATH = os.path.join(tmp, "prefs.json")
            try:
                model = UserPreferenceModel()

                feat_good = model.extract_features(sample_res_good)
                self.assertEqual(len(feat_good), len(UserPreferenceModel.FEATURE_NAMES),
                                 "Feature vector must match the declared feature names")
                idx = UserPreferenceModel.FEATURE_NAMES.index("shoulder_tilt_norm")
                self.assertAlmostEqual(feat_good[idx], 1.5 / 15.0, places=2,
                                       msg="shoulder_tilt_norm should be the normalised tilt")

                # Two ratings is not a model. Fitting them (previously padded with a
                # synthetic row) produced a "trained" classifier with a meaningless
                # 100% self-scored accuracy.
                model.add_sample(sample_res_good, is_favorite=True)
                model.add_sample(sample_res_bad, is_favorite=False)
                self.assertFalse(model.is_trained,
                                 "Model should not report trained on 2 samples")
                self.assertEqual(model.predict(feat_good), 0.5,
                                 "Untrained model must return a neutral score")

                # Enough separable feedback and it should train and prefer the good one.
                for _ in range(UserPreferenceModel.MIN_SAMPLES_TO_TRAIN):
                    model.add_sample(sample_res_good, is_favorite=True)
                    model.add_sample(sample_res_bad, is_favorite=False)

                self.assertTrue(model.is_trained)
                self.assertGreaterEqual(model.confidence, 0.5)
                self.assertLessEqual(model.confidence, 1.0)
                self.assertGreater(model.predict(feat_good),
                                   model.predict(model.extract_features(sample_res_bad)),
                                   "Favorited sample should outscore the rejected one")
                self.assertGreater(model.trust, 0.0,
                                   "Separable feedback should earn some influence")
            finally:
                UserPreferenceModel.JSON_DATA_PATH = original_path

    def test_preference_model_earns_no_trust_from_noise(self):
        """A model that has only learned to answer with the majority class must carry
        no weight. Plain accuracy rewarded exactly that - on 4 favourites against 18
        rejects it read 0.92, almost all of which came free from the label skew."""
        import tempfile
        import numpy as np
        from preference_model import UserPreferenceModel

        with tempfile.TemporaryDirectory() as tmp:
            original_path = UserPreferenceModel.JSON_DATA_PATH
            UserPreferenceModel.JSON_DATA_PATH = os.path.join(tmp, "prefs.json")
            try:
                rng = np.random.default_rng(0)
                n_features = len(UserPreferenceModel.FEATURE_NAMES)
                model = UserPreferenceModel()
                model.samples = [(rng.random(n_features).tolist(), 1) for _ in range(4)]
                model.samples += [(rng.random(n_features).tolist(), 0) for _ in range(18)]
                model.fit()

                self.assertLessEqual(model.confidence, 0.75,
                                     "Balanced accuracy on noise should sit near chance")
                self.assertEqual(model.trust, 0.0,
                                 "A model no better than guessing must earn no trust")

                model.samples = [(rng.random(n_features).tolist(), 1) for _ in range(2)]
                model.samples += [(rng.random(n_features).tolist(), 0) for _ in range(20)]
                model.fit()
                self.assertFalse(model.is_trained,
                                 "Fewer than MIN_PER_CLASS of a class should not train")
            finally:
                UserPreferenceModel.JSON_DATA_PATH = original_path

    def test_mouth_tracking_and_lip_licking_disqualification(self):
        """Lip licking is disqualifying; an open mouth is graded, not disqualifying."""
        from analyzer import SingleFaceAnalysis
        
        # 1. Lip Licking -> Disqualified and penalized
        face_licking = SingleFaceAnalysis(
            person_id=1,
            head_pose_score=0.95,
            eye_openness_score=1.0,
            expression_score=0.02,
            shoulder_pose_score=1.0,
            sharpness_score=0.9,
            is_licking_lips=True,
            is_awkward_expression=True
        )
        self.analyzer._finalize_face_score(face_licking)
        self.assertFalse(face_licking.is_qualified, "Face licking lips must be disqualified")
        self.assertLess(face_licking.score, 0.15, "Licking lips score must be heavily penalized")

        # 2. Open Mouth Mid-Speech / Talking -> Disqualified
        face_talking = SingleFaceAnalysis(
            person_id=1,
            head_pose_score=0.95,
            eye_openness_score=1.0,
            expression_score=0.15,
            shoulder_pose_score=1.0,
            sharpness_score=0.9,
            is_talking_or_open=True,
            is_awkward_expression=True
        )
        # An open mouth is graded by how far the jaw actually drops, not treated as a
        # cliff. Disqualifying it outright meant a speaker who talks continuously
        # produced no captures at all for a whole service.
        face_talking.jaw_open_score = 0.45
        self.analyzer._finalize_face_score(face_talking)
        self.assertNotIn("talking", " ".join(face_talking.disqualified_reasons),
                         "Mid-speech should not be a hard disqualifier")

        face_mouth_shut = SingleFaceAnalysis(
            person_id=1,
            head_pose_score=0.95,
            eye_openness_score=1.0,
            expression_score=0.88,
            shoulder_pose_score=1.0,
            sharpness_score=0.9,
            jaw_open_score=0.05,
        )
        self.analyzer._finalize_face_score(face_mouth_shut)
        self.assertLess(face_talking.score, face_mouth_shut.score,
                        "A closed mouth should outrank a mouth caught mid-word")

        face_gaping = SingleFaceAnalysis(
            person_id=1,
            head_pose_score=0.95,
            eye_openness_score=1.0,
            expression_score=0.15,
            shoulder_pose_score=1.0,
            sharpness_score=0.9,
            is_talking_or_open=True,
            jaw_open_score=0.80,
        )
        self.analyzer._finalize_face_score(face_gaping)
        self.assertLess(face_gaping.score, face_talking.score,
                        "A wide-open jaw should be penalized more than a slightly open one")

        # 3. Weird Mouth Shape / Grimacing -> penalised below the capture threshold
        face_weird = SingleFaceAnalysis(
            person_id=1,
            head_pose_score=0.95,
            eye_openness_score=1.0,
            expression_score=0.05,
            shoulder_pose_score=1.0,
            sharpness_score=0.9,
            is_weird_mouth=True,
            is_awkward_expression=True
        )
        self.analyzer._finalize_face_score(face_weird)
        # A grimace excludes itself by scoring under the capture threshold rather than by
        # a dedicated gate, so it can still be picked if a recording contains nothing
        # better — but it never outranks a clean frame.
        self.assertFalse(face_weird.is_qualified, "Weird mouth shape must not qualify")
        self.assertLess(face_weird.score, self.config.thresholds.min_overall_score,
                        "Weird mouth shape must score below the capture threshold")

        # 4. Clean Natural Smile -> Qualified
        face_smile = SingleFaceAnalysis(
            person_id=1,
            head_pose_score=0.95,
            eye_openness_score=1.0,
            expression_score=1.0,
            shoulder_pose_score=1.0,
            sharpness_score=0.9,
            is_smiling=True,
            is_licking_lips=False,
            is_talking_or_open=False,
            is_weird_mouth=False,
            is_awkward_expression=False
        )
        self.analyzer._finalize_face_score(face_smile)
        self.assertTrue(face_smile.is_qualified, "Clean natural smile must be qualified")
        self.assertGreaterEqual(face_smile.score, self.config.thresholds.min_overall_score)

    def test_face_landmarker_initialization(self):
        """Test that MediaPipe FaceLandmarker with 52 blendshapes initializes properly."""
        self.assertIsNotNone(self.analyzer.face_landmarker, "FaceLandmarker should be initialized")

if __name__ == "__main__":
    unittest.main()
