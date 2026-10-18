import cv2
import numpy as np
import math
import os
import sys
import time
import contextlib
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from config import QualityWeights, QualityThresholds
from preference_model import UserPreferenceModel

# Where the downloaded ML weights live. Overridable so a container can bake them into
# an image path that isn't shadowed by a mounted volume — with the weights under a bind
# mount, a fresh host directory hides them and the container re-downloads on every
# start, or fails outright when it has no outbound internet.
MODELS_DIR = os.environ.get("TG_MODELS_DIR", "").strip() or os.path.join(os.getcwd(), "models")


@contextlib.contextmanager
def _quiet_native_startup():
    """Silences the MediaPipe/TFLite C++ chatter emitted while models are built.

    Those warnings ("Feedback manager requires a model with a single signature",
    "Sets FaceBlendshapesGraph acceleration to xnnpack") are informational and
    unavoidable — they are written straight to file descriptor 2 by the native layer,
    so GLOG_minloglevel and Python's logging module cannot reach them. Seven lines per
    analyzer are enough to bury the capture log and any genuine error, so fd 2 is
    redirected for the duration of model construction only. Per-frame analysis is left
    untouched, and the loaders report their own failures on stdout, so a model that
    fails to load still says so.

    Set TG_VERBOSE_MEDIAPIPE=1 to see the native output.
    """
    if os.environ.get("TG_VERBOSE_MEDIAPIPE", "").strip():
        yield
        return
    try:
        sys.stderr.flush()
        saved_fd = os.dup(2)
    except Exception:
        # No real fd 2 to juggle (embedded/pythonw); just run without suppression.
        yield
        return
    devnull_fd = None
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        try:
            os.dup2(saved_fd, 2)
        finally:
            os.close(saved_fd)
            if devnull_fd is not None:
                os.close(devnull_fd)

# Standard 21 Hand Landmark skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

@dataclass
class SingleFaceAnalysis:
    person_id: int = 1
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    # Unsmoothed detection box. bbox is EMA-smoothed for a steady HUD, which lags
    # during movement — crops for landmarking must use the raw box or they drift.
    raw_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    score: float = 0.0
    is_qualified: bool = False
    is_selected: bool = True       # Whether this person is actively selected for scoring
    
    # Angles (degrees)
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    
    # Shoulder / Posture Metrics
    shoulder_tilt_deg: float = 0.0
    is_shoulders_level: bool = True
    is_shoulder_tilted: bool = False
    shoulder_detected: bool = False
    shoulder_score: float = 1.0
    shoulder_points: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    
    # Metrics
    avg_ear: float = 0.35
    is_blinking: bool = False
    smile_ratio: float = 0.50
    is_smiling: bool = False
    is_awkward_expression: bool = False
    is_looking_down: bool = False
    mar: float = 0.20
    
    # Precise Mouth & Expression Blendshape Metrics
    is_licking_lips: bool = False
    is_weird_mouth: bool = False
    is_talking_or_open: bool = False
    jaw_open_score: float = 0.0
    smile_blendshape_score: float = 0.0
    tongue_out_score: float = 0.0
    eye_squint_score: float = 0.0
    eye_look_down_score: float = 0.0      # Gaze dropped, distinct from the head being down
    mesh_from_blendshapes: bool = False   # True when the 478-pt mesh actually resolved
    disqualified_reasons: List[str] = field(default_factory=list)

    # Cutout suitability — how well this frame survives having its background removed
    # and being dropped onto a slide, which is the end use for these captures.
    cutout_score: float = 0.5
    subject_bbox: Optional[Tuple[int, int, int, int]] = None
    subject_height_ratio: float = 0.0   # Subject height as a fraction of frame height
    is_subject_clipped: bool = False    # Subject runs off an edge of the frame
    hands_on_torso: bool = False        # Hands overlap the body, which muddies the matte
    arms_clear: bool = False            # Arms held wide or at the sides - cuts out cleanly
    torso_clear_score: float = 1.0      # 1.0 = nothing crossing the body, 0.0 = heavily crossed
    torso_crossing: float = 0.0         # Raw measure: widest edge run across the body
    background_score: float = 1.0       # 1.0 = clean/plain behind the subject
    pose_points: Optional[Dict[str, Any]] = None
    is_prop_moment: bool = False        # True when speaker is holding/presenting a prop on stage
    prop_bbox: Optional[Tuple[int, int, int, int]] = None
    
    # Score Breakdown
    head_pose_score: float = 0.0
    eye_openness_score: float = 0.0
    expression_score: float = 0.0
    shoulder_pose_score: float = 1.0
    sharpness_score: float = 0.0
    exposure_score: float = 1.0
    centering_score: float = 0.5
    
    laplacian_variance: float = 0.0
    face_brightness: float = 128.0
    mesh_points: Optional[np.ndarray] = None

@dataclass
class AnalysisResult:
    face_detected: bool = False
    people_detected_count: int = 0
    target_person_count: int = 1
    overall_score: float = 0.0
    smoothed_score: float = 0.0    # EMA-smoothed composite score
    ml_score: float = 0.5          # Machine learning preference model score (0.0 - 1.0)
    ml_trained: bool = False       # Whether ML model is trained
    ml_weight_applied: float = 0.0 # How much the learned model actually contributed
    is_qualified: bool = False
    is_stable: bool = False        # Whether score is stable (low variance over recent frames)
    
    # Primary (First) Face Detailed Breakdown (for backward compatibility & single-person telemetry)
    head_pose_score: float = 0.0
    eye_openness_score: float = 0.0
    expression_score: float = 0.0
    shoulder_pose_score: float = 1.0
    sharpness_score: float = 0.0
    hand_pose_score: float = 1.0
    exposure_score: float = 1.0
    centering_score: float = 0.5
    
    # Angles (Primary Face)
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    
    # Shoulder / Posture Metrics (Primary Face)
    shoulder_tilt_deg: float = 0.0
    is_shoulders_level: bool = True
    is_shoulder_tilted: bool = False
    shoulder_detected: bool = False
    shoulder_points: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    
    # Metrics (Primary Face)
    left_ear: float = 0.35
    right_ear: float = 0.35
    avg_ear: float = 0.35
    is_blinking: bool = False
    
    mar: float = 0.20
    smile_ratio: float = 0.50
    is_smiling: bool = False
    is_awkward_expression: bool = False
    is_licking_lips: bool = False
    is_weird_mouth: bool = False
    is_talking_or_open: bool = False
    
    laplacian_variance: float = 0.0
    hand_laplacian_variance: float = 0.0   # Normalised, comparable with laplacian_variance
    face_brightness: float = 128.0
    cutout_score: float = 0.5
    subject_height_ratio: float = 0.0
    is_subject_clipped: bool = False
    hands_on_torso: bool = False
    background_score: float = 1.0
    torso_clear_score: float = 1.0
    torso_crossing: float = 0.0
    is_prop_moment: bool = False
    prop_bbox: Optional[Tuple[int, int, int, int]] = None
    disqualified_reasons: List[str] = field(default_factory=list)
    hands_covering_face: bool = False
    is_hand_blurry: bool = False
    is_hand_weird: bool = False
    is_looking_down: bool = False
    hands_detected_count: int = 0
    
    # Primary Face Bounding Box & Landmarks
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    mesh_points: Optional[np.ndarray] = None
    hand_skeletons: Optional[List[List[Tuple[int, int]]]] = field(default_factory=list)
    pose_skeletons: Optional[List[Dict[str, Any]]] = field(default_factory=list)
    
    # List of all analyzed individual faces (up to 6)
    faces: List[SingleFaceAnalysis] = field(default_factory=list)

class PersonQualityAnalyzer:
    YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    YUNET_PATH = os.path.join(MODELS_DIR, "yunet.onnx")
    
    FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    FACE_LANDMARKER_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")

    HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    HAND_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")

    POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    POSE_MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")

    def __init__(self, weights: Optional[QualityWeights] = None, thresholds: Optional[QualityThresholds] = None,
                 ema_alpha: float = 0.3, stability_window: int = 10, stability_threshold: float = 0.08,
                 blink_suppression_ms: int = 300, max_people_target: int = 1,
                 selected_person_ids: Optional[List[int]] = None,
                 require_temporal_stability: bool = True,
                 pose_hand_interval: int = 1,
                 use_ml_preference: bool = True,
                 ml_weight: float = 0.35,
                 min_consecutive_qualified: int = 2,
                 require_exact_people_count: bool = False,
                 capture_prop_moments: bool = True):
        self.weights = weights or QualityWeights()
        self.thresholds = thresholds or QualityThresholds()
        self.max_people_target = max(1, min(6, max_people_target))  # Clamp between 1 and 6
        # Live capture waits for the score to settle before firing, so it doesn't save
        # the frame either side of a good moment. Scanning a file has the whole timeline
        # available and should judge each frame on its own merits — with these gates on,
        # a single excellent frame surrounded by bad ones is thrown away.
        self.require_temporal_stability = require_temporal_stability
        self.use_ml_preference = bool(use_ml_preference)
        # Ceiling on how much the learned preference model may contribute once it is
        # fully matured; the effective weight ramps up to this with samples and accuracy.
        self.weights_ml_cap = max(0.0, min(1.0, float(ml_weight)))
        self.min_consecutive_qualified = max(1, int(min_consecutive_qualified))
        self.require_exact_people_count = bool(require_exact_people_count)
        self.capture_prop_moments = bool(capture_prop_moments)

        # Hand/pose re-detection cadence, and the cached results reused in between.
        # The cached landmarks are re-matched against the current faces every frame, so
        # only their positions go stale — by at most one analysis interval.
        self.pose_hand_interval = max(1, int(pose_hand_interval))
        self._detail_tick: int = 0
        self._cached_hand_result = None
        self._cached_pose_result = None
        self.selected_person_ids: set = set(selected_person_ids if selected_person_ids is not None else [1, 2, 3, 4, 5, 6])
        
        # User Preference ML Model (Scikit-Learn)
        self.preference_model = UserPreferenceModel()
        
        # Temporal smoothing state
        self._ema_alpha = ema_alpha
        self._ema_score: Optional[float] = None
        
        # Score stability tracking
        self._stability_window = stability_window
        self._stability_threshold = stability_threshold
        self._recent_scores: deque = deque(maxlen=stability_window)
        
        # Blink suppression
        self._blink_suppression_ms = blink_suppression_ms
        self._last_blink_time: float = 0.0
        self._blink_suppressed: bool = False
        
        # Consecutive qualified frame counter
        self._consecutive_qualified: int = 0
        
        # Face detection persistence (caches up to 6 faces)
        self._face_persistence_frames: int = 8
        self._frames_since_last_face: int = 999
        self._last_detected_faces: List[np.ndarray] = []
        self._smooth_bboxes: Dict[int, Tuple[float, float, float, float]] = {}  # Per-person EMA smoothed bboxes
        self._bbox_ema_alpha: float = 0.20
        
        self._ensure_models_exist()

        # The face detector is not optional — without it nothing else runs. Say so
        # clearly rather than letting OpenCV raise about an unreadable file path.
        if not os.path.exists(self.YUNET_PATH):
            raise RuntimeError(
                f"Face detector model missing at '{self.YUNET_PATH}' and it could not be "
                f"downloaded. If this machine has no internet access, fetch the models on "
                f"another machine and place them in that directory (or point TG_MODELS_DIR "
                f"at one that has them)."
            )

        # Calibrated YuNet score threshold (0.45) + robust skin chroma check for 100% human face recognition
        self.detector = cv2.FaceDetectorYN.create(
            self.YUNET_PATH, "", (320, 320), score_threshold=0.45, nms_threshold=0.3
        )
        self.current_input_size = (320, 320)
        
        # Model construction is the only place the native layer is chatty.
        with _quiet_native_startup():
            self._init_landmarkers()

        # 3D Canonical Head Model for solvePnP
        self.model_3d_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Right eye center
            (225.0, 170.0, -135.0),      # Left eye center
            (-150.0, -150.0, -125.0),    # Right mouth corner
            (150.0, -150.0, -125.0)      # Left mouth corner
        ], dtype=np.float64)

    def _init_landmarkers(self):
        """Builds the MediaPipe face / hand / pose landmarkers.

        Each loader degrades to None on failure so the analyzer still runs with
        whatever did load, rather than refusing to start.
        """
        # Initialize MediaPipe FaceLandmarker with 52 Blendshapes
        self.face_landmarker = None
        try:
            face_base_options = mp_python.BaseOptions(model_asset_path=self.FACE_LANDMARKER_MODEL_PATH)
            face_options = mp_vision.FaceLandmarkerOptions(
                base_options=face_base_options,
                output_face_blendshapes=True,
                # One face per call: the landmarker is run on a crop of each detected
                # face rather than on the whole frame (see _landmark_face).
                num_faces=1,
                min_face_detection_confidence=0.30,
                min_face_presence_confidence=0.30,
                min_tracking_confidence=0.30
            )
            self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_options)
        except Exception as e:
            print(f"[PersonQualityAnalyzer] Warning initializing FaceLandmarker: {e}")

        # Initialize MediaPipe HandLandmarker
        self.hand_landmarker = None
        try:
            base_options = mp_python.BaseOptions(model_asset_path=self.HAND_MODEL_PATH)
            hand_options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4
            )
            self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)
        except Exception as e:
            print(f"[PersonQualityAnalyzer] Warning initializing HandLandmarker: {e}")

        # Initialize MediaPipe PoseLandmarker
        self.pose_landmarker = None
        try:
            pose_base_options = mp_python.BaseOptions(model_asset_path=self.POSE_MODEL_PATH)
            pose_options = mp_vision.PoseLandmarkerOptions(
                base_options=pose_base_options,
                num_poses=6,
                min_pose_detection_confidence=0.35,
                min_pose_presence_confidence=0.35,
                min_tracking_confidence=0.35
            )
            self.pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)
        except Exception as e:
            print(f"[PersonQualityAnalyzer] Warning initializing PoseLandmarker: {e}")

    def _ensure_models_exist(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        if not os.path.exists(self.YUNET_PATH):
            print(f"[PersonQualityAnalyzer] Downloading YuNet Face Detector to '{self.YUNET_PATH}'...")
            try:
                urllib.request.urlretrieve(self.YUNET_URL, self.YUNET_PATH)
            except Exception as e:
                print(f"Error downloading YuNet: {e}")

        if not os.path.exists(self.FACE_LANDMARKER_MODEL_PATH):
            print(f"[PersonQualityAnalyzer] Downloading MediaPipe FaceLandmarker to '{self.FACE_LANDMARKER_MODEL_PATH}'...")
            try:
                urllib.request.urlretrieve(self.FACE_LANDMARKER_URL, self.FACE_LANDMARKER_MODEL_PATH)
            except Exception as e:
                print(f"Error downloading FaceLandmarker: {e}")

        if not os.path.exists(self.HAND_MODEL_PATH):
            print(f"[PersonQualityAnalyzer] Downloading MediaPipe HandLandmarker to '{self.HAND_MODEL_PATH}'...")
            try:
                urllib.request.urlretrieve(self.HAND_MODEL_URL, self.HAND_MODEL_PATH)
            except Exception as e:
                print(f"Error downloading HandLandmarker: {e}")

        if not os.path.exists(self.POSE_MODEL_PATH):
            print(f"[PersonQualityAnalyzer] Downloading MediaPipe PoseLandmarker to '{self.POSE_MODEL_PATH}'...")
            try:
                urllib.request.urlretrieve(self.POSE_MODEL_URL, self.POSE_MODEL_PATH)
            except Exception as e:
                print(f"Error downloading PoseLandmarker: {e}")

    @staticmethod
    def compute_shoulder_tilt_angle(pt_left: Tuple[float, float], pt_right: Tuple[float, float]) -> float:
        """Computes the absolute angle in degrees of the shoulder line relative to horizontal.
        
        0.0° is perfectly level.
        """
        dx = abs(float(pt_right[0]) - float(pt_left[0]))
        dy = abs(float(pt_right[1]) - float(pt_left[1]))
        if dx < 1e-6:
            return 90.0 if dy > 0 else 0.0
        angle_rad = math.atan2(dy, dx)
        return round(math.degrees(angle_rad), 1)

    def compute_shoulder_score(self, tilt_deg: float) -> Tuple[float, bool]:
        """Returns (shoulder_score: float, is_level: bool) based on max_shoulder_tilt_deg threshold."""
        max_tilt = max(1.0, self.thresholds.max_shoulder_tilt_deg)
        is_level = tilt_deg <= max_tilt
        if tilt_deg <= 3.0:
            score = 1.0
        elif tilt_deg <= max_tilt:
            score = 1.0 - 0.35 * ((tilt_deg - 3.0) / max(0.1, max_tilt - 3.0))
        else:
            score = max(0.0, 0.65 - 0.65 * ((tilt_deg - max_tilt) / max_tilt))
        return round(max(0.0, min(1.0, score)), 2), is_level

    def _finalize_face_score(self, f_res: SingleFaceAnalysis):
        """Computes the composite score, then decides whether the frame is usable at all.

        The distinction matters. Anything merely imperfect — mid-word, shoulders tilted,
        gaze dropped — is scored down and left in the running, because ranking picks the
        best of what actually happened and retention replaces it when something better
        comes along. Only frames that cannot be used no matter what else is available are
        disqualified outright. Gating on imperfections instead produced no captures at
        all from a speaker who talks continuously.
        """
        min_sharp = self.thresholds.min_sharpness_laplacian
        if f_res.laplacian_variance == 0.0 and f_res.sharpness_score > 0.0:
            f_res.laplacian_variance = self._score_to_sharpness(f_res.sharpness_score)

        w = self.weights
        prop_active = bool(f_res.is_prop_moment and getattr(self, "capture_prop_moments", True))
        # hand_pose used to be multiplied by a hardcoded 1.0, so its weight was simply
        # dead score. It now reflects whether the hands are clear of the body.
        # When actively presenting a prop, hands in front of the torso are intentional.
        if prop_active:
            hand_pose_score = 1.0
        else:
            hand_pose_score = 0.70 if f_res.hands_on_torso else (1.0 if f_res.arms_clear else 0.90)
        raw_score = (
            w.head_pose * f_res.head_pose_score +
            w.eye_openness * f_res.eye_openness_score +
            w.expression * f_res.expression_score +
            w.shoulder_pose * f_res.shoulder_pose_score +
            w.sharpness * f_res.sharpness_score +
            w.hand_pose * hand_pose_score +
            w.exposure * f_res.exposure_score +
            w.cutout * f_res.cutout_score
        )

        # --- Graded penalties + learned operator comment multipliers ----------
        penalties = getattr(self.preference_model, "learned_penalties", {}) or {}
        p_head_down = float(penalties.get("head_down", 1.0))
        p_eyes = float(penalties.get("eyes_closed", 1.0))
        p_mouth = float(penalties.get("mouth_open", 1.0))
        p_hand = float(penalties.get("hand_obstruction", 1.0))
        p_blur = float(penalties.get("blur", 1.0))

        # Mouth openness tapers rather than falling off a cliff. A slightly parted mouth
        # is normal in a good shot; a jaw dropped mid-word is not, but is still worth
        # keeping if the alternative is nothing.
        jaw = f_res.jaw_open_score
        if jaw <= 0.30:
            mouth_mult = 1.0
        elif jaw >= 0.65:
            mouth_mult = 0.45
        else:
            mouth_mult = 1.0 - 0.55 * ((jaw - 0.30) / 0.35)
        if mouth_mult < 1.0 and p_mouth > 1.0:
            mouth_mult = max(0.15, mouth_mult ** p_mouth)
        raw_score *= mouth_mult

        if f_res.is_licking_lips:
            raw_score *= 0.15
        elif f_res.is_weird_mouth:
            raw_score *= max(0.20, 0.55 / p_mouth)

        # Gaze dropped & head bowed: NEVER waive a bowed head or hidden eyes, even if
        # holding a prop! A usable thumbnail requires visible eyes and an engaged head.
        effective_looking_down = (
            f_res.is_looking_down
            or f_res.pitch_deg < -14.0
            or f_res.eye_look_down_score > 0.52
        )
        if f_res.eye_look_down_score > 0.35:
            gaze_pen = 0.30 * min(1.0, (f_res.eye_look_down_score - 0.35) / 0.35) * p_head_down
            raw_score *= max(0.30, 1.0 - gaze_pen)
        if effective_looking_down:
            raw_score *= max(0.15, 0.45 / p_head_down)
        if f_res.avg_ear < 0.28 and not f_res.is_blinking:
            # Squinting or eyes partially obscured by glasses frames when tilting down
            raw_score *= max(0.35, 0.75 / p_eyes)
        if f_res.is_blinking:
            raw_score *= max(0.05, 0.20 / p_eyes)
        if f_res.is_shoulder_tilted:
            raw_score *= 0.80
        if f_res.hands_on_torso and not prop_active and p_hand > 1.0:
            raw_score *= max(0.45, 1.0 / p_hand)

        # Something crossing the body - a table, lectern or music stand - is the worst
        # thing for a cutout: the matte either keeps it or punches a hole through the
        # subject. Graded rather than disqualifying, because some speakers never step
        # away from the table, and for them this scales every frame alike and leaves the
        # ranking untouched.
        torso_factor = 0.55 + 0.45 * f_res.torso_clear_score
        if torso_factor < 0.95 and p_hand > 1.0:
            torso_factor = max(0.30, torso_factor ** p_hand)
        raw_score *= torso_factor

        # Heavy penalty for motion blur / low sharpness
        effective_min_sharp = min_sharp * (1.0 + 0.25 * (p_blur - 1.0))
        is_blurry = f_res.laplacian_variance < effective_min_sharp
        if is_blurry:
            sharp_ratio = max(0.05, f_res.laplacian_variance / max(1.0, effective_min_sharp))
            raw_score *= (sharp_ratio ** 1.5)

        f_res.score = round(max(0.0, min(1.0, raw_score)), 3)

        # --- Hard disqualifiers: unusable, not merely imperfect ---------------
        reasons: List[str] = []
        if f_res.score < self.thresholds.min_overall_score:
            reasons.append(f"score {f_res.score:.0%} below {self.thresholds.min_overall_score:.0%}")
        effective_max_yaw = self.thresholds.max_yaw_deg * (1.35 if prop_active else 1.0)
        if abs(f_res.yaw_deg) > effective_max_yaw:
            reasons.append(f"head turned {abs(f_res.yaw_deg):.0f}deg")
        if f_res.is_blinking:
            reasons.append("eyes closed")
        if effective_looking_down:
            reasons.append("head down")
        if f_res.is_licking_lips:
            reasons.append("tongue/lip")
        if is_blurry:
            reasons.append(f"blurry ({f_res.laplacian_variance:.0f} < {effective_min_sharp:.0f})")

        f_res.disqualified_reasons = reasons
        f_res.is_qualified = not reasons

    def analyze_frame(self, frame: np.ndarray) -> AnalysisResult:
        result = AnalysisResult()
        result.target_person_count = self.max_people_target
        orig_h, orig_w, c = frame.shape
        if orig_h == 0 or orig_w == 0:
            return result
        
        # Downscale frame for analysis (max width 640)
        target_w = 640
        if orig_w > target_w:
            scale = target_w / float(orig_w)
            target_h = int(orig_h * scale)
            proc_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            scale = 1.0
            target_h, target_w = orig_h, orig_w
            proc_frame = frame

        # Update YuNet detector input size
        if self.current_input_size != (target_w, target_h):
            self.detector.setInputSize((target_w, target_h))
            self.current_input_size = (target_w, target_h)
            
        # Detect all faces in frame
        faces_detected = self.detector.detect(proc_frame)
        
        # === Spatial Centroid Tracking & Slot Lock ===
        # Match incoming face detections to persistent person_id slots (1 to 6) based on spatial distance.
        # This prevents ID swapping / jumping face windows when people move or scale slightly.
        current_detections = []
        if faces_detected[1] is not None and len(faces_detected[1]) > 0:
            for f in faces_detected[1]:
                fw, fh = int(f[2]), int(f[3])
                fx, fy = int(f[0]), int(f[1])
                if fw > 15 and fh > 15:
                    aspect = float(fw) / float(fh)
                    if 0.40 <= aspect <= 1.60:
                        # Verify skin tone presence to reject cymbals, drums & hardware
                        if self._is_valid_human_skin_roi(proc_frame, fx, fy, fw, fh):
                            cx = fx + fw / 2.0
                            cy = fy + fh / 2.0
                            current_detections.append({'face': f, 'cx': cx, 'cy': cy, 'area': fw * fh})

        # Match new detections to existing tracked slots (1 to 6)
        matched_slots: Dict[int, np.ndarray] = {}
        unmatched_detections = list(current_detections)
        
        # 1. Match existing active slots to nearest new detection
        if hasattr(self, '_tracked_slots'):
            for pid, slot_info in list(self._tracked_slots.items()):
                if not unmatched_detections:
                    break
                tcx, tcy = slot_info['cx'], slot_info['cy']
                # Find nearest detection within 140px
                best_match_idx = None
                min_dist = 999999.0
                for idx, det in enumerate(unmatched_detections):
                    dist = math.hypot(det['cx'] - tcx, det['cy'] - tcy)
                    if dist < 140.0 and dist < min_dist:
                        min_dist = dist
                        best_match_idx = idx

                if best_match_idx is not None:
                    matched_det = unmatched_detections.pop(best_match_idx)
                    matched_slots[pid] = matched_det['face']
                    self._tracked_slots[pid] = {
                        'cx': matched_det['cx'], 'cy': matched_det['cy'],
                        'face': matched_det['face'], 'missed': 0
                    }
                else:
                    # Increment missed frames grace period
                    slot_info['missed'] += 1
                    if slot_info['missed'] <= self._face_persistence_frames:
                        matched_slots[pid] = slot_info['face']
                    else:
                        del self._tracked_slots[pid]
                        if pid in self._smooth_bboxes:
                            del self._smooth_bboxes[pid]
        else:
            self._tracked_slots = {}

        # 2. Assign remaining unmatched detections to free slots (1 to 6)
        # Sort remaining unmatched by area descending
        unmatched_detections.sort(key=lambda d: d['area'], reverse=True)
        for det in unmatched_detections:
            # Find lowest available slot ID between 1 and max_people_target
            free_pid = None
            for candidate_pid in range(1, self.max_people_target + 1):
                if candidate_pid not in matched_slots and candidate_pid not in self._tracked_slots:
                    free_pid = candidate_pid
                    break
            
            if free_pid is not None:
                matched_slots[free_pid] = det['face']
                self._tracked_slots[free_pid] = {
                    'cx': det['cx'], 'cy': det['cy'],
                    'face': det['face'], 'missed': 0
                }

        # Filter slots to up to max_people_target
        active_pids = sorted([p for p in matched_slots.keys() if p <= self.max_people_target])

        if not active_pids:
            result.face_detected = False
            result.people_detected_count = 0
            self._smooth_bboxes.clear()
            return result

        result.face_detected = True
        result.people_detected_count = len(active_pids)
        
        # Analyze each individual face (up to 6)
        analyzed_faces: List[SingleFaceAnalysis] = []
        any_blinking = False

        for pid in active_pids:
            face_data = matched_slots[pid]
            face_analysis = self._analyze_single_face(pid, face_data, frame, orig_w, orig_h, scale)
            face_analysis.is_selected = (pid in self.selected_person_ids)
            analyzed_faces.append(face_analysis)
            if face_analysis.is_selected and face_analysis.is_blinking:
                any_blinking = True

        result.faces = analyzed_faces
        
        # MediaPipe Hand, Pose, and Face Tracking across the frame
        result.hand_skeletons = []
        result.pose_skeletons = []
        result.hands_covering_face = False
        result.is_hand_blurry = False
        result.is_hand_weird = False
        result.hands_detected_count = 0
        
        mp_image = None
        if self.face_landmarker is not None or self.hand_landmarker is not None or self.pose_landmarker is not None:
            try:
                rgb_proc = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_proc)
            except Exception:
                pass

        # 1. MediaPipe FaceLandmarker with 52 Blendshapes (Mouth, Lip Licking, Talking & Blink Analysis)
        # Run per detected face on an upscaled crop. Run on the whole frame it finds
        # nothing whenever the subject is more than a few feet from the camera, which
        # silently drops every blendshape metric back to the crude 2D fallbacks.
        if self.face_landmarker is not None:
            for f_item in analyzed_faces:
                self._landmark_face(frame, f_item)

        # Update blink suppression cooldown after blendshapes have refined blink state
        any_blinking = any(f.is_selected and f.is_blinking for f in analyzed_faces)
        if self.require_temporal_stability and self._blink_suppression_ms > 0:
            now_ts = time.time()
            if any_blinking:
                self._last_blink_time = now_ts
                self._blink_suppressed = False
            elif self._last_blink_time > 0.0:
                elapsed_ms = (now_ts - self._last_blink_time) * 1000.0
                self._blink_suppressed = elapsed_ms < self._blink_suppression_ms
            else:
                self._blink_suppressed = False
        else:
            self._blink_suppressed = False

        # 2. Hand Tracking
        if self.hand_landmarker is not None and mp_image is not None:
            try:
                if self._detail_tick % self.pose_hand_interval == 0:
                    self._cached_hand_result = self.hand_landmarker.detect(mp_image)
                hand_result = self._cached_hand_result
                if hand_result and hand_result.hand_landmarks:
                    result.hands_detected_count = len(hand_result.hand_landmarks)
                    for hand_pts in hand_result.hand_landmarks:
                        pts_full_res = []
                        xs, ys = [], []
                        for pt in hand_pts:
                            px = int(pt.x * target_w / scale)
                            py = int(pt.y * target_h / scale)
                            pts_full_res.append((px, py))
                            xs.append(px)
                            ys.append(py)
                            
                            # Check against all detected face bboxes
                            for f_item in analyzed_faces:
                                bx, by, bw, bh = f_item.bbox
                                if bx <= px <= bx + bw and by <= py <= by + bh:
                                    result.hands_covering_face = True

                        result.hand_skeletons.append(pts_full_res)
                        
                        min_x, max_x = max(0, min(xs)), min(orig_w, max(xs))
                        min_y, max_y = max(0, min(ys)), min(orig_h, max(ys))
                        hand_crop = frame[min_y:max_y, min_x:max_x]
                        
                        if hand_crop.size > 0:
                            # Measured the same way as the face - resampled to a fixed
                            # height first - so the two are comparable. Previously the
                            # hand was a raw full-resolution Laplacian while the face
                            # had been normalised, so the relative test compared two
                            # different scales and never fired: visibly motion-blurred
                            # hands were reported as sharp.
                            hand_gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
                            hand_norm = self._resize_to_height(
                                hand_gray, self.thresholds.sharpness_face_height)
                            hand_lap_var = float(cv2.Laplacian(hand_norm, cv2.CV_64F).var())
                            # Keep the worst hand: with two hands in frame, reporting
                            # whichever happened to be measured last hid the blurry one.
                            if (result.hand_laplacian_variance <= 0.0
                                    or hand_lap_var < result.hand_laplacian_variance):
                                result.hand_laplacian_variance = round(hand_lap_var, 1)

                            face_lap = analyzed_faces[0].laplacian_variance
                            # A gesturing hand smears while the face stays sharp, so the
                            # giveaway is the ratio, not an absolute floor.
                            if (hand_lap_var < self.thresholds.min_hand_sharpness or
                                    (face_lap > 1.0 and
                                     hand_lap_var < face_lap * self.thresholds.hand_sharpness_ratio)):
                                result.is_hand_blurry = True

                        index_tip = np.array(pts_full_res[8])
                        pinky_tip = np.array(pts_full_res[20])
                        hand_span = np.linalg.norm(index_tip - pinky_tip)
                        
                        primary_bbox = analyzed_faces[0].bbox
                        if primary_bbox is not None:
                            bx, by, bw, bh = primary_bbox
                            if min_y <= by + bh and hand_span > bw * 0.5:
                                result.is_hand_weird = True
            except Exception:
                pass

        result.hand_pose_score = 1.0

        # 3. Pose & Shoulder Tracking
        if self.pose_landmarker is not None and mp_image is not None:
            try:
                if self._detail_tick % self.pose_hand_interval == 0:
                    self._cached_pose_result = self.pose_landmarker.detect(mp_image)
                pose_result = self._cached_pose_result
                if pose_result and pose_result.pose_landmarks:
                    for pose_lms in pose_result.pose_landmarks:
                        if len(pose_lms) > 12:
                            lm_nose = pose_lms[0]
                            lm_ls = pose_lms[11]  # Left shoulder (anatomical left)
                            lm_rs = pose_lms[12]  # Right shoulder (anatomical right)
                            
                            nose_pt = (int(lm_nose.x * target_w / scale), int(lm_nose.y * target_h / scale))
                            ls_pt = (int(lm_ls.x * target_w / scale), int(lm_ls.y * target_h / scale))
                            rs_pt = (int(lm_rs.x * target_w / scale), int(lm_rs.y * target_h / scale))
                            
                            ls_vis = getattr(lm_ls, 'visibility', 1.0)
                            rs_vis = getattr(lm_rs, 'visibility', 1.0)
                            ls_pres = getattr(lm_ls, 'presence', 1.0)
                            rs_pres = getattr(lm_rs, 'presence', 1.0)
                            
                            shoulders_visible = (
                                (ls_vis is None or ls_vis > 0.3) and
                                (rs_vis is None or rs_vis > 0.3) and
                                (ls_pres is None or ls_pres > 0.3) and
                                (rs_pres is None or rs_pres > 0.3)
                            )
                            
                            # Match this pose to closest face in analyzed_faces
                            matched_face = None
                            min_dist = float('inf')
                            for f_item in analyzed_faces:
                                bx, by, bw, bh = f_item.bbox
                                fcx = bx + bw / 2.0
                                fcy = by + bh / 2.0
                                mid_sx = (ls_pt[0] + rs_pt[0]) / 2.0
                                mid_sy = (ls_pt[1] + rs_pt[1]) / 2.0
                                dist_nose = math.hypot(nose_pt[0] - fcx, nose_pt[1] - fcy)
                                dist_mid = math.hypot(mid_sx - fcx, mid_sy - (by + bh))
                                dist = min(dist_nose, dist_mid)
                                
                                if dist < max(bw * 2.5, bh * 2.5) and dist < min_dist:
                                    min_dist = dist
                                    matched_face = f_item

                            tilt_deg = 0.0
                            is_level = True
                            sh_score = 1.0
                            if shoulders_visible:
                                tilt_deg = self.compute_shoulder_tilt_angle(ls_pt, rs_pt)
                                sh_score, is_level = self.compute_shoulder_score(tilt_deg)
                                
                            pose_dict = {
                                'left_shoulder': ls_pt,
                                'right_shoulder': rs_pt,
                                'tilt_deg': tilt_deg,
                                'is_level': is_level,
                                'score': sh_score,
                                'visible': shoulders_visible,
                                'person_id': matched_face.person_id if matched_face else None
                            }
                            if len(pose_lms) > 14:
                                pose_dict['left_elbow'] = (int(pose_lms[13].x * target_w / scale), int(pose_lms[13].y * target_h / scale))
                                pose_dict['right_elbow'] = (int(pose_lms[14].x * target_w / scale), int(pose_lms[14].y * target_h / scale))

                            # Wrists and hips drive the cutout checks (hands over torso,
                            # subject extent), so pull them out when they are visible.
                            for name, idx in (('left_wrist', 15), ('right_wrist', 16),
                                              ('left_hip', 23), ('right_hip', 24)):
                                if len(pose_lms) > idx:
                                    lm = pose_lms[idx]
                                    if getattr(lm, 'visibility', 1.0) > 0.5:
                                        pose_dict[name] = (int(lm.x * target_w / scale),
                                                           int(lm.y * target_h / scale))

                            # Full visible-landmark cloud, used to bound the subject.
                            visible_pts = [
                                (int(lm.x * target_w / scale), int(lm.y * target_h / scale))
                                for lm in pose_lms if getattr(lm, 'visibility', 1.0) > 0.5
                            ]
                            pose_dict['visible_points'] = visible_pts
                            result.pose_skeletons.append(pose_dict)
                            
                            if matched_face is not None:
                                matched_face.pose_points = pose_dict
                                matched_face.shoulder_detected = shoulders_visible
                                matched_face.shoulder_points = (ls_pt, rs_pt) if shoulders_visible else None
                                matched_face.shoulder_tilt_deg = tilt_deg
                                matched_face.is_shoulders_level = is_level
                                matched_face.is_shoulder_tilted = not is_level
                                matched_face.shoulder_score = sh_score
                                matched_face.shoulder_pose_score = sh_score
            except Exception:
                pass

        self._detail_tick += 1

        # Finalize score and qualification across all analyzed faces (incorporating shoulder metrics)
        # Cutout scoring needs the pose landmarks matched above, so it runs here.
        for f_item in analyzed_faces:
            self._compute_cutout_score(frame, f_item, result.hand_skeletons)
            self._finalize_face_score(f_item)

        # Populate root level fields from Primary (First) Face for telemetry & backward compatibility
        primary = analyzed_faces[0]
        result.face_bbox = primary.bbox
        result.head_pose_score = primary.head_pose_score
        result.eye_openness_score = primary.eye_openness_score
        result.expression_score = primary.expression_score
        result.shoulder_pose_score = primary.shoulder_pose_score
        result.sharpness_score = primary.sharpness_score
        result.exposure_score = primary.exposure_score
        result.centering_score = primary.centering_score
        result.yaw_deg = primary.yaw_deg
        result.pitch_deg = primary.pitch_deg
        result.roll_deg = primary.roll_deg
        result.shoulder_tilt_deg = primary.shoulder_tilt_deg
        result.is_shoulders_level = primary.is_shoulders_level
        result.is_shoulder_tilted = primary.is_shoulder_tilted
        result.shoulder_detected = primary.shoulder_detected
        result.shoulder_points = primary.shoulder_points
        result.left_ear = primary.avg_ear
        result.right_ear = primary.avg_ear
        result.avg_ear = primary.avg_ear
        result.is_blinking = primary.is_blinking
        result.mar = primary.mar
        result.smile_ratio = primary.smile_ratio
        result.is_smiling = primary.is_smiling
        result.is_awkward_expression = primary.is_awkward_expression
        result.is_licking_lips = primary.is_licking_lips
        result.is_weird_mouth = primary.is_weird_mouth
        result.is_talking_or_open = primary.is_talking_or_open
        result.is_looking_down = primary.is_looking_down
        result.laplacian_variance = primary.laplacian_variance
        result.face_brightness = primary.face_brightness
        result.cutout_score = primary.cutout_score
        result.subject_height_ratio = primary.subject_height_ratio
        result.is_subject_clipped = primary.is_subject_clipped
        result.hands_on_torso = primary.hands_on_torso
        result.background_score = primary.background_score
        result.torso_clear_score = primary.torso_clear_score
        result.torso_crossing = primary.torso_crossing
        result.is_prop_moment = any(f.is_selected and f.is_prop_moment for f in analyzed_faces) or primary.is_prop_moment
        result.prop_bbox = primary.prop_bbox
        if result.is_prop_moment and result.prop_bbox is None:
            for f_item in analyzed_faces:
                if f_item.is_selected and f_item.prop_bbox is not None:
                    result.prop_bbox = f_item.prop_bbox
                    break
        result.hand_pose_score = 1.0 if primary.is_prop_moment else (
            0.70 if primary.hands_on_torso else (1.0 if primary.arms_clear else 0.90)
        )
        result.mesh_points = primary.mesh_points

        # Composite Overall Score Calculation strictly for ACTIVE SELECTED faces
        selected_face_scores = [f.score for f in analyzed_faces if f.is_selected]
        
        if not selected_face_scores:
            # All detected faces are unselected / ignored
            raw_score = 0.0
        elif len(selected_face_scores) == 1:
            raw_score = selected_face_scores[0]
        else:
            min_s = min(selected_face_scores)
            avg_s = sum(selected_face_scores) / len(selected_face_scores)
            raw_score = 0.60 * min_s + 0.40 * avg_s

        # Apply global environmental/gesture penalties
        if result.is_hand_weird and not result.is_prop_moment:
            raw_score *= 0.55
        if result.hands_covering_face:
            raw_score *= 0.10
        if result.is_hand_blurry:
            # Graded by how smeared the hand is relative to the face, not a flat cut.
            # A speaker gesturing while they talk has some hand motion in most frames,
            # so a fixed 0.45 was a near-constant halving that buried everything else
            # and left no way for the sharpest-handed frame of a moment to win. Now a
            # slightly soft hand costs little and a badly smeared one costs real score,
            # which is what lets buffer refinement pick the crisp version.
            face_lap = max(1.0, result.laplacian_variance)
            ratio = max(0.0, min(1.0, result.hand_laplacian_variance / face_lap))
            limit = max(0.01, self.thresholds.hand_sharpness_ratio)
            raw_score *= 0.70 + 0.30 * min(1.0, ratio / limit)

        rule_score = max(0.0, min(1.0, raw_score))
        
        # Scikit-Learn Machine Learning Preference Prediction
        try:
            features = UserPreferenceModel.extract_features(primary if primary else result, self.thresholds)
            ml_pred = self.preference_model.predict(features)
            result.ml_score = ml_pred
            result.ml_trained = self.preference_model.is_trained
            
            # Blend the rule score with the learned preference, but scale the learned
            # half by how much it has actually earned. At a flat 0.35 a model trained on
            # a couple of dozen ratings was overriding the heuristics outright - pulling
            # a frame the rules scored 0.87 down to 0.34. Weighting by sample count and
            # cross-validated accuracy lets it grow into that influence as it learns.
            if self.use_ml_preference and self.preference_model.is_trained:
                maturity = min(1.0, len(self.preference_model.samples) / 50.0)
                # trust is the distance past chance on balanced accuracy, so a model
                # that has only learned to answer with the majority class gets zero
                # influence rather than the high number raw accuracy would have given it.
                trust = self.preference_model.trust
                w = self.weights_ml_cap * maturity * trust
                final_score = (1.0 - w) * rule_score + w * ml_pred
                result.ml_weight_applied = round(w, 3)
            else:
                final_score = rule_score
                result.ml_weight_applied = 0.0
        except Exception:
            result.ml_score = 0.5
            result.ml_trained = False
            final_score = rule_score

        result.overall_score = round(max(0.0, min(1.0, final_score)), 3)
        
        # Temporal EMA Smoothing
        if self._ema_score is None:
            self._ema_score = result.overall_score
        else:
            self._ema_score = self._ema_alpha * result.overall_score + (1.0 - self._ema_alpha) * self._ema_score
        result.smoothed_score = round(self._ema_score, 3)
        
        # Score Stability Check
        self._recent_scores.append(result.overall_score)
        if len(self._recent_scores) >= 3:
            score_std = float(np.std(list(self._recent_scores)))
            result.is_stable = score_std <= self._stability_threshold
        else:
            result.is_stable = False
        
        # Qualification check: ALL active selected faces must be individually qualified!
        selected_faces = [f for f in analyzed_faces if f.is_selected]
        all_faces_qualified = len(selected_faces) > 0 and all(f.is_qualified for f in selected_faces)
        
        # Collect why this frame can't be captured, so "score is high but nothing saves"
        # is answerable from the HUD instead of by reading the scoring code.
        frame_reasons: List[str] = []
        for f_item in selected_faces:
            for reason in f_item.disqualified_reasons:
                tag = reason if len(selected_faces) == 1 else f"P{f_item.person_id}: {reason}"
                if tag not in frame_reasons:
                    frame_reasons.append(tag)
        if not selected_faces:
            frame_reasons.append("no person selected")
        exact_count_mismatch = (
            self.require_exact_people_count and
            result.people_detected_count != self.max_people_target
        )
        if exact_count_mismatch:
            frame_reasons.append(
                f"people count {result.people_detected_count} != {self.max_people_target}")
        # Only mention the frame score when no individual face already explained it,
        # otherwise a single-person shot reports the same problem twice.
        if (result.overall_score < self.thresholds.min_overall_score
                and not any("score" in r for r in frame_reasons)):
            frame_reasons.append(
                f"frame score {result.overall_score:.0%} below {self.thresholds.min_overall_score:.0%}")
        if result.hands_covering_face:
            frame_reasons.append("hand over face")
        if self._blink_suppressed:
            frame_reasons.append("blink cooldown")

        basic_qualified = (
            result.overall_score >= self.thresholds.min_overall_score and
            all_faces_qualified and
            not exact_count_mismatch and
            not result.hands_covering_face and
            not self._blink_suppressed
        )
        
        if basic_qualified:
            self._consecutive_qualified += 1
        else:
            self._consecutive_qualified = 0

        min_consec = max(1, getattr(self, "min_consecutive_qualified", 2))
        if self.require_temporal_stability:
            if basic_qualified and not result.is_stable:
                frame_reasons.append("score still settling")
            if basic_qualified and self._consecutive_qualified < min_consec:
                frame_reasons.append("waiting for a 2nd good frame")
            result.is_qualified = (
                basic_qualified and
                result.is_stable and
                self._consecutive_qualified >= min_consec
            )
        else:
            result.is_qualified = basic_qualified

        result.disqualified_reasons = frame_reasons

        return result

    def _analyze_single_face(self, person_id: int, best_face: np.ndarray, frame: np.ndarray,
                            orig_w: int, orig_h: int, scale: float) -> SingleFaceAnalysis:
        """Analyzes quality metrics for one specific face in the frame."""
        f_res = SingleFaceAnalysis(person_id=person_id)
        
        fx_s, fy_s, fw_s, fh_s = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])
        
        fx = int(fx_s / scale)
        fy = int(fy_s / scale)
        fw = int(fw_s / scale)
        fh = int(fh_s / scale)
        
        fx, fy = max(0, fx), max(0, fy)
        fw, fh = min(orig_w - fx, fw), min(orig_h - fy, fh)
        
        f_res.raw_bbox = (fx, fy, fw, fh)

        # EMA Bounding Box Smoothing per person slot
        raw_bbox = (float(fx), float(fy), float(fw), float(fh))
        if person_id not in self._smooth_bboxes:
            self._smooth_bboxes[person_id] = raw_bbox
        else:
            a = self._bbox_ema_alpha
            prev = self._smooth_bboxes[person_id]
            self._smooth_bboxes[person_id] = (
                a * raw_bbox[0] + (1 - a) * prev[0],
                a * raw_bbox[1] + (1 - a) * prev[1],
                a * raw_bbox[2] + (1 - a) * prev[2],
                a * raw_bbox[3] + (1 - a) * prev[3],
            )
        
        sm = self._smooth_bboxes[person_id]
        f_res.bbox = (int(sm[0]), int(sm[1]), int(sm[2]), int(sm[3]))
        
        right_eye = (int(best_face[4] / scale), int(best_face[5] / scale))
        left_eye = (int(best_face[6] / scale), int(best_face[7] / scale))
        nose_tip = (int(best_face[8] / scale), int(best_face[9] / scale))
        right_mouth = (int(best_face[10] / scale), int(best_face[11] / scale))
        left_mouth = (int(best_face[12] / scale), int(best_face[13] / scale))
        
        mouth_center_x = (right_mouth[0] + left_mouth[0]) / 2.0
        mouth_center_y = (right_mouth[1] + left_mouth[1]) / 2.0
        chin = (int(mouth_center_x + (mouth_center_x - nose_tip[0]) * 0.3), int(mouth_center_y + (mouth_center_y - nose_tip[1]) * 1.5))
        
        landmarks_2d = np.array([nose_tip, chin, right_eye, left_eye, right_mouth, left_mouth], dtype=np.float64)
        f_res.mesh_points = landmarks_2d.astype(int)

        # 1. Image Sharpness on Face ROI & Eyes Region
        # Measured on a fixed-height copy of the face so the number means the same thing
        # whether the subject is 3ft or 30ft from the camera. Measured raw, a big near
        # face always beat a small far one no matter which was actually in focus.
        face_roi = frame[fy:fy+fh, fx:fx+fw]
        if face_roi.size > 0:
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            gray_norm = self._resize_to_height(gray_roi, self.thresholds.sharpness_face_height)
            lap_var_face = float(cv2.Laplacian(gray_norm, cv2.CV_64F).var())

            # Eye region sharpness (upper 15% to 55% of face ROI)
            roi_h, roi_w = gray_norm.shape
            eye_top = int(roi_h * 0.15)
            eye_bottom = int(roi_h * 0.55)
            eye_roi = gray_norm[eye_top:eye_bottom, :]
            if eye_roi.size > 40:
                lap_var_eyes = float(cv2.Laplacian(eye_roi, cv2.CV_64F).var())
                # Sharpness is weighted between eyes and full face ROI
                lap_var = 0.60 * lap_var_eyes + 0.40 * lap_var_face
            else:
                lap_var = lap_var_face
        else:
            gray_roi = None
            lap_var = 0.0

        f_res.laplacian_variance = lap_var
        f_res.sharpness_score = self._sharpness_to_score(lap_var)

        # 2. Exposure & Centering
        f_res.exposure_score, f_res.face_brightness = self._compute_exposure_score(face_roi, gray_roi)
        f_res.centering_score = self._compute_centering_score(fx, fy, fw, fh, orig_w, orig_h)

        # 3. Head Pose Estimation
        yaw, pitch, roll, pose_score = self._compute_head_pose(landmarks_2d, orig_w, orig_h)
        f_res.yaw_deg = yaw
        f_res.pitch_deg = pitch
        f_res.roll_deg = roll

        eye_mid_y = (right_eye[1] + left_eye[1]) / 2.0
        nose_eye_v = nose_tip[1] - eye_mid_y
        nose_mouth_v = mouth_center_y - nose_tip[1]

        is_looking_down_2d = False
        if nose_eye_v > 5:
            vert_ratio = nose_mouth_v / nose_eye_v
            is_looking_down_2d = vert_ratio < 0.50

        f_res.is_looking_down = is_looking_down_2d or pitch < -15.0
        f_res.head_pose_score = round(pose_score, 2)

        # 4. Eye Openness & Blink
        eye_dist = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
        is_blink, avg_ear = self._detect_blink(
            right_eye, left_eye, nose_tip, face_roi, gray_roi, fx, fy, fw, fh, eye_dist
        )
        f_res.avg_ear = round(avg_ear, 2)
        f_res.is_blinking = is_blink
        f_res.eye_openness_score = 0.0 if is_blink else 1.0

        # 5. Expression Analysis
        mouth_width = np.linalg.norm(np.array(left_mouth) - np.array(right_mouth))
        smile_ratio = mouth_width / max(1.0, eye_dist)
        f_res.smile_ratio = round(smile_ratio, 2)
        is_smiling = (smile_ratio >= 0.72)
        f_res.is_smiling = is_smiling

        mouth_is_open = False
        mar = 0.20
        if face_roi.size > 0 and gray_roi is not None and gray_roi.shape[0] > 10:
            m_left = max(0, int(min(right_mouth[0], left_mouth[0]) - fx))
            m_right = min(fw, int(max(right_mouth[0], left_mouth[0]) - fx))
            pad_x = int((m_right - m_left) * 0.15)
            m_left = m_left + pad_x
            m_right = m_right - pad_x
            m_top = max(0, int(nose_tip[1] - fy + (mouth_center_y - nose_tip[1]) * 0.25))
            m_bottom = min(fh, int(mouth_center_y - fy + (mouth_center_y - nose_tip[1]) * 0.7))

            if m_right > m_left + 4 and m_bottom > m_top + 4:
                mouth_roi = gray_roi[m_top:m_bottom, m_left:m_right]
                if mouth_roi.size > 20:
                    mh = mouth_roi.shape[0]
                    if mh >= 6:
                        upper_band = mouth_roi[:mh // 3, :]
                        center_band = mouth_roi[mh // 3: 2 * mh // 3, :]
                        if center_band.size > 0 and upper_band.size > 0:
                            center_mean = float(np.mean(center_band))
                            upper_mean = float(np.mean(upper_band))
                            if upper_mean > 10:
                                darkness_ratio = center_mean / upper_mean
                                mouth_is_open = darkness_ratio < 0.70
                                mar = round(max(0.0, min(1.0, 1.0 - darkness_ratio)), 2)

                    mouth_std = float(np.std(mouth_roi))
                    if mouth_std > 42.0:
                        mouth_is_open = True
                        mar = max(mar, 0.40)

        f_res.mar = mar
        f_res.is_awkward_expression = mouth_is_open and not is_smiling

        if is_smiling:
            expr_score = 1.0
        elif not mouth_is_open:
            expr_score = 0.95
        else:
            expr_score = 0.30

        f_res.expression_score = round(expr_score, 2)

        # Default Shoulder / Posture values (updated when pose landmarker runs on full frame)
        f_res.shoulder_tilt_deg = 0.0
        f_res.is_shoulders_level = True
        f_res.is_shoulder_tilted = False
        f_res.shoulder_detected = False
        f_res.shoulder_score = 1.0
        f_res.shoulder_pose_score = 1.0
        f_res.shoulder_points = None

        return f_res

    def _compute_cutout_score(self, frame: np.ndarray, f_res: SingleFaceAnalysis,
                              hand_skeletons: Optional[List[List[Tuple[int, int]]]] = None) -> None:
        """Scores how usable this frame is once the background is removed.

        A shot can be a flawless headshot and still be useless for a slide: the subject
        runs off the edge of frame, they are too small to enlarge, their hands are
        crossed over their chest, or they are standing against clutter that no matting
        tool will cut cleanly. Those are the failures this scores.
        """
        h, w = frame.shape[:2]
        pose = f_res.pose_points or {}
        pts = pose.get('visible_points') or []

        fx, fy, fw, fh = f_res.raw_bbox if (f_res.raw_bbox and f_res.raw_bbox[2] > 0) else f_res.bbox

        # Match hand skeletons that plausibly belong to this person
        matched_hands: List[List[Tuple[int, int]]] = []
        if hand_skeletons and fw > 0 and fh > 0:
            fcx = fx + fw / 2.0
            for hand in hand_skeletons:
                if not hand:
                    continue
                hcx = sum(pt[0] for pt in hand) / float(len(hand))
                hcy = sum(pt[1] for pt in hand) / float(len(hand))
                if abs(hcx - fcx) <= fw * 4.0 and (fy - fh * 1.5) <= hcy <= (fy + fh * 7.5):
                    matched_hands.append(hand)
        elif hand_skeletons:
            matched_hands = [hand for hand in hand_skeletons if hand]

        # --- Subject extent -------------------------------------------------
        # Prefer the pose landmark cloud; fall back to a rough body estimate from the
        # face box when the pose model found nothing.
        if len(pts) >= 4:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            sx0, sy0, sx1, sy1 = min(xs), min(ys), max(xs), max(ys)
            # Landmarks sit on the skeleton, not the silhouette; pad out to the body.
            pad_x = int((sx1 - sx0) * 0.18)
            sx0, sx1 = sx0 - pad_x, sx1 + pad_x
            sy0 = min(sy0, fy)          # include the top of the head
        elif fw > 0 and fh > 0:
            # ~7.5 head-heights for a standing adult, ~3 face-widths across the shoulders
            sx0, sx1 = fx - int(fw * 1.0), fx + fw + int(fw * 1.0)
            sy0, sy1 = fy, fy + int(fh * 7.5)
        else:
            f_res.cutout_score = 0.0
            return

        # Include matched hand landmarks so gesturing hands are never sliced off in _subject.png
        for hand in matched_hands:
            h_xs = [pt[0] for pt in hand]
            h_ys = [pt[1] for pt in hand]
            h_pad = max(8, int(fw * 0.15)) if fw > 0 else 12
            sx0 = min(sx0, min(h_xs) - h_pad)
            sx1 = max(sx1, max(h_xs) + h_pad)
            sy0 = min(sy0, min(h_ys) - h_pad)
            sy1 = max(sy1, max(h_ys) + h_pad)

        # Detect whether the speaker is holding/presenting a prop on stage
        if getattr(self, "capture_prop_moments", True):
            is_prop, prop_bbox = self._detect_held_prop(frame, f_res, matched_hands)
        else:
            is_prop, prop_bbox = False, None
        f_res.is_prop_moment = is_prop
        f_res.prop_bbox = prop_bbox

        if prop_bbox is not None:
            px, py, pw, ph = prop_bbox
            sx0 = min(sx0, px)
            sy0 = min(sy0, py)
            sx1 = max(sx1, px + pw)
            sy1 = max(sy1, py + ph)

        sx0, sy0 = max(0, sx0), max(0, sy0)
        sx1, sy1 = min(w, sx1), min(h, sy1)
        f_res.subject_bbox = (sx0, sy0, sx1 - sx0, sy1 - sy0)

        subj_h = max(1, sy1 - sy0)
        f_res.subject_height_ratio = round(subj_h / float(h), 3)

        # --- 1. Completeness: does the subject fit inside the frame? ---------
        # A silhouette sliced by the frame edge cuts off flat in the cutout.
        margin = max(4, int(w * 0.01))
        clipped_sides = 0
        if sx0 <= margin:
            clipped_sides += 1
        if sx1 >= w - margin:
            clipped_sides += 1
        if sy0 <= margin:                       # head cropped — the worst case
            clipped_sides += 2
        f_res.is_subject_clipped = clipped_sides > 0
        # The bottom edge is expected (subjects are cut at the waist), so it is ignored.
        completeness = max(0.0, 1.0 - 0.32 * clipped_sides)

        # --- 2. Size: enough pixels to enlarge on a slide? -------------------
        # Below ~35% of frame height the cutout has to be upscaled hard and goes soft.
        r = f_res.subject_height_ratio
        if r >= 0.75:
            size_score = 1.0
        elif r <= 0.20:
            size_score = 0.0
        else:
            size_score = (r - 0.20) / 0.55

        # --- 3. Hands clear of the torso ------------------------------------
        # Hands crossed over the chest merge into the body and read as a blob.
        hands_on_torso = False
        arms_clear = False
        ls, rs = pose.get('left_shoulder'), pose.get('right_shoulder')
        lh, rh = pose.get('left_hip'), pose.get('right_hip')
        if ls and rs:
            torso_x0, torso_x1 = min(ls[0], rs[0]), max(ls[0], rs[0])
            torso_y0 = min(ls[1], rs[1])
            if lh and rh:
                torso_y1 = max(lh[1], rh[1])
            else:
                torso_y1 = torso_y0 + int(abs(torso_x1 - torso_x0) * 1.4)
            # A hand anywhere across the body merges with it in the cutout, so this
            # trims only a sliver off each side rather than a wide margin.
            inset = int((torso_x1 - torso_x0) * 0.05)
            # Arms hanging naturally rest against the hips, which is normal and cuts out
            # fine. Only the chest and midriff count, or every relaxed pose gets flagged.
            torso_y1 = torso_y0 + int((torso_y1 - torso_y0) * 0.70)
            torso_x0, torso_x1 = torso_x0 + inset, torso_x1 - inset

            def _in_torso(pt):
                return torso_x0 <= pt[0] <= torso_x1 and torso_y0 <= pt[1] <= torso_y1

            # Prefer the hand landmarks: the pose model marks the wrist joint, which can
            # sit outboard of the body while the fingers are spread across the chest.
            for hand in (matched_hands or []):
                if not hand:
                    continue
                cx = sum(pt[0] for pt in hand) / float(len(hand))
                cy = sum(pt[1] for pt in hand) / float(len(hand))
                if _in_torso((cx, cy)):
                    hands_on_torso = True
                    break
            if not hands_on_torso:
                for wrist in (pose.get('left_wrist'), pose.get('right_wrist')):
                    if wrist and _in_torso(wrist):
                        hands_on_torso = True
                        break
            # Arms held clear of the body - gesturing wide, or hanging at the sides -
            # cut out cleanly and read as open posture. Worth rewarding rather than
            # merely not penalising: a speaker who gestures constantly has hands near
            # the torso in most frames, so "not on the torso" alone barely separates
            # them. The rare wide-arm moment is what should win.
            span = max(1.0, float(torso_x1 - torso_x0))
            for wrist in (pose.get('left_wrist'), pose.get('right_wrist')):
                if not wrist:
                    continue
                outboard = (wrist[0] < torso_x0 - span * 0.25) or (wrist[0] > torso_x1 + span * 0.25)
                below = wrist[1] > torso_y1
                if outboard or below:
                    arms_clear = True
                    break

        if is_prop:
            # Holding/presenting a prop is an intentional candid pose, not an accidental
            # chest-crossing gesture.
            hands_on_torso = False
            hands_score = 1.00
        else:
            if hands_on_torso:
                hands_score = 0.70
            elif arms_clear:
                hands_score = 1.00
            else:
                hands_score = 0.90

        f_res.hands_on_torso = hands_on_torso
        f_res.arms_clear = arms_clear

        # --- 4. Background cleanliness --------------------------------------
        f_res.background_score = self._background_cleanliness(frame, (sx0, sy0, sx1, sy1))

        # --- 5. Nothing crossing the body -----------------------------------
        crossing = self._torso_crossing(frame, pose)
        f_res.torso_crossing = round(crossing, 3)
        # Calibrated on real captures: ~0.07 standing clear, ~0.29 with a table across
        # the lower torso, ~0.39 pressed against it with both hands on top.
        torso_clear = max(0.0, min(1.0, 1.0 - (crossing - 0.05) / 0.35))
        if is_prop:
            # A held prop in front of the speaker creates horizontal edges that should
            # not be mistaken for a lectern or table cutting the body in half.
            torso_clear = max(0.90, torso_clear)
        f_res.torso_clear_score = round(torso_clear, 3)

        # Not folded into the cutout composite: at a fifth of an eighteen percent term it
        # moved the final score by about three points, while the frames it flagged were
        # the ones a matte cannot handle at all. It is applied as a multiplier below,
        # alongside the other things that spoil a frame outright.
        cutout = (
            0.35 * completeness +
            0.25 * size_score +
            0.15 * hands_score +
            0.25 * f_res.background_score
        )
        f_res.cutout_score = round(max(0.0, min(1.0, cutout)), 3)

    def _detect_held_prop(self, frame: np.ndarray, f_res: SingleFaceAnalysis,
                          matched_hands: Optional[List[List[Tuple[int, int]]]] = None
                          ) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """Detects when a speaker is actively holding or presenting a physical prop on stage.

        Evaluates raised/presenting hands (from 21-point hand skeletons or pose wrists)
        and checks whether the region around or between the hands contains a structured,
        non-skin, non-face object with distinct edges and contrast against the backdrop.
        """
        h, w = frame.shape[:2]
        fx, fy, fw, fh = f_res.raw_bbox if (f_res.raw_bbox and f_res.raw_bbox[2] > 0) else f_res.bbox
        if fw <= 0 or fh <= 0 or h < 32 or w < 32:
            return False, None

        # A candid prop moment still requires an engaged speaker: never trigger prop mode
        # when the speaker's head is bowed down, eyes are closed/hidden, or mouth is open mid-word.
        if (f_res.is_looking_down or f_res.is_blinking or f_res.pitch_deg < -13.0
                or f_res.eye_look_down_score > 0.45 or f_res.avg_ear < 0.28
                or f_res.is_talking_or_open):
            return False, None

        pose = f_res.pose_points or {}
        ls, rs = pose.get('left_shoulder'), pose.get('right_shoulder')
        lh, rh = pose.get('left_hip'), pose.get('right_hip')

        if ls and rs:
            sh_y = min(ls[1], rs[1])
            hip_y = max(lh[1], rh[1]) if (lh and rh) else sh_y + int(abs(ls[0] - rs[0]) * 1.4)
            t_span = max(1, hip_y - sh_y)
            max_active_y = sh_y + int(t_span * 0.72)
        else:
            max_active_y = fy + int(fh * 3.4)
        min_active_y = fy - int(fh * 0.8)

        # Handheld microphone exclusion box directly under the mouth/chin: a hand centred
        # right below the face holding a mic is normal speaking posture, not a stage prop.
        mic_x0 = fx + int(fw * 0.05)
        mic_x1 = fx + int(fw * 0.95)
        mic_y0 = fy + int(fh * 0.75)
        mic_y1 = fy + int(fh * 2.05)

        def _is_mic_hand(hcx: float, hcy: float) -> bool:
            return (mic_x0 <= hcx <= mic_x1) and (mic_y0 <= hcy <= mic_y1)

        # Collect active (raised/presenting) hand bounding boxes: (hx0, hy0, hx1, hy1, hcx, hcy)
        active_hands: List[Tuple[int, int, int, int, float, float]] = []
        for hand in (matched_hands or []):
            if not hand:
                continue
            xs = [pt[0] for pt in hand]
            ys = [pt[1] for pt in hand]
            hx0, hy0, hx1, hy1 = min(xs), min(ys), max(xs), max(ys)
            hcx = sum(xs) / float(len(xs))
            hcy = sum(ys) / float(len(ys))
            # Ignore hands directly covering the face or holding a mic under the chin
            if fx <= hcx <= fx + fw and fy <= hcy <= fy + fh:
                continue
            if _is_mic_hand(hcx, hcy):
                continue
            if min_active_y <= hcy <= max_active_y:
                active_hands.append((hx0, hy0, hx1, hy1, hcx, hcy))

        # Fall back to raised pose wrists when 21-pt hand landmarks didn't fire
        if not active_hands and pose:
            half_w = max(14, int(fw * 0.35))
            for wrist_key in ('left_wrist', 'right_wrist'):
                wrist = pose.get(wrist_key)
                if not wrist:
                    continue
                wx, wy = wrist
                if fx <= wx <= fx + fw and fy <= wy <= fy + fh:
                    continue
                if _is_mic_hand(float(wx), float(wy)):
                    continue
                if min_active_y <= wy <= max_active_y:
                    active_hands.append((wx - half_w, wy - half_w, wx + half_w, wy + half_w, float(wx), float(wy)))

        if not active_hands:
            return False, None

        # Build candidate prop regions: (px0, py0, px1, py1, is_two_hand, hand_union_box)
        candidates: List[Tuple[int, int, int, int, bool, Tuple[int, int, int, int]]] = []
        if len(active_hands) >= 2:
            h1, h2 = active_hands[0], active_hands[1]
            if abs(h1[4] - h2[4]) <= fw * 3.8 and abs(h1[5] - h2[5]) <= fh * 2.2:
                ux0, uy0 = min(h1[0], h2[0]), min(h1[1], h2[1])
                ux1, uy1 = max(h1[2], h2[2]), max(h1[3], h2[3])
                px0 = ux0 - max(12, int(fw * 0.25))
                py0 = uy0 - max(16, int(fh * 0.45))
                px1 = ux1 + max(12, int(fw * 0.25))
                py1 = uy1 + max(12, int(fh * 0.35))
                candidates.append((px0, py0, px1, py1, True, (ux0, uy0, ux1, uy1)))

        for hx0, hy0, hx1, hy1, hcx, hcy in active_hands:
            hw, hh = max(16, hx1 - hx0), max(16, hy1 - hy0)
            pad_x = max(18, int(max(fw * 0.55, hw * 0.80)))
            pad_top = max(18, int(max(fh * 0.60, hh * 0.85)))
            pad_bot = max(14, int(max(fh * 0.40, hh * 0.55)))
            candidates.append((
                hx0 - pad_x, hy0 - pad_top, hx1 + pad_x, hy1 + pad_bot,
                False, (hx0, hy0, hx1, hy1)
            ))

        # Estimate baseline torso/background brightness & colour to distinguish a held prop
        # from a plain shirt or plain curtain behind an empty gesturing hand.
        bg_sample_std = 0.0
        if ls and rs:
            tx0, tx1 = max(0, min(ls[0], rs[0])), min(w, max(ls[0], rs[0]))
            ty0 = max(0, min(ls[1], rs[1]))
            ty1 = min(h, ty0 + max(16, int((tx1 - tx0) * 0.8)))
            if tx1 - tx0 > 12 and ty1 - ty0 > 12:
                bg_sample_std = float(np.std(cv2.cvtColor(frame[ty0:ty1, tx0:tx1], cv2.COLOR_BGR2GRAY)))

        face_pad_x, face_pad_y = int(fw * 0.12), int(fh * 0.15)
        fx0, fy0 = max(0, fx - face_pad_x), max(0, fy - face_pad_y)
        fx1, fy1 = min(w, fx + fw + face_pad_x), min(h, fy + fh + face_pad_y)

        prop_strictness = float((getattr(self.preference_model, "learned_penalties", {}) or {}).get("prop_strictness", 1.0))

        for raw_px0, raw_py0, raw_px1, raw_py1, is_two_hand, (ux0, uy0, ux1, uy1) in candidates:
            px0, py0 = max(0, raw_px0), max(0, raw_py0)
            px1, py1 = min(w, raw_px1), min(h, raw_py1)
            cw, ch = px1 - px0, py1 - py0
            if cw < 24 or ch < 24:
                continue

            crop = frame[py0:py1, px0:px1]
            # 1. Build skin mask so bare hands/arms are excluded from the prop check
            ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
            cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
            ycrcb_skin = (cr >= 133) & (cr <= 177) & (cb >= 77) & (cb <= 127)
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            hsv_skin = (hue <= 22) & (sat >= 25) & (sat <= 170) & (val >= 40)
            skin_mask = (ycrcb_skin & hsv_skin).astype(np.uint8) * 255

            # Mask out the face box as well
            if fx0 < px1 and fx1 > px0 and fy0 < py1 and fy1 > py0:
                skin_mask[max(0, fy0 - py0):min(ch, fy1 - py0),
                          max(0, fx0 - px0):min(cw, fx1 - px0)] = 255

            # Dilate exclusion mask so the hand/finger silhouette edge itself isn't counted
            kernel = np.ones((7, 7), np.uint8)
            excl_dilated = cv2.dilate(skin_mask, kernel, iterations=1)
            valid_mask = excl_dilated == 0
            valid_count = int(np.count_nonzero(valid_mask))
            if valid_count < 100:
                continue

            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_crop, 50, 140)
            obj_edges = (edges > 0) & valid_mask
            edge_count = int(np.count_nonzero(obj_edges))
            edge_density = edge_count / float(valid_count)

            # Require structured non-skin edges AND luminance/colour contrast
            valid_pixels = gray_crop[valid_mask]
            intensity_std = float(np.std(valid_pixels)) if valid_pixels.size > 0 else 0.0
            min_edge_density = (0.022 if is_two_hand else 0.032) * prop_strictness

            if edge_density >= min_edge_density and intensity_std >= max(16.0 * prop_strictness, bg_sample_std * 0.45):
                ys_e, xs_e = np.where(obj_edges)
                if xs_e.size >= 12:
                    # Trim 5% outliers to bound the held object cleanly
                    ox0 = px0 + int(np.percentile(xs_e, 5))
                    oy0 = py0 + int(np.percentile(ys_e, 5))
                    ox1 = px0 + int(np.percentile(xs_e, 95))
                    oy1 = py0 + int(np.percentile(ys_e, 95))
                    # Reject objects clipped at the very bottom of the frame (e.g. a box or
                    # notes sitting on a table at the bottom edge of the camera view).
                    if oy1 >= h - 10 or uy1 >= h - 8:
                        continue
                    obj_w, obj_h = ox1 - ox0, oy1 - oy0
                    if obj_w >= max(12, int(fw * 0.20)) and obj_h >= max(12, int(fh * 0.20)):
                        # Union of the object bounds and the holding hand(s)
                        bx0 = max(0, min(ox0, ux0) - 6)
                        by0 = max(0, min(oy0, uy0) - 6)
                        bx1 = min(w, max(ox1, ux1) + 6)
                        by1 = min(h, max(oy1, uy1) + 6)
                        return True, (bx0, by0, bx1 - bx0, by1 - by0)

        return False, None

    @staticmethod
    def _torso_crossing(frame: np.ndarray, pose: Dict[str, Any]) -> float:
        """How far a horizontal edge runs across the body below the chest.

        A table, lectern or music stand in front of the speaker reads as a long,
        near-horizontal, high-contrast line spanning most of their width - and it is the
        single worst thing for a cutout, because the matte either keeps it or punches a
        hole through the body. Printed shirt text is high-contrast too, but it breaks
        into short runs, so measuring the *fraction of body width* a single row crosses
        separates the two.

        Deliberately returns a graded value rather than a verdict. Some speakers never
        step away from the table; for them every frame scores alike and the ranking is
        unaffected, which is the right outcome. For a speaker who does move clear, the
        clear frames win.
        """
        ls, rs = pose.get('left_shoulder'), pose.get('right_shoulder')
        if not (ls and rs):
            return 0.0
        lh, rh = pose.get('left_hip'), pose.get('right_hip')
        x0, x1 = min(ls[0], rs[0]), max(ls[0], rs[0])
        y_top = min(ls[1], rs[1])
        y_bot = max(lh[1], rh[1]) if (lh and rh) else y_top + int((x1 - x0) * 1.4)
        span = max(1, y_bot - y_top)

        h, w = frame.shape[:2]
        pad = int((x1 - x0) * 0.15)
        xa, xb = max(0, x0 - pad), min(w, x1 + pad)
        # From mid-chest down past the hips - where a table or lectern edge falls.
        ya, yb = max(0, int(y_top + span * 0.35)), min(h, int(y_bot + span * 0.45))
        if xb - xa < 20 or yb - ya < 10:
            return 0.0

        roi = cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_BGR2GRAY)
        gy = np.abs(cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3))
        peak = float(gy.max())
        if peak <= 1e-6:
            return 0.0
        strong = (gy / peak) > 0.35
        row_fraction = strong.mean(axis=1)
        # Average of the worst few rows, so a single noisy line doesn't decide it.
        worst = np.sort(row_fraction)[-5:]
        return float(worst.mean())

    @staticmethod
    def _background_cleanliness(frame: np.ndarray, subject_rect: Tuple[int, int, int, int]) -> float:
        """Rates how easy the area around the subject is to key out.

        Samples a ring just outside the subject and measures edge density there. A plain
        dark curtain scores near 1.0; a drum kit, furniture or lit signage scores low,
        because that is where automatic background removal leaves fringes and holes.

        This is a proxy, not a matte: clutter falling within the subject's own bounding
        box is masked out along with the subject, so something directly behind their
        head is not counted. It is weighted accordingly.
        """
        h, w = frame.shape[:2]
        sx0, sy0, sx1, sy1 = subject_rect
        band = max(12, int((sx1 - sx0) * 0.35))

        rx0, ry0 = max(0, sx0 - band), max(0, sy0 - band)
        rx1, ry1 = min(w, sx1 + band), min(h, sy1 + band)
        if rx1 - rx0 < 8 or ry1 - ry0 < 8:
            return 1.0

        outer = frame[ry0:ry1, rx0:rx1]
        gray = cv2.cvtColor(outer, cv2.COLOR_BGR2GRAY)
        # Mask out the subject so we measure only what is behind and beside them.
        mask = np.ones(gray.shape, dtype=np.uint8)
        mask[max(0, sy0 - ry0):max(0, sy1 - ry0), max(0, sx0 - rx0):max(0, sx1 - rx0)] = 0
        if int(mask.sum()) < 64:
            return 1.0

        edges = cv2.Canny(gray, 60, 160)
        edge_ratio = float(np.count_nonzero(edges & (mask * 255))) / float(mask.sum())

        # ~2% edge pixels is a clean backdrop; ~12% is heavy clutter.
        score = 1.0 - (edge_ratio - 0.02) / 0.10
        return round(max(0.0, min(1.0, score)), 3)

    @staticmethod
    def _resize_to_height(gray: np.ndarray, target_h: int) -> np.ndarray:
        """Resamples a grayscale ROI to a fixed height, preserving aspect ratio."""
        h, w = gray.shape[:2]
        if h < 2 or w < 2 or h == target_h:
            return gray
        scale = target_h / float(h)
        new_w = max(2, int(round(w * scale)))
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        return cv2.resize(gray, (new_w, target_h), interpolation=interp)

    def _sharpness_to_score(self, lap_var: float) -> float:
        """Maps normalised Laplacian variance onto 0..1 across the range real footage
        actually occupies, so soft and crisp frames get different scores.
        """
        floor = self.thresholds.sharpness_floor
        good = max(floor + 1.0, self.thresholds.sharpness_good)
        return round(max(0.0, min(1.0, (lap_var - floor) / (good - floor))), 3)

    def _score_to_sharpness(self, score: float) -> float:
        """Inverse of _sharpness_to_score, used to back-fill a plausible variance when
        only a score was supplied (synthetic results in tests and the ML feature path).
        """
        floor = self.thresholds.sharpness_floor
        good = max(floor + 1.0, self.thresholds.sharpness_good)
        return floor + score * (good - floor)

    # Width the face crop is resampled to before landmarking. The mesh model wants a
    # face filling a decent share of its input; 384 is enough for stable blendshapes
    # on a ~100px source face without paying for a larger inference.
    LANDMARK_CROP_WIDTH = 384
    LANDMARK_CROP_PAD = 0.85   # padding around the detection box, as a fraction of its longer side

    def _landmark_face(self, frame: np.ndarray, f_item: SingleFaceAnalysis) -> bool:
        """Runs the FaceLandmarker on a padded crop around one face and applies the
        blendshape-derived expression / eye metrics to `f_item`.

        Returns True if a mesh was found. On a miss the crude 2D metrics already set by
        _analyze_single_face are left untouched.
        """
        h, w = frame.shape[:2]
        bx, by, bw, bh = f_item.raw_bbox
        if bw <= 0 or bh <= 0:
            return False

        pad = int(max(bw, bh) * self.LANDMARK_CROP_PAD)
        x0, y0 = max(0, bx - pad), max(0, by - pad)
        x1, y1 = min(w, bx + bw + pad), min(h, by + bh + pad)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return False

        crop = frame[y0:y1, x0:x1]
        crop_w, crop_h = x1 - x0, y1 - y0

        # Normalise crop size so the mesh model sees a consistently scaled face.
        target_w = self.LANDMARK_CROP_WIDTH
        interp = cv2.INTER_CUBIC if crop_w < target_w else cv2.INTER_AREA
        target_h = max(1, int(crop_h * target_w / float(crop_w)))
        try:
            crop_rs = cv2.resize(crop, (target_w, target_h), interpolation=interp)
            mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(crop_rs, cv2.COLOR_BGR2RGB))
            res = self.face_landmarker.detect(mp_crop)
        except Exception:
            return False

        if not res or not res.face_landmarks:
            return False

        lms = res.face_landmarks[0]
        # Landmarks are normalised to the crop, so they map back to the source frame
        # through the crop rect alone — the resize factor cancels out.
        f_item.mesh_points = np.array(
            [[int(x0 + lm.x * crop_w), int(y0 + lm.y * crop_h)] for lm in lms], dtype=np.int32
        )

        bl = {}
        if getattr(res, 'face_blendshapes', None):
            bl = {b.category_name: b.score for b in res.face_blendshapes[0]}
        if not bl:
            return True

        g = bl.get
        jaw_open = g('jawOpen', 0.0)
        mouth_close = g('mouthClose', 0.0)
        tongue_out = g('tongueOut', 0.0)
        smile_score = (g('mouthSmileLeft', 0.0) + g('mouthSmileRight', 0.0)) / 2.0
        pucker = g('mouthPucker', 0.0)
        funnel = g('mouthFunnel', 0.0)
        roll_lower = g('mouthRollLower', 0.0)
        roll_upper = g('mouthRollUpper', 0.0)
        shrug_lower = g('mouthShrugLower', 0.0)
        shrug_upper = g('mouthShrugUpper', 0.0)
        press_l = g('mouthPressLeft', 0.0)
        press_r = g('mouthPressRight', 0.0)
        blink_l = g('eyeBlinkLeft', 0.0)
        blink_r = g('eyeBlinkRight', 0.0)
        squint = (g('eyeSquintLeft', 0.0) + g('eyeSquintRight', 0.0)) / 2.0
        look_down = (g('eyeLookDownLeft', 0.0) + g('eyeLookDownRight', 0.0)) / 2.0

        # Exact lip licking detection (tongue actually protruding or lower lip sucked)
        is_licking = (tongue_out > 0.16) or (roll_lower > 0.55 and jaw_open > 0.15)

        # Weird mouth shape / pursing / grimacing / speaking
        is_weird = (
            is_licking or
            pucker > 0.45 or
            funnel > 0.45 or
            (roll_lower > 0.55) or
            (roll_upper > 0.55) or
            ((shrug_lower > 0.50 or shrug_upper > 0.50) and smile_score < 0.20) or
            (press_l > 0.60 or press_r > 0.60)
        )

        # Open mouth mid-speech vs genuine smile
        is_talking = (jaw_open > 0.28 and smile_score < 0.25 and mouth_close < 0.30) or (jaw_open > 0.55)

        # Genuine smile check
        is_smile = (smile_score >= 0.25) and not is_weird and not is_licking and (jaw_open <= 0.50)

        # A half-closed eye reads as badly as a full blink in a still, and it is common
        # under stage lighting, so openness is graded rather than treated as a boolean.
        blink_max = max(blink_l, blink_r)
        is_blink = (blink_l > 0.50 and blink_r > 0.50) or blink_max > 0.70
        if blink_max <= 0.25:
            eye_open = 1.0
        elif blink_max >= 0.60:
            eye_open = 0.0
        else:
            eye_open = 1.0 - (blink_max - 0.25) / 0.35
        # Hard squinting still leaves the eyes readable, so it only trims the score.
        if squint > 0.55:
            eye_open *= max(0.55, 1.0 - (squint - 0.55))

        f_item.is_licking_lips = is_licking
        f_item.is_weird_mouth = is_weird
        f_item.is_talking_or_open = is_talking
        f_item.is_smiling = is_smile
        f_item.is_blinking = is_blink
        f_item.is_awkward_expression = is_weird or is_licking or is_talking
        # Eye gaze dropping is not the same as the head being down: a speaker glancing at
        # notes with a level head still photographs fine, so it is tracked separately and
        # only nudges the score. is_looking_down stays reserved for the head itself.
        f_item.eye_look_down_score = round(look_down, 2)
        f_item.smile_ratio = round(smile_score, 2)
        f_item.mar = round(jaw_open, 2)
        f_item.jaw_open_score = jaw_open
        f_item.smile_blendshape_score = smile_score
        f_item.tongue_out_score = tongue_out
        f_item.eye_squint_score = round(squint, 2)
        f_item.avg_ear = round(0.40 * eye_open, 2)

        if is_licking:
            expr_score = 0.05
        elif is_weird:
            expr_score = 0.15
        elif is_talking:
            expr_score = max(0.10, 0.35 - jaw_open * 0.4)
        elif is_smile:
            expr_score = min(1.0, 0.88 + smile_score * 0.12)
        else:
            expr_score = 0.88  # Natural resting closed mouth

        f_item.expression_score = round(expr_score, 2)
        f_item.eye_openness_score = round(max(0.0, min(1.0, eye_open)), 2)
        f_item.mesh_from_blendshapes = True
        return True

    def _compute_exposure_score(self, face_roi: np.ndarray, gray_roi: Optional[np.ndarray]) -> Tuple[float, float]:
        if face_roi is None or face_roi.size == 0 or gray_roi is None:
            return 0.5, 128.0
        
        mean_brightness = float(np.mean(gray_roi))
        min_b = self.thresholds.min_brightness
        max_b = self.thresholds.max_brightness
        ideal_low = 80.0
        ideal_high = 180.0
        
        if ideal_low <= mean_brightness <= ideal_high:
            exposure_score = 1.0
        elif mean_brightness < min_b:
            exposure_score = max(0.0, mean_brightness / min_b * 0.5)
        elif mean_brightness > max_b:
            exposure_score = max(0.0, (255.0 - mean_brightness) / (255.0 - max_b) * 0.5)
        elif mean_brightness < ideal_low:
            exposure_score = 0.5 + 0.5 * (mean_brightness - min_b) / (ideal_low - min_b)
        else:
            exposure_score = 0.5 + 0.5 * (max_b - mean_brightness) / (max_b - ideal_high)
        
        return round(max(0.0, min(1.0, exposure_score)), 2), round(mean_brightness, 1)

    def _compute_centering_score(self, fx: int, fy: int, fw: int, fh: int,
                                  frame_w: int, frame_h: int) -> float:
        face_cx = fx + fw / 2.0
        face_cy = fy + fh / 2.0
        
        frame_cx = frame_w / 2.0
        frame_cy = frame_h / 2.0
        
        dx = abs(face_cx - frame_cx) / frame_cx
        dy = abs(face_cy - frame_cy) / frame_cy
        
        center_dist = 0.6 * dx + 0.4 * dy
        centering_score = max(0.0, 1.0 - center_dist * 1.2)
        
        face_area_ratio = (fw * fh) / (frame_w * frame_h)
        if face_area_ratio < 0.02:
            centering_score *= 0.5
        elif face_area_ratio > 0.5:
            centering_score *= 0.7
        
        return round(max(0.0, min(1.0, centering_score)), 2)

    def _detect_blink(self, right_eye, left_eye, nose_tip, face_roi, gray_roi,
                      fx, fy, fw, fh, eye_dist) -> Tuple[bool, float]:
        if face_roi is None or face_roi.size == 0 or gray_roi is None:
            return False, 0.30
        
        eye_results = []
        for eye_center in [right_eye, left_eye]:
            ex = eye_center[0] - fx
            ey = eye_center[1] - fy
            
            eye_half_w = max(8, int(eye_dist * 0.15))
            eye_half_h = max(5, int(eye_dist * 0.08))
            
            ey1 = max(0, ey - eye_half_h)
            ey2 = min(gray_roi.shape[0], ey + eye_half_h)
            ex1 = max(0, ex - eye_half_w)
            ex2 = min(gray_roi.shape[1], ex + eye_half_w)
            
            eye_patch = gray_roi[ey1:ey2, ex1:ex2]
            if eye_patch.size < 10:
                eye_results.append(0.30)
                continue
            
            patch_h = eye_patch.shape[0]
            if patch_h >= 3:
                upper_third = eye_patch[:patch_h // 3, :]
                center_third = eye_patch[patch_h // 3: 2 * patch_h // 3, :]
                lower_third = eye_patch[2 * patch_h // 3:, :]
                
                upper_mean = float(np.mean(upper_third)) if upper_third.size > 0 else 128.0
                center_mean = float(np.mean(center_third)) if center_third.size > 0 else 128.0
                lower_mean = float(np.mean(lower_third)) if lower_third.size > 0 else 128.0
                
                surround_mean = (upper_mean + lower_mean) / 2.0
                if surround_mean > 5:
                    contrast = (surround_mean - center_mean) / surround_mean
                else:
                    contrast = 0.0
                
                eye_std = float(np.std(eye_patch))
                approx_ear = 0.15 + 0.25 * min(1.0, contrast * 3.0) + 0.10 * min(1.0, eye_std / 30.0)
                eye_results.append(approx_ear)
            else:
                eye_results.append(0.30)
        
        avg_ear = sum(eye_results) / len(eye_results) if eye_results else 0.30
        
        pixel_blink = False
        if gray_roi is not None and gray_roi.shape[0] > fh * 0.4:
            eye_roi = gray_roi[int(fh * 0.2):int(fh * 0.4), :]
            if eye_roi.size > 0:
                eye_std = float(np.std(eye_roi))
                pixel_blink = eye_std < 10.0
        
        geometric_blink = avg_ear < self.thresholds.min_ear
        
        if geometric_blink and pixel_blink:
            is_blink = True
        elif geometric_blink:
            is_blink = avg_ear < (self.thresholds.min_ear - 0.03)
        else:
            is_blink = False
        
        return is_blink, avg_ear

    def _compute_head_pose(self, landmarks_2d: np.ndarray, frame_w: int, frame_h: int) -> Tuple[float, float, float, float]:
        focal_length = frame_w
        center = (frame_w / 2, frame_h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            self.model_3d_points, landmarks_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        if sy >= 1e-6:
            pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
            yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
            roll = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
        else:
            pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
            yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
            roll = 0.0

        yaw_deg = self._normalize_angle(round(yaw, 1))
        pitch_deg = self._normalize_angle(round(pitch, 1))
        roll_deg = self._normalize_angle(round(roll, 1))

        yaw_pen = min(1.0, abs(yaw_deg) / self.thresholds.max_yaw_deg)
        pitch_pen = min(1.0, abs(pitch_deg) / self.thresholds.max_pitch_deg)
        roll_pen = min(1.0, abs(roll_deg) / self.thresholds.max_roll_deg)
        
        pose_score = round(max(0.0, 1.0 - (0.45 * yaw_pen + 0.35 * pitch_pen + 0.20 * roll_pen)), 2)

        return yaw_deg, pitch_deg, roll_deg, pose_score

    @staticmethod
    def _normalize_angle(angle_deg: float) -> float:
        while angle_deg > 180.0:
            angle_deg -= 360.0
        while angle_deg < -180.0:
            angle_deg += 360.0
        
        if angle_deg > 150.0:
            angle_deg = angle_deg - 180.0
        elif angle_deg < -150.0:
            angle_deg = angle_deg + 180.0
        
        return round(angle_deg, 1)

    @staticmethod
    def _is_valid_human_skin_roi(frame_bgr: np.ndarray, fx: int, fy: int, fw: int, fh: int) -> bool:
        """Verifies that candidate face ROI contains human skin tone pixels.
        
        Uses dual YCrCb + HSV warm-chroma checks.
        Metallic objects like cymbals, snare drum heads, chrome hardware, and speakers have
        0.0% skin chroma, whereas human faces easily exceed 15-40% skin pixels.
        Rejects false detections on stage equipment while passing all real faces.
        """
        h, w = frame_bgr.shape[:2]
        fx, fy = max(0, fx), max(0, fy)
        fw, fh = min(w - fx, fw), min(h - fy, fh)
        
        if fw < 8 or fh < 8:
            return False
            
        crop = frame_bgr[fy:fy+fh, fx:fx+fw]
        if crop.size == 0:
            return False
            
        # 1. YCrCb Skin Mask (broadened range for all skin tones and lighting)
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        ycrcb_mask = (cr >= 125) & (cr <= 185) & (cb >= 70) & (cb <= 140)
        
        # 2. HSV Warm Chroma Mask (Hue: 0 to 25 or 165 to 180)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        hsv_mask = ((hue <= 25) | (hue >= 165)) & (sat >= 20) & (val >= 30)
        
        combined_mask = ycrcb_mask | hsv_mask
        skin_pixel_ratio = np.count_nonzero(combined_mask) / float(crop.shape[0] * crop.shape[1])
        
        # 4% threshold passes all human faces under any lighting while rejecting pure grey/metallic objects
        return skin_pixel_ratio >= 0.04
