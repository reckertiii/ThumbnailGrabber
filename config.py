import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any

_PROJECT_ROOT = Path(__file__).resolve().parent

def _load_dotenv():
    """Load KEY=VALUE pairs from a local .env file into os.environ.

    Keeps camera credentials out of source control. Existing environment
    variables always win, so you can override per-shell.
    """
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as e:
        print(f"[config] Could not read .env: {e}")

_load_dotenv()

def _default_stream_source() -> str:
    """Stream source from TG_STREAM_SOURCE, else webcam 0.

    Set TG_STREAM_SOURCE in .env, e.g.
        TG_STREAM_SOURCE=rtsp://user:pass@192.168.50.93:554/media/video1
    """
    return os.environ.get("TG_STREAM_SOURCE", "").strip() or "0"

def _default_output_dir() -> str:
    """TG_OUTPUT_DIR if set, else Pictures/ThumbnailGrabber, else ./captures."""
    override = os.environ.get("TG_OUTPUT_DIR", "").strip()
    if override:
        Path(override).mkdir(parents=True, exist_ok=True)
        return override

    pictures = Path.home() / "Pictures" / "ThumbnailGrabber"
    if pictures.parent.exists():
        pictures.mkdir(parents=True, exist_ok=True)
        return str(pictures)
    return os.path.join(os.getcwd(), "captures")

@dataclass
class QualityWeights:
    head_pose: float = 0.20      # Weight for head orientation (looking straight at camera)
    eye_openness: float = 0.20   # Weight for eyes open (not blinking)
    expression: float = 0.16     # Weight for smile / pleasant expression
    shoulder_pose: float = 0.10  # Weight for level speaker shoulders / posture
    sharpness: float = 0.14      # Weight for image clarity / sharpness
    hand_pose: float = 0.10      # Weight for clean hand posture / not blurry or awkward
    exposure: float = 0.05       # Weight for proper lighting / exposure
    centering: float = 0.05      # Weight for face centered in frame

@dataclass
class QualityThresholds:
    min_overall_score: float = 0.55  # Capture threshold (55%)
    max_yaw_deg: float = 30.0        # Max head turn (yaw) from center (30 degrees)
    max_pitch_deg: float = 25.0      # Max head tilt (pitch) up/down in degrees
    max_roll_deg: float = 20.0       # Max head roll side-to-side in degrees
    max_shoulder_tilt_deg: float = 12.0 # Max shoulder tilt / slant from horizontal in degrees
    min_ear: float = 0.20            # Minimum Eye Aspect Ratio (below this is closed/blinking)
    min_mar_smile: float = 0.15      # Mouth Aspect Ratio indicator for natural smile
    max_mar_awkward: float = 0.60    # Excessively wide/open mouth threshold (yawning / mid-speech)
    min_sharpness_laplacian: float = 40.0 # Min Laplacian variance for motion blur check
    min_brightness: float = 40.0     # Min mean brightness of face ROI (underexposure threshold)
    max_brightness: float = 220.0    # Max mean brightness of face ROI (overexposure threshold)

@dataclass
class AppConfig:
    # Camera / Stream settings
    # Sony SRG-A40 RTSP format with auth: rtsp://<user>:<pass>@<IP>:554/media/video1
    # Configured via TG_STREAM_SOURCE in .env — never hardcode credentials here.
    sony_srg_a40_preset_url: str = field(default_factory=_default_stream_source)
    stream_source: str = field(default_factory=_default_stream_source)


    # Analysis & Capture settings
    analysis_fps: int = 15                 # Sub-sample frames per second for CV analysis
    burst_window_seconds: float = 2.5      # Time window to find peak score candidate frame
    cooldown_seconds: float = 2.0          # Minimum cooldown between auto-saved screenshots
    single_winner_mode: bool = True        # Automatically delete inferior shots beyond Top 5
    max_retained_captures: int = 5         # Keep Top 5 best final screenshots
    output_dir: str = field(default_factory=_default_output_dir)
    
    # Multi-Person Settings
    max_people_target: int = 1             # Number of people to track and score (1 to 6)
    require_exact_people_count: bool = False # If True, requires finding exactly max_people_target people
    
    # Machine Learning User Preference
    use_ml_preference: bool = True          # Enable user preference machine learning
    ml_weight: float = 0.35                 # Weight for ML preference score vs rule-based score
    
    # Temporal Smoothing & Score Stability
    ema_alpha: float = 0.3                 # Exponential moving average smoothing factor (0=full smooth, 1=raw)
    score_stability_window: int = 10       # Number of recent frames to check for score stability
    score_stability_threshold: float = 0.18  # Max score standard deviation to consider "stable"
    blink_suppression_ms: int = 300        # Milliseconds to suppress capture after blink detected
    min_consecutive_qualified: int = 2     # Require N consecutive qualified frames before capture

    # RTSP Reconnection
    rtsp_max_retries: int = 10             # Max reconnection attempts before giving up
    rtsp_backoff_base: float = 1.0         # Initial retry delay in seconds
    rtsp_backoff_max: float = 30.0         # Maximum retry delay in seconds
    
    # Duplicate Frame Detection
    duplicate_mse_threshold: float = 50.0  # Max MSE between frames to consider them duplicates

    # Weights & Thresholds
    weights: QualityWeights = field(default_factory=QualityWeights)
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_source": self.stream_source,
            "analysis_fps": self.analysis_fps,
            "burst_window_seconds": self.burst_window_seconds,
            "cooldown_seconds": self.cooldown_seconds,
            "output_dir": self.output_dir,
            "ema_alpha": self.ema_alpha,
            "score_stability_window": self.score_stability_window,
            "score_stability_threshold": self.score_stability_threshold,
            "weights": self.weights.__dict__,
            "thresholds": self.thresholds.__dict__,
        }
