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
        TG_STREAM_SOURCE=rtsp://user:pass@192.168.1.100:554/media/video1
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
    # Tuned for cutouts: these frames get their background removed and dropped onto a
    # slide, so how cleanly the subject separates matters as much as the face itself.
    sharpness: float = 0.20      # Weight for image clarity / sharpness (critical for crisp thumbnails)
    head_pose: float = 0.16      # Weight for head orientation (looking straight at camera)
    eye_openness: float = 0.16   # Weight for eyes open (not blinking or half-shut)
    expression: float = 0.14     # Weight for smile / pleasant expression
    cutout: float = 0.18         # Weight for cutout suitability (framing, size, clean background)
    shoulder_pose: float = 0.08  # Weight for level speaker shoulders / posture
    hand_pose: float = 0.04      # Weight for hands clear of the torso
    exposure: float = 0.04       # Weight for proper lighting / exposure
    centering: float = 0.0       # Face-centred framing: reported only, not scored.
                                 # The subject is normally off-centre by design, and it
                                 # is irrelevant once they are cut out and repositioned.

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
    # Sharpness is measured on the face ROI after it is resampled to a fixed height, so
    # the numbers below are comparable across camera distances and source resolutions.
    # A raw (unnormalised) Laplacian grows with face size, which made a distant subject
    # look "blurry" and a close one always look perfect.
    # Calibrated against lossless PNG captures, where a crisp face measures 100-160 and
    # horizontal motion blur drops it fast: ~61 at 3px of smear, ~41 at 5px, ~33 at 7px.
    # An earlier 'good' of 70 was derived from JPEG captures, whose compression softens
    # the image; against PNG every real frame sat above it and scored a flat 1.00, so
    # sharpness stopped separating a crisp frame from a soft one.
    sharpness_face_height: int = 160  # Face ROI is resized to this height before measuring
    min_sharpness_laplacian: float = 45.0 # Below this the frame is treated as motion-blurred
    sharpness_floor: float = 8.0     # Normalised variance scoring 0.0 (heavy blur)
    sharpness_good: float = 150.0    # Normalised variance scoring 1.0 (crisp)
    # Hands are measured the same normalised way. A gesturing hand smears while the face
    # stays sharp, so the ratio against the face is the reliable signal.
    min_hand_sharpness: float = 25.0      # Absolute floor for a hand region
    hand_sharpness_ratio: float = 0.45    # Blurry if the hand is below this share of the face
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
    # A full analysis pass costs roughly 90-120ms per 1080p frame on a desktop CPU and
    # more on a NAS, so anything above ~8 is unreachable: the loop simply never sleeps
    # and pins a core. 6 leaves headroom for the MJPEG encode and the web server.
    analysis_fps: int = 8                  # Sub-sample frames per second for CV analysis
    # The preview deliberately runs faster than analysis. Scoring a frame is expensive,
    # but the video feed should still look live — so the newest grabbed frame is shown
    # with the most recent overlay drawn on it, rather than only showing scored frames.
    preview_fps: int = 15                  # Frames per second for the web MJPEG preview
    # Hand and pose inference together cost ~80ms of a ~120ms analysis pass. Hands and
    # shoulders move far more slowly than eyes and mouths, so they are re-detected every
    # Nth analysed frame and the previous landmarks are re-matched in between. Set to 1
    # to detect them on every frame.
    pose_hand_interval: int = 2
    # Rolling frame buffer ("DVR"). Analysis runs at analysis_fps while the camera sends
    # ~30, so most frames are never examined and a brief good expression can fall between
    # samples. The buffer keeps every frame for a few seconds so a chosen moment can be
    # re-examined at full frame rate for the crispest version of itself.
    # Memory is the constraint: a 1080p frame is 6.22 MB, so 5s x 30fps is about 0.93 GB.
    # 0 disables the buffer and the refinement pass.
    dvr_buffer_seconds: float = 5.0
    dvr_buffer_fps: int = 30               # Frames per second retained (sizes the buffer)
    dvr_refine_window_sec: float = 0.3     # Scrub +/- this around the chosen moment
    dvr_refine_candidates: int = 3         # Fully re-score this many of the sharpest frames
    # Candidates found while a judge review is in flight are queued rather than dropped,
    # so a slow reviewer costs latency instead of lost moments.
    review_queue_size: int = 10

    burst_window_seconds: float = 2.5      # Time window to find peak score candidate frame
    cooldown_seconds: float = 2.0          # Minimum cooldown between auto-saved screenshots
    single_winner_mode: bool = True        # Automatically delete inferior shots beyond Top 5
    max_retained_captures: int = 5         # Keep Top 5 best final screenshots
    output_dir: str = field(default_factory=_default_output_dir)

    # Output format. These frames get their background removed and dropped onto a slide,
    # and JPEG ringing lands exactly on the hair and shoulder edges the matting tool has
    # to cut — so PNG is the default despite the larger files.
    capture_format: str = "png"            # "png" (lossless) or "jpg"
    jpeg_quality: int = 95                 # Only used when capture_format is "jpg"
    save_subject_crop: bool = True         # Also write a tight crop around the subject
    subject_crop_padding: float = 0.08     # Crop margin, as a fraction of subject size
    
    # Multi-Person Settings
    max_people_target: int = 1             # Number of people to track and score (1 to 6)
    require_exact_people_count: bool = False # If True, requires finding exactly max_people_target people
    
    # AI judge — a vision model reviews each candidate that would enter the kept set.
    # Only runs when Anthropic credentials resolve (ANTHROPIC_API_KEY, or an `ant auth
    # login` profile); otherwise it stays off and says so once at startup. Cost is a
    # few dozen calls per service, not one per frame: a candidate reaches the judge
    # only after clearing the heuristics and beating the weakest kept capture.
    ai_judge_enabled: bool = True
    # "auto" uses the Claude API when credentials resolve, otherwise a local Ollama if
    # one is reachable with the model pulled, otherwise off. Force one with
    # "anthropic" or "ollama"; "off" disables the judge entirely.
    ai_judge_backend: str = field(default_factory=lambda: os.environ.get("TG_AI_JUDGE_BACKEND", "auto"))
    ai_judge_model: str = "claude-opus-5"      # claude-sonnet-5 is the cheaper option
    # Local backend. qwen3-vl:4b (3.3 GB) is the sensible default on a CPU-only box;
    # qwen3-vl:8b (6.1 GB) judges noticeably better if there is RAM for it.
    ai_judge_ollama_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ai_judge_ollama_model: str = field(default_factory=lambda: os.environ.get("TG_AI_JUDGE_OLLAMA_MODEL", "qwen3-vl:4b"))
    # How long Ollama keeps the weights loaded after a review. "-1" pins them in memory
    # permanently — the right setting when RAM is dedicated to it, since the first
    # review of a service otherwise pays a cold load of the whole model from disk.
    ai_judge_ollama_keep_alive: str = field(default_factory=lambda: os.environ.get("TG_AI_JUDGE_OLLAMA_KEEP_ALIVE", "30m"))
    # Threads Ollama should use for a review. Inside a container with a CPU limit this
    # MUST match the limit: Ollama otherwise defaults to the host's physical core count,
    # overruns the quota, and gets throttled — measured at ~3x slower. 0 leaves it to
    # Ollama (fine when it runs on bare metal with no limit).
    ai_judge_ollama_num_thread: int = int(os.environ.get("TG_AI_JUDGE_OLLAMA_NUM_THREAD", "0") or 0)
    ai_judge_timeout_sec: float = 300.0        # CPU inference plus a possible 20 GB cold load
    ai_judge_effort: str = "medium"            # low | medium | high
    ai_judge_weight: float = 0.5               # Share of the final score from the judge
    ai_judge_image_max_px: int = 768           # Longest side sent to the model
    ai_judge_min_interval_sec: float = 3.0     # Never review more often than this
    ai_judge_max_calls_per_session: int = 300  # Hard cost guard per process
    # How long the judge stays active after startup. A service is an hour or two, so
    # past that it stops reviewing and unloads the model from Ollama rather than
    # holding ~20GB of weights on the host until the next restart. 0 = no window.
    # The web UI can re-arm it without restarting.
    ai_judge_active_hours: float = float(os.environ.get("TG_AI_JUDGE_ACTIVE_HOURS", "2") or 2)

    # Machine Learning User Preference
    use_ml_preference: bool = True          # Enable user preference machine learning
    ml_weight: float = 0.35                 # Weight for ML preference score vs rule-based score
    
    # Temporal Smoothing & Score Stability
    ema_alpha: float = 0.3                 # Exponential moving average smoothing factor (0=full smooth, 1=raw)
    score_stability_window: int = 10       # Number of recent frames to check for score stability
    score_stability_threshold: float = 0.18  # Max score standard deviation to consider "stable"
    blink_suppression_ms: int = 300        # Milliseconds to suppress capture after blink detected
    min_consecutive_qualified: int = 2     # Require N consecutive qualified frames before capture

    # Web server path safety
    # Directories the web UI is allowed to read videos from, as an OS-path-separator
    # separated list in TG_VIDEO_ROOTS. Empty means "anywhere the process can read",
    # which is the historical behaviour — set it when the port is reachable by anyone
    # you don't trust, since these endpoints have no authentication.
    video_scan_roots: str = field(default_factory=lambda: os.environ.get("TG_VIDEO_ROOTS", "").strip())
    max_upload_mb: int = int(os.environ.get("TG_MAX_UPLOAD_MB", "4096"))
    # Directories the web UI's folder picker may browse, OS-path-separator separated.
    # Empty means anywhere the process can read, which is the historical behaviour for
    # these endpoints. Set it alongside TG_VIDEO_ROOTS when the port is exposed.
    output_browse_roots: str = field(default_factory=lambda: os.environ.get("TG_OUTPUT_ROOTS", "").strip())

    # RTSP Reconnection
    rtsp_max_retries: int = 10             # Max reconnection attempts before giving up
    rtsp_backoff_base: float = 1.0         # Initial retry delay in seconds
    rtsp_backoff_max: float = 30.0         # Maximum retry delay in seconds
    
    # Capture spacing. Two keepers from the same few seconds are one shot with
    # variations, so retention prefers spread-out captures and a new one only displaces
    # a nearby one if it scores higher. 0 disables spacing.
    min_capture_spacing_sec: float = 20.0

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
