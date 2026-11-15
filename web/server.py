"""FastAPI web front-end for the Thumbnail Grabber.

Runs on the same LAN as the camera, serves the analyzed video as MJPEG and the
live telemetry over a WebSocket, so any browser on the network can drive it.

Run with:
    python main.py                  (from the project root)
    python -m web.server            (from the project root)
    python web/server.py --port 8000
"""
import asyncio
import base64
import os
import posixpath
import secrets
import sys
import tempfile
import threading
import time
import urllib.parse
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

# Allow "python web/server.py" as well as "python -m web.server"
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
from fastapi import (
    Body,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

import camera_probe
import camera_store
import settings_store
from analyzer import AnalysisResult
from audio_analyzer import SermonNotesGenerator
from config import AppConfig, home_dir, is_redacted, redact_credentials
from ndi_source import discover_ndi_sources, ndi_status
from overlay import draw_overlay
from postprocess import cutout_path_for, get_postprocess_engine, save_cutout_png
from sony_ptz import SonyPTZController
from stage_zones import StageZoneConfig
from stream_processor import CapturedScreenshot, StreamProcessor, subject_crop_rect
from video_scanner import ScanProgress, VideoInfo, VideoRecordingScanner
from visual_embedder import get_visual_embedder

STATIC_DIR = Path(__file__).resolve().parent / "static"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".ts", ".mts")


# ---------------------------------------------------------------------------
# Filesystem & Mount Persistence Helpers
# ---------------------------------------------------------------------------

def _mount_points() -> set:
    """Every path this process has a filesystem mounted at.

    Read per call rather than cached: a volume can be added and the app
    redeployed without this module being reimported in between.
    """
    points = set()
    try:
        with open("/proc/self/mountinfo", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) > 4:
                    points.add(parts[4])
    except OSError:
        pass            # not Linux, or no procfs - caller treats this as "unknown"
    return points


def is_ephemeral(path) -> bool:
    """True when `path` lives in the container's own writable layer.

    Anything written there survives a restart but is destroyed when the app is
    recreated, which is every update. The failure is silent and very convincing:
    the app creates its own output folder, writes to it, serves the files back,
    and reports everything as fine, while the host sees an empty dataset. That
    happens whenever a bind mount misses -- most easily by mounting a pool root
    and expecting child datasets to come with it, which a non-recursive bind
    mount does not do.

    Walks POSIX paths explicitly rather than through pathlib's native flavour:
    mountinfo entries are POSIX whatever the host running the tests is.

    Returns False when it cannot tell, so this only ever reports a problem it
    has actually established.
    """
    points = _mount_points()
    if not points:
        return False                      # no procfs: say nothing
    raw = str(path)
    if os.name == "posix":
        try:
            raw = os.path.realpath(raw)   # follow symlinks out of the layer
        except OSError:
            pass
    target = PurePosixPath(posixpath.normpath(raw.replace("\\", "/")))
    for c in [target] + list(target.parents):
        if str(c) in points:
            # The deepest enclosing mount. Only the image root means ephemeral.
            return str(c) == "/"
    return False


def _persistence_warning(path) -> str:
    """The sentence to show about a folder, or "" when there is nothing to say."""
    if not is_ephemeral(path):
        return ""
    return (f"{path} is inside the container, not on a mounted volume. Captures "
            f"written here are lost the next time the app is updated. Add a bind "
            f"mount for this path - and note that mounting a pool root does not "
            f"bring its child datasets with it; each dataset needs its own mount.")


def _is_writable(path: Path) -> bool:
    return os.access(str(path), os.W_OK | os.X_OK)


def _browse_roots_for(raw_roots: str) -> List[Path]:
    raw = (raw_roots or "").strip()
    roots = []
    for part in raw.split(os.pathsep):
        part = part.strip()
        if part:
            try:
                roots.append(Path(part).resolve())
            except Exception:
                continue
    return roots


def _check_basic_auth(auth_header: Optional[str], expected_password: str) -> bool:
    if not expected_password:
        return True
    if not auth_header or not auth_header.lower().startswith("basic "):
        return False
    token = auth_header[6:].strip()
    try:
        decoded = base64.b64decode(token).decode("utf-8")
    except Exception:
        return False
    if ":" not in decoded:
        return False
    _, _, given_pass = decoded.partition(":")
    return secrets.compare_digest(given_pass, expected_password)


class AppState:
    """Holds the single shared StreamProcessor, view preferences, and video scanner state."""

    def __init__(self):
        self.config = AppConfig()
        self.processor: Optional[StreamProcessor] = None
        self.show_mesh: bool = True
        self.show_hud: bool = True
        self.is_capturing: bool = False
        self.capture_started_at: Optional[float] = None
        # Captures only start once someone has confirmed where they land. Once confirmed,
        # the choice is persisted in TG_DATA_DIR so Watchtower container updates don't
        # silently disarm capture readiness every night.
        self.output_confirmed: bool = False
        self._lock = threading.Lock()

        # Viewer & idle-standby tracking
        self.active_ws_clients: int = 0
        self.active_mjpeg_clients: int = 0
        self.last_client_activity: float = time.time()

        # Shared MJPEG encode cache so multiple browser tabs share a single overlay + JPEG pass
        self._mjpeg_lock = threading.Lock()
        self._mjpeg_cache_key: Optional[Tuple[Any, ...]] = None
        self._mjpeg_cache_bytes: Optional[bytes] = None

        # Video file scanner state
        self.is_scanning_video: bool = False
        self.scan_progress: Optional[ScanProgress] = None
        self.scan_cancel_event: Optional[threading.Event] = None
        self.scan_thread: Optional[threading.Thread] = None
        self.scan_results: List[CapturedScreenshot] = []
        self.scan_error: Optional[str] = None
        self.scan_video_path: Optional[str] = None
        self.scan_output_dir: Optional[str] = None
        self._scan_champ_id: int = 0
        self._scan_champ_b64: Optional[str] = None

        # Latest frame + analysis pushed from the analysis thread
        self._frame: Optional[np.ndarray] = None
        self._analysis: Optional[AnalysisResult] = None
        self._frame_seq: int = 0

        # Stage zones (#9) & Sermon Notes (#8)
        self.stage_zones = StageZoneConfig()
        self._stage_zones_rev: int = 0
        self.sermon_notes_gen = SermonNotesGenerator()
        self.latest_sermon_notes: Optional[Dict[str, Any]] = None

        self.apply_persisted_ui_settings()

    def apply_persisted_ui_settings(self) -> None:
        """Restores UI preferences and confirmed output folder from TG_DATA_DIR."""
        saved = settings_store.load_ui_settings()
        if not saved:
            return
        cfg = self.config

        if "show_mesh" in saved:
            self.show_mesh = bool(saved["show_mesh"])
        if "show_hud" in saved:
            self.show_hud = bool(saved["show_hud"])
        if "single_winner" in saved:
            cfg.single_winner_mode = bool(saved["single_winner"])
        if "max_retained" in saved:
            try:
                cfg.max_retained_captures = max(1, min(200, int(saved["max_retained"])))
            except (TypeError, ValueError):
                pass
        if "capture_prop_moments" in saved:
            cfg.capture_prop_moments = bool(saved["capture_prop_moments"])
        if "mood_buckets_enabled" in saved:
            cfg.mood_buckets_enabled = bool(saved["mood_buckets_enabled"])
        if "auto_save_cutout" in saved:
            cfg.auto_save_cutout = bool(saved["auto_save_cutout"])
        if "broadcast_polish_enabled" in saved:
            cfg.broadcast_polish_enabled = bool(saved["broadcast_polish_enabled"])
        if "upscale_4k_enabled" in saved:
            cfg.upscale_4k_enabled = bool(saved["upscale_4k_enabled"])
        if "audio_gating_enabled" in saved:
            cfg.audio_gating_enabled = bool(saved["audio_gating_enabled"])
        if "stage_zones" in saved and isinstance(saved["stage_zones"], dict):
            try:
                self.stage_zones = StageZoneConfig.from_dict(saved["stage_zones"])
            except Exception:
                pass
        if "dvr_buffer_seconds" in saved:
            try:
                max_dvr = max(60.0, float(getattr(cfg, "dvr_max_seconds", 180.0) or 180.0))
                cfg.dvr_buffer_seconds = max(0.0, min(max_dvr, float(saved["dvr_buffer_seconds"])))
            except (TypeError, ValueError):
                pass
        if "dvr_buffer_fps" in saved and saved["dvr_buffer_fps"] is not None:
            try:
                cfg.dvr_buffer_fps = max(1, min(60, int(saved["dvr_buffer_fps"])))
            except (TypeError, ValueError):
                pass
        if "capture_max_minutes" in saved:
            try:
                cfg.capture_max_minutes = max(0.0, min(1440.0, float(saved["capture_max_minutes"])))
            except (TypeError, ValueError):
                pass
        if "min_score" in saved:
            try:
                cfg.thresholds.min_overall_score = max(0.1, min(1.0, float(saved["min_score"]) / 100.0))
            except (TypeError, ValueError):
                pass
        if "max_yaw" in saved:
            try:
                cfg.thresholds.max_yaw_deg = float(saved["max_yaw"])
            except (TypeError, ValueError):
                pass
        if "max_shoulder_tilt" in saved:
            try:
                cfg.thresholds.max_shoulder_tilt_deg = float(saved["max_shoulder_tilt"])
            except (TypeError, ValueError):
                pass
        if "burst_window" in saved:
            try:
                cfg.burst_window_seconds = float(saved["burst_window"])
            except (TypeError, ValueError):
                pass
        if "cooldown" in saved:
            try:
                cfg.cooldown_seconds = float(saved["cooldown"])
            except (TypeError, ValueError):
                pass
        if "people_target" in saved:
            try:
                cfg.max_people_target = max(1, min(6, int(saved["people_target"])))
            except (TypeError, ValueError):
                pass

        saved_dir = str(saved.get("output_dir") or "").strip()
        if saved_dir:
            try:
                cand = Path(saved_dir).expanduser().resolve()
                roots = _browse_roots_for(cfg.output_browse_roots)
                in_roots = (not roots) or any(r == cand or r in cand.parents for r in roots)
                if cand.is_dir() and _is_writable(cand) and in_roots and not is_ephemeral(cand):
                    cfg.output_dir = str(cand)
                    if bool(saved.get("output_confirmed", False)):
                        self.output_confirmed = True
            except Exception:
                pass

    # --- processor & activity lifecycle --------------------------------------

    def touch_activity(self) -> None:
        """Records live UI/API interaction and wakes from Zero-CPU Idle Standby."""
        self.last_client_activity = time.time()
        proc = self.processor
        if proc is not None and getattr(proc, "idle_standby", False):
            proc.set_idle_standby(False)

    def capture_seconds_remaining(self) -> Optional[int]:
        if not self.is_capturing or self.capture_started_at is None:
            return None
        max_min = float(getattr(self.config, "capture_max_minutes", 0.0) or 0.0)
        if max_min <= 0:
            return None
        rem = (max_min * 60.0) - (time.time() - self.capture_started_at)
        return max(0, int(rem))

    def check_lifecycle_tick(self, idle_grace_sec: float = 20.0) -> Dict[str, Any]:
        """Evaluates Auto-Stop Capture timer and Zero-CPU Idle Standby."""
        actions: Dict[str, Any] = {"auto_stopped": False, "idle_standby": False}
        proc = self.processor
        now = time.time()

        # 1. Max Capture Window / Auto-Stop Timer
        max_min = float(getattr(self.config, "capture_max_minutes", 0.0) or 0.0)
        if self.is_capturing and self.capture_started_at is not None and max_min > 0:
            if (now - self.capture_started_at) >= (max_min * 60.0):
                self.is_capturing = False
                self.capture_started_at = None
                if proc is not None:
                    proc.auto_capture_enabled = False
                    judge = getattr(proc, "judge", None)
                    if judge is not None and judge.enabled and not judge._expiry_handled:
                        judge.sleep_now(reason="capture timer elapsed")
                print(f"[Lifecycle] Auto-stopped capture after {max_min:.0f} min window.")
                actions["auto_stopped"] = True

        # 2. Zero-CPU Idle Standby when no viewers, not capturing, and not scanning
        if proc is not None:
            if getattr(self.config, "idle_standby_enabled", True):
                has_viewers = (self.active_ws_clients > 0 or self.active_mjpeg_clients > 0)
                recently_active = (now - self.last_client_activity) < idle_grace_sec
                should_standby = (
                    not self.is_capturing
                    and not self.is_scanning_video
                    and not has_viewers
                    and not recently_active
                )
            else:
                should_standby = False

            if bool(getattr(proc, "idle_standby", False)) != should_standby:
                proc.set_idle_standby(should_standby)
            actions["idle_standby"] = bool(getattr(proc, "idle_standby", False))

        return actions

    def start_processor(self):
        with self._lock:
            if self.processor:
                self.processor.stop()
            self.processor = StreamProcessor(
                config=self.config,
                on_frame_callback=self._on_frame,
            )
            self.processor.analyzer.stage_zones = self.stage_zones
            self.processor.enable_persistence(adopt_orphans=True)
            self.processor.start()
            self.processor.auto_capture_enabled = self.is_capturing

    def stop_processor(self):
        with self._lock:
            if self.processor:
                self.processor.stop()
                self.processor = None

    def _on_frame(self, frame: np.ndarray, analysis: AnalysisResult):
        # Called on the analysis thread — keep it cheap, just hand off references.
        self._frame = frame
        self._analysis = analysis
        self._frame_seq += 1

    def snapshot(self):
        """Newest frame off the camera, with the most recent analysis to draw over it."""
        proc = self.processor
        if proc is not None:
            with proc.frame_lock:
                frame = proc.latest_frame
                seq = proc.frames_grabbed
            if frame is not None:
                return frame, proc.latest_analysis, seq
        return self._frame, self._analysis, self._frame_seq

    def encode_preview_jpeg(self, frame: np.ndarray, analysis: Optional[AnalysisResult],
                            seq: int, max_width: int, quality: int) -> Optional[bytes]:
        """Renders overlay + downscaled JPEG once per unique frame/view key for all tabs."""
        key = (seq, max_width, quality, self.show_mesh, self.show_hud, id(analysis), self._stage_zones_rev)
        with self._mjpeg_lock:
            if self._mjpeg_cache_key == key and self._mjpeg_cache_bytes is not None:
                return self._mjpeg_cache_bytes
        sz = (getattr(self.processor.analyzer, "stage_zones", None) if (self.processor and getattr(self.processor, "analyzer", None)) else None) or self.stage_zones
        annotated = draw_overlay(frame, analysis, self.show_mesh, self.show_hud, stage_zones=sz)
        h, w = annotated.shape[:2]
        if max_width and w > max_width:
            scale = max_width / float(w)
            annotated = cv2.resize(annotated, (max_width, int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return None
        jpg_bytes = buf.tobytes()
        with self._mjpeg_lock:
            self._mjpeg_cache_key = key
            self._mjpeg_cache_bytes = jpg_bytes
        return jpg_bytes


state = AppState()


async def _lifecycle_monitor(stop_event: asyncio.Event):
    while not stop_event.is_set():
        try:
            state.check_lifecycle_tick()
        except Exception as e:
            print(f"[Lifecycle] Monitor tick error: {e}")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state.start_processor()
    stop_event = asyncio.Event()
    monitor_task = asyncio.create_task(_lifecycle_monitor(stop_event))
    try:
        yield
    finally:
        stop_event.set()
        try:
            await monitor_task
        except Exception:
            pass
        state.stop_processor()


app = FastAPI(title="Thumbnail Grabber", lifespan=lifespan)


@app.middleware("http")
async def basic_auth_middleware(request: Request, call_next):
    pwd = (state.config.web_password or "").strip()
    if pwd and request.url.path != "/api/health":
        if not _check_basic_auth(request.headers.get("authorization"), pwd):
            return Response(
                content="Authentication required",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Thumbnail Grabber"'},
            )
    return await call_next(request)


@app.get("/api/health")
def health_check():
    proc = state.processor
    return {
        "ok": True,
        "status": proc.connection_status if proc else "starting",
        "is_capturing": bool(state.is_capturing),
        "is_scanning_video": bool(state.is_scanning_video),
        "idle_standby": bool(getattr(proc, "idle_standby", False)) if proc else False,
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _face_to_dict(f) -> Dict[str, Any]:
    bx, by, bw, bh = f.bbox
    return {
        "person_id": f.person_id,
        "bbox": [int(bx), int(by), int(bw), int(bh)],
        "score": round(float(f.score), 3),
        "is_qualified": bool(f.is_qualified),
        "is_selected": bool(f.is_selected),
        "is_blinking": bool(f.is_blinking),
        "is_smiling": bool(f.is_smiling),
        "is_awkward_expression": bool(f.is_awkward_expression),
        "is_looking_down": bool(f.is_looking_down),
        "is_shoulder_tilted": bool(f.is_shoulder_tilted),
        "is_prop_moment": bool(getattr(f, "is_prop_moment", False)),
        "speaker_name": str(getattr(f, "speaker_name", "") or ""),
        "speaker_similarity": round(float(getattr(f, "speaker_similarity", 0.0) or 0.0), 3),
        "mood_bucket": str(getattr(f, "mood_bucket", "") or ""),
        "outside_safe_zone": bool(getattr(f, "outside_safe_zone", False)),
    }


def _telemetry() -> Dict[str, Any]:
    proc = state.processor
    frame, analysis, _ = state.snapshot()
    cfg = state.config
    pm = getattr(proc.analyzer, "preference_model", None) if (proc and getattr(proc, "analyzer", None)) else None
    spk = getattr(proc.analyzer, "speaker_id", None) if (proc and getattr(proc, "analyzer", None)) else None
    aud_mon = getattr(proc, "audio_monitor", None) if proc else None
    aud_tel = aud_mon.telemetry() if aud_mon is not None else None
    sz = (getattr(proc.analyzer, "stage_zones", None) if (proc and getattr(proc, "analyzer", None)) else None) or state.stage_zones

    payload: Dict[str, Any] = {
        "status": proc.connection_status if proc else StreamProcessor.STATUS_DISCONNECTED,
        "reconnect_attempt": proc.reconnect_attempt if proc else 0,
        "idle_standby": bool(getattr(proc, "idle_standby", False)) if proc else False,
        "fps": round(proc.current_fps, 1) if proc else 0.0,
        "frames_processed": proc.frames_processed if proc else 0,
        "is_capturing": state.is_capturing,
        "capture_max_minutes": round(float(getattr(cfg, "capture_max_minutes", 120.0) or 0.0), 1),
        "capture_seconds_remaining": state.capture_seconds_remaining(),
        "auto_capture": bool(proc.auto_capture_enabled) if proc else False,
        "single_winner": cfg.single_winner_mode,
        "capture_prop_moments": bool(getattr(cfg, "capture_prop_moments", True)),
        "mood_buckets_enabled": bool(getattr(cfg, "mood_buckets_enabled", True)),
        "auto_save_cutout": bool(getattr(cfg, "auto_save_cutout", False)),
        "broadcast_polish_enabled": bool(getattr(cfg, "broadcast_polish_enabled", True)),
        "upscale_4k_enabled": bool(getattr(cfg, "upscale_4k_enabled", False)),
        "audio_gating_enabled": bool(getattr(cfg, "audio_gating_enabled", False)),
        "dvr_max_seconds": round(float(getattr(cfg, "dvr_max_seconds", 180.0) or 180.0), 0),
        "show_mesh": state.show_mesh,
        "show_hud": state.show_hud,
        # Redacted: camera credentials never leave the server in readable form.
        "source": redact_credentials(cfg.stream_source),
        "output_dir": cfg.output_dir,
        "output_confirmed": bool(state.output_confirmed),
        "buffer": proc.buffer_stats() if proc and hasattr(proc, "buffer_stats") else None,
        "ai_judge": proc.judge.status_line() if proc and getattr(proc, "judge", None) else "AI judge off",
        "ai_judge_active": bool(proc.judge.available) if proc and getattr(proc, "judge", None) else False,
        "ai_judge_minutes_left": (
            None if not (proc and getattr(proc, "judge", None))
            else (None if proc.judge.seconds_remaining == float("inf")
                  else int(proc.judge.seconds_remaining // 60))),
        "ml_trained": bool(pm.is_trained) if pm else False,
        "ml_confidence": round(float(pm.confidence), 3) if pm else 0.0,
        "ml_trust": round(float(pm.trust), 3) if pm else 0.0,
        "ml_samples": len(pm.samples) if pm else 0,
        "ml_notes_count": len(getattr(pm, "feedback_notes", []) or []) if pm else 0,
        "ml_penalties": dict(getattr(pm, "learned_penalties", {}) or {}) if pm else {},
        "speaker_lock": spk.status() if spk is not None else {"locked": False, "profiles": []},
        "stage_zones": sz.to_dict() if sz is not None else {"safe_zone": [], "obstruction_zones": []},
        "audio": (aud_tel.__dict__ if aud_tel is not None else {"active": False, "rms_db": -60.0, "in_speech_pause": False, "is_laughter_burst": False}),
        "frame_size": [int(frame.shape[1]), int(frame.shape[0])] if frame is not None else None,
        "settings": {
            "min_score": int(cfg.thresholds.min_overall_score * 100),
            "max_yaw": int(cfg.thresholds.max_yaw_deg),
            "max_shoulder_tilt": int(cfg.thresholds.max_shoulder_tilt_deg),
            "burst_window": round(cfg.burst_window_seconds, 1),
            "cooldown": round(cfg.cooldown_seconds, 1),
            "people_target": cfg.max_people_target,
            "max_retained": int(cfg.max_retained_captures),
            "capture_max_minutes": round(float(getattr(cfg, "capture_max_minutes", 120.0) or 0.0), 1),
        },
        "selected_people": sorted(proc.analyzer.selected_person_ids) if proc else [],
        "face_detected": False,
    }

    if analysis is not None:
        payload.update({
            "face_detected": bool(analysis.face_detected),
            "people_detected": int(analysis.people_detected_count),
            "overall_score": round(float(analysis.overall_score), 3),
            "smoothed_score": round(float(analysis.smoothed_score), 3),
            "ml_score": round(float(analysis.ml_score), 3),
            "is_qualified": bool(analysis.is_qualified),
            "is_stable": bool(analysis.is_stable),
            "yaw": round(float(analysis.yaw_deg), 1),
            "pitch": round(float(analysis.pitch_deg), 1),
            "roll": round(float(analysis.roll_deg), 1),
            "avg_ear": round(float(analysis.avg_ear), 3),
            "is_blinking": bool(analysis.is_blinking),
            "smile_ratio": round(float(analysis.smile_ratio), 3),
            "is_smiling": bool(analysis.is_smiling),
            "is_awkward_expression": bool(analysis.is_awkward_expression),
            "eye_gaze_away_score": round(float(getattr(analysis, "eye_gaze_away_score", 0.0)), 3),
            "mouth_velocity": round(float(getattr(analysis, "mouth_velocity", 0.0)), 3),
            "shoulder_detected": bool(analysis.shoulder_detected),
            "shoulder_tilt": round(float(analysis.shoulder_tilt_deg), 1),
            "is_shoulders_level": bool(analysis.is_shoulders_level),
            "sharpness": int(analysis.laplacian_variance),
            "brightness": round(float(analysis.face_brightness), 0),
            "exposure_score": round(float(analysis.exposure_score), 3),
            "centering_score": round(float(analysis.centering_score), 3),
            "cutout_score": round(float(analysis.cutout_score), 3),
            "subject_height_ratio": round(float(analysis.subject_height_ratio), 3),
            "is_subject_clipped": bool(analysis.is_subject_clipped),
            "hands_on_torso": bool(analysis.hands_on_torso),
            "is_prop_moment": bool(getattr(analysis, "is_prop_moment", False)),
            "mood_bucket": str(getattr(analysis, "mood_bucket", "") or ""),
            "speaker_name": str(getattr(analysis, "speaker_name", "") or ""),
            "color_cast_score": round(float(getattr(analysis, "color_cast_score", 1.0)), 3),
            "background_score": round(float(analysis.background_score), 3),
            "disqualified_reasons": list(getattr(analysis, "disqualified_reasons", []) or []),
            "hands_covering_face": bool(analysis.hands_covering_face),
            "is_hand_blurry": bool(analysis.is_hand_blurry),
            "faces": [_face_to_dict(f) for f in (analysis.faces or [])],
            "ml_weight_applied": round(float(getattr(analysis, "ml_weight_applied", 0.0)), 3),
        })

    return payload


def _rel_capture_path(path: str) -> str:
    """Relative POSIX path inside output_dir when possible, else basename."""
    try:
        base = Path(state.config.output_dir).resolve()
        target = Path(path).resolve()
        if base in target.parents:
            return target.relative_to(base).as_posix()
    except Exception:
        pass
    return os.path.basename(path)


def _capture_url(path: Optional[str]) -> Optional[str]:
    """Cache-busted /captures URL for a written file, or None if it isn't there."""
    if not path or not os.path.exists(path):
        return None
    rel = _rel_capture_path(path)
    return f"/captures/{rel}?t={int(os.path.getmtime(path))}"


def _visible_captures() -> List[CapturedScreenshot]:
    proc = state.processor
    if not proc:
        return []
    with getattr(proc, "captures_lock", threading.RLock()):
        all_caps = [
            c for c in sorted(proc.saved_captures, key=lambda c: c.overall_score, reverse=True)
            if c.filepath and os.path.exists(c.filepath)
        ]
    if state.config.single_winner_mode:
        unpinned_count = 0
        limit = state.config.max_retained_captures
        caps = []
        for c in all_caps:
            if getattr(c, "is_pinned", False) or getattr(c, "from_video_scan", False):
                caps.append(c)
            elif unpinned_count < limit:
                caps.append(c)
                unpinned_count += 1
        return caps
    return all_caps


def _captures_payload() -> List[Dict[str, Any]]:
    caps = _visible_captures()
    proc = state.processor
    burst_cache = getattr(proc, "_burst_cache", {}) if proc else {}
    out = []
    for idx, c in enumerate(caps):
        name = os.path.basename(c.filepath)
        rel = _rel_capture_path(c.filepath)
        a = c.analysis
        is_prop = bool(getattr(c, "is_prop_moment", False) or getattr(a, "is_prop_moment", False))
        is_pinned = bool(getattr(c, "is_pinned", False))
        mood = str(getattr(c, "mood_bucket", "") or getattr(a, "mood_bucket", "") or "")
        cpath = getattr(c, "cutout_filepath", None) or cutout_path_for(c.filepath)
        has_burst = bool(name in burst_cache and len(burst_cache[name].get("frames") or []) > 1)
        out.append({
            "rank": idx + 1,
            "filename": name,
            "url": f"/captures/{rel}?t={int(os.path.getmtime(c.filepath)) if os.path.exists(c.filepath) else 0}",
            "crop_url": _capture_url(c.crop_filepath),
            "cutout_url": _capture_url(cpath) if (cpath and os.path.exists(cpath)) else None,
            "score": round(float(c.overall_score), 3),
            "timestamp": c.timestamp_str,
            "cutout_score": round(float(getattr(a, "cutout_score", 0.0)), 3),
            "heuristic_score": round(float(getattr(c, "heuristic_score", c.overall_score) or c.overall_score), 3),
            "judge_score": (round(float(c.judge_score), 3) if getattr(c, "judge_score", None) is not None else None),
            "judge_summary": getattr(c, "judge_summary", "") or "",
            "judge_issues": list(getattr(c, "judge_issues", []) or []),
            "is_prop_moment": is_prop,
            "is_pinned": is_pinned,
            "mood_bucket": mood,
            "finals_badge": str(getattr(c, "finals_badge", "") or ""),
            "speaker_name": str(getattr(a, "speaker_name", "") or ""),
            "has_burst": has_burst,
            "yaw": round(float(a.yaw_deg), 1),
            "shoulder_tilt": round(float(a.shoulder_tilt_deg), 1),
            "shoulder_detected": bool(a.shoulder_detected),
            "avg_ear": round(float(a.avg_ear), 2),
            "smile_ratio": round(float(a.smile_ratio), 2),
            "sharpness": int(a.laplacian_variance),
        })
    return out


def _find_capture(filename: str) -> CapturedScreenshot:
    proc = state.processor
    if proc:
        with getattr(proc, "captures_lock", threading.RLock()):
            for c in proc.saved_captures:
                if os.path.basename(c.filepath) == filename or _rel_capture_path(c.filepath) == filename:
                    return c
    raise HTTPException(status_code=404, detail="Capture not found")


# ---------------------------------------------------------------------------
# Video + telemetry (Shared Async MJPEG Broadcaster)
# ---------------------------------------------------------------------------

def _placeholder_jpeg(status: str) -> Optional[bytes]:
    blank = np.zeros((360, 640, 3), dtype=np.uint8)
    if status == StreamProcessor.STATUS_UNCONFIGURED:
        msg = "No camera configured - click Camera to set up"
        pos = (65, 190)
    elif status == StreamProcessor.STATUS_STANDBY:
        msg = "Camera in standby (waiting for power)..."
        pos = (85, 190)
    else:
        msg = "Waiting for camera feed..."
        pos = (135, 190)
    cv2.putText(blank, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (148, 163, 184), 2)
    ok, buf = cv2.imencode(".jpg", blank)
    return buf.tobytes() if ok else None


async def _async_mjpeg_generator(request: Request, max_width: int = 1280, quality: int = 80):
    """Async multipart MJPEG stream sharing a single encoded JPEG cache across viewers."""
    boundary = b"--frame\r\n"
    last_seq = -1
    target_fps = max(1, getattr(state.config, "preview_fps", 15))
    min_interval = 1.0 / target_fps

    state.active_mjpeg_clients += 1
    state.touch_activity()
    try:
        while True:
            if await request.is_disconnected():
                break
            loop_start = time.time()
            frame, analysis, seq = state.snapshot()

            if frame is None:
                proc = state.processor
                status = proc.connection_status if proc else StreamProcessor.STATUS_DISCONNECTED
                jpg = _placeholder_jpeg(status)
                if jpg:
                    yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                await asyncio.sleep(0.5)
                continue

            if seq != last_seq:
                last_seq = seq
                jpg = await asyncio.to_thread(
                    state.encode_preview_jpeg, frame, analysis, seq, max_width, quality
                )
                if jpg:
                    yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"

            elapsed = time.time() - loop_start
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
    finally:
        state.active_mjpeg_clients = max(0, state.active_mjpeg_clients - 1)
        state.last_client_activity = time.time()


@app.get("/video.mjpg")
async def video_feed(request: Request, w: int = 1280, q: int = 80):
    """Live annotated preview. ?w=640&q=60 for phones on weak Wi-Fi."""
    return StreamingResponse(
        _async_mjpeg_generator(
            request,
            max_width=max(160, min(1920, w)),
            quality=max(30, min(95, q)),
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


@app.websocket("/ws")
async def telemetry_socket(ws: WebSocket):
    pwd = (state.config.web_password or "").strip()
    if pwd and not _check_basic_auth(ws.headers.get("authorization"), pwd):
        await ws.close(code=1008)
        return

    await ws.accept()
    state.active_ws_clients += 1
    state.touch_activity()
    last_capture_sig = None
    try:
        while True:
            payload = _telemetry()
            proc = state.processor
            if proc:
                with getattr(proc, "captures_lock", threading.RLock()):
                    sig = (
                        state.config.output_dir,
                        state.config.single_winner_mode,
                        state.config.max_retained_captures,
                        tuple(
                            (
                                c.filepath,
                                round(float(c.overall_score), 3),
                                c.judge_score,
                                getattr(c, "is_prop_moment", False),
                                getattr(c, "is_pinned", False),
                                getattr(c, "finals_badge", ""),
                                getattr(c, "cutout_filepath", None),
                                int(os.path.getmtime(c.filepath)) if (c.filepath and os.path.exists(c.filepath)) else 0,
                            )
                            for c in proc.saved_captures
                        ),
                    )
            else:
                sig = ()
            if sig != last_capture_sig:
                last_capture_sig = sig
                payload["captures"] = _captures_payload()
            await ws.send_json(payload)
            await asyncio.sleep(0.1)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        state.active_ws_clients = max(0, state.active_ws_clients - 1)
        state.last_client_activity = time.time()


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def _sanitize_upload_name(raw: str) -> str:
    """Reduces a client-supplied filename to a bare, safe basename."""
    name = os.path.basename(str(raw or "").replace("\\", "/").strip())
    name = "".join(ch for ch in name if ch.isalnum() or ch in " ._-()[]")
    name = name.strip(". ")
    if not name:
        name = f"upload_{int(time.time())}.mp4"
    return name[:180]


def _allowed_video_roots() -> List[Path]:
    """Directories videos may be read from, or [] when unrestricted."""
    raw = (state.config.video_scan_roots or "").strip()
    if not raw:
        return []
    roots = _browse_roots_for(raw)
    # Uploads always land inside the output dir, so that has to stay reachable.
    roots.append((Path(state.config.output_dir) / "uploads").resolve())
    return roots


def _resolve_video_path(raw: str) -> Path:
    """Resolves a requested video path, enforcing the root allowlist if one is set."""
    if not raw:
        raise HTTPException(status_code=400, detail="No video path given")
    try:
        target = Path(raw).expanduser().resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid video path")

    roots = _allowed_video_roots()
    if roots and not any(root == target or root in target.parents for root in roots):
        raise HTTPException(
            status_code=403,
            detail="Path is outside the directories allowed by TG_VIDEO_ROOTS"
        )

    if not target.is_file():
        raise HTTPException(status_code=400, detail="Video file not found")

    return target


def _is_network_or_device_source(src: str) -> bool:
    s = (src or "").strip()
    low = s.lower()
    return s.isdigit() or low.startswith(("rtsp://", "rtsps://", "http://", "https://", "ndi://"))


def _browse_roots() -> List[Path]:
    """Directories the output folder picker may browse, or [] when unrestricted."""
    return _browse_roots_for(state.config.output_browse_roots)


def _check_browsable(target: Path) -> None:
    roots = _browse_roots()
    if roots and not any(r == target or r in target.parents for r in roots):
        raise HTTPException(status_code=403,
                            detail="Path is outside the directories allowed by TG_OUTPUT_ROOTS")


def _resolve_scan_output_dir(raw: str) -> str:
    """Resolves scan output folder, allowing relative subfolders or browsable server directories."""
    base = Path(state.config.output_dir).resolve()
    if not raw:
        return str(base)
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        resolved = candidate.resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid output directory")

    _check_browsable(resolved)
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not create output directory: {e}")

    if not resolved.is_dir() or not _is_writable(resolved):
        raise HTTPException(status_code=400, detail="Output directory is not writable")
    return str(resolved)


# ---------------------------------------------------------------------------
# Control API
# ---------------------------------------------------------------------------

@app.post("/api/source")
def set_source(payload: Dict[str, Any] = Body(...)):
    state.touch_activity()
    src = str(payload.get("source", "")).strip()
    if not src:
        raise HTTPException(status_code=400, detail="source is required")
    if is_redacted(src):
        raise HTTPException(
            status_code=400,
            detail="Cannot connect with redacted credentials - use the Camera dialog to edit settings.",
        )
    if not _is_network_or_device_source(src):
        src = str(_resolve_video_path(src))

    state.config.stream_source = src
    if state.processor:
        state.processor.update_source(src)
    else:
        state.start_processor()
    return {"ok": True, "source": redact_credentials(src)}


# ---------------------------------------------------------------------------
# Camera connection
# ---------------------------------------------------------------------------

def _settings_from_payload(payload: Dict[str, Any]) -> camera_store.CameraSettings:
    """Builds settings from a request, keeping the saved password when the
    request omits it -- the UI is never given the password, so it cannot send it
    back, and a blank field on a reconnect must not wipe it."""
    saved = camera_store.load()
    kind = str(payload.get("kind", "rtsp")).strip().lower()
    if kind not in ("rtsp", "webcam", "file", "ndi"):
        raise HTTPException(status_code=400, detail=f"Unknown camera kind '{kind}'")

    password = payload.get("password", None)
    if password is None or password == "":
        password = saved.password if (saved and saved.kind == kind) else ""

    try:
        port = int(payload.get("port") or camera_probe.DEFAULT_PORT)
        http_port = int(payload.get("http_port") or (saved.http_port if saved else 80) or 80)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Port must be a number")

    return camera_store.CameraSettings(
        kind=kind,
        host=str(payload.get("host", "") or "").strip(),
        port=port,
        http_port=http_port,
        path=str(payload.get("path", "") or "").strip(),
        username=str(payload.get("username", "") or "").strip(),
        password=str(password),
        webcam_index=int(payload.get("webcam_index") or 0),
        file_path=str(payload.get("file_path", "") or "").strip(),
        ndi_source_name=str(payload.get("ndi_source_name", "") or "").strip(),
    )


@app.get("/api/camera")
def get_camera():
    """Saved settings, without the password."""
    saved = camera_store.load()
    return {
        "configured": bool(saved and saved.is_configured()),
        "settings": saved.public() if saved else camera_store.CameraSettings().public(),
        "active_source": redact_credentials(state.config.stream_source),
        "common_paths": camera_probe.COMMON_PATHS,
        "ndi": ndi_status(),
    }


@app.post("/api/camera/test")
def test_camera(payload: Dict[str, Any] = Body(...)):
    """Asks the camera directly, without touching the running stream."""
    state.touch_activity()
    cfg = _settings_from_payload(payload)

    if cfg.kind == "ndi":
        st = ndi_status()
        if not cfg.ndi_source_name:
            return {
                "ok": False, "stage": "ndi", "code": "missing",
                "message": "Enter or discover an NDI source name.",
                "hint": st.get("setup_hint", ""),
            }
        if not st.get("available"):
            return {
                "ok": False, "stage": "ndi", "code": "no_runtime",
                "message": "NDI runtime library (libndi) is not installed yet.",
                "hint": st.get("setup_hint", ""),
            }
        disc = discover_ndi_sources(timeout_ms=1000)
        names = [s["name"] for s in (disc.get("sources") or [])]
        matched = any(cfg.ndi_source_name.lower() in n.lower() for n in names)
        return {
            "ok": True,
            "stage": "ndi",
            "code": "ok" if matched else "unverified",
            "message": (
                f"NDI source '{cfg.ndi_source_name}' found on network."
                if matched else f"NDI runtime ready; '{cfg.ndi_source_name}' will be resolved on connect."
            ),
            "hint": "",
        }

    if cfg.kind == "file":
        if not cfg.file_path:
            return {"ok": False, "stage": "file", "code": "missing",
                    "message": "No file path provided.",
                    "hint": "Browse or enter the path as the server sees it."}
        try:
            target = Path(cfg.file_path).expanduser().resolve()
        except Exception:
            return {"ok": False, "stage": "file", "code": "invalid",
                    "message": f"Invalid file path: {cfg.file_path}",
                    "hint": "Check the path syntax."}
        roots = _allowed_video_roots()
        if roots and not any(root == target or root in target.parents for root in roots):
            raise HTTPException(
                status_code=403,
                detail="Path is outside the directories allowed by TG_VIDEO_ROOTS"
            )
        if not target.is_file():
            return {"ok": False, "stage": "file", "code": "missing",
                    "message": f"No file at {cfg.file_path}.",
                    "hint": "Check the path as the server sees it, not your own machine."}
        return {"ok": True, "stage": "file", "code": "ok",
                "message": f"Found {target.name}.", "hint": ""}

    if cfg.kind == "webcam":
        cap = cv2.VideoCapture(cfg.webcam_index,
                               cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        try:
            opened = cap.isOpened()
        finally:
            cap.release()
        return {"ok": opened, "stage": "webcam", "code": "ok" if opened else "no_device",
                "message": (f"Webcam {cfg.webcam_index} opened."
                            if opened else f"No webcam at index {cfg.webcam_index}."),
                "hint": "" if opened else "Try index 0 or 1."}

    if not cfg.host:
        raise HTTPException(status_code=400, detail="Enter the camera's address")

    result = camera_probe.probe(
        host=cfg.host, port=cfg.port, path=cfg.path,
        user=cfg.username, password=cfg.password,
        discover=True,
    )
    return result.as_dict()


@app.post("/api/camera/connect")
def connect_camera(payload: Dict[str, Any] = Body(...)):
    """Tests, saves, then switches the stream over."""
    state.touch_activity()
    cfg = _settings_from_payload(payload)
    force = bool(payload.get("force", False))

    if cfg.kind == "rtsp" and not cfg.host:
        raise HTTPException(status_code=400, detail="Enter the camera's address")
    if cfg.kind == "ndi" and not cfg.ndi_source_name:
        raise HTTPException(status_code=400, detail="Enter or select an NDI source name")
    if cfg.kind == "file":
        if not cfg.file_path:
            raise HTTPException(status_code=400, detail="Enter the path to the video file")
        cfg.file_path = str(_resolve_video_path(cfg.file_path))

    verdict: Dict[str, Any] = {}
    if cfg.kind == "rtsp" and not force:
        result = camera_probe.probe(
            host=cfg.host, port=cfg.port, path=cfg.path,
            user=cfg.username, password=cfg.password, discover=True)
        verdict = result.as_dict()
        if not result.ok:
            raise HTTPException(status_code=409, detail={
                "message": result.message, "hint": result.hint,
                "code": result.code, "stage": result.stage,
            })
        if result.working_path:
            cfg.path = result.working_path

    if not cfg.is_configured():
        raise HTTPException(
            status_code=400,
            detail="Enter the stream path, or press Test to find it automatically")

    source = cfg.url()
    if not source:
        raise HTTPException(status_code=400, detail="Could not build a stream URL")

    try:
        path_written = camera_store.save(cfg)
    except OSError as e:
        raise HTTPException(status_code=500,
                            detail=f"Could not save camera settings: {e}")

    state.config.stream_source = source
    if state.processor:
        state.processor.update_source(source)
    else:
        state.start_processor()

    print(f"[Camera] Connected to {redact_credentials(source)} "
          f"(settings saved to {path_written})")
    return {
        "ok": True,
        "source": redact_credentials(source),
        "settings": cfg.public(),
        "verdict": verdict,
    }


@app.post("/api/camera/forget")
def forget_camera():
    """Drops the saved camera and falls back to the environment stream source."""
    state.touch_activity()
    removed = camera_store.clear()
    fallback = os.environ.get("TG_STREAM_SOURCE", "0").strip() if "TG_STREAM_SOURCE" in os.environ else "0"
    state.config.stream_source = fallback
    if state.processor:
        state.processor.update_source(fallback)
    return {
        "ok": True,
        "removed": removed,
        "active_source": redact_credentials(fallback),
    }


@app.post("/api/capture/toggle")
def toggle_capture(payload: Dict[str, Any] = Body(...)):
    state.touch_activity()
    if "max_minutes" in payload and payload["max_minutes"] is not None:
        try:
            mm = max(0.0, min(1440.0, float(payload["max_minutes"])))
            state.config.capture_max_minutes = mm
            settings_store.save_ui_settings({"capture_max_minutes": mm})
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="max_minutes must be a number")

    enabled = bool(payload.get("enabled", False))
    if enabled and not state.output_confirmed:
        raise HTTPException(
            status_code=409,
            detail=f"Confirm the output folder before capturing. Currently "
                   f"{state.config.output_dir} - open Output Folder and choose it, or pick another.")
    state.is_capturing = enabled
    state.capture_started_at = time.time() if enabled else None

    proc = state.processor
    if proc:
        proc.auto_capture_enabled = enabled
        if enabled:
            judge = getattr(proc, "judge", None)
            if judge is not None and judge.enabled:
                min_h = (state.config.capture_max_minutes / 60.0) if state.config.capture_max_minutes > 0 else None
                judge.wake(min_hours=min_h)

    return {
        "ok": True,
        "is_capturing": enabled,
        "capture_max_minutes": round(float(state.config.capture_max_minutes), 1),
        "capture_seconds_remaining": state.capture_seconds_remaining(),
    }


@app.post("/api/capture/manual")
def manual_capture():
    state.touch_activity()
    proc = state.processor
    if not proc:
        raise HTTPException(status_code=409, detail="Stream not running")
    if not state.output_confirmed:
        raise HTTPException(
            status_code=409,
            detail=f"Confirm the output folder first. Currently {state.config.output_dir}.")
    frame = None
    with proc.frame_lock:
        if proc.latest_frame is not None:
            frame = proc.latest_frame.copy()
    analysis = proc.latest_analysis
    if frame is None or analysis is None:
        raise HTTPException(status_code=409, detail="No frame available yet")
    cap = proc.save_screenshot(frame, analysis)
    return {"ok": True, "filename": os.path.basename(cap.filepath)}


@app.post("/api/settings")
def update_settings(payload: Dict[str, Any] = Body(...)):
    state.touch_activity()
    cfg = state.config
    patch: Dict[str, Any] = {}
    if "min_score" in payload:
        val_ms = float(payload["min_score"])
        cfg.thresholds.min_overall_score = val_ms / 100.0
        patch["min_score"] = val_ms
    if "max_yaw" in payload:
        val_yaw = float(payload["max_yaw"])
        cfg.thresholds.max_yaw_deg = val_yaw
        patch["max_yaw"] = val_yaw
    if "max_shoulder_tilt" in payload:
        val_st = float(payload["max_shoulder_tilt"])
        cfg.thresholds.max_shoulder_tilt_deg = val_st
        patch["max_shoulder_tilt"] = val_st
    if "burst_window" in payload:
        val_bw = float(payload["burst_window"])
        cfg.burst_window_seconds = val_bw
        patch["burst_window"] = val_bw
    if "cooldown" in payload:
        val_cd = float(payload["cooldown"])
        cfg.cooldown_seconds = val_cd
        patch["cooldown"] = val_cd
    if "people_target" in payload:
        val = max(1, min(6, int(payload["people_target"])))
        cfg.max_people_target = val
        patch["people_target"] = val
        if state.processor:
            state.processor.analyzer.max_people_target = val
            if state.processor._refine_analyzer is not None:
                state.processor._refine_analyzer.max_people_target = val
    if "max_retained" in payload:
        val = max(1, min(200, int(payload["max_retained"])))
        cfg.max_retained_captures = val
        patch["max_retained"] = val
        if state.processor:
            state.processor.config.max_retained_captures = val
            if cfg.single_winner_mode:
                state.processor.prune_to_best_n(val)
    if "capture_max_minutes" in payload:
        val_mm = max(0.0, min(1440.0, float(payload["capture_max_minutes"])))
        cfg.capture_max_minutes = val_mm
        patch["capture_max_minutes"] = val_mm
    for bool_key in (
        "mood_buckets_enabled",
        "auto_save_cutout",
        "broadcast_polish_enabled",
        "upscale_4k_enabled",
        "audio_gating_enabled",
    ):
        if bool_key in payload:
            bval = bool(payload[bool_key])
            setattr(cfg, bool_key, bval)
            patch[bool_key] = bval
            if state.processor:
                setattr(state.processor.config, bool_key, bval)
                if bool_key == "audio_gating_enabled":
                    aud = getattr(state.processor, "audio_monitor", None)
                    if aud is not None:
                        if bval and cfg.stream_source:
                            aud.start(cfg.stream_source)
                        elif not bval:
                            aud.stop()
    if patch:
        settings_store.save_ui_settings(patch)
    return {"ok": True}


@app.post("/api/view")
def update_view(payload: Dict[str, Any] = Body(...)):
    state.touch_activity()
    patch: Dict[str, Any] = {}
    if "show_mesh" in payload:
        state.show_mesh = bool(payload["show_mesh"])
        patch["show_mesh"] = state.show_mesh
    if "show_hud" in payload:
        state.show_hud = bool(payload["show_hud"])
        patch["show_hud"] = state.show_hud
    if "single_winner" in payload:
        state.config.single_winner_mode = bool(payload["single_winner"])
        patch["single_winner"] = state.config.single_winner_mode
        if state.config.single_winner_mode and state.processor:
            state.processor.prune_to_best_n(state.config.max_retained_captures)
    if "max_retained" in payload:
        val = max(1, min(200, int(payload["max_retained"])))
        state.config.max_retained_captures = val
        patch["max_retained"] = val
        if state.processor:
            state.processor.config.max_retained_captures = val
            if state.config.single_winner_mode:
                state.processor.prune_to_best_n(val)
    if "capture_prop_moments" in payload:
        val = bool(payload["capture_prop_moments"])
        state.config.capture_prop_moments = val
        patch["capture_prop_moments"] = val
        if state.processor:
            state.processor.analyzer.capture_prop_moments = val
            if state.processor._refine_analyzer is not None:
                state.processor._refine_analyzer.capture_prop_moments = val
    for bool_key in (
        "mood_buckets_enabled",
        "auto_save_cutout",
        "broadcast_polish_enabled",
        "upscale_4k_enabled",
        "audio_gating_enabled",
    ):
        if bool_key in payload:
            bval = bool(payload[bool_key])
            setattr(state.config, bool_key, bval)
            patch[bool_key] = bval
            if state.processor:
                setattr(state.processor.config, bool_key, bval)
                if bool_key == "audio_gating_enabled":
                    aud = getattr(state.processor, "audio_monitor", None)
                    if aud is not None:
                        if bval and state.config.stream_source:
                            aud.start(state.config.stream_source)
                        elif not bval:
                            aud.stop()
    if patch:
        settings_store.save_ui_settings(patch)
    return {"ok": True}


@app.post("/api/person/{person_id}")
def toggle_person(person_id: int, payload: Dict[str, Any] = Body(default={})):
    state.touch_activity()
    proc = state.processor
    if not proc:
        raise HTTPException(status_code=409, detail="Stream not running")
    ids = proc.analyzer.selected_person_ids
    if "selected" in payload:
        selected = bool(payload["selected"])
    else:
        selected = person_id not in ids  # plain toggle
    if selected:
        ids.add(person_id)
    else:
        ids.discard(person_id)
    if proc._refine_analyzer is not None:
        proc._refine_analyzer.selected_person_ids = set(ids)
    return {"ok": True, "selected_people": sorted(ids)}


@app.post("/api/click")
def click_video(payload: Dict[str, Any] = Body(...)):
    """Toggle a person or lock onto a target speaker by clicking their face box on the video."""
    state.touch_activity()
    proc = state.processor
    frame, analysis, _ = state.snapshot()
    if not proc or analysis is None or frame is None:
        raise HTTPException(status_code=409, detail="No active analysis")
    fx = float(payload.get("x_frac", -1)) * frame.shape[1]
    fy = float(payload.get("y_frac", -1)) * frame.shape[0]
    mode = str(payload.get("mode", "toggle") or "toggle").strip().lower()
    speaker_name = str(payload.get("name", "") or "").strip()
    for f in (analysis.faces or []):
        bx, by, bw, bh = f.bbox
        if bx <= fx <= bx + bw and by <= fy <= by + bh:
            if mode == "lock_speaker":
                spk = proc.analyzer.speaker_id
                res = spk.lock_from_frame(frame, f.bbox, save_as_name=speaker_name, download_sface=True)
                if proc._refine_analyzer is not None:
                    proc._refine_analyzer.speaker_id = spk
                return res
            return toggle_person(f.person_id, {})
    return {"ok": False, "detail": "No face at that point"}


@app.post("/api/buffer")
def set_buffer(payload: Dict[str, Any] = Body(...)):
    """Resizes the rolling frame buffer and reports the measured memory it now holds."""
    state.touch_activity()
    proc = state.processor
    if not proc:
        raise HTTPException(status_code=409, detail="Stream not running")
    try:
        seconds = float(payload.get("seconds", proc.config.dvr_buffer_seconds))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="seconds must be a number")
    fps = payload.get("fps")
    try:
        fps = int(fps) if fps is not None else None
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="fps must be a whole number")
    stats = proc.set_buffer(seconds, fps)
    settings_store.save_ui_settings({
        "dvr_buffer_seconds": proc.config.dvr_buffer_seconds,
        "dvr_buffer_fps": proc.config.dvr_buffer_fps,
    })
    return {"ok": True, "buffer": stats}


@app.post("/api/judge/wake")
def wake_judge():
    """Re-arms the judge for another full active window."""
    state.touch_activity()
    proc = state.processor
    if not proc or not getattr(proc, "judge", None):
        raise HTTPException(status_code=409, detail="Stream not running")
    min_h = (state.config.capture_max_minutes / 60.0) if state.config.capture_max_minutes > 0 else None
    proc.judge.wake(min_hours=min_h)
    return {"ok": True, "status": proc.judge.status_line(),
            "minutes_left": (None if proc.judge.seconds_remaining == float("inf")
                             else int(proc.judge.seconds_remaining // 60))}


@app.post("/api/judge/sleep")
def sleep_judge():
    """Ends the window now and releases the model on the Ollama host."""
    state.touch_activity()
    proc = state.processor
    if not proc or not getattr(proc, "judge", None):
        raise HTTPException(status_code=409, detail="Stream not running")
    proc.judge.sleep_now()
    return {"ok": True, "status": proc.judge.status_line()}


# ---------------------------------------------------------------------------
# Gallery, Pins, Downloads & Feedback
# ---------------------------------------------------------------------------

@app.get("/api/captures")
def list_captures():
    return {"captures": _captures_payload()}


@app.get("/api/captures/download-zip")
def download_captures_zip(mode: str = "all"):
    """Bundles gallery captures into a downloadable .zip archive."""
    mode = (mode or "all").strip().lower()
    if mode not in ("all", "pinned", "unpinned"):
        raise HTTPException(status_code=400, detail="mode must be 'all', 'pinned', or 'unpinned'")

    caps = _visible_captures()
    if mode == "pinned":
        caps = [c for c in caps if getattr(c, "is_pinned", False)]
    elif mode == "unpinned":
        caps = [c for c in caps if not getattr(c, "is_pinned", False)]

    valid = [c for c in caps if c.filepath and os.path.isfile(c.filepath)]
    if not valid:
        raise HTTPException(status_code=404, detail="No captures available to download")

    spool = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024)
    used_names = set()
    with zipfile.ZipFile(spool, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for idx, c in enumerate(valid, start=1):
            base_name = os.path.basename(c.filepath)
            arcname = f"{idx:02d}_{base_name}"
            if arcname in used_names:
                stem, ext = os.path.splitext(base_name)
                arcname = f"{idx:02d}_{stem}_{idx}{ext}"
            used_names.add(arcname)
            zf.write(c.filepath, arcname=arcname)
    spool.seek(0)

    def _iter_spool():
        try:
            while True:
                chunk = spool.read(256 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            spool.close()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_label = Path(state.config.output_dir).name or "captures"
    safe_label = "".join(ch for ch in folder_label if ch.isalnum() or ch in "_-")[:32] or "captures"
    zip_name = f"thumbnails_{safe_label}_{ts}.zip"
    return StreamingResponse(
        _iter_spool(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


def _load_capture_crop_bgr(cap: CapturedScreenshot) -> Optional[np.ndarray]:
    """Loads the subject crop (or full frame fallback) for visual prototype learning."""
    cand_path = cap.crop_filepath if (cap.crop_filepath and os.path.isfile(cap.crop_filepath)) else cap.filepath
    if not cand_path or not os.path.isfile(cand_path):
        return None
    img = cv2.imread(cand_path, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None
    if cand_path == cap.filepath and cap.analysis is not None:
        h, w = img.shape[:2]
        rect = subject_crop_rect(cap.analysis, w, h, state.config.subject_crop_padding)
        if rect:
            sub = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            if sub.size > 0:
                return sub
    return img


@app.get("/api/captures/{filename}/download")
def download_single_capture(filename: str, learn: int = 1):
    """Downloads a single captured frame with Content-Disposition: attachment."""
    cap = _find_capture(filename)
    if not cap.filepath or not os.path.isfile(cap.filepath):
        raise HTTPException(status_code=404, detail="Capture file missing on disk")
    proc = state.processor
    if learn and proc and getattr(proc, "analyzer", None):
        pm = proc.analyzer.preference_model
        peers = _visible_captures()
        pm.record_implicit_feedback(cap, peer_captures=peers, action="download", crop_bgr=_load_capture_crop_bgr(cap))
    ext = os.path.splitext(cap.filepath)[1].lower()
    media = "image/png" if ext == ".png" else "image/jpeg"
    return FileResponse(
        cap.filepath,
        media_type=media,
        filename=os.path.basename(cap.filepath),
    )


@app.post("/api/captures/{filename}/feedback")
def capture_feedback(filename: str, payload: Dict[str, Any] = Body(...)):
    """Train the preference model from a thumbnail's star / thumbs-down and optional comment/reasons."""
    state.touch_activity()
    proc = state.processor
    if not proc:
        raise HTTPException(status_code=409, detail="Stream not running")
    cap = _find_capture(filename)
    is_favorite = bool(payload.get("favorite", True))
    comment = str(payload.get("comment", "") or "").strip()
    raw_reasons = payload.get("reasons") or []
    reasons = [str(r).strip() for r in raw_reasons if str(r).strip()] if isinstance(raw_reasons, list) else []
    remove = bool(payload.get("remove", False))

    pm = proc.analyzer.preference_model
    crop_bgr = _load_capture_crop_bgr(cap)
    sample_id = pm.add_sample(
        cap.analysis,
        is_favorite=is_favorite,
        comment=comment,
        reasons=reasons,
        crop_bgr=crop_bgr,
        source="human",
        weight=1.0,
        filename=cap.filepath,
    )

    # Pairwise session learning: when starring a shot, record preference over unpinned session peers
    if is_favorite:
        for peer in _visible_captures()[:8]:
            if peer is not cap and not getattr(peer, "is_pinned", False) and peer.analysis is not None:
                pm.add_pairwise_preference(cap.analysis, peer.analysis, weight=0.85)
        pm.fit()
        pm.save()
        proc.set_capture_pinned(cap.filepath, True)

    removed = False
    if remove:
        removed = proc.remove_capture(cap.filepath)
        if state.scan_results:
            state.scan_results = [r for r in state.scan_results if r.filepath != cap.filepath]

    return {
        "ok": True,
        "sample_id": sample_id,
        "favorite": is_favorite,
        "is_pinned": bool(getattr(cap, "is_pinned", False)) if not removed else False,
        "comment": comment,
        "reasons": reasons,
        "removed": removed,
        "samples": len(pm.samples),
        "notes_count": len(getattr(pm, "feedback_notes", []) or []),
        "learned_penalties": dict(getattr(pm, "learned_penalties", {}) or {}),
        "confidence": round(float(pm.confidence), 3),
        "trust": round(float(pm.trust), 3),
        "captures": _captures_payload(),
    }


@app.post("/api/captures/{filename}/pin")
def pin_capture(filename: str, payload: Dict[str, Any] = Body(default={})):
    """Flags or unflags a capture so it is protected from auto-deletion when pruning to Top N."""
    state.touch_activity()
    proc = state.processor
    if not proc:
        raise HTTPException(status_code=409, detail="Stream not running")
    cap = _find_capture(filename)
    desired = payload.get("pinned", None) if isinstance(payload, dict) else None
    pinned = proc.set_capture_pinned(cap.filepath, None if desired is None else bool(desired))
    if pinned is None:
        raise HTTPException(status_code=404, detail="Capture not found")
    if pinned and getattr(proc, "analyzer", None):
        proc.analyzer.preference_model.record_implicit_feedback(
            cap, peer_captures=_visible_captures(), action="pin", crop_bgr=_load_capture_crop_bgr(cap)
        )
    return {
        "ok": True,
        "filename": os.path.basename(cap.filepath),
        "is_pinned": pinned,
        "captures": _captures_payload(),
    }


@app.delete("/api/captures/{filename}")
def delete_single_capture(filename: str, learn: int = 1):
    """Deletes a capture from gallery & disk and optionally records implicit negative feedback."""
    state.touch_activity()
    proc = state.processor
    if not proc:
        raise HTTPException(status_code=409, detail="Stream not running")
    cap = _find_capture(filename)
    if learn and getattr(proc, "analyzer", None):
        proc.analyzer.preference_model.record_implicit_feedback(
            cap, action="delete", crop_bgr=_load_capture_crop_bgr(cap)
        )
    removed = proc.remove_capture(cap.filepath)
    if state.scan_results:
        state.scan_results = [r for r in state.scan_results if r.filepath != cap.filepath]
    return {"ok": True, "removed": removed, "captures": _captures_payload()}


@app.post("/api/captures/clear")
def clear_captures_endpoint(payload: Dict[str, Any] = Body(default={})):
    """Clears unpinned top shots (to reset between services while keeping 📌 Saved shots) or deletes pinned shots."""
    state.touch_activity()
    proc = state.processor
    if not proc:
        raise HTTPException(status_code=409, detail="Stream not running")
    mode = str((payload or {}).get("mode", "unpinned") or "unpinned").strip().lower()
    if mode not in ("unpinned", "pinned", "all"):
        raise HTTPException(status_code=400, detail="mode must be 'unpinned', 'pinned', or 'all'")
    summary = proc.clear_captures(mode=mode)
    if state.scan_results:
        kept_paths = {c.filepath for c in proc.saved_captures}
        state.scan_results = [r for r in state.scan_results if r.filepath in kept_paths]
    return {
        "ok": True,
        "mode": mode,
        **summary,
        "captures": _captures_payload(),
    }


@app.get("/captures/{filename:path}")
def get_capture_image(filename: str):
    # Resolve against the output dir and refuse anything that escapes it
    base = Path(state.config.output_dir).resolve()
    target = (base / filename).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    media = "image/png" if target.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(str(target), media_type=media)


# ---------------------------------------------------------------------------
# Folder & Server Video Browsing
# ---------------------------------------------------------------------------

def _drive_list() -> List[Dict[str, str]]:
    """Windows drive letters, so the picker can escape the current tree."""
    if os.name != "nt":
        return [{"name": "/", "path": "/"}]
    out = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        d = f"{letter}:\\"
        if os.path.exists(d):
            out.append({"name": f"{letter}:", "path": d})
    return out


@app.get("/api/browse")
def browse_folders(path: str = ""):
    """Lists subdirectories of `path` for the output folder picker."""
    if not path:
        target = Path(state.config.output_dir)
    else:
        target = Path(path)
    try:
        target = target.expanduser().resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {target}")
    _check_browsable(target)

    entries = []
    try:
        with os.scandir(target) as it:
            for e in it:
                if e.name.startswith("."):
                    continue
                try:
                    if e.is_dir():
                        entries.append({"name": e.name, "path": str(Path(e.path))})
                except OSError:
                    continue
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {target}")
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Could not read directory: {e}")

    entries.sort(key=lambda d: d["name"].lower())
    roots = _browse_roots()
    parent_cand = target.parent
    if parent_cand != target and (not roots or any(r == parent_cand or r in parent_cand.parents for r in roots)):
        parent = str(parent_cand)
    else:
        parent = None

    shortcuts = []
    if roots:
        for r in roots:
            if r.is_dir():
                shortcuts.append({"name": r.name or str(r), "path": str(r)})
    else:
        home = home_dir()
        if home is not None:
            shortcuts.append({"name": "Home", "path": str(home)})
            for sub in ("Pictures", "Videos", "Desktop", "Downloads"):
                q = home / sub
                if q.is_dir():
                    shortcuts.append({"name": sub, "path": str(q)})
    shortcuts.append({"name": "Current output", "path": str(Path(state.config.output_dir))})

    return {
        "path": str(target),
        "parent": parent,
        "entries": entries[:1000],
        "drives": [] if roots else _drive_list(),
        "shortcuts": shortcuts,
        "writable": _is_writable(target),
        "is_current": str(target) == str(Path(state.config.output_dir).resolve()),
    }


@app.get("/api/browse/videos")
def browse_videos(path: str = ""):
    """Lists directories and video files allowed by TG_VIDEO_ROOTS for zero-typing selection."""
    roots = _allowed_video_roots()
    if not path:
        target = None
        if roots:
            for r in roots:
                if r.is_dir():
                    target = r
                    break
            if target is None:
                uploads = Path(state.config.output_dir) / "uploads"
                uploads.mkdir(parents=True, exist_ok=True)
                target = uploads.resolve()
        else:
            home = home_dir()
            vids = (home / "Videos") if home is not None else None
            target = vids if (vids and vids.is_dir()) else Path(state.config.output_dir)
    else:
        target = Path(path)

    try:
        target = target.expanduser().resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")

    if roots and not any(r == target or r in target.parents for r in roots):
        raise HTTPException(
            status_code=403,
            detail="Path is outside the directories allowed by TG_VIDEO_ROOTS"
        )
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {target}")

    dirs: List[Dict[str, Any]] = []
    videos: List[Dict[str, Any]] = []
    try:
        with os.scandir(target) as it:
            for e in it:
                if e.name.startswith("."):
                    continue
                try:
                    if e.is_dir():
                        dirs.append({"kind": "dir", "name": e.name, "path": str(Path(e.path))})
                    elif e.is_file() and e.name.lower().endswith(VIDEO_EXTENSIONS):
                        st = e.stat()
                        videos.append({
                            "kind": "video",
                            "name": e.name,
                            "path": str(Path(e.path)),
                            "size_mb": round(st.st_size / (1024 * 1024), 1),
                            "mtime": int(st.st_mtime),
                        })
                except OSError:
                    continue
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {target}")
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Could not read directory: {e}")

    dirs.sort(key=lambda d: d["name"].lower())
    videos.sort(key=lambda v: (-v["mtime"], v["name"].lower()))

    parent_cand = target.parent
    if parent_cand != target and (not roots or any(r == parent_cand or r in parent_cand.parents for r in roots)):
        parent = str(parent_cand)
    else:
        parent = None

    shortcuts = []
    if roots:
        for r in roots:
            if r.is_dir():
                shortcuts.append({"name": r.name or str(r), "path": str(r)})
    else:
        home = home_dir()
        if home is not None:
            for sub in ("Videos", "Downloads", "Desktop"):
                q = home / sub
                if q.is_dir():
                    shortcuts.append({"name": sub, "path": str(q)})
        shortcuts.append({"name": "Output Folder", "path": str(Path(state.config.output_dir))})
        uploads_dir = Path(state.config.output_dir) / "uploads"
        if uploads_dir.is_dir():
            shortcuts.append({"name": "Uploads", "path": str(uploads_dir)})

    return {
        "path": str(target),
        "parent": parent,
        "entries": (dirs + videos)[:1000],
        "shortcuts": shortcuts,
    }


@app.post("/api/output")
def set_output_dir(payload: Dict[str, Any] = Body(...)):
    """Points captures at a different folder, switching the folder-scoped gallery & manifest."""
    state.touch_activity()
    raw = str(payload.get("path", "")).strip()
    if not raw:
        raise HTTPException(status_code=400, detail="No path given")
    try:
        target = Path(raw).expanduser().resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")

    _check_browsable(target)

    if not target.exists():
        if not bool(payload.get("create", False)):
            raise HTTPException(status_code=404, detail="Folder does not exist")
        try:
            target.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not create folder: {e}")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Not a directory")
    if not _is_writable(target):
        raise HTTPException(status_code=400, detail="Folder is not writable by the server")

    warning = _persistence_warning(target)
    if warning:
        print(f"[Output] WARNING: {warning}")

    state.config.output_dir = str(target)
    if state.processor:
        state.processor.switch_output_dir(str(target), adopt_orphans=True)
    state.output_confirmed = True
    if not warning:
        settings_store.save_ui_settings({
            "output_dir": str(target),
            "output_confirmed": True,
        })
    return {
        "ok": True,
        "output_dir": str(target),
        "output_confirmed": True,
        "warning": warning,
        "captures": _captures_payload(),
    }


# ---------------------------------------------------------------------------
# Video Recording Scanning API
# ---------------------------------------------------------------------------

def _frame_to_base64(frame: Optional[np.ndarray], max_w: int = 480) -> Optional[str]:
    if frame is None:
        return None
    try:
        h, w = frame.shape[:2]
        if w > max_w:
            scale = max_w / float(w)
            small = cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = frame
        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok:
            return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception:
        pass
    return None


def _is_managed_upload(video_path: str) -> bool:
    try:
        v = Path(video_path).resolve()
        uploads = (Path(state.config.output_dir) / "uploads").resolve()
        return v.parent == uploads
    except Exception:
        return False


def _cleanup_stale_uploads(uploads_dir: Path, max_age_sec: float = 86400.0) -> None:
    if not uploads_dir.is_dir():
        return
    now = time.time()
    active_scan = os.path.normcase(os.path.abspath(state.scan_video_path or "")) if state.scan_video_path else ""
    active_stream = os.path.normcase(os.path.abspath(state.config.stream_source or "")) if state.config.stream_source else ""
    try:
        for f in uploads_dir.iterdir():
            if not f.is_file():
                continue
            norm_f = os.path.normcase(os.path.abspath(str(f)))
            if norm_f in (active_scan, active_stream):
                continue
            if (now - f.stat().st_mtime) > max_age_sec:
                f.unlink(missing_ok=True)
    except OSError:
        pass


def _run_video_scan_worker(video_path: str, top_n: int, min_score: float,
                           sample_fps: float, min_interval_sec: float, out_dir: str,
                           start_sec: float = 0.0, end_sec: Optional[float] = None,
                           cleanup_upload: bool = True):
    state.is_scanning_video = True
    state.scan_error = None
    state.scan_results = []
    state.scan_video_path = video_path
    state.scan_output_dir = out_dir
    state._scan_champ_id = 0
    state._scan_champ_b64 = None

    # Pause live RTSP MediaPipe analysis during offline video scanning so the two
    # don't fight over the same CPU cores and cut scan speed in half.
    if state.processor:
        state.processor.paused = True
        judge = getattr(state.processor, "judge", None)
        if judge is not None and judge.enabled:
            judge.wake()

    def progress_cb(p: ScanProgress):
        state.scan_progress = p

    try:
        proc = state.processor
        shared_judge = getattr(proc, "judge", None) if proc else None
        shared_spk = getattr(proc.analyzer, "speaker_id", None) if proc else None
        shared_zones = getattr(proc.analyzer, "stage_zones", None) if proc else state.stage_zones
        burst_cb = proc.cache_burst_around if proc else None
        scanner = VideoRecordingScanner(
            state.config,
            judge=shared_judge,
            speaker_id=shared_spk,
            stage_zones=shared_zones,
            burst_cache_callback=burst_cb,
        )
        results = scanner.scan_video(
            video_path=video_path,
            top_n=top_n,
            min_score=min_score,
            sample_fps=sample_fps,
            min_interval_sec=min_interval_sec,
            output_dir=out_dir,
            progress_callback=progress_cb,
            cancel_event=state.scan_cancel_event,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        state.scan_results = results
        if state.processor:
            state.processor.add_scan_captures(results, target_dir=out_dir)
    except Exception as e:
        state.scan_error = str(e)
    finally:
        if state.processor:
            state.processor.paused = False
        state.is_scanning_video = False
        if cleanup_upload and _is_managed_upload(video_path):
            active_stream = os.path.normcase(os.path.abspath(state.config.stream_source or "")) if state.config.stream_source else ""
            if os.path.normcase(os.path.abspath(video_path)) != active_stream:
                try:
                    Path(video_path).unlink(missing_ok=True)
                except OSError:
                    pass


@app.post("/api/video/info")
def video_info(payload: Dict[str, Any] = Body(...)):
    state.touch_activity()
    v_path = str(_resolve_video_path(str(payload.get("path", "")).strip()))

    info = VideoRecordingScanner.get_video_info(v_path)
    if not info:
        raise HTTPException(status_code=400, detail="Could not read video metadata")

    dur_str = VideoRecordingScanner.format_time_display(info.duration_sec)
    return {
        "ok": True,
        "filename": info.filename,
        "path": info.filepath,
        "duration_sec": round(info.duration_sec, 2),
        "duration_str": dur_str,
        "width": info.width,
        "height": info.height,
        "fps": round(info.fps, 1),
        "total_frames": info.total_frames
    }


@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    state.touch_activity()
    out_dir = Path(state.config.output_dir) / "uploads"
    out_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_uploads(out_dir)

    dest_path = out_dir / _sanitize_upload_name(file.filename)

    limit_bytes = max(1, int(state.config.max_upload_mb)) * 1024 * 1024
    written = 0
    try:
        with open(dest_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit_bytes:
                    f.close()
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the {state.config.max_upload_mb} MB limit"
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    info = VideoRecordingScanner.get_video_info(str(dest_path))
    if not info:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid video")

    return {
        "ok": True,
        "path": str(dest_path),
        "filename": info.filename,
        "duration_sec": round(info.duration_sec, 2),
        "duration_str": VideoRecordingScanner.format_time_display(info.duration_sec),
        "width": info.width,
        "height": info.height,
        "fps": round(info.fps, 1),
        "total_frames": info.total_frames
    }


@app.post("/api/video/scan")
def start_video_scan(payload: Dict[str, Any] = Body(...)):
    state.touch_activity()
    if state.is_scanning_video:
        raise HTTPException(status_code=409, detail="A video scan is already running")

    v_path = str(_resolve_video_path(str(payload.get("path", "")).strip()))

    top_n = max(1, min(100, int(payload.get("top_n", 5))))
    min_score = float(payload.get("min_score", state.config.thresholds.min_overall_score))
    sample_fps = max(0.2, min(60.0, float(payload.get("sample_fps", 5.0))))
    min_interval_sec = max(0.0, float(payload.get("min_interval_sec", 2.0)))
    out_dir = _resolve_scan_output_dir(str(payload.get("output_dir", "")).strip())

    raw_start = payload.get("start_time", payload.get("start_sec", ""))
    raw_end = payload.get("end_time", payload.get("end_sec", ""))
    try:
        start_sec = VideoRecordingScanner.parse_timestamp_sec(raw_start) or 0.0
        end_sec = VideoRecordingScanner.parse_timestamp_sec(raw_end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if end_sec is not None and end_sec <= start_sec:
        raise HTTPException(status_code=400, detail="End time must be greater than start time")

    cleanup_upload = bool(payload.get("cleanup_upload", True))

    state.scan_cancel_event = threading.Event()
    state.scan_thread = threading.Thread(
        target=_run_video_scan_worker,
        args=(v_path, top_n, min_score, sample_fps, min_interval_sec, out_dir,
              start_sec, end_sec, cleanup_upload),
        daemon=True
    )
    state.scan_thread.start()

    return {
        "ok": True,
        "message": "Video scan started",
        "video": v_path,
        "output_dir": out_dir,
        "start_sec": start_sec,
        "end_sec": end_sec,
    }


@app.get("/api/video/scan/status")
def get_video_scan_status():
    p = state.scan_progress
    resp: Dict[str, Any] = {
        "is_scanning": state.is_scanning_video,
        "status": p.status if p else ("scanning" if state.is_scanning_video else "idle"),
        "video_path": state.scan_video_path,
        "scan_output_dir": state.scan_output_dir,
        "progress_pct": round(p.progress_pct, 1) if p else 0.0,
        "current_frame_idx": p.current_frame_idx if p else 0,
        "total_frames": p.total_frames if p else 0,
        "current_time_sec": round(p.current_time_sec, 2) if p else 0.0,
        "current_time_str": VideoRecordingScanner.format_time_display(p.current_time_sec) if p else "00:00",
        "total_duration_sec": round(p.total_duration_sec, 2) if p else 0.0,
        "total_duration_str": VideoRecordingScanner.format_time_display(p.total_duration_sec) if p else "00:00",
        "fps_speed": round(p.fps_speed, 1) if p else 0.0,
        "eta_sec": int(p.estimated_time_remaining_sec) if p else 0,
        "best_score": round(float(p.best_score_found), 3) if p else 0.0,
        "candidates_analyzed": p.total_candidates_analyzed if p else 0,
        "error": state.scan_error or (p.error_message if p else None),
        "results_count": len(state.scan_results),
    }

    if p:
        resp["current_frame_preview"] = _frame_to_base64(p.current_frame, max_w=400)
        if p.champion_frame is not None:
            cid = id(p.champion_frame)
            if cid != state._scan_champ_id:
                state._scan_champ_id = cid
                state._scan_champ_b64 = _frame_to_base64(p.champion_frame, max_w=400)
        resp["champion_frame_preview"] = state._scan_champ_b64

    if not state.is_scanning_video and state.scan_results:
        results_payload = []
        for idx, c in enumerate(state.scan_results):
            name = os.path.basename(c.filepath)
            cpath = getattr(c, "cutout_filepath", None) or cutout_path_for(c.filepath)
            results_payload.append({
                "rank": idx + 1,
                "filename": name,
                "url": _capture_url(c.filepath),
                "crop_url": _capture_url(c.crop_filepath),
                "cutout_url": _capture_url(cpath) if (cpath and os.path.exists(cpath)) else None,
                "score": round(float(c.overall_score), 3),
                "timestamp": c.timestamp_str,
                "is_prop_moment": bool(getattr(c, "is_prop_moment", False)),
                "mood_bucket": str(getattr(c, "mood_bucket", "") or ""),
                "finals_badge": str(getattr(c, "finals_badge", "") or ""),
            })
        resp["results"] = results_payload

    return resp


@app.post("/api/video/scan/cancel")
def cancel_video_scan():
    state.touch_activity()
    if state.scan_cancel_event:
        state.scan_cancel_event.set()
    return {"ok": True, "message": "Cancellation requested"}


# ---------------------------------------------------------------------------
# Next-Gen Studio, ML Brain, Speaker Lock, Stage Zones, Audio & PTZ APIs
# ---------------------------------------------------------------------------

def _check_bootstrap_folder_allowed(target: Path) -> None:
    """Allows folders under either TG_OUTPUT_ROOTS or TG_VIDEO_ROOTS when restricted."""
    out_roots = _browse_roots()
    vid_roots = _allowed_video_roots()
    combined = out_roots + [r for r in vid_roots if r not in out_roots]
    if combined and not any(r == target or r in target.parents for r in combined):
        raise HTTPException(
            status_code=403,
            detail="Folder is outside the directories allowed by TG_OUTPUT_ROOTS / TG_VIDEO_ROOTS",
        )


@app.get("/api/ml/brain")
def get_ml_brain():
    state.touch_activity()
    proc = state.processor
    if not proc or not getattr(proc, "analyzer", None):
        raise HTTPException(status_code=409, detail="Analyzer not initialized")
    pm = proc.analyzer.preference_model
    summary = pm.get_brain_summary()
    summary["visual_backbone"] = get_visual_embedder().status()
    return summary


@app.post("/api/ml/undo")
def ml_undo_rating(payload: Dict[str, Any] = Body(...)):
    state.touch_activity()
    proc = state.processor
    if not proc or not getattr(proc, "analyzer", None):
        raise HTTPException(status_code=409, detail="Analyzer not initialized")
    sid = str(payload.get("sample_id", "") or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="sample_id is required")
    ok = proc.analyzer.preference_model.undo_rating(sid)
    if not ok:
        raise HTTPException(status_code=404, detail="Sample not found")
    return {"ok": True, **proc.analyzer.preference_model.get_brain_summary()}


@app.post("/api/ml/flip")
def ml_flip_rating(payload: Dict[str, Any] = Body(...)):
    state.touch_activity()
    proc = state.processor
    if not proc or not getattr(proc, "analyzer", None):
        raise HTTPException(status_code=409, detail="Analyzer not initialized")
    sid = str(payload.get("sample_id", "") or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="sample_id is required")
    ok = proc.analyzer.preference_model.flip_rating(sid)
    if not ok:
        raise HTTPException(status_code=404, detail="Sample not found")
    return {"ok": True, **proc.analyzer.preference_model.get_brain_summary()}


@app.post("/api/ml/reset")
def ml_reset_brain():
    state.touch_activity()
    proc = state.processor
    if not proc or not getattr(proc, "analyzer", None):
        raise HTTPException(status_code=409, detail="Analyzer not initialized")
    proc.analyzer.preference_model.reset_all()
    return {"ok": True, **proc.analyzer.preference_model.get_brain_summary()}


@app.post("/api/ml/bootstrap-folder")
def ml_bootstrap_folder(payload: Dict[str, Any] = Body(...)):
    """Pre-trains the ML preference model from a folder of past winning thumbnails (ML-3)."""
    state.touch_activity()
    proc = state.processor
    if not proc or not getattr(proc, "analyzer", None):
        raise HTTPException(status_code=409, detail="Analyzer not initialized")
    raw = str(payload.get("path", "") or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Folder path is required")
    try:
        target = Path(raw).expanduser().resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid folder path")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Folder not found: {target}")
    _check_bootstrap_folder_allowed(target)

    max_imgs = max(1, min(300, int(payload.get("max_images", 80) or 80)))
    ref_analyzer = proc._get_or_create_refine_analyzer()
    res = proc.analyzer.preference_model.bootstrap_from_folder(
        str(target), ref_analyzer, max_images=max_imgs
    )
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("error", "Bootstrap failed"))
    return {**res, "brain": proc.analyzer.preference_model.get_brain_summary()}


@app.post("/api/ml/download-backbone")
def ml_download_backbone():
    state.touch_activity()
    res = get_visual_embedder().download_backbone()
    return res


# --- Speaker Face Lock (#3 & ML-5) ------------------------------------------

@app.get("/api/speaker")
def get_speaker_status():
    proc = state.processor
    if not proc or not getattr(proc, "analyzer", None):
        return {"locked": False, "profiles": []}
    return proc.analyzer.speaker_id.status()


@app.post("/api/speaker/lock")
def lock_speaker_endpoint(payload: Dict[str, Any] = Body(default={})):
    state.touch_activity()
    proc = state.processor
    frame, analysis, _ = state.snapshot()
    if not proc or frame is None or analysis is None or not analysis.faces:
        raise HTTPException(status_code=409, detail="No detected face on live feed to lock onto")

    pid = payload.get("person_id")
    save_name = str(payload.get("name", "") or "").strip()
    target_face = None
    if pid is not None:
        target_face = next((f for f in analysis.faces if f.person_id == int(pid)), None)
    if target_face is None:
        selected = [f for f in analysis.faces if f.is_selected]
        pool = selected or analysis.faces
        target_face = max(pool, key=lambda f: f.bbox[2] * f.bbox[3])

    spk = proc.analyzer.speaker_id
    res = spk.lock_from_frame(frame, target_face.bbox, save_as_name=save_name, download_sface=True)
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("error", "Could not lock speaker"))
    if proc._refine_analyzer is not None:
        proc._refine_analyzer.speaker_id = spk
    return res


@app.post("/api/speaker/activate")
def activate_speaker_profile(payload: Dict[str, Any] = Body(...)):
    state.touch_activity()
    proc = state.processor
    if not proc or not getattr(proc, "analyzer", None):
        raise HTTPException(status_code=409, detail="Stream not running")
    prof_id = str(payload.get("profile_id", "") or "").strip()
    res = proc.analyzer.speaker_id.activate_profile(prof_id)
    if not res.get("ok"):
        raise HTTPException(status_code=404, detail=res.get("error", "Speaker profile not found"))
    return res


@app.post("/api/speaker/unlock")
def unlock_speaker_endpoint():
    state.touch_activity()
    proc = state.processor
    if not proc or not getattr(proc, "analyzer", None):
        raise HTTPException(status_code=409, detail="Stream not running")
    res = proc.analyzer.speaker_id.unlock()
    proc.analyzer.speaker_id.save_profiles()
    return res


@app.delete("/api/speaker/profiles/{profile_id}")
def delete_speaker_profile(profile_id: str):
    state.touch_activity()
    proc = state.processor
    if not proc or not getattr(proc, "analyzer", None):
        raise HTTPException(status_code=409, detail="Stream not running")
    removed = proc.analyzer.speaker_id.delete_profile(profile_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Speaker profile not found")
    return {"ok": True, **proc.analyzer.speaker_id.status()}


# --- Stage Safe-Zone & Obstruction Polygons (#9) ----------------------------

@app.get("/api/stage-zones")
def get_stage_zones():
    sz = (getattr(state.processor.analyzer, "stage_zones", None) if (state.processor and getattr(state.processor, "analyzer", None)) else None) or state.stage_zones
    return {"ok": True, **sz.to_dict()}


@app.post("/api/stage-zones")
def set_stage_zones(payload: Dict[str, Any] = Body(...)):
    state.touch_activity()
    new_zones = StageZoneConfig.from_dict(payload)
    state.stage_zones = new_zones
    state._stage_zones_rev += 1
    if state.processor and getattr(state.processor, "analyzer", None):
        state.processor.analyzer.stage_zones = new_zones
        if state.processor._refine_analyzer is not None:
            state.processor._refine_analyzer.stage_zones = new_zones
    settings_store.save_ui_settings({"stage_zones": new_zones.to_dict()})
    return {"ok": True, **new_zones.to_dict()}


# --- Time-Machine Replay Mining (#4b), Burst Scrub (#7) & AI Finals (#4a) ---

@app.post("/api/buffer/mine")
def mine_buffer_endpoint(payload: Dict[str, Any] = Body(default={})):
    state.touch_activity()
    proc = state.processor
    if not proc:
        raise HTTPException(status_code=409, detail="Stream not running")
    if not state.output_confirmed:
        raise HTTPException(
            status_code=409,
            detail=f"Confirm the output folder first. Currently {state.config.output_dir}.",
        )
    seconds = max(5.0, min(float(getattr(state.config, "dvr_max_seconds", 180.0)), float(payload.get("seconds", 60.0) or 60.0)))
    max_picks = max(1, min(12, int(payload.get("max_picks", 4) or 4)))
    saved = proc.mine_dvr_window(lookback_seconds=seconds, max_picks=max_picks)
    return {
        "ok": True,
        "saved_count": len(saved),
        "filenames": [os.path.basename(c.filepath) for c in saved],
        "captures": _captures_payload(),
    }


@app.get("/api/captures/{filename}/burst")
def get_capture_burst(filename: str):
    proc = state.processor
    if not proc:
        raise HTTPException(status_code=409, detail="Stream not running")
    _find_capture(filename)
    return proc.get_burst_frames(filename)


@app.get("/api/captures/{filename}/burst/{idx}")
def get_capture_burst_frame(filename: str, idx: int):
    proc = state.processor
    if not proc:
        raise HTTPException(status_code=409, detail="Stream not running")
    jb = proc.get_burst_frame_jpeg(filename, idx)
    if jb is None:
        raise HTTPException(status_code=404, detail="Burst frame not cached")
    return Response(content=jb, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@app.post("/api/captures/{filename}/burst/{idx}/select")
def select_capture_burst_frame(filename: str, idx: int):
    state.touch_activity()
    proc = state.processor
    if not proc:
        raise HTTPException(status_code=409, detail="Stream not running")
    cap = proc.select_burst_frame(filename, idx)
    if cap is None:
        raise HTTPException(status_code=404, detail="Burst frame not available")
    return {
        "ok": True,
        "filename": os.path.basename(cap.filepath),
        "score": round(float(cap.overall_score), 3),
        "captures": _captures_payload(),
    }


@app.post("/api/captures/ai-finals")
def run_ai_finals_endpoint():
    state.touch_activity()
    proc = state.processor
    if not proc:
        raise HTTPException(status_code=409, detail="Stream not running")
    judge = getattr(proc, "judge", None)
    if judge is not None and judge.enabled and not judge.available:
        judge.wake()
    res = proc.run_ai_finals(include_scan=True)
    if not res.get("ok"):
        raise HTTPException(status_code=409, detail=res.get("error", "AI Finals failed"))
    return {**res, "captures": _captures_payload()}


# --- Studio Transparent Cutouts (#1) & Broadcast Polish (#11) ---------------

@app.get("/api/captures/{filename}/cutout")
def get_capture_cutout(filename: str, crop: int = 0, download: int = 1):
    state.touch_activity()
    cap = _find_capture(filename)
    if not cap.filepath or not os.path.isfile(cap.filepath):
        raise HTTPException(status_code=404, detail="Capture file missing on disk")

    stem, _ = os.path.splitext(cap.filepath)
    dest_path = f"{stem}_subject_cutout.png" if crop else cutout_path_for(cap.filepath)
    if not os.path.isfile(dest_path):
        img = cv2.imread(cap.filepath, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=500, detail="Could not read capture image")
        written = save_cutout_png(img, cap.analysis, dest_path, crop_to_subject=bool(crop), polish_first=True)
        if not written or not os.path.isfile(written):
            raise HTTPException(status_code=500, detail="Failed to generate transparent cutout")
        if not crop:
            cap.cutout_filepath = written
            if state.processor:
                state.processor._persist_manifest()

    if download:
        return FileResponse(dest_path, media_type="image/png", filename=os.path.basename(dest_path))
    return FileResponse(dest_path, media_type="image/png")


@app.get("/api/captures/{filename}/polish")
def get_capture_polished(filename: str, upscale: int = 1, download: int = 1):
    state.touch_activity()
    cap = _find_capture(filename)
    if not cap.filepath or not os.path.isfile(cap.filepath):
        raise HTTPException(status_code=404, detail="Capture file missing on disk")

    stem, _ = os.path.splitext(cap.filepath)
    suffix = "_4k.png" if upscale else "_polished.png"
    dest_path = f"{stem}{suffix}"
    if not os.path.isfile(dest_path):
        img = cv2.imread(cap.filepath, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=500, detail="Could not read capture image")
        polished = get_postprocess_engine().polish_broadcast_frame(
            img, cap.analysis, deblock=True, neutralize_led=True, upscale_2x=bool(upscale)
        )
        ok = cv2.imwrite(dest_path, polished, [int(cv2.IMWRITE_PNG_COMPRESSION), 2])
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to write polished image")

    if download:
        return FileResponse(dest_path, media_type="image/png", filename=os.path.basename(dest_path))
    return FileResponse(dest_path, media_type="image/png")


@app.post("/api/postprocess/warm-models")
def warm_neural_models():
    state.touch_activity()
    mat = get_postprocess_engine().ensure_matting_model()
    vis = get_visual_embedder().download_backbone()
    spk = (
        state.processor.analyzer.speaker_id.ensure_sface_ready()
        if (state.processor and getattr(state.processor, "analyzer", None))
        else {"ok": False}
    )
    return {"ok": True, "matting": mat, "visual_backbone": vis, "sface": spk}


# --- Audio Pause Gating & Whisper Sermon Notes (#8) -------------------------

@app.get("/api/audio/status")
def get_audio_status():
    proc = state.processor
    aud = getattr(proc, "audio_monitor", None) if proc else None
    tel = aud.telemetry().__dict__ if aud is not None else {"active": False, "rms_db": -60.0}
    return {
        "ok": True,
        "audio_gating_enabled": bool(getattr(state.config, "audio_gating_enabled", False)),
        "whisper_available": SermonNotesGenerator.whisper_available(),
        "whisper_model": state.config.whisper_model_size,
        "telemetry": tel,
        "has_latest_notes": state.latest_sermon_notes is not None,
    }


@app.post("/api/sermon-notes/generate")
def generate_sermon_notes(payload: Dict[str, Any] = Body(default={})):
    state.touch_activity()
    src_mode = str(payload.get("source", "live") or "live").strip().lower()
    raw_transcript = str(payload.get("transcript", "") or "").strip()
    media_path: Optional[str] = None
    audio_mon = None

    if src_mode == "video":
        raw_v = str(payload.get("video_path", "") or state.scan_video_path or "").strip()
        media_path = str(_resolve_video_path(raw_v))
    elif src_mode == "live" and not raw_transcript:
        proc = state.processor
        audio_mon = getattr(proc, "audio_monitor", None) if proc else None
        # If live source is actually a local video file and no PCM chunks buffered yet, transcribe file directly
        if (audio_mon is None or audio_mon.telemetry().buffered_minutes <= 0.01) and state.config.stream_source:
            if not _is_network_or_device_source(state.config.stream_source) and os.path.isfile(state.config.stream_source):
                media_path = state.config.stream_source

    if not raw_transcript and media_path is None:
        if audio_mon is None or audio_mon.telemetry().buffered_minutes <= 0.01:
            raise HTTPException(
                status_code=400,
                detail="No buffered live audio yet (enable Audio Pause Gating during stream, select a video file, or paste a transcript).",
            )
        if not SermonNotesGenerator.whisper_available():
            raise HTTPException(
                status_code=400,
                detail="faster-whisper is not installed in this Python environment (paste a transcript or run via the Docker image).",
            )
    elif not raw_transcript and media_path is not None and not SermonNotesGenerator.whisper_available():
        raise HTTPException(
            status_code=400,
            detail="faster-whisper is not installed in this Python environment (paste a transcript or run via the Docker image).",
        )

    res = state.sermon_notes_gen.generate_and_save(
        output_dir=state.config.output_dir if state.output_confirmed else "",
        media_path=media_path,
        audio_monitor=audio_mon,
        raw_transcript=raw_transcript,
        whisper_model=state.config.whisper_model_size,
        ollama_url=state.config.ai_judge_ollama_url,
        ollama_model=state.config.ai_judge_ollama_model,
        num_thread=state.config.ai_judge_ollama_num_thread,
    )
    state.latest_sermon_notes = res
    return {"ok": True, **res}


@app.get("/api/sermon-notes/latest")
def get_latest_sermon_notes():
    notes = state.latest_sermon_notes or state.sermon_notes_gen.last_notes
    return {"ok": True, "notes": notes}


# --- NDI Discovery (#12) & Sony SRG-A40 PTZ / AI Auto-Framing (#10) ---------

@app.get("/api/ndi/discover")
def api_discover_ndi(timeout_ms: int = 1200):
    state.touch_activity()
    return discover_ndi_sources(timeout_ms=max(200, min(4000, int(timeout_ms))))


def _ptz_controller() -> SonyPTZController:
    cam = camera_store.load()
    if cam and cam.kind == "rtsp" and cam.host:
        return SonyPTZController.from_camera_settings(cam)
    src = (state.config.stream_source or "").strip()
    if src.lower().startswith(("rtsp://", "rtsps://")):
        parts = urllib.parse.urlsplit(src)
        if parts.hostname:
            return SonyPTZController(
                host=parts.hostname,
                username=urllib.parse.unquote(parts.username or ""),
                password=urllib.parse.unquote(parts.password or ""),
                http_port=80,
            )
    return SonyPTZController()


@app.get("/api/ptz/status")
def ptz_status():
    return _ptz_controller().status()


@app.post("/api/ptz/move")
def ptz_move(payload: Dict[str, Any] = Body(...)):
    state.touch_activity()
    ctrl = _ptz_controller()
    if not ctrl.configured:
        raise HTTPException(status_code=409, detail="Connect an RTSP IP camera first to use Sony PTZ controls")
    direction = str(payload.get("direction", "stop") or "stop")
    speed = int(payload.get("speed", 8) or 8)
    dur = payload.get("duration_ms", 260)
    dur_ms = int(dur) if dur is not None else None
    res = ctrl.move(direction=direction, speed=speed, duration_ms=dur_ms)
    if not res.get("ok"):
        raise HTTPException(status_code=502, detail=res.get("error", "PTZ move failed"))
    return res


@app.post("/api/ptz/zoom")
def ptz_zoom(payload: Dict[str, Any] = Body(...)):
    state.touch_activity()
    ctrl = _ptz_controller()
    if not ctrl.configured:
        raise HTTPException(status_code=409, detail="Connect an RTSP IP camera first to use Sony PTZ controls")
    direction = str(payload.get("direction", "stop") or "stop")
    speed = int(payload.get("speed", 4) or 4)
    dur = payload.get("duration_ms", 240)
    dur_ms = int(dur) if dur is not None else None
    res = ctrl.zoom(direction=direction, speed=speed, duration_ms=dur_ms)
    if not res.get("ok"):
        raise HTTPException(status_code=502, detail=res.get("error", "PTZ zoom failed"))
    return res


@app.post("/api/ptz/home")
def ptz_home():
    state.touch_activity()
    ctrl = _ptz_controller()
    if not ctrl.configured:
        raise HTTPException(status_code=409, detail="Connect an RTSP IP camera first to use Sony PTZ controls")
    res = ctrl.home()
    if not res.get("ok"):
        raise HTTPException(status_code=502, detail=res.get("error", "PTZ home failed"))
    return res


@app.post("/api/ptz/preset")
def ptz_preset(payload: Dict[str, Any] = Body(...)):
    state.touch_activity()
    ctrl = _ptz_controller()
    if not ctrl.configured:
        raise HTTPException(status_code=409, detail="Connect an RTSP IP camera first to use Sony PTZ controls")
    preset_id = int(payload.get("preset_id", 1) or 1)
    action = str(payload.get("action", "recall") or "recall").strip().lower()
    if action == "save":
        res = ctrl.save_preset(preset_id, name=str(payload.get("name", "") or ""))
    else:
        res = ctrl.recall_preset(preset_id)
    if not res.get("ok"):
        raise HTTPException(status_code=502, detail=res.get("error", "PTZ preset command failed"))
    return res


@app.post("/api/ptz/auto-framing")
def ptz_auto_framing(payload: Dict[str, Any] = Body(...)):
    state.touch_activity()
    ctrl = _ptz_controller()
    if not ctrl.configured:
        raise HTTPException(status_code=409, detail="Connect an RTSP IP camera first to use Sony AI Auto-Framing")
    enabled = bool(payload.get("enabled", True))
    res = ctrl.set_auto_framing(enabled)
    if not res.get("ok"):
        raise HTTPException(status_code=502, detail=res.get("error", "Sony AI Auto-Framing command failed"))
    return res


@app.get("/api/state")
def get_state():
    state.touch_activity()
    payload = _telemetry()
    payload["captures"] = _captures_payload()
    return payload


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    state.touch_activity()
    return FileResponse(str(STATIC_DIR / "index.html"),
                        headers={"Cache-Control": "no-cache"})


class _RevalidatingStatics(StaticFiles):
    """Serves the UI with `Cache-Control: no-cache` so browser tabs revalidate after updates."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", _RevalidatingStatics(directory=str(STATIC_DIR)), name="static")


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Thumbnail Grabber web server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: all interfaces)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--source", help="Override video source (RTSP URL, webcam index, or file)")
    args = parser.parse_args()

    if args.source:
        state.config.stream_source = args.source

    print(f"\n  Thumbnail Grabber web UI:  http://localhost:{args.port}")
    print("  On your LAN, use this machine's IP address with the same port.\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
