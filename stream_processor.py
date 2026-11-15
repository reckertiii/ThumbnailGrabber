import cv2
import time
import os
import socket
import types
import urllib.parse
import threading
import contextlib
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List, Tuple
import numpy as np

from config import AppConfig, redact_credentials
from analyzer import PersonQualityAnalyzer, AnalysisResult
from ai_judge import AIJudge, JudgeResult
from postprocess import get_postprocess_engine, cutout_path_for, save_cutout_png
from audio_analyzer import LiveAudioMonitor
import settings_store

@dataclass
class CapturedScreenshot:
    filepath: str
    timestamp_str: str
    overall_score: float
    analysis: AnalysisResult
    crop_filepath: Optional[str] = None   # Subject-tight crop written alongside, if enabled
    captured_at: float = 0.0              # time.time() at capture, for spacing decisions
    heuristic_score: float = 0.0          # Analyzer score before the judge weighed in
    judge_score: Optional[float] = None   # Vision-model score, if the judge ran
    judge_summary: str = ""
    judge_issues: List[str] = field(default_factory=list)
    is_prop_moment: bool = False          # True when speaker is holding/presenting a stage prop
    from_video_scan: bool = False         # True when produced by the offline video scanner
    is_pinned: bool = False               # True when flagged/locked by operator to protect from auto-deletion
    cutout_filepath: Optional[str] = None # Transparent RGBA *_cutout.png written alongside, if enabled
    finals_badge: str = ""                # Tournament Finals badge (e.g. "#1 Overall", "Best Smile")
    mood_bucket: str = ""                 # Expression mood bucket (warm_smile, passionate, calm_portrait, prop)


def subject_crop_rect(analysis: AnalysisResult, frame_w: int, frame_h: int,
                      padding: float = 0.08) -> Optional[Tuple[int, int, int, int]]:
    """Rectangle tightly framing the subject, for the companion crop.

    Uses the pose-derived subject box when available, falling back to an estimate from
    the face box. Returns None if neither is usable.
    """
    faces = analysis.faces or []
    primary = next((f for f in faces if getattr(f, "is_selected", True)), faces[0]) if faces else None
    rect = getattr(primary, "subject_bbox", None) if primary else None

    if not rect or rect[2] <= 0 or rect[3] <= 0:
        bbox = (getattr(primary, "bbox", None) if primary else None) or analysis.face_bbox
        if not bbox or bbox[2] <= 0 or bbox[3] <= 0:
            return None
        fx, fy, fw, fh = bbox
        # Roughly head-to-thigh from the face box alone.
        rect = (fx - fw, fy - int(fh * 0.4), fw * 3, int(fh * 8.0))

    x, y, w, h = rect
    pad_x, pad_y = int(w * padding), int(h * padding)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(frame_w, x + w + pad_x)
    y1 = min(frame_h, y + h + pad_y)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def write_capture(frame: np.ndarray, analysis: AnalysisResult, out_dir: str,
                  base_name: str, config: AppConfig) -> Tuple[str, Optional[str]]:
    """Writes a capture (and optionally its subject crop) and returns both paths.

    Shared by live capture and the offline video scanner so both produce identical
    output, in the same format, with the same naming.
    """
    os.makedirs(out_dir, exist_ok=True)
    fmt = (config.capture_format or "png").lower().lstrip(".")
    if fmt not in ("png", "jpg", "jpeg"):
        fmt = "png"
    ext = ".png" if fmt == "png" else ".jpg"
    params = (
        [int(cv2.IMWRITE_PNG_COMPRESSION), 1]
        if fmt == "png"
        else [int(cv2.IMWRITE_JPEG_QUALITY), int(config.jpeg_quality)]
    )

    polish_on = bool(getattr(config, "broadcast_polish_enabled", False))
    upscale_on = bool(getattr(config, "upscale_4k_enabled", False))
    base_frame = frame
    if polish_on:
        base_frame = get_postprocess_engine().polish_broadcast_frame(
            frame, analysis, deblock=True, neutralize_led=True, upscale_2x=False
        )

    out_frame = (
        get_postprocess_engine().polish_broadcast_frame(
            base_frame, analysis, deblock=False, neutralize_led=False, upscale_2x=True
        )
        if upscale_on
        else base_frame
    )

    filepath = os.path.join(out_dir, base_name + ext)
    cv2.imwrite(filepath, out_frame, params)

    crop_path = None
    if config.save_subject_crop:
        h, w = base_frame.shape[:2]
        rect = subject_crop_rect(analysis, w, h, config.subject_crop_padding)
        if rect:
            cx, cy, cw, ch = rect
            crop = base_frame[cy:cy + ch, cx:cx + cw]
            if crop.size > 0:
                if upscale_on:
                    crop = get_postprocess_engine().polish_broadcast_frame(
                        crop, None, deblock=False, neutralize_led=False, upscale_2x=True
                    )
                crop_path = os.path.join(out_dir, base_name + "_subject" + ext)
                cv2.imwrite(crop_path, crop, params)

    if getattr(config, "auto_save_cutout", False):
        save_cutout_png(
            base_frame,
            analysis,
            cutout_path_for(filepath),
            crop_to_subject=False,
            polish_first=False,
        )

    return filepath, crop_path

class StreamProcessor:
    # Connection status constants
    STATUS_DISCONNECTED = "disconnected"
    STATUS_UNCONFIGURED = "unconfigured"
    STATUS_CONNECTING = "connecting"
    STATUS_CONNECTED = "connected"
    STATUS_RECONNECTING = "reconnecting"
    STATUS_STANDBY = "standby"
    STATUS_FAILED = "failed"

    def __init__(self, config: AppConfig, on_frame_callback: Optional[Callable[[np.ndarray, AnalysisResult], None]] = None, on_capture_callback: Optional[Callable[[CapturedScreenshot], None]] = None):
        self.config = config
        self.analyzer = PersonQualityAnalyzer(
            config.weights, config.thresholds,
            ema_alpha=config.ema_alpha,
            stability_window=config.score_stability_window,
            stability_threshold=config.score_stability_threshold,
            blink_suppression_ms=config.blink_suppression_ms,
            max_people_target=config.max_people_target,
            pose_hand_interval=getattr(config, "pose_hand_interval", 1),
            use_ml_preference=getattr(config, "use_ml_preference", True),
            ml_weight=getattr(config, "ml_weight", 0.35),
            min_consecutive_qualified=getattr(config, "min_consecutive_qualified", 2),
            require_exact_people_count=getattr(config, "require_exact_people_count", False),
            capture_prop_moments=getattr(config, "capture_prop_moments", True),
        )
        self.on_frame_callback = on_frame_callback
        self.on_capture_callback = on_capture_callback

        self.running = False
        self._stop_event = threading.Event()
        self.grabber_thread: Optional[threading.Thread] = None
        self.analysis_thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None

        # Thread-safe atomic frame store
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_analysis: Optional[AnalysisResult] = None
        self.frame_lock = threading.Lock()
        self.captures_lock = threading.RLock()
        
        # Rolling raw-frame buffer. The grabber fills it at camera rate; a chosen moment
        # is then re-examined across it at full frame rate, rather than being stuck with
        # whichever of the analysis_fps samples happened to land.
        dvr_frames = int(max(0.0, config.dvr_buffer_seconds) * max(1, config.dvr_buffer_fps))
        self._dvr: deque = deque(maxlen=dvr_frames) if dvr_frames > 0 else deque(maxlen=0)
        self._dvr_lock = threading.Lock()
        # Refinement scores buffered frames, which would otherwise churn the live
        # analyzer's EMA, stability window and consecutive-qualified counter. A separate
        # instance keeps that state clean, and judges each frame standalone - which is
        # exactly what picking the best single frame of a moment calls for. Built on
        # first use so it costs nothing when the buffer is off.
        self._refine_analyzer: Optional[PersonQualityAnalyzer] = None
        if dvr_frames > 0:
            est_mb = (dvr_frames * 0.36) if getattr(config, "dvr_jpeg_compression", True) else (dvr_frames * 6.22)
            print(f"[StreamProcessor] Frame buffer: {config.dvr_buffer_seconds:.0f}s "
                  f"({dvr_frames} frames, ~{est_mb:.0f} MB at 1080p"
                  f"{', Q95 JPEG' if getattr(config, 'dvr_jpeg_compression', True) else ''})")

        # Lifecycle & persistence flags
        self.paused: bool = False
        self.idle_standby: bool = False
        self.persist_manifests: bool = False

        # Buffer for peak detection & Single Winner tracking
        self.candidate_buffer = deque()
        self.saved_captures: List[CapturedScreenshot] = []
        self.last_capture_time = 0.0
        self.auto_capture_enabled = True
        
        # Score trend tracking for local peak detection
        self._score_history: deque = deque(maxlen=20)

        # Second-opinion review. The analysis thread must never wait on the network,
        # so candidates are handed to a single worker that reviews and then saves.
        self.judge = AIJudge(config)
        self._review_lock = threading.Lock()
        self._review_in_flight: bool = False
        # Candidates found while a review is running are queued, not discarded. A review
        # on a local model can take a minute; dropping everything for that minute meant
        # good moments passed by unseen. The queue is bounded and evicts oldest first,
        # so a reviewer that falls permanently behind loses the stalest work, not the
        # freshest.
        self._review_queue: deque = deque(maxlen=max(1, int(getattr(config, "review_queue_size", 10))))
        self._review_worker: Optional[threading.Thread] = None

        # Audio-aware pause/laughter monitor & RAM speech ring buffer (#8)
        self.audio_monitor = LiveAudioMonitor()

        # Bounded LRU cache of +-15 surrounding high-res frames per saved capture (#7)
        self._burst_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._burst_lock = threading.Lock()
        self._last_refined_timestamp: Optional[float] = None
        
        # Connection status tracking
        self.connection_status: str = self.STATUS_DISCONNECTED
        self.reconnect_attempt: int = 0
        self.on_status_callback: Optional[Callable[[str, int], None]] = None
        
        # Stats
        self.frames_grabbed = 0        # Frames read off the stream (drives the preview)
        self.frames_processed = 0      # Frames actually scored
        self.captures_count = 0
        self.current_fps = 0.0

    def start(self):
        """Starts the video grabber and analysis background threads."""
        if self.running:
            return
        self._stop_event.clear()
        self.running = True
        self.connection_status = self.STATUS_CONNECTING

        if getattr(self.config, "audio_gating_enabled", False) and getattr(self, "audio_monitor", None) is not None:
            src = (self.config.stream_source or "").strip()
            if src:
                self.audio_monitor.start(src)
        
        # 1. Grabber Thread (Drains RTSP stream continuously to prevent buffer lag)
        self.grabber_thread = threading.Thread(target=self._grabber_loop, daemon=True)
        self.grabber_thread.start()
        
        # 2. Analysis Thread (Processes latest frame at target analysis FPS)
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()

    def stop(self):
        """Stops processing loops and releases camera resources."""
        # Release the judge's model first: with keep_alive=-1 a stopped app would
        # otherwise leave ~20GB pinned on the Ollama host indefinitely.
        judge = getattr(self, "judge", None)
        if judge is not None and judge.backend == "ollama" and not judge._expiry_handled:
            judge.sleep_now()
        if getattr(self, "audio_monitor", None) is not None:
            self.audio_monitor.stop()
        self.running = False
        self._stop_event.set()  # interrupts an in-progress reconnect backoff
        if self.grabber_thread and self.grabber_thread.is_alive():
            self.grabber_thread.join(timeout=5.0)
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=5.0)
        # Release only after the grabber thread has exited, so it can't read
        # from a capture we just freed.
        if self.cap:
            self.cap.release()
            self.cap = None
        self.connection_status = self.STATUS_DISCONNECTED

    def update_source(self, new_source: str):
        """Switches video input source (e.g. RTSP URL, Webcam, File).

        Always restarts. This used to start again only if it had been running,
        which left the processor dead exactly when it mattered: a source that
        fails to open stops the grabber loop and clears `running`, so pointing it
        at a camera that works found `running` already false and never started.
        The result was a correctly saved camera, a "connected" message, and a feed
        that stayed empty forever.

        There is no separate stop control for the stream -- capture is toggled
        independently via auto_capture_enabled -- so the stream running whenever a
        source is set is the behaviour the UI already implies.
        """
        # stop() releases the judge's model, which is right for a shutdown but
        # not for a two-second reconnect -- the judge used to come back
        # permanently asleep. Carry the remaining window across the restart.
        judge = getattr(self, "judge", None)
        left = None
        if judge is not None and judge.enabled and not judge.expired:
            left = judge.suspend()

        self.stop()
        self.config.stream_source = new_source
        self.reconnect_attempt = 0
        self.start()

        if judge is not None and left is not None:
            judge.resume(left)

    def _set_status(self, status: str):
        """Update connection status and notify GUI."""
        self.connection_status = status
        if self.on_status_callback:
            try:
                self.on_status_callback(status, self.reconnect_attempt)
            except Exception:
                pass

    def enable_persistence(self, adopt_orphans: bool = True) -> None:
        """Enables folder-scoped capture/pin manifest persistence in TG_DATA_DIR."""
        self.persist_manifests = True
        with getattr(self, "captures_lock", contextlib.nullcontext()):
            loaded = settings_store.load_captures_manifest(
                self.config.output_dir, adopt_orphans=adopt_orphans
            )
            self.saved_captures = loaded
            self.captures_count = len([c for c in loaded if not getattr(c, "from_video_scan", False)])
            if loaded:
                self._persist_manifest()

    def _persist_manifest(self) -> None:
        if getattr(self, "persist_manifests", False) and getattr(self, "config", None) and self.config.output_dir:
            settings_store.save_captures_manifest(self.config.output_dir, self.saved_captures)

    def switch_output_dir(self, new_dir: str, adopt_orphans: bool = True) -> List[CapturedScreenshot]:
        """Switches output directory cleanly, saving the previous folder's manifest and loading the new one."""
        with getattr(self, "captures_lock", contextlib.nullcontext()):
            if getattr(self, "persist_manifests", False) and self.config.output_dir:
                settings_store.save_captures_manifest(self.config.output_dir, self.saved_captures)
            self.config.output_dir = new_dir
            if hasattr(self, "candidate_buffer") and self.candidate_buffer is not None:
                self.candidate_buffer.clear()
            if hasattr(self, "_review_lock") and hasattr(self, "_review_queue"):
                with self._review_lock:
                    self._review_queue.clear()
            if getattr(self, "persist_manifests", False):
                loaded = settings_store.load_captures_manifest(new_dir, adopt_orphans=adopt_orphans)
                self.saved_captures = loaded
                self.captures_count = len([c for c in loaded if not getattr(c, "from_video_scan", False)])
                if loaded:
                    self._persist_manifest()
            else:
                self.saved_captures = []
                self.captures_count = 0
            return list(self.saved_captures)

    def add_scan_captures(self, new_captures: List[CapturedScreenshot], target_dir: Optional[str] = None) -> None:
        """Records offline video-scan captures into the appropriate folder's gallery/manifest."""
        if not new_captures:
            return
        dest = os.path.abspath(target_dir or self.config.output_dir)
        curr = os.path.abspath(self.config.output_dir) if self.config.output_dir else ""
        same_dir = os.path.normcase(dest) == os.path.normcase(curr)
        if same_dir:
            with getattr(self, "captures_lock", contextlib.nullcontext()):
                existing_paths = {os.path.normcase(c.filepath) for c in self.saved_captures}
                for cap in new_captures:
                    if os.path.normcase(cap.filepath) not in existing_paths:
                        self.saved_captures.append(cap)
                        existing_paths.add(os.path.normcase(cap.filepath))
                self._persist_manifest()
        elif getattr(self, "persist_manifests", False):
            other = settings_store.load_captures_manifest(dest, adopt_orphans=False)
            existing_paths = {os.path.normcase(c.filepath) for c in other}
            for cap in new_captures:
                if os.path.normcase(cap.filepath) not in existing_paths:
                    other.append(cap)
                    existing_paths.add(os.path.normcase(cap.filepath))
            settings_store.save_captures_manifest(dest, other)

    def set_idle_standby(self, enabled: bool) -> bool:
        """Toggles Zero-CPU Idle Standby (pauses CV inference & frees DVR RAM when idle)."""
        new_val = bool(enabled)
        prev = getattr(self, "idle_standby", False)
        self.idle_standby = new_val
        if new_val and not prev:
            with self._dvr_lock:
                self._dvr.clear()
            if hasattr(self, "candidate_buffer") and self.candidate_buffer is not None:
                self.candidate_buffer.clear()
            self.current_fps = 0.0
            print("[StreamProcessor] Entered idle standby (0 viewers, capture off — CV paused, DVR released).")
        elif not new_val and prev:
            print("[StreamProcessor] Woke from idle standby.")
        return self.idle_standby

    @staticmethod
    def _rtsp_tcp_reachable(url: str, timeout: float = 1.5) -> bool:
        """Fast 1-socket check before invoking FFmpeg while a powered-off camera is in standby."""
        try:
            parsed = urllib.parse.urlsplit(url)
            host = parsed.hostname
            port = parsed.port or (322 if parsed.scheme.lower() == "rtsps" else 554)
            if not host:
                return True
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    @staticmethod
    def _dvr_entry_nbytes(entry: tuple) -> int:
        payload = entry[1]
        if isinstance(payload, (bytes, bytearray)):
            return len(payload)
        return int(getattr(payload, "nbytes", 0))

    @staticmethod
    def _decode_dvr_frame(entry: tuple) -> Optional[np.ndarray]:
        payload = entry[1]
        if isinstance(payload, np.ndarray):
            return payload
        if isinstance(payload, (bytes, bytearray)):
            arr = np.frombuffer(payload, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return None

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        source = self.config.stream_source.strip()
        if source.lower().startswith("ndi://"):
            from ndi_source import NDIVideoCapture
            cap = NDIVideoCapture(source)
            if not cap.isOpened():
                print(f"[StreamProcessor] Error: Failed to open NDI source: '{source}'")
                return None
            return cap  # type: ignore[return-value]
        
        # Low latency OpenCV FFMPEG environment settings
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000|reorder_queue_size;0"
        
        if source.isdigit():
            cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        else:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            
        if not cap.isOpened():
            print("[StreamProcessor] Error: Failed to open video source: "
                  f"'{redact_credentials(source)}'")
            return None
            
        # Minimal buffer size to ensure zero frame delay
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _wait_for_rtsp_standby(self, source: str) -> Optional[cv2.VideoCapture]:
        """Polls a powered-off RTSP/NDI camera lightly until it comes online or stop() is called."""
        poll_sec = max(1.0, float(getattr(self.config, "rtsp_standby_poll_sec", 15.0) or 15.0))
        self._set_status(self.STATUS_STANDBY)
        print(f"[StreamProcessor] Camera offline ({redact_credentials(source)}); "
              f"standing by and polling every {poll_sec:.0f}s until powered on.")
        # Only pre-probe TCP when _open_capture hasn't been replaced by a test stub and source is RTSP.
        is_ndi = source.lower().startswith("ndi://")
        use_tcp_probe = (not is_ndi) and isinstance(getattr(self, "_open_capture", None), types.MethodType)
        while not self._stop_event.is_set():
            if self._stop_event.wait(poll_sec):
                return None
            if use_tcp_probe and not self._rtsp_tcp_reachable(source, timeout=1.5):
                continue
            cap = self._open_capture()
            if cap is not None:
                print("[StreamProcessor] Camera online — connected from standby.")
                return cap
        return None

    def _grabber_loop(self):
        """Dedicated high-speed loop reading camera frames as fast as they arrive."""
        source = (self.config.stream_source or "").strip()
        if not source:
            self._set_status(self.STATUS_UNCONFIGURED)
            self._stop_event.wait()
            return

        is_rtsp = source.lower().startswith(("rtsp://", "rtsps://", "ndi://"))
        self.cap = self._open_capture()
        if not self.cap:
            if is_rtsp:
                self.cap = self._wait_for_rtsp_standby(source)
                if not self.cap:
                    return
            else:
                self._set_status(self.STATUS_FAILED)
                self.running = False
                self._stop_event.set()  # also stops the analysis loop
                return

        self._set_status(self.STATUS_CONNECTED)
        self.reconnect_attempt = 0
        idle_tick = 0

        while not self._stop_event.is_set():
            # Zero-CPU Idle Standby: drain live camera socket with cheap grab() without
            # full H.264/BGR decode on most frames, decoding 1 frame ~every 1.5s.
            if getattr(self, "idle_standby", False) and (is_rtsp or source.isdigit()) and hasattr(self.cap, "grab"):
                idle_tick += 1
                if idle_tick % 45 != 0:
                    ok_grab = self.cap.grab()
                    if ok_grab:
                        time.sleep(0.015)
                        continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                if is_rtsp:
                    # === RTSP Reconnection with Exponential Backoff ===
                    self.reconnect_attempt += 1
                    
                    if self.reconnect_attempt > self.config.rtsp_max_retries:
                        if self.cap:
                            self.cap.release()
                            self.cap = None
                        self.cap = self._wait_for_rtsp_standby(source)
                        if not self.cap:
                            return
                        self._set_status(self.STATUS_CONNECTED)
                        self.reconnect_attempt = 0
                        continue
                    
                    # Exponential backoff: 1s, 2s, 4s, 8s, ... capped at rtsp_backoff_max
                    backoff = min(
                        self.config.rtsp_backoff_base * (2 ** (self.reconnect_attempt - 1)),
                        self.config.rtsp_backoff_max
                    )
                    print(f"[StreamProcessor] RTSP connection dropped, reconnecting... "
                          f"(attempt {self.reconnect_attempt}/{self.config.rtsp_max_retries}, "
                          f"backoff {backoff:.1f}s)")
                    self._set_status(self.STATUS_RECONNECTING)

                    # Interruptible sleep — stop() can cut this short.
                    if self._stop_event.wait(backoff):
                        return

                    if self.cap:
                        self.cap.release()
                    self.cap = self._open_capture()

                    if self.cap is None:
                        continue

                    if self.cap:
                        self._set_status(self.STATUS_CONNECTED)
                        self.reconnect_attempt = 0
                        print("[StreamProcessor] RTSP reconnected successfully.")
                    continue
                else:
                    # Video file loop
                    if self.cap and hasattr(self.cap, "get") and self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break

            # Store latest frame
            now_grab = time.time()
            with self.frame_lock:
                self.latest_frame = frame
                self.frames_grabbed += 1
            if self._dvr.maxlen and not getattr(self, "idle_standby", False):
                dvr_item = (now_grab, frame)
                if getattr(self.config, "dvr_jpeg_compression", True):
                    ok_enc, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    if ok_enc:
                        dvr_item = (now_grab, enc.tobytes(), frame.shape)
                with self._dvr_lock:
                    # deque(maxlen) evicts the oldest, so memory stays bounded without
                    # any trimming pass on the hot path.
                    self._dvr.append(dvr_item)

            # Yield slightly to prevent 100% CPU lock in grabber thread
            time.sleep(0.001)

    def _analysis_loop(self):
        """Processes latest frame, evaluates face quality, and updates GUI."""
        fps_target = self.config.analysis_fps
        frame_delay = 1.0 / max(1, fps_target)
        
        last_time = time.time()
        frame_counter = 0

        while not self._stop_event.is_set():
            if getattr(self, "paused", False) or getattr(self, "idle_standby", False):
                time.sleep(0.25)
                continue
            start_time = time.time()
            
            # Fetch latest frame atomically
            frame = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()
            
            if frame is None:
                time.sleep(0.01)
                continue

            self.frames_processed += 1
            frame_counter += 1
            
            # FPS Calculation
            elapsed_fps = time.time() - last_time
            if elapsed_fps >= 1.0:
                self.current_fps = frame_counter / elapsed_fps
                frame_counter = 0
                last_time = time.time()

            # Analyze frame quality
            analysis = self.analyzer.analyze_frame(frame)
            self.latest_analysis = analysis

            # Track score history for trend analysis
            if analysis.face_detected:
                self._score_history.append(analysis.overall_score)
            
            # Rolling peak candidate selection. Frames whose *only* problem is blur are
            # admitted too: the buffer very likely holds a sharp version of the same
            # moment a frame or two away, and refinement will find it. Without this a
            # fast gesture or a camera pan disqualified the moment outright and it was
            # gone, which is the "good shot passed by" case. Anything refinement can't
            # rescue is rejected again after the fact.
            now = time.time()
            faces = analysis.faces or []
            primary = next((f for f in faces if getattr(f, "is_selected", True)), faces[0]) if faces else None
            reasons = list(getattr(primary, "disqualified_reasons", []) or []) if primary else []
            # Blur also drags the score under the threshold, so a smeared frame lists
            # both reasons. Treat it as blur-only when every reason is blur or the
            # score it caused - but require at least one actual blur reason, or every
            # low-scoring frame would be admitted.
            has_blur = any("blurry" in r for r in reasons)
            blur_only = has_blur and all(
                ("blurry" in r) or r.startswith("score ") for r in reasons)
            if analysis.face_detected and (analysis.is_qualified or blur_only):
                self.candidate_buffer.append((now, frame.copy(), analysis))
            
            # Clean old entries outside burst window
            while self.candidate_buffer and (now - self.candidate_buffer[0][0]) > self.config.burst_window_seconds:
                self.candidate_buffer.popleft()

            # Check if auto-capture trigger conditions are met
            if self.auto_capture_enabled and (now - self.last_capture_time) >= self.config.cooldown_seconds:
                self._evaluate_and_trigger_capture(now)

            # Invoke callback for GUI frame preview
            if self.on_frame_callback:
                self.on_frame_callback(frame, analysis)

            # Throttle loop to target analysis FPS
            process_duration = time.time() - start_time
            sleep_needed = frame_delay - process_duration
            if sleep_needed > 0:
                time.sleep(sleep_needed)
            else:
                # Analysis is already slower than the requested rate. Yield anyway so
                # this thread can't starve the grabber and the web server on a NAS.
                time.sleep(0.002)

    def _is_at_local_peak(self) -> bool:
        """Check if the score trend is at or near a local peak or high plateau."""
        if len(self._score_history) < 3:
            return True  # Not enough history, allow capture
        
        scores = list(self._score_history)
        recent = scores[-5:]
        
        max_s = max(recent)
        min_s = min(recent)
        curr_s = recent[-1]
        
        # If score is steady/plateau at a high level (range <= 0.08 and curr >= min_overall_score)
        if (max_s - min_s) <= 0.08 and curr_s >= self.config.thresholds.min_overall_score:
            return True

        peak_idx = int(np.argmax(recent))
        # Allow peak at position 1, 2, 3, or 4
        return peak_idx >= 1

    def buffer_stats(self) -> Dict[str, Any]:
        """Measured memory held by the rolling buffer.

        Reports what the frames actually occupy rather than a guess from the configured
        resolution, so the number stays honest if the camera changes format - and so a
        NAS deployment can be tuned against the real figure.
        """
        with self._dvr_lock:
            frames = len(self._dvr)
            held = sum(self._dvr_entry_nbytes(entry) for entry in self._dvr)
            newest_entry = self._dvr[-1] if frames else None
            span = (self._dvr[-1][0] - self._dvr[0][0]) if frames > 1 else 0.0
        capacity = self._dvr.maxlen or 0
        per_frame = self._dvr_entry_nbytes(newest_entry) if newest_entry is not None else 0
        # What the camera is really delivering. dvr_buffer_fps only sizes the buffer;
        # if the camera runs faster, the same frame count covers proportionally less
        # time, so the configured seconds quietly become fewer. Reporting both makes
        # that visible instead of leaving a buffer that is half the requested depth.
        measured_fps = round(frames / span, 1) if span > 0.2 else 0.0
        return {
            "measured_fps": measured_fps,
            "seconds_short": bool(measured_fps and span + 0.3 < float(self.config.dvr_buffer_seconds)),
            "enabled": capacity > 0,
            "compressed": bool(getattr(self.config, "dvr_jpeg_compression", True)),
            "seconds": float(self.config.dvr_buffer_seconds),
            "fps": int(self.config.dvr_buffer_fps),
            "frames_held": frames,
            "frames_capacity": capacity,
            "span_seconds": round(span, 2),
            "bytes_held": held,
            "bytes_capacity": per_frame * capacity,
            "frame_bytes": per_frame,
            "mb_held": round(held / (1024 * 1024), 1),
            "mb_capacity": round(per_frame * capacity / (1024 * 1024), 1),
        }

    def set_buffer(self, seconds: float, fps: Optional[int] = None) -> Dict[str, Any]:
        """Resizes the rolling buffer in place, keeping the most recent frames.

        A deque's maxlen is fixed at construction, so this builds a new one. The newest
        frames are the useful ones, so a shrink drops from the front.
        """
        max_sec = max(60.0, float(getattr(self.config, "dvr_max_seconds", 180.0) or 180.0))
        seconds = max(0.0, min(max_sec, float(seconds)))
        if fps is not None:
            self.config.dvr_buffer_fps = max(1, min(60, int(fps)))
        self.config.dvr_buffer_seconds = seconds
        new_max = int(seconds * self.config.dvr_buffer_fps)

        with self._dvr_lock:
            kept = list(self._dvr)[-new_max:] if new_max > 0 else []
            self._dvr = deque(kept, maxlen=new_max) if new_max > 0 else deque(maxlen=0)
        stats = self.buffer_stats()
        print(f"[StreamProcessor] Frame buffer -> {seconds:.1f}s @ {self.config.dvr_buffer_fps}fps "
              f"({new_max} frames, ~{stats['mb_capacity']:.0f} MB at this resolution)")
        return stats

    def _get_or_create_refine_analyzer(self) -> PersonQualityAnalyzer:
        """Returns a standalone analyzer synced with live speaker lock, stage zones, and ML model."""
        if getattr(self, "_refine_analyzer", None) is None:
            self._refine_analyzer = PersonQualityAnalyzer(
                self.config.weights, self.config.thresholds,
                max_people_target=getattr(self.analyzer, "max_people_target", 1),
                selected_person_ids=list(getattr(self.analyzer, "selected_person_ids", [])),
                pose_hand_interval=1,
                require_temporal_stability=False,
                use_ml_preference=getattr(self.config, "use_ml_preference", True),
                ml_weight=getattr(self.config, "ml_weight", 0.35),
                require_exact_people_count=getattr(self.analyzer, "require_exact_people_count", False),
                capture_prop_moments=getattr(self.analyzer, "capture_prop_moments", True),
                fast_reject=True,
                speaker_id=getattr(self.analyzer, "speaker_id", None),
                stage_zones=getattr(self.analyzer, "stage_zones", None),
            )
            self._refine_analyzer.preference_model = getattr(self.analyzer, "preference_model", None)
        else:
            self._refine_analyzer.max_people_target = getattr(self.analyzer, "max_people_target", 1)
            self._refine_analyzer.selected_person_ids = set(getattr(self.analyzer, "selected_person_ids", []))
            self._refine_analyzer.require_exact_people_count = getattr(self.analyzer, "require_exact_people_count", False)
            self._refine_analyzer.capture_prop_moments = getattr(self.analyzer, "capture_prop_moments", True)
            self._refine_analyzer.preference_model = getattr(self.analyzer, "preference_model", None)
            self._refine_analyzer.speaker_id = getattr(self.analyzer, "speaker_id", None)
            self._refine_analyzer.stage_zones = getattr(self.analyzer, "stage_zones", None)
        return self._refine_analyzer

    def _refine_from_buffer(self, chosen_at: float, chosen_frame: np.ndarray,
                           chosen_analysis: AnalysisResult):
        """Looks for a crisper version of the chosen moment in the rolling buffer.

        Returns (frame, analysis) - the originals if nothing better is found.
        """
        self._last_refined_timestamp = chosen_at
        window = float(getattr(self.config, "dvr_refine_window_sec", 0.0) or 0.0)
        top_n = int(getattr(self.config, "dvr_refine_candidates", 0) or 0)
        if window <= 0 or top_n <= 0 or not self._dvr.maxlen:
            return chosen_frame, chosen_analysis

        with self._dvr_lock:
            nearby_entries = [entry for entry in self._dvr if abs(entry[0] - chosen_at) <= window]
        if len(nearby_entries) < 2:
            return chosen_frame, chosen_analysis

        faces = chosen_analysis.faces or []
        primary_face = next((f for f in faces if getattr(f, "is_selected", True)), faces[0]) if faces else None
        rect = getattr(primary_face, "subject_bbox", None) if primary_face else None
        if not rect or rect[2] <= 0 or rect[3] <= 0:
            rect = (getattr(primary_face, "bbox", None) if primary_face else None) or chosen_analysis.face_bbox
        if not rect:
            return chosen_frame, chosen_analysis
        rx, ry, rw, rh = rect
        height = self.analyzer.thresholds.sharpness_face_height

        scored = []
        for entry in nearby_entries:
            t = entry[0]
            f = self._decode_dvr_frame(entry)
            if f is None:
                continue
            roi = f[max(0, ry):ry + rh, max(0, rx):rx + rw]
            if roi.size == 0:
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            norm = self.analyzer._resize_to_height(gray, height)
            scored.append((float(cv2.Laplacian(norm, cv2.CV_64F).var()), t, f))
        if not scored:
            return chosen_frame, chosen_analysis

        scored.sort(key=lambda item: item[0], reverse=True)
        ref_analyzer = self._get_or_create_refine_analyzer()

        started_qualified = chosen_analysis.is_qualified
        best_frame, best_analysis = chosen_frame, chosen_analysis
        best_score = chosen_analysis.overall_score if started_qualified else -1.0
        best_time = chosen_at

        for _sharp, cand_t, f in scored[:top_n]:
            try:
                a = ref_analyzer.analyze_frame(f)
            except Exception:
                continue
            if a.face_detected and a.is_qualified and a.overall_score > best_score:
                best_score, best_frame, best_analysis, best_time = a.overall_score, f, a, cand_t

        self._last_refined_timestamp = best_time
        if best_analysis is not chosen_analysis:
            print(f"[Refine] {chosen_analysis.overall_score:.0%} -> {best_score:.0%} "
                  f"(sharpness {chosen_analysis.laplacian_variance:.0f} -> "
                  f"{best_analysis.laplacian_variance:.0f}) from {len(nearby_entries)} buffered frames")
        return best_frame, best_analysis

    def mine_dvr_window(
        self,
        lookback_seconds: float = 60.0,
        max_picks: int = 3,
        min_spacing_sec: float = 3.0,
        window_seconds: Optional[float] = None,
    ) -> List[CapturedScreenshot]:
        """Retroactively mines the RAM Time-Machine buffer for great recent moments (#4b)."""
        if window_seconds is not None:
            lookback_seconds = float(window_seconds)
        if not getattr(self, "_dvr", None):
            return []
        with self._dvr_lock:
            entries = list(self._dvr)
        if not entries:
            return []

        newest_ts = entries[-1][0]
        cutoff = newest_ts - max(1.0, float(lookback_seconds))
        window_entries = [e for e in entries if e[0] >= cutoff]
        if not window_entries:
            return []

        # Sample at ~4 fps across the window so 60s (=1800 frames) runs in <1.5 seconds,
        # then refine the top winners at full 30fps!
        step = max(1, len(window_entries) // max(12, int(lookback_seconds * 4)))
        sampled = window_entries[::step]
        ref_analyzer = self._get_or_create_refine_analyzer()

        candidates: List[Tuple[float, float, np.ndarray, AnalysisResult]] = []
        for entry in sampled:
            ts = entry[0]
            frame = self._decode_dvr_frame(entry)
            if frame is None:
                continue
            try:
                analysis = ref_analyzer.analyze_frame(frame)
            except Exception:
                continue
            if analysis.face_detected and analysis.overall_score > 0.20:
                eff = analysis.overall_score if analysis.is_qualified else (analysis.overall_score * 0.65)
                candidates.append((eff, ts, frame, analysis))

        if not candidates:
            return []

        candidates.sort(key=lambda item: item[0], reverse=True)
        picked: List[Tuple[float, np.ndarray, AnalysisResult]] = []
        for _eff, ts, frame, analysis in candidates:
            if len(picked) >= max(1, int(max_picks)):
                break
            if any(abs(ts - p[0]) < min_spacing_sec for p in picked):
                continue
            r_frame, r_analysis = self._refine_from_buffer(ts, frame, analysis)
            r_ts = getattr(self, "_last_refined_timestamp", None) or ts
            picked.append((r_ts, r_frame, r_analysis))

        saved: List[CapturedScreenshot] = []
        for r_ts, r_frame, r_analysis in picked:
            cap_item = self.save_screenshot(r_frame, r_analysis, captured_at=r_ts)
            saved.append(cap_item)
        return saved

    # --- +-15 Frame Micro-Scrub Burst Filmstrip (#7) -------------------------

    @staticmethod
    def _entry_to_jpeg_bytes(entry: tuple) -> Optional[bytes]:
        payload = entry[1]
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, bytearray):
            return bytes(payload)
        if isinstance(payload, np.ndarray):
            ok, enc = cv2.imencode(".jpg", payload, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            return enc.tobytes() if ok else None
        return None

    def cache_burst_around(
        self,
        filename: str,
        center_ts: float,
        fallback_frame: Optional[np.ndarray] = None,
        extra_frames: Optional[List[Tuple[float, bytes]]] = None,
    ) -> None:
        """Caches +-N high-res frames around `center_ts` in RAM for instant micro-scrubbing."""
        lock = getattr(self, "_burst_lock", None)
        cache = getattr(self, "_burst_cache", None)
        if lock is None or cache is None or not filename:
            return
        half_n = max(3, int(getattr(self.config, "burst_filmstrip_frames", 15) or 15))
        frames_list: List[Tuple[float, bytes]] = []

        if extra_frames:
            frames_list = list(extra_frames)
        elif getattr(self, "_dvr", None):
            with self._dvr_lock:
                dvr_snapshot = list(self._dvr)
            if dvr_snapshot:
                best_idx = min(range(len(dvr_snapshot)), key=lambda i: abs(dvr_snapshot[i][0] - center_ts))
                if abs(dvr_snapshot[best_idx][0] - center_ts) <= 2.5:
                    lo = max(0, best_idx - half_n)
                    hi = min(len(dvr_snapshot), best_idx + half_n + 1)
                    for e in dvr_snapshot[lo:hi]:
                        jb = self._entry_to_jpeg_bytes(e)
                        if jb:
                            frames_list.append((float(e[0]), jb))

        if not frames_list and fallback_frame is not None:
            ok, enc = cv2.imencode(".jpg", fallback_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if ok:
                frames_list.append((float(center_ts), enc.tobytes()))

        if not frames_list:
            return

        sel_idx = min(range(len(frames_list)), key=lambda i: abs(frames_list[i][0] - center_ts))
        key = os.path.basename(filename)
        with lock:
            cache[key] = {
                "center_ts": float(center_ts),
                "selected_idx": int(sel_idx),
                "frames": frames_list,
            }
            cache.move_to_end(key)
            while len(cache) > 35:
                cache.popitem(last=False)

    def get_burst_frames(self, filename: str) -> Dict[str, Any]:
        key = os.path.basename(filename)
        lock = getattr(self, "_burst_lock", None)
        cache = getattr(self, "_burst_cache", None)
        if lock is None or cache is None:
            return {"available": False, "frames": []}
        with lock:
            item = cache.get(key)
            if not item:
                return {"available": False, "frames": []}
            sel = int(item.get("selected_idx", 0))
            base_ts = item["frames"][sel][0] if item["frames"] else item["center_ts"]
            meta = [
                {
                    "index": idx,
                    "offset_ms": int(round((ts - base_ts) * 1000.0)),
                    "is_selected": (idx == sel),
                }
                for idx, (ts, _jb) in enumerate(item["frames"])
            ]
            return {"available": len(meta) > 0, "selected_idx": sel, "count": len(meta), "frames": meta}

    def get_burst_frame_jpeg(self, filename: str, idx: int) -> Optional[bytes]:
        key = os.path.basename(filename)
        lock = getattr(self, "_burst_lock", None)
        cache = getattr(self, "_burst_cache", None)
        if lock is None or cache is None:
            return None
        with lock:
            item = cache.get(key)
            if not item:
                return None
            frames = item.get("frames") or []
            if 0 <= idx < len(frames):
                return frames[idx][1]
        return None

    def select_burst_frame(self, filename: str, idx: int) -> Optional[CapturedScreenshot]:
        """Swaps a saved capture to frame `idx` from its cached burst filmstrip (#7 + ML-2)."""
        jb = self.get_burst_frame_jpeg(filename, idx)
        if jb is None:
            return None
        arr = np.frombuffer(jb, dtype=np.uint8)
        new_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if new_frame is None:
            return None

        key = os.path.basename(filename)
        with getattr(self, "captures_lock", contextlib.nullcontext()):
            target_cap = next(
                (c for c in self.saved_captures if os.path.basename(c.filepath) == key),
                None,
            )
            if target_cap is None:
                return None

            ref_analyzer = self._get_or_create_refine_analyzer()
            new_analysis = ref_analyzer.analyze_frame(new_frame)

            out_dir = os.path.dirname(target_cap.filepath)
            stem, _ext = os.path.splitext(os.path.basename(target_cap.filepath))
            new_fp, new_crop = write_capture(new_frame, new_analysis, out_dir, stem, self.config)

            # Update existing cutout PNG if one already existed even when auto_save_cutout is off
            cpath = cutout_path_for(new_fp)
            if target_cap.cutout_filepath and not os.path.isfile(cpath):
                save_cutout_png(new_frame, new_analysis, cpath, crop_to_subject=False)

            target_cap.filepath = new_fp
            target_cap.crop_filepath = new_crop
            target_cap.cutout_filepath = cpath if os.path.isfile(cpath) else None
            target_cap.analysis = new_analysis
            if new_analysis.face_detected:
                target_cap.heuristic_score = new_analysis.overall_score
                target_cap.overall_score = new_analysis.overall_score
                target_cap.mood_bucket = getattr(new_analysis, "mood_bucket", "") or target_cap.mood_bucket
                target_cap.is_prop_moment = bool(getattr(new_analysis, "is_prop_moment", False))

            lock = getattr(self, "_burst_lock", None)
            cache = getattr(self, "_burst_cache", None)
            if lock is not None and cache is not None:
                with lock:
                    if key in cache:
                        cache[key]["selected_idx"] = int(idx)

            # Implicit positive ML feedback: operator hand-picked this exact sub-second frame!
            pm = getattr(getattr(self, "analyzer", None), "preference_model", None)
            if pm is not None and hasattr(pm, "record_implicit_feedback") and new_analysis.face_detected:
                h, w = new_frame.shape[:2]
                rect = subject_crop_rect(new_analysis, w, h, self.config.subject_crop_padding)
                crop = new_frame if rect is None else new_frame[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
                pm.record_implicit_feedback(target_cap, peer_captures=list(self.saved_captures), action="burst_swap", crop_bgr=crop)

            self._persist_manifest()
            return target_cap

    @staticmethod
    def _item_mood(c: CapturedScreenshot) -> str:
        mb = getattr(c, "mood_bucket", "") or getattr(getattr(c, "analysis", None), "mood_bucket", "") or ""
        return str(mb)

    def _would_improve_retained(self, score: float, now: float,
                                is_prop_moment: bool = False,
                                is_smiling: bool = False,
                                mood_bucket: str = "") -> bool:
        """Is this capture worth writing, given what is already kept?"""
        if not self.config.single_winner_mode:
            return True
        n = max(1, self.config.max_retained_captures)
        with getattr(self, "captures_lock", contextlib.nullcontext()):
            kept = [
                c for c in self.saved_captures
                if not getattr(c, "from_video_scan", False) and not getattr(c, "is_pinned", False)
            ]
        if len(kept) < n:
            return True

        prop_enabled = bool(getattr(self.config, "capture_prop_moments", True)) and n > 1
        prop_cap = min(max(0, int(getattr(self.config, "max_prop_captures", 2))), max(0, n - 1)) if prop_enabled else 0
        kept_props = [c for c in kept if getattr(c, "is_prop_moment", False)]
        non_props = [c for c in kept if not getattr(c, "is_prop_moment", False)]
        has_kept_smile = any(getattr(getattr(c, "analysis", None), "is_smiling", False) for c in kept)
        mood_enabled = bool(getattr(self.config, "mood_buckets_enabled", True)) and n >= 4
        kept_moods = {self._item_mood(c) for c in kept if self._item_mood(c)}

        spacing = max(0.0, float(getattr(self.config, "min_capture_spacing_sec", 0.0)))
        if spacing > 0:
            nearby = [c for c in kept if abs(now - c.captured_at) < spacing]
            if nearby:
                if is_prop_moment and prop_enabled:
                    nearby_props = [c for c in nearby if getattr(c, "is_prop_moment", False)]
                    if nearby_props:
                        return score > max(c.overall_score for c in nearby_props)
                    if len(kept_props) < prop_cap:
                        return True
                if is_smiling and n >= 3 and not has_kept_smile and score >= self.config.thresholds.min_overall_score:
                    return True
                if mood_enabled and mood_bucket and mood_bucket not in kept_moods and score >= self.config.thresholds.min_overall_score:
                    return True
                return score > max(c.overall_score for c in nearby)

        if is_prop_moment and prop_enabled:
            if len(kept_props) < prop_cap:
                return True
            if kept_props:
                return score > min(c.overall_score for c in kept_props) or score > min(c.overall_score for c in kept)

        if is_smiling and n >= 3 and not has_kept_smile and score >= self.config.thresholds.min_overall_score:
            return True

        # Mood Buckets (#6): admit an unrepresented expression mood (e.g. passionate or calm_portrait)
        if mood_enabled and mood_bucket and mood_bucket not in kept_moods and score >= self.config.thresholds.min_overall_score:
            return True

        if not is_prop_moment and prop_enabled and kept_props and len(kept_props) <= prop_cap and non_props:
            return score > min(c.overall_score for c in non_props)

        return score > min(c.overall_score for c in kept)

    def _evaluate_and_trigger_capture(self, now: float):
        if not self.candidate_buffer:
            return

        # Pick candidate frame with highest overall score among strictly qualified frames
        best_candidate = max(self.candidate_buffer, key=lambda item: item[2].overall_score)
        cand_time, best_frame, best_analysis = best_candidate

        # === Score Trend: require local peak ===
        if not self._is_at_local_peak():
            return
        
        # === Scrub the buffer for a crisper version of this same moment ===
        best_frame, best_analysis = self._refine_from_buffer(cand_time, best_frame, best_analysis)
        refined_ts = getattr(self, "_last_refined_timestamp", None) or cand_time

        if not best_analysis.is_qualified:
            return

        # Audio-aware pause/laughter tie-breaker (#8)
        aud_mon = getattr(self, "audio_monitor", None)
        if getattr(self.config, "audio_gating_enabled", False) and aud_mon is not None:
            tel = aud_mon.telemetry()
            if tel.active and tel.capture_bonus > 0 and not best_analysis.is_awkward_expression:
                best_analysis.overall_score = round(min(1.0, best_analysis.overall_score + tel.capture_bonus * 0.5), 3)

        is_prop = bool(getattr(best_analysis, "is_prop_moment", False))
        is_smile = bool(getattr(best_analysis, "is_smiling", False))
        mood_b = str(getattr(best_analysis, "mood_bucket", "") or "")

        # === Would this actually improve what we're keeping? ===
        if not self._would_improve_retained(
            best_analysis.overall_score, refined_ts,
            is_prop_moment=is_prop, is_smiling=is_smile, mood_bucket=mood_b
        ):
            return

        # All checks passed — either save now, or send it for review first.
        if self.judge.available:
            if not self._start_review(best_frame, best_analysis, now, cand_time=refined_ts):
                return   # a review is already running; the next candidate will come along
        else:
            self.save_screenshot(best_frame, best_analysis, captured_at=refined_ts)
        # Reports the terms that actually drive the pick. Plain "deg" rather than the
        # degree sign, which mojibakes on a default Windows console codepage.
        flags = []
        if is_prop:
            flags.append("prop-moment")
        if best_analysis.is_smiling:
            flags.append("smile")
        if best_analysis.hands_on_torso:
            flags.append("hands-on-body")
        if best_analysis.is_subject_clipped:
            flags.append("clipped")
        print(f"[Capture] Score={best_analysis.overall_score:.0%} "
              f"Sharp={best_analysis.sharpness_score:.2f}({int(best_analysis.laplacian_variance)}) "
              f"Cutout={best_analysis.cutout_score:.2f} "
              f"Subject={best_analysis.subject_height_ratio:.0%}h "
              f"Bg={best_analysis.background_score:.2f} "
              f"Yaw={best_analysis.yaw_deg:.0f}deg "
              f"Pitch={best_analysis.pitch_deg:.0f}deg "
              f"Exposure={best_analysis.exposure_score:.2f}"
              + (f" [{', '.join(flags)}]" if flags else ""))
        self.last_capture_time = now
        self.candidate_buffer.clear()

    def _start_review(self, frame: np.ndarray, analysis: AnalysisResult, now: float,
                      cand_time: Optional[float] = None) -> bool:
        """Queues a candidate for judge review and makes sure the worker is running."""
        actual_time = cand_time if cand_time is not None else now
        with self._review_lock:
            dropped = len(self._review_queue) == self._review_queue.maxlen
            self._review_queue.append((frame.copy(), analysis, actual_time))
            if dropped:
                print(f"[AIJudge] Review queue full ({self._review_queue.maxlen}); "
                      f"dropped the oldest pending candidate.")
            if self._review_worker is None or not self._review_worker.is_alive():
                self._review_worker = threading.Thread(target=self._review_worker_loop, daemon=True)
                self._review_worker.start()
        # Cooldown starts at hand-off rather than at save, so the analysis loop paces
        # candidates by the capture cooldown instead of by how slow the reviewer is.
        self.last_capture_time = now
        self.candidate_buffer.clear()
        return True

    def _review_worker_loop(self) -> None:
        """Drains the review queue one candidate at a time."""
        while not self._stop_event.is_set():
            with self._review_lock:
                if not self._review_queue:
                    self._review_in_flight = False
                    return
                item = self._review_queue.popleft()
                if len(item) == 3:
                    frame, analysis, cand_time = item
                else:
                    frame, analysis = item
                    cand_time = time.time()
                self._review_in_flight = True
            try:
                self._review_and_save(frame, analysis, cand_time=cand_time)
            except Exception as e:  # noqa: BLE001 - one bad candidate must not end the worker
                print(f"[AIJudge] Review worker error: {e}")

    def _review_and_save(self, frame: np.ndarray, analysis: AnalysisResult,
                         cand_time: Optional[float] = None) -> None:
        try:
            actual_time = cand_time if cand_time is not None else time.time()
            h, w = frame.shape[:2]
            rect = subject_crop_rect(analysis, w, h, self.config.subject_crop_padding)
            crop = frame if rect is None else frame[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            notes = list(getattr(analysis, "disqualified_reasons", []) or [])
            if getattr(analysis, "is_prop_moment", False):
                notes.append("speaker appears to be holding or presenting a stage prop")
            elif analysis.hands_on_torso:
                notes.append("hands over the torso")
            if analysis.is_subject_clipped:
                notes.append("subject runs off the frame edge")

            pm = getattr(getattr(self, "analyzer", None), "preference_model", None)
            fb_prompt = pm.get_judge_feedback_prompt() if pm is not None and hasattr(pm, "get_judge_feedback_prompt") else ""
            if fb_prompt:
                try:
                    result = self.judge.judge(crop, analysis.overall_score, notes, user_feedback_prompt=fb_prompt)
                except TypeError:
                    result = self.judge.judge(crop, analysis.overall_score, notes)
            else:
                result = self.judge.judge(crop, analysis.overall_score, notes)

            if result.verdict is not None:
                # Teacher -> Student ML Distillation (ML-4)
                if getattr(self.config, "ai_judge_distill_to_ml", True) and pm is not None and hasattr(pm, "record_judge_distillation"):
                    try:
                        pm.record_judge_distillation(analysis, result.verdict, crop_bgr=crop)
                    except Exception:
                        pass

                if getattr(result.verdict, "holding_prop", False):
                    if getattr(self.config, "capture_prop_moments", True):
                        analysis.is_prop_moment = True
                        analysis.mood_bucket = "prop"
                else:
                    issue_text = " ".join(result.verdict.issues or []).lower()
                    if any(kw in issue_text for kw in (
                        "head bowed", "head slightly bowed", "bowed down", "looking down",
                        "gaze not engaged", "not engaged", "eyes closed", "microphone",
                    )):
                        analysis.is_prop_moment = False

            # "Unusable" is a hard no only once there is something better to keep. Early
            # in a service it is a heavy penalty instead, so a strict judge can never
            # leave the gallery empty - the same reason the heuristics don't hard-gate.
            if result.verdict is not None and not result.verdict.usable:
                n = max(1, self.config.max_retained_captures)
                with getattr(self, "captures_lock", contextlib.nullcontext()):
                    live_kept = [c for c in self.saved_captures if not getattr(c, "from_video_scan", False)]
                if not self.config.single_winner_mode or len(live_kept) >= n:
                    print(f"[AIJudge] Rejected ({analysis.overall_score:.0%} -> judge "
                          f"{result.verdict.score:.0%}): {result.verdict.summary}")
                    return
                result.blended_score = round(min(result.blended_score, analysis.overall_score * 0.5), 3)

            is_prop = bool(getattr(analysis, "is_prop_moment", False))
            is_smile = bool(getattr(analysis, "is_smiling", False))
            mood_b = str(getattr(analysis, "mood_bucket", "") or "")

            # Re-check against the kept set with the blended score: the review took a
            # few seconds, and a better capture may have landed meanwhile.
            if not self._would_improve_retained(
                result.blended_score, actual_time,
                is_prop_moment=is_prop, is_smiling=is_smile, mood_bucket=mood_b
            ):
                return

            self.save_screenshot(frame, analysis, review=result, captured_at=actual_time)
        except Exception as e:  # noqa: BLE001 - a review failure must not kill the worker
            print(f"[AIJudge] Review failed for one candidate: {e}")

    def remove_capture(self, filepath: str) -> bool:
        """Removes a capture from `saved_captures` and deletes its files from disk."""
        target_name = os.path.basename(filepath)
        removed = False
        with getattr(self, "captures_lock", contextlib.nullcontext()):
            keep: List[CapturedScreenshot] = []
            for c in self.saved_captures:
                if c.filepath == filepath or os.path.basename(c.filepath) == target_name:
                    removed = True
                    extra_cutout = cutout_path_for(c.filepath) if c.filepath else None
                    for path in (c.filepath, c.crop_filepath, getattr(c, "cutout_filepath", None), extra_cutout):
                        if path and os.path.exists(path):
                            try:
                                os.remove(path)
                            except Exception as e:
                                print(f"[StreamProcessor] Error deleting rejected capture {path}: {e}")
                else:
                    keep.append(c)
            if removed:
                self.saved_captures = keep
                self.captures_count = len([c for c in keep if not getattr(c, "from_video_scan", False)])
                self._persist_manifest()
        return removed

    def set_capture_pinned(self, filepath: str, pinned: Optional[bool] = None) -> Optional[bool]:
        """Pins or unpins a capture so it is protected from auto-deletion when pruning to Top N."""
        target_name = os.path.basename(filepath)
        new_state: Optional[bool] = None
        with getattr(self, "captures_lock", contextlib.nullcontext()):
            for c in self.saved_captures:
                if c.filepath == filepath or os.path.basename(c.filepath) == target_name:
                    new_state = (not getattr(c, "is_pinned", False)) if pinned is None else bool(pinned)
                    c.is_pinned = new_state
                    break
            if new_state is False and self.config.single_winner_mode:
                self.prune_to_best_n(self.config.max_retained_captures)
            elif new_state is not None:
                self._persist_manifest()
        return new_state

    def clear_captures(self, mode: str = "unpinned") -> Dict[str, int]:
        """Clears captures from memory and deletes their files from the output folder."""
        mode = (mode or "unpinned").strip().lower()
        deleted_count = 0
        with getattr(self, "captures_lock", contextlib.nullcontext()):
            keep: List[CapturedScreenshot] = []
            to_delete: List[CapturedScreenshot] = []
            for c in self.saved_captures:
                pinned = bool(getattr(c, "is_pinned", False))
                if mode == "unpinned":
                    if pinned:
                        keep.append(c)
                    else:
                        to_delete.append(c)
                elif mode == "pinned":
                    if pinned:
                        to_delete.append(c)
                    else:
                        keep.append(c)
                else:  # "all"
                    to_delete.append(c)

            for c in to_delete:
                deleted_any = False
                stem, ext = os.path.splitext(c.filepath)
                legacy_crop = f"{stem}_subject{ext}" if stem and ext else None
                legacy_cutout = f"{stem}_cutout.png" if stem else None
                for path in (c.filepath, c.crop_filepath, legacy_crop, getattr(c, "cutout_filepath", None), legacy_cutout):
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                            deleted_any = True
                        except Exception as e:
                            print(f"[StreamProcessor] Error deleting capture {path}: {e}")
                if deleted_any or c.filepath:
                    deleted_count += 1

            # Also clean up any leftover unpinned best_shot_* files in output_dir when resetting
            # for the next service ("unpinned" or "all"), while strictly protecting kept files.
            if mode in ("unpinned", "all") and getattr(self, "config", None) and self.config.output_dir:
                out_dir = self.config.output_dir
                if os.path.isdir(out_dir):
                    protected_names = set()
                    for c in keep:
                        if c.filepath:
                            protected_names.add(os.path.basename(c.filepath))
                            s, e = os.path.splitext(os.path.basename(c.filepath))
                            protected_names.add(f"{s}_subject{e}")
                            protected_names.add(f"{s}_cutout.png")
                        if c.crop_filepath:
                            protected_names.add(os.path.basename(c.crop_filepath))
                        if getattr(c, "cutout_filepath", None):
                            protected_names.add(os.path.basename(c.cutout_filepath))
                    try:
                        for fname in os.listdir(out_dir):
                            low = fname.lower()
                            if fname in protected_names:
                                continue
                            if low.startswith("best_shot_") and low.endswith((".png", ".jpg", ".jpeg")):
                                fpath = os.path.join(out_dir, fname)
                                if os.path.isfile(fpath):
                                    try:
                                        os.remove(fpath)
                                    except Exception:
                                        pass
                    except Exception:
                        pass

            self.saved_captures = keep
            self.captures_count = len([c for c in keep if not getattr(c, "from_video_scan", False)])
            self._persist_manifest()

        if mode in ("unpinned", "all"):
            if hasattr(self, "candidate_buffer") and self.candidate_buffer is not None:
                self.candidate_buffer.clear()
            if hasattr(self, "_review_lock") and hasattr(self, "_review_queue"):
                with self._review_lock:
                    self._review_queue.clear()
            self.last_capture_time = 0.0

        pinned_remaining = len([c for c in self.saved_captures if getattr(c, "is_pinned", False)])
        return {
            "deleted_count": deleted_count,
            "remaining_count": len(self.saved_captures),
            "pinned_remaining": pinned_remaining,
        }

    def save_screenshot(self, frame: np.ndarray, analysis: AnalysisResult,
                        review: Optional[JudgeResult] = None,
                        captured_at: Optional[float] = None) -> CapturedScreenshot:
        """Saves screenshot frame to disk and prunes inferior shots if Single Winner Mode is enabled."""
        heuristic = analysis.overall_score
        final_score = heuristic
        verdict = review.verdict if review else None
        if verdict is not None:
            final_score = review.blended_score

        is_prop = bool(
            getattr(analysis, "is_prop_moment", False) or
            (verdict is not None and getattr(verdict, "holding_prop", False) and getattr(self.config, "capture_prop_moments", True))
        )
        analysis.is_prop_moment = is_prop
        if is_prop and not getattr(analysis, "mood_bucket", ""):
            analysis.mood_bucket = "prop"

        cap_ts = captured_at if captured_at is not None else time.time()
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(cap_ts))
        ms_str = f"{int((cap_ts % 1) * 1000):03d}"
        prop_tag = "_prop" if is_prop else ""
        base_name = f"best_shot_{timestamp_str}_{ms_str}{prop_tag}_score_{int(final_score * 100)}"

        filepath, crop_path = write_capture(
            frame, analysis, self.config.output_dir, base_name, self.config
        )
        cpath = cutout_path_for(filepath)
        cutout_fp = cpath if os.path.isfile(cpath) else None
        mood_b = str(getattr(analysis, "mood_bucket", "") or "")

        capture_item = CapturedScreenshot(
            filepath=filepath,
            timestamp_str=timestamp_str,
            overall_score=final_score,
            analysis=analysis,
            crop_filepath=crop_path,
            captured_at=cap_ts,
            heuristic_score=heuristic,
            judge_score=(float(verdict.score) if verdict is not None else None),
            judge_summary=(verdict.summary if verdict is not None else ""),
            judge_issues=(list(verdict.issues) if verdict is not None else []),
            is_prop_moment=is_prop,
            cutout_filepath=cutout_fp,
            mood_bucket=mood_b,
        )
        # Cache +-15 surrounding frames from RAM DVR for Micro-Scrub UI (#7)
        self.cache_burst_around(os.path.basename(filepath), cap_ts, fallback_frame=frame)

        if verdict is not None:
            print(f"[AIJudge] Kept ({heuristic:.0%} -> judge {verdict.score:.0%} = {final_score:.0%}, "
                  f"{review.latency_sec:.1f}s): {verdict.summary}")
        
        with getattr(self, "captures_lock", contextlib.nullcontext()):
            self.saved_captures.append(capture_item)
            
            # In Single Winner Mode, compare against existing captures and keep ONLY the top N best screenshots on disk
            if self.config.single_winner_mode:
                self.prune_to_best_n(self.config.max_retained_captures)
            else:
                self.captures_count += 1
                self._persist_manifest()
                if self.on_capture_callback:
                    self.on_capture_callback(capture_item)
            
        return capture_item

    def prune_to_best_n(self, n: int = 1):
        """Analyzes all captures, keeps the top N highest-scoring screenshots (plus any pinned shots), and deletes inferior candidate files from disk."""
        with getattr(self, "captures_lock", contextlib.nullcontext()):
            if not self.saved_captures:
                return

            scan_items = [c for c in self.saved_captures if getattr(c, "from_video_scan", False)]
            pinned_items = [
                c for c in self.saved_captures
                if not getattr(c, "from_video_scan", False) and getattr(c, "is_pinned", False)
            ]
            live_items = [
                c for c in self.saved_captures
                if not getattr(c, "from_video_scan", False) and not getattr(c, "is_pinned", False)
            ]
            if not live_items:
                kept_live = sorted(pinned_items, key=lambda item: item.overall_score, reverse=True)
                self.saved_captures = kept_live + scan_items
                self.captures_count = len(kept_live)
                self._persist_manifest()
                return

            ordered = sorted(live_items, key=lambda item: item.overall_score, reverse=True)
            spacing = max(0.0, float(getattr(self.config, "min_capture_spacing_sec", 0.0)))
            winners: List[CapturedScreenshot] = []

            # Reserve up to max_prop_captures slots for candid prop moments when enabled
            prop_enabled = bool(getattr(self.config, "capture_prop_moments", True)) and n > 1
            prop_cap = min(max(0, int(getattr(self.config, "max_prop_captures", 2))), max(0, n - 1)) if prop_enabled else 0
            if prop_cap > 0:
                prop_candidates = [c for c in ordered if getattr(c, "is_prop_moment", False)]
                for enforce in (True, False):
                    for c in prop_candidates:
                        if len(winners) >= prop_cap:
                            break
                        if c in winners:
                            continue
                        if enforce and spacing > 0 and any(
                            abs(c.captured_at - w.captured_at) < spacing for w in winners
                        ):
                            continue
                        winners.append(c)

            # Expression diversity: when retaining >=3 shots and no smile is in winners yet,
            # reserve 1 slot for the strongest qualified smiling shot so a warm smile is kept.
            min_qual_score = float(getattr(getattr(self.config, "thresholds", None), "min_overall_score", 0.55))
            if n >= 3 and len(winners) < n and not any(
                getattr(getattr(w, "analysis", None), "is_smiling", False) for w in winners
            ):
                smile_candidates = [
                    c for c in ordered
                    if c not in winners
                    and getattr(getattr(c, "analysis", None), "is_smiling", False)
                    and c.overall_score >= min_qual_score
                ]
                picked_smile = False
                for enforce in (True, False):
                    if picked_smile:
                        break
                    for c in smile_candidates:
                        if enforce and spacing > 0 and any(
                            abs(c.captured_at - w.captured_at) < spacing for w in winners
                        ):
                            continue
                        winners.append(c)
                        picked_smile = True
                        break

            # Mood Buckets (#6): when retaining >=4 shots, ensure passionate & calm_portrait
            # moods each get up to 1 slot if explicitly classified and qualified.
            if bool(getattr(self.config, "mood_buckets_enabled", True)) and n >= 4:
                for target_mood in ("warm_smile", "passionate", "calm_portrait"):
                    if len(winners) >= n - 1:  # always leave at least 1 pure highest-score slot
                        break
                    if any(self._item_mood(w) == target_mood for w in winners):
                        continue
                    mood_cands = [
                        c for c in ordered
                        if c not in winners
                        and self._item_mood(c) == target_mood
                        and c.overall_score >= min_qual_score
                    ]
                    picked_m = False
                    for enforce in (True, False):
                        if picked_m:
                            break
                        for c in mood_cands:
                            if enforce and spacing > 0 and any(
                                abs(c.captured_at - w.captured_at) < spacing for w in winners
                            ):
                                continue
                            winners.append(c)
                            picked_m = True
                            break

            for enforce in (True, False):
                for c in ordered:
                    if len(winners) >= n:
                        break
                    if c in winners:
                        continue
                    if enforce and spacing > 0 and any(
                        abs(c.captured_at - w.captured_at) < spacing for w in winners
                    ):
                        continue
                    winners.append(c)
            losers = [c for c in ordered if c not in winners]
            kept_live = sorted(pinned_items + winners, key=lambda item: item.overall_score, reverse=True)
            
            # Delete inferior screenshot files from disk (pinned shots are never in losers)
            for loser in losers:
                extra_cutout = cutout_path_for(loser.filepath) if loser.filepath else None
                for path in (loser.filepath, loser.crop_filepath, getattr(loser, "cutout_filepath", None), extra_cutout):
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                            print(f"[StreamProcessor] Pruned inferior shot: {path} (Score: {loser.overall_score})")
                        except Exception as e:
                            print(f"[StreamProcessor] Error deleting file {path}: {e}")

            self.saved_captures = kept_live + scan_items
            self.captures_count = len(kept_live)
            self._persist_manifest()

        # Notify callback with the champion winner
        cb = getattr(self, "on_capture_callback", None)
        if kept_live and cb:
            cb(kept_live[0])

    def run_ai_finals(self, include_scan: bool = True) -> Dict[str, Any]:
        """Runs a side-by-side contact-sheet Tournament Finals pass across top gallery shots (#4a)."""
        judge = getattr(self, "judge", None)
        if judge is None or not judge.enabled:
            return {"ok": False, "error": "AI Judge is disabled"}

        with getattr(self, "captures_lock", contextlib.nullcontext()):
            pool = [
                c for c in self.saved_captures
                if os.path.isfile(c.filepath) and (include_scan or not getattr(c, "from_video_scan", False))
            ]
        pool.sort(key=lambda c: c.overall_score, reverse=True)
        contenders = pool[:8]
        if len(contenders) < 2:
            return {"ok": False, "error": "Need at least 2 gallery shots to run Tournament Finals"}

        crops: List[np.ndarray] = []
        valid_contenders: List[CapturedScreenshot] = []
        for c in contenders:
            img_path = c.crop_filepath if (c.crop_filepath and os.path.isfile(c.crop_filepath)) else c.filepath
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                continue
            if img_path == c.filepath and c.analysis is not None:
                h, w = img.shape[:2]
                rect = subject_crop_rect(c.analysis, w, h, self.config.subject_crop_padding)
                if rect:
                    sub = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
                    if sub.size > 0:
                        img = sub
            crops.append(img)
            valid_contenders.append(c)

        if len(crops) < 2:
            return {"ok": False, "error": "Could not load contender images"}

        pm = getattr(getattr(self, "analyzer", None), "preference_model", None)
        fb_prompt = pm.get_judge_feedback_prompt() if pm is not None and hasattr(pm, "get_judge_feedback_prompt") else ""
        verdict = judge.judge_finals(crops, user_feedback_prompt=fb_prompt)
        if verdict is None:
            return {"ok": False, "error": "AI Judge did not return a Finals verdict"}

        badges_applied: Dict[str, str] = {}
        with getattr(self, "captures_lock", contextlib.nullcontext()):
            for c in valid_contenders:
                c.finals_badge = ""

            w_idx = int(verdict.winner_index) - 1
            if 0 <= w_idx < len(valid_contenders):
                champ = valid_contenders[w_idx]
                champ.finals_badge = "#1 Overall"
                max_other = max((c.overall_score for j, c in enumerate(valid_contenders) if j != w_idx), default=champ.overall_score)
                if champ.overall_score <= max_other:
                    champ.overall_score = round(min(1.0, max_other + 0.012), 3)
                badges_applied[os.path.basename(champ.filepath)] = champ.finals_badge

            s_idx = int(verdict.best_smile_index) - 1
            if 0 <= s_idx < len(valid_contenders):
                sm = valid_contenders[s_idx]
                sm.finals_badge = (f"{sm.finals_badge} · Best Smile" if sm.finals_badge else "Best Smile")
                badges_applied[os.path.basename(sm.filepath)] = sm.finals_badge

            g_idx = int(verdict.best_gesture_index) - 1
            if 0 <= g_idx < len(valid_contenders):
                gs = valid_contenders[g_idx]
                gs.finals_badge = (f"{gs.finals_badge} · Best Gesture" if gs.finals_badge else "Best Gesture")
                badges_applied[os.path.basename(gs.filepath)] = gs.finals_badge

            self.saved_captures.sort(key=lambda item: item.overall_score, reverse=True)
            self._persist_manifest()

        winner_file = (
            os.path.basename(valid_contenders[int(verdict.winner_index) - 1].filepath)
            if 1 <= int(verdict.winner_index) <= len(valid_contenders)
            else ""
        )
        return {
            "ok": True,
            "winner_index": int(verdict.winner_index),
            "winner_filename": winner_file,
            "reasoning": verdict.reasoning,
            "badges": badges_applied,
        }
