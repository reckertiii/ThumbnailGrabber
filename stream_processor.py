import cv2
import time
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List, Tuple
import numpy as np

from config import AppConfig, redact_credentials
from analyzer import PersonQualityAnalyzer, AnalysisResult
from ai_judge import AIJudge, JudgeResult

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


def subject_crop_rect(analysis: AnalysisResult, frame_w: int, frame_h: int,
                      padding: float = 0.08) -> Optional[Tuple[int, int, int, int]]:
    """Rectangle tightly framing the subject, for the companion crop.

    Uses the pose-derived subject box when available, falling back to an estimate from
    the face box. Returns None if neither is usable.
    """
    faces = analysis.faces or []
    primary = faces[0] if faces else None
    rect = getattr(primary, "subject_bbox", None) if primary else None

    if not rect or rect[2] <= 0 or rect[3] <= 0:
        bbox = analysis.face_bbox
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
    params = [] if fmt == "png" else [int(cv2.IMWRITE_JPEG_QUALITY), int(config.jpeg_quality)]

    filepath = os.path.join(out_dir, base_name + ext)
    cv2.imwrite(filepath, frame, params)

    crop_path = None
    if config.save_subject_crop:
        h, w = frame.shape[:2]
        rect = subject_crop_rect(analysis, w, h, config.subject_crop_padding)
        if rect:
            cx, cy, cw, ch = rect
            crop = frame[cy:cy + ch, cx:cx + cw]
            if crop.size > 0:
                crop_path = os.path.join(out_dir, base_name + "_subject" + ext)
                cv2.imwrite(crop_path, crop, params)

    return filepath, crop_path

class StreamProcessor:
    # Connection status constants
    STATUS_DISCONNECTED = "disconnected"
    STATUS_CONNECTING = "connecting"
    STATUS_CONNECTED = "connected"
    STATUS_RECONNECTING = "reconnecting"
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
            pose_hand_interval=getattr(config, "pose_hand_interval", 1)
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
            print(f"[StreamProcessor] Frame buffer: {config.dvr_buffer_seconds:.0f}s "
                  f"({dvr_frames} frames, ~{dvr_frames * 6.22 / 1024:.2f} GB at 1080p)")

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
        """Switches video input source (e.g. RTSP URL, Webcam, File)."""
        was_running = self.running
        self.stop()
        self.config.stream_source = new_source
        self.reconnect_attempt = 0
        if was_running:
            self.start()

    def _set_status(self, status: str):
        """Update connection status and notify GUI."""
        self.connection_status = status
        if self.on_status_callback:
            try:
                self.on_status_callback(status, self.reconnect_attempt)
            except Exception:
                pass

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        source = self.config.stream_source.strip()
        
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

    def _grabber_loop(self):
        """Dedicated high-speed loop reading camera frames as fast as they arrive."""
        self.cap = self._open_capture()
        if not self.cap:
            self._set_status(self.STATUS_FAILED)
            self.running = False
            self._stop_event.set()  # also stops the analysis loop
            return

        is_rtsp = self.config.stream_source.startswith("rtsp://")
        self._set_status(self.STATUS_CONNECTED)
        self.reconnect_attempt = 0

        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if is_rtsp:
                    # === RTSP Reconnection with Exponential Backoff ===
                    self.reconnect_attempt += 1
                    
                    if self.reconnect_attempt > self.config.rtsp_max_retries:
                        print(f"[StreamProcessor] RTSP reconnection failed after {self.config.rtsp_max_retries} attempts. Giving up.")
                        self._set_status(self.STATUS_FAILED)
                        self.running = False
                        self._stop_event.set()  # also stops the analysis loop
                        return
                    
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
                    if self.cap and self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break

            # Store latest frame
            now_grab = time.time()
            with self.frame_lock:
                self.latest_frame = frame
                self.frames_grabbed += 1
            if self._dvr.maxlen:
                with self._dvr_lock:
                    # deque(maxlen) evicts the oldest, so memory stays bounded without
                    # any trimming pass on the hot path.
                    self._dvr.append((now_grab, frame))

            # Yield slightly to prevent 100% CPU lock in grabber thread
            time.sleep(0.001)

    def _analysis_loop(self):
        """Processes latest frame, evaluates face quality, and updates GUI."""
        fps_target = self.config.analysis_fps
        frame_delay = 1.0 / max(1, fps_target)
        
        last_time = time.time()
        frame_counter = 0

        while not self._stop_event.is_set():
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
            primary = (analysis.faces or [None])[0]
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
            held = sum(f.nbytes for _, f in self._dvr)
            newest = self._dvr[-1][1] if frames else None
            span = (self._dvr[-1][0] - self._dvr[0][0]) if frames > 1 else 0.0
        capacity = self._dvr.maxlen or 0
        per_frame = int(newest.nbytes) if newest is not None else 0
        # What the camera is really delivering. dvr_buffer_fps only sizes the buffer;
        # if the camera runs faster, the same frame count covers proportionally less
        # time, so the configured seconds quietly become fewer. Reporting both makes
        # that visible instead of leaving a buffer that is half the requested depth.
        measured_fps = round(frames / span, 1) if span > 0.2 else 0.0
        return {
            "measured_fps": measured_fps,
            "seconds_short": bool(measured_fps and span + 0.3 < float(self.config.dvr_buffer_seconds)),
            "enabled": capacity > 0,
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
        seconds = max(0.0, min(60.0, float(seconds)))
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

    def _refine_from_buffer(self, chosen_at: float, chosen_frame: np.ndarray,
                           chosen_analysis: AnalysisResult):
        """Looks for a crisper version of the chosen moment in the rolling buffer.

        Analysis samples at analysis_fps while the camera sends roughly four times that,
        so the frame that got scored is rarely the sharpest one available - a gesturing
        hand smears badly from one frame to the next. This scrubs a short window around
        the chosen moment, ranks every buffered frame by a cheap sharpness proxy, and
        then fully re-scores only the best few. Sharpness alone would be the wrong thing
        to trust: half a second is enough to start a blink, so the winner has to earn it
        on the whole score, not just on clarity.

        Returns (frame, analysis) - the originals if nothing better is found.
        """
        window = float(getattr(self.config, "dvr_refine_window_sec", 0.0) or 0.0)
        top_n = int(getattr(self.config, "dvr_refine_candidates", 0) or 0)
        if window <= 0 or top_n <= 0 or not self._dvr.maxlen:
            return chosen_frame, chosen_analysis

        with self._dvr_lock:
            nearby = [(t, f) for (t, f) in self._dvr if abs(t - chosen_at) <= window]
        if len(nearby) < 2:
            return chosen_frame, chosen_analysis

        # Cheap proxy first: sharpness over the subject region, normalised the same way
        # the analyzer does it so the numbers mean the same thing.
        faces = chosen_analysis.faces or []
        rect = getattr(faces[0], "subject_bbox", None) if faces else None
        if not rect:
            rect = chosen_analysis.face_bbox
        if not rect:
            return chosen_frame, chosen_analysis
        rx, ry, rw, rh = rect
        height = self.analyzer.thresholds.sharpness_face_height

        scored = []
        for t, f in nearby:
            roi = f[max(0, ry):ry + rh, max(0, rx):rx + rw]
            if roi.size == 0:
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            norm = self.analyzer._resize_to_height(gray, height)
            scored.append((float(cv2.Laplacian(norm, cv2.CV_64F).var()), t, f))
        if not scored:
            return chosen_frame, chosen_analysis

        scored.sort(key=lambda item: item[0], reverse=True)

        if self._refine_analyzer is None:
            self._refine_analyzer = PersonQualityAnalyzer(
                self.config.weights, self.config.thresholds,
                max_people_target=self.config.max_people_target,
                pose_hand_interval=getattr(self.config, "pose_hand_interval", 1),
                require_temporal_stability=False,
            )

        started_qualified = chosen_analysis.is_qualified
        best_frame, best_analysis = chosen_frame, chosen_analysis
        best_score = chosen_analysis.overall_score if started_qualified else -1.0

        for _sharp, _t, f in scored[:top_n]:
            try:
                a = self._refine_analyzer.analyze_frame(f)
            except Exception:
                continue
            if a.face_detected and a.is_qualified and a.overall_score > best_score:
                best_score, best_frame, best_analysis = a.overall_score, f, a

        if best_analysis is not chosen_analysis:
            print(f"[Refine] {chosen_analysis.overall_score:.0%} -> {best_score:.0%} "
                  f"(sharpness {chosen_analysis.laplacian_variance:.0f} -> "
                  f"{best_analysis.laplacian_variance:.0f}) from {len(nearby)} buffered frames")
        return best_frame, best_analysis

    def _would_improve_retained(self, score: float, now: float) -> bool:
        """Is this capture worth writing, given what is already kept?

        This replaced a check that compared each candidate against the *last saved*
        score and demanded a 2% improvement over it unless the frame looked visually
        different. That was a ratchet: one strong capture pinned the bar high, a seated
        speaker rarely looks "different" enough at 160x120, and capturing quietly
        stopped for the rest of the service — even for frames that would have replaced
        the weakest of the five kept. The question that matters is whether this frame
        improves the kept set, so that is what is asked.
        """
        if not self.config.single_winner_mode:
            return True
        n = max(1, self.config.max_retained_captures)
        kept = self.saved_captures
        if len(kept) < n:
            return True

        spacing = max(0.0, float(getattr(self.config, "min_capture_spacing_sec", 0.0)))
        if spacing > 0:
            # A better version of a moment already represented replaces it; a weaker
            # one is not worth a second copy of the same few seconds.
            nearby = [c for c in kept if abs(now - c.captured_at) < spacing]
            if nearby:
                return score > max(c.overall_score for c in nearby)

        return score > min(c.overall_score for c in kept)

    def _evaluate_and_trigger_capture(self, now: float):
        if not self.candidate_buffer:
            return

        # Pick candidate frame with highest overall score among strictly qualified frames
        best_candidate = max(self.candidate_buffer, key=lambda item: item[2].overall_score)
        cand_time, best_frame, best_analysis = best_candidate

        # The score threshold is deliberately NOT checked here. Blur pushes a frame under
        # it, and refinement below may well replace that frame with a sharp version of
        # the same moment that clears it comfortably. The post-refinement is_qualified
        # guard enforces the threshold on whatever frame actually wins.

        # === Score Trend: require local peak ===
        if not self._is_at_local_peak():
            return
        
        # === Scrub the buffer for a crisper version of this same moment ===
        # Done before the retention test, since refinement can only raise the score and
        # a sharper frame may clear a bar the sampled one missed.
        best_frame, best_analysis = self._refine_from_buffer(cand_time, best_frame, best_analysis)

        # Blur-only candidates were admitted on the chance the buffer held a sharp
        # version of the same moment. If it didn't, the frame is still blurry - drop it.
        if not best_analysis.is_qualified:
            return

        # === Would this actually improve what we're keeping? ===
        if not self._would_improve_retained(best_analysis.overall_score, now):
            return

        # All checks passed — either save now, or send it for review first.
        if self.judge.available:
            if not self._start_review(best_frame, best_analysis, now):
                return   # a review is already running; the next candidate will come along
        else:
            self.save_screenshot(best_frame, best_analysis)
        # Reports the terms that actually drive the pick. Plain "deg" rather than the
        # degree sign, which mojibakes on a default Windows console codepage.
        flags = []
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

    def _start_review(self, frame: np.ndarray, analysis: AnalysisResult, now: float) -> bool:
        """Queues a candidate for judge review and makes sure the worker is running."""
        with self._review_lock:
            dropped = len(self._review_queue) == self._review_queue.maxlen
            self._review_queue.append((frame.copy(), analysis))
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
                frame, analysis = self._review_queue.popleft()
                self._review_in_flight = True
            try:
                self._review_and_save(frame, analysis)
            except Exception as e:  # noqa: BLE001 - one bad candidate must not end the worker
                print(f"[AIJudge] Review worker error: {e}")

    def _review_and_save(self, frame: np.ndarray, analysis: AnalysisResult) -> None:
        try:
            h, w = frame.shape[:2]
            rect = subject_crop_rect(analysis, w, h, self.config.subject_crop_padding)
            crop = frame if rect is None else frame[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            notes = list(getattr(analysis, "disqualified_reasons", []) or [])
            if analysis.hands_on_torso:
                notes.append("hands over the torso")
            if analysis.is_subject_clipped:
                notes.append("subject runs off the frame edge")

            result = self.judge.judge(crop, analysis.overall_score, notes)

            # "Unusable" is a hard no only once there is something better to keep. Early
            # in a service it is a heavy penalty instead, so a strict judge can never
            # leave the gallery empty - the same reason the heuristics don't hard-gate.
            if result.verdict is not None and not result.verdict.usable:
                n = max(1, self.config.max_retained_captures)
                if not self.config.single_winner_mode or len(self.saved_captures) >= n:
                    print(f"[AIJudge] Rejected ({analysis.overall_score:.0%} -> judge "
                          f"{result.verdict.score:.0%}): {result.verdict.summary}")
                    return
                result.blended_score = round(min(result.blended_score, analysis.overall_score * 0.5), 3)

            # Re-check against the kept set with the blended score: the review took a
            # few seconds, and a better capture may have landed meanwhile.
            if not self._would_improve_retained(result.blended_score, time.time()):
                return

            self.save_screenshot(frame, analysis, review=result)
        except Exception as e:  # noqa: BLE001 - a review failure must not kill the worker
            print(f"[AIJudge] Review failed for one candidate: {e}")

    def save_screenshot(self, frame: np.ndarray, analysis: AnalysisResult,
                        review: Optional[JudgeResult] = None) -> CapturedScreenshot:
        """Saves screenshot frame to disk and prunes inferior shots if Single Winner Mode is enabled."""
        heuristic = analysis.overall_score
        final_score = heuristic
        verdict = review.verdict if review else None
        if verdict is not None:
            final_score = review.blended_score

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        ms_str = f"{int((time.time() % 1) * 1000):03d}"
        base_name = f"best_shot_{timestamp_str}_{ms_str}_score_{int(final_score * 100)}"

        filepath, crop_path = write_capture(
            frame, analysis, self.config.output_dir, base_name, self.config
        )

        capture_item = CapturedScreenshot(
            filepath=filepath,
            timestamp_str=timestamp_str,
            overall_score=final_score,
            analysis=analysis,
            crop_filepath=crop_path,
            captured_at=time.time(),
            heuristic_score=heuristic,
            judge_score=(float(verdict.score) if verdict is not None else None),
            judge_summary=(verdict.summary if verdict is not None else ""),
            judge_issues=(list(verdict.issues) if verdict is not None else []),
        )
        if verdict is not None:
            print(f"[AIJudge] Kept ({heuristic:.0%} -> judge {verdict.score:.0%} = {final_score:.0%}, "
                  f"{review.latency_sec:.1f}s): {verdict.summary}")
        
        self.saved_captures.append(capture_item)
        
        # In Single Winner Mode, compare against existing captures and keep ONLY the top #1 best screenshot on disk
        if self.config.single_winner_mode:
            self.prune_to_best_n(self.config.max_retained_captures)
        else:
            self.captures_count += 1
            if self.on_capture_callback:
                self.on_capture_callback(capture_item)
            
        return capture_item

    def prune_to_best_n(self, n: int = 1):
        """Analyzes all captures, keeps the top N highest-scoring screenshots, and deletes inferior candidate files from disk."""
        if not self.saved_captures:
            return

        # Best first, but five shots from the same ten seconds are one shot with
        # variations. Spread the keepers out, then relax the spacing only if there
        # aren't enough well-separated shots to fill the slots.
        ordered = sorted(self.saved_captures, key=lambda item: item.overall_score, reverse=True)
        spacing = max(0.0, float(getattr(self.config, "min_capture_spacing_sec", 0.0)))
        winners: List[CapturedScreenshot] = []
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
        winners.sort(key=lambda item: item.overall_score, reverse=True)
        
        # Delete inferior screenshot files from disk
        for loser in losers:
            # Remove the companion subject crop alongside the full frame, otherwise
            # pruning leaves orphaned crops of shots that were deleted.
            for path in (loser.filepath, loser.crop_filepath):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        print(f"[StreamProcessor] Pruned inferior shot: {path} (Score: {loser.overall_score})")
                    except Exception as e:
                        print(f"[StreamProcessor] Error deleting file {path}: {e}")

        self.saved_captures = winners
        self.captures_count = len(winners)

        # Notify callback with the champion winner
        if winners and self.on_capture_callback:
            self.on_capture_callback(winners[0])
