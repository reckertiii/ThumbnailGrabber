import cv2
import time
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List, Tuple
import numpy as np

from config import AppConfig
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
            print(f"[StreamProcessor] Error: Failed to open video source: '{source}'")
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
            with self.frame_lock:
                self.latest_frame = frame
                self.frames_grabbed += 1

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
            
            # Rolling peak candidate selection - strictly require qualified clean frames
            now = time.time()
            if analysis.face_detected and analysis.is_qualified:
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

        # Must exceed threshold
        if best_analysis.overall_score < self.config.thresholds.min_overall_score:
            return
        
        # === Score Trend: require local peak ===
        if not self._is_at_local_peak():
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
        """Kicks off an asynchronous judge review. Returns False if one is already running."""
        with self._review_lock:
            if self._review_in_flight:
                return False
            self._review_in_flight = True
        # Cooldown starts at hand-off, not at save, so the analysis loop doesn't queue
        # up a second candidate while the first is still being reviewed.
        self.last_capture_time = now
        self.candidate_buffer.clear()
        threading.Thread(target=self._review_and_save, args=(frame.copy(), analysis),
                         daemon=True).start()
        return True

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
            print(f"[AIJudge] Review worker error: {e}")
        finally:
            with self._review_lock:
                self._review_in_flight = False

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
