import cv2
import time
import os
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List
import numpy as np

from config import AppConfig
from analyzer import PersonQualityAnalyzer, AnalysisResult

@dataclass
class CapturedScreenshot:
    filepath: str
    timestamp_str: str
    overall_score: float
    analysis: AnalysisResult

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
            max_people_target=config.max_people_target
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
        
        # Duplicate frame detection: store last captured frame's fingerprint
        self._last_captured_frame: Optional[np.ndarray] = None
        
        # Connection status tracking
        self.connection_status: str = self.STATUS_DISCONNECTED
        self.reconnect_attempt: int = 0
        self.on_status_callback: Optional[Callable[[str, int], None]] = None
        
        # Stats
        self.frames_processed = 0
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

    def _is_duplicate_frame(self, frame: np.ndarray) -> bool:
        """Check if the new frame is nearly identical to the last captured frame.
        
        Uses Mean Squared Error (MSE) on downsampled frames for speed.
        """
        if self._last_captured_frame is None:
            return False
        
        try:
            # Downsample both frames for fast comparison
            target_size = (160, 120)
            new_small = cv2.resize(frame, target_size, interpolation=cv2.INTER_NEAREST)
            old_small = cv2.resize(self._last_captured_frame, target_size, interpolation=cv2.INTER_NEAREST)
            
            mse = float(np.mean((new_small.astype(float) - old_small.astype(float)) ** 2))
            return mse < self.config.duplicate_mse_threshold
        except Exception:
            return False

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
        
        # === Duplicate Frame Detection ===
        if self._is_duplicate_frame(best_frame):
            print(f"[StreamProcessor] Skipped duplicate frame (too similar to last capture)")
            return
        
        # All checks passed — save the screenshot
        self._last_captured_frame = best_frame.copy()
        self.save_screenshot(best_frame, best_analysis)
        print(f"[Capture] Score={best_analysis.overall_score:.0%} "
              f"Smoothed={best_analysis.smoothed_score:.0%} "
              f"Stable={best_analysis.is_stable} "
              f"Pitch={best_analysis.pitch_deg:.1f}° "
              f"Roll={best_analysis.roll_deg:.1f}° "
              f"LookDown={best_analysis.is_looking_down} "
              f"MouthOpen={best_analysis.is_awkward_expression} "
              f"HandWeird={best_analysis.is_hand_weird} "
              f"HandBlur={best_analysis.is_hand_blurry} "
              f"MAR={best_analysis.mar:.2f} "
              f"Exposure={best_analysis.exposure_score:.2f} "
              f"Centering={best_analysis.centering_score:.2f}")
        self.last_capture_time = now
        self.candidate_buffer.clear()

    def save_screenshot(self, frame: np.ndarray, analysis: AnalysisResult) -> CapturedScreenshot:
        """Saves screenshot frame to disk and prunes inferior shots if Single Winner Mode is enabled."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        ms_str = f"{int((time.time() % 1) * 1000):03d}"
        filename = f"best_shot_{timestamp_str}_{ms_str}_score_{int(analysis.overall_score * 100)}.jpg"
        filepath = os.path.join(self.config.output_dir, filename)

        # Save JPEG at maximum quality (95)
        cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        capture_item = CapturedScreenshot(
            filepath=filepath,
            timestamp_str=timestamp_str,
            overall_score=analysis.overall_score,
            analysis=analysis
        )
        
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

        # Sort captures by overall_score descending
        self.saved_captures.sort(key=lambda item: item.overall_score, reverse=True)
        
        # Keep top N
        winners = self.saved_captures[:n]
        losers = self.saved_captures[n:]
        
        # Delete inferior screenshot files from disk
        for loser in losers:
            if os.path.exists(loser.filepath):
                try:
                    os.remove(loser.filepath)
                    print(f"[StreamProcessor] Pruned inferior shot: {loser.filepath} (Score: {loser.overall_score})")
                except Exception as e:
                    print(f"[StreamProcessor] Error deleting file {loser.filepath}: {e}")

        self.saved_captures = winners
        self.captures_count = len(winners)

        # Notify callback with the champion winner
        if winners and self.on_capture_callback:
            self.on_capture_callback(winners[0])
