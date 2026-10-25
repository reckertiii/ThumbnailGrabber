import os
import cv2
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any, Tuple
import numpy as np

from config import AppConfig, QualityWeights, QualityThresholds
from analyzer import PersonQualityAnalyzer, AnalysisResult
from stream_processor import CapturedScreenshot, write_capture, subject_crop_rect
from ai_judge import AIJudge, JudgeVerdict

@dataclass
class VideoInfo:
    filepath: str
    filename: str
    total_frames: int
    fps: float
    duration_sec: float
    width: int
    height: int

@dataclass
class ScanProgress:
    status: str  # "initializing", "scanning", "completed", "cancelled", "failed"
    progress_pct: float  # 0.0 to 100.0
    current_frame_idx: int
    total_frames: int
    current_time_sec: float
    total_duration_sec: float
    fps_speed: float
    estimated_time_remaining_sec: float
    total_candidates_analyzed: int
    best_score_found: float
    current_frame: Optional[np.ndarray] = None
    champion_frame: Optional[np.ndarray] = None
    champion_analysis: Optional[AnalysisResult] = None
    error_message: Optional[str] = None

@dataclass
class ScannedCandidate:
    frame_idx: int
    timestamp_sec: float
    timestamp_str: str
    frame: np.ndarray
    analysis: AnalysisResult
    overall_score: float
    rank_score: float = 0.0   # overall_score weighted by sharpness, used for ordering
    meets_min_score: bool = True
    heuristic_score: float = 0.0
    verdict: Optional[JudgeVerdict] = None

    def __post_init__(self):
        if not self.heuristic_score:
            self.heuristic_score = self.overall_score
        # Derived here rather than at each call site so a candidate built directly can
        # never carry a stale or zero ordering score.
        if not self.rank_score:
            sharpness = float(getattr(self.analysis, "sharpness_score", 0.0) or 0.0)
            self.rank_score = self.overall_score * (0.50 + 0.50 * sharpness)

class VideoRecordingScanner:
    """Offline scanner that analyzes pre-recorded video files (e.g. MP4, MOV, MKV, AVI, WEBM)
    and extracts the highest-quality thumbnails / best screenshots.
    """

    def __init__(self, config: Optional[AppConfig] = None, judge: Optional[AIJudge] = None):
        self.config = config or AppConfig()
        self.judge = judge if judge is not None else AIJudge(self.config)
        self.analyzer = PersonQualityAnalyzer(
            self.config.weights,
            self.config.thresholds,
            ema_alpha=self.config.ema_alpha,
            stability_window=self.config.score_stability_window,
            stability_threshold=self.config.score_stability_threshold,
            blink_suppression_ms=self.config.blink_suppression_ms,
            max_people_target=self.config.max_people_target,
            # Offline scanning judges each sampled frame independently.
            require_temporal_stability=False,
            pose_hand_interval=2,
            use_ml_preference=getattr(self.config, "use_ml_preference", True),
            ml_weight=getattr(self.config, "ml_weight", 0.35),
            require_exact_people_count=getattr(self.config, "require_exact_people_count", False),
            capture_prop_moments=getattr(self.config, "capture_prop_moments", True),
        )

    @staticmethod
    def get_video_info(video_path: str) -> Optional[VideoInfo]:
        """Inspects a video file and returns its technical metadata."""
        if not os.path.exists(video_path):
            return None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if fps > 0 else 0.0
        cap.release()

        return VideoInfo(
            filepath=os.path.abspath(video_path),
            filename=os.path.basename(video_path),
            total_frames=total_frames,
            fps=fps,
            duration_sec=duration_sec,
            width=width,
            height=height,
        )

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Formats seconds into readable string (e.g. 01m24s350ms or 01h02m15s000ms)."""
        total_millis = int(round(seconds * 1000))
        hours = total_millis // 3600000
        rem = total_millis % 3600000
        minutes = rem // 60000
        rem = rem % 60000
        secs = rem // 1000
        millis = rem % 1000
        if hours > 0:
            return f"{hours:02d}h{minutes:02d}m{secs:02d}s{millis:03d}ms"
        return f"{minutes:02d}m{secs:02d}s{millis:03d}ms"

    @staticmethod
    def format_time_display(seconds: float) -> str:
        """Formats seconds into standard MM:SS or HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def scan_video(
        self,
        video_path: str,
        top_n: int = 5,
        min_score: Optional[float] = None,
        sample_fps: float = 5.0,
        min_interval_sec: float = 2.0,
        output_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[ScanProgress], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> List[CapturedScreenshot]:
        """Scans the video file, analyzes sampled frames, selects the top diverse thumbnails,
        and saves them to output_dir.
        
        Args:
            video_path: Path to input video file.
            top_n: Maximum number of best thumbnails to extract.
            min_score: Minimum quality threshold (defaults to config.thresholds.min_overall_score).
            sample_fps: Frame analysis rate (e.g. 5.0 means 5 frames per second of video).
                        Set <= 0 or None to analyze every single frame.
            min_interval_sec: Minimum temporal separation between chosen thumbnails to prevent duplicates.
            output_dir: Directory where picked thumbnails are saved.
            progress_callback: Callback receiving ScanProgress updates.
            cancel_event: threading.Event to signal cancellation.
            
        Returns:
            List of CapturedScreenshot objects for the top selected frames, sorted by score descending.
        """
        if min_score is None:
            min_score = self.config.thresholds.min_overall_score
            
        out_dir = output_dir or self.config.output_dir
        os.makedirs(out_dir, exist_ok=True)

        info = self.get_video_info(video_path)
        if not info or info.total_frames <= 0:
            if progress_callback:
                progress_callback(ScanProgress(
                    status="failed",
                    progress_pct=0.0,
                    current_frame_idx=0,
                    total_frames=0,
                    current_time_sec=0.0,
                    total_duration_sec=0.0,
                    fps_speed=0.0,
                    estimated_time_remaining_sec=0.0,
                    total_candidates_analyzed=0,
                    best_score_found=0.0,
                    error_message=f"Failed to open or decode video file: {video_path}"
                ))
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        # Determine frame step based on sample_fps
        video_fps = info.fps if info.fps > 0 else 30.0
        if sample_fps and sample_fps > 0 and sample_fps < video_fps:
            frame_step = max(1, int(round(video_fps / sample_fps)))
        else:
            frame_step = 1

        # Disable missed-face slot persistence when stepping across non-adjacent video frames
        self.analyzer._face_persistence_frames = 0 if frame_step > 1 else 2

        total_frames = info.total_frames
        total_duration = info.duration_sec

        champion_score = 0.0
        champion_frame: Optional[np.ndarray] = None
        champion_analysis: Optional[AnalysisResult] = None

        start_time = time.time()
        frames_analyzed = 0
        current_frame_idx = 0

        # Candidates are kept as best-per-time-bucket rather than in one flat list.
        # A flat list holds every qualifying frame at full resolution — ~6 MB each at
        # 1080p — which runs into many gigabytes on a full-length recording. Bucketing
        # also guarantees the survivors are spread across the timeline instead of
        # clustered in whichever few seconds happened to score well.
        bucket_sec = max(0.5, min_interval_sec)
        buckets: Dict[int, ScannedCandidate] = {}
        bucket_cap = max(top_n * 10, 50)

        def _consider(cand: ScannedCandidate) -> None:
            key = int(cand.timestamp_sec / bucket_sec)
            held = buckets.get(key)
            if held is None or cand.rank_score > held.rank_score:
                buckets[key] = cand
                if len(buckets) > bucket_cap:
                    worst = min(buckets, key=lambda k: buckets[k].rank_score)
                    del buckets[worst]

        def _analyze_and_collect(f_idx: int, f_time: float, f_img: np.ndarray) -> None:
            nonlocal champion_score, champion_frame, champion_analysis
            analysis = self.analyzer.analyze_frame(f_img)
            if not analysis.face_detected:
                return
            # Rank every frame that has a face rather than gating on is_qualified.
            # The per-frame score already penalises blinks, mid-speech mouths and blur
            # heavily, so ranking sorts them out — while a hard gate threw away the
            # whole recording when nothing cleared every check at once.
            cand = ScannedCandidate(
                frame_idx=f_idx,
                timestamp_sec=f_time,
                timestamp_str=self.format_timestamp(f_time),
                frame=f_img.copy(),
                analysis=analysis,
                overall_score=analysis.overall_score,
                meets_min_score=(analysis.is_qualified and analysis.overall_score >= min_score),
            )
            _consider(cand)
            if analysis.overall_score > champion_score:
                champion_score = analysis.overall_score
                champion_frame = cand.frame
                champion_analysis = analysis

        # Initial progress emit
        if progress_callback:
            progress_callback(ScanProgress(
                status="scanning",
                progress_pct=0.0,
                current_frame_idx=0,
                total_frames=total_frames,
                current_time_sec=0.0,
                total_duration_sec=total_duration,
                fps_speed=0.0,
                estimated_time_remaining_sec=0.0,
                total_candidates_analyzed=0,
                best_score_found=0.0,
            ))

        last_progress_emit = time.time()

        while True:
            if cancel_event and cancel_event.is_set():
                break

            # Skip intermediate frames without decoding BGR pixels or running full-frame
            # Laplacian filters. cap.grab() advances the demuxer/decoder state in <0.1ms,
            # whereas cap.read() + BGR conversion + 1080p Laplacian on every skipped frame
            # bottlenecks 60fps 1080p videos to ~0.5x real-time (~1 hour for a 36m video).
            if current_frame_idx % frame_step != 0:
                if not cap.grab():
                    break
                current_frame_idx += 1
                continue

            # Decode only the sampled frame
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            current_time_sec = current_frame_idx / video_fps

            frames_analyzed += 1
            _analyze_and_collect(current_frame_idx, current_time_sec, frame)

            # Progress notification throttled to ~5 updates/sec
            now = time.time()
            if now - last_progress_emit >= 0.20 or current_frame_idx >= total_frames - frame_step:
                last_progress_emit = now
                elapsed = now - start_time
                fps_speed = (current_frame_idx + 1) / elapsed if elapsed > 0 else 0.0
                
                frames_remaining = max(0, total_frames - current_frame_idx)
                eta_sec = frames_remaining / fps_speed if fps_speed > 0 else 0.0
                pct = min(100.0, (current_frame_idx / float(max(1, total_frames))) * 100.0)

                if progress_callback:
                    progress_callback(ScanProgress(
                        status="scanning",
                        progress_pct=pct,
                        current_frame_idx=current_frame_idx,
                        total_frames=total_frames,
                        current_time_sec=current_time_sec,
                        total_duration_sec=total_duration,
                        fps_speed=fps_speed,
                        estimated_time_remaining_sec=eta_sec,
                        total_candidates_analyzed=frames_analyzed,
                        best_score_found=champion_score,
                        current_frame=frame,
                        champion_frame=champion_frame,
                        champion_analysis=champion_analysis,
                    ))

            current_frame_idx += 1

        cap.release()

        # Check if cancelled
        is_cancelled = cancel_event.is_set() if cancel_event else False

        # === Non-Maximum Suppression (Temporal Diversity Filtering) ===
        # Filter candidate frames so that picked thumbnails aren't all clustered in the same second
        pool = list(buckets.values())
        if self.judge.available and pool and not is_cancelled:
            # Review a short list of finalists — twice the requested count, spaced out —
            # rather than every bucket, so a long recording costs a handful of calls.
            finalists = self._select_diverse_top_candidates(
                pool, top_n=top_n * 2, min_interval_sec=min_interval_sec
            )
            self._review_finalists(finalists, progress_callback, info, total_frames,
                                   current_frame_idx, frames_analyzed, champion_score,
                                   champion_frame, champion_analysis)

        selected_candidates = self._select_diverse_top_candidates(
            pool, top_n=top_n, min_interval_sec=min_interval_sec
        )

        # Save selected thumbnails to disk
        video_base_name = os.path.splitext(info.filename)[0]
        results: List[CapturedScreenshot] = []

        for rank_idx, cand in enumerate(selected_candidates, 1):
            score_int = int(round(cand.overall_score * 100))
            is_prop = bool(
                getattr(cand.analysis, "is_prop_moment", False) or
                (cand.verdict is not None and getattr(cand.verdict, "holding_prop", False) and getattr(self.config, "capture_prop_moments", True))
            )
            prop_tag = "_prop" if is_prop else ""
            base_name = f"{video_base_name}_ts_{cand.timestamp_str}_rank{rank_idx}{prop_tag}_score{score_int}"

            out_filepath, crop_path = write_capture(
                cand.frame, cand.analysis, out_dir, base_name, self.config
            )

            cap_item = CapturedScreenshot(
                filepath=out_filepath,
                timestamp_str=cand.timestamp_str,
                overall_score=cand.overall_score,
                analysis=cand.analysis,
                crop_filepath=crop_path,
                captured_at=cand.timestamp_sec,
                heuristic_score=cand.heuristic_score,
                judge_score=(float(cand.verdict.score) if cand.verdict else None),
                judge_summary=(cand.verdict.summary if cand.verdict else ""),
                judge_issues=(list(cand.verdict.issues) if cand.verdict else []),
                is_prop_moment=is_prop,
                from_video_scan=True,
            )
            results.append(cap_item)

        # Final progress emit
        final_status = "cancelled" if is_cancelled else "completed"
        if progress_callback:
            progress_callback(ScanProgress(
                status=final_status,
                progress_pct=100.0 if not is_cancelled else min(100.0, (current_frame_idx / float(max(1, total_frames))) * 100.0),
                current_frame_idx=current_frame_idx,
                total_frames=total_frames,
                current_time_sec=info.duration_sec,
                total_duration_sec=total_duration,
                fps_speed=(current_frame_idx / (time.time() - start_time)) if (time.time() - start_time) > 0 else 0.0,
                estimated_time_remaining_sec=0.0,
                total_candidates_analyzed=frames_analyzed,
                best_score_found=champion_score,
                champion_frame=champion_frame,
                champion_analysis=champion_analysis,
            ))

        return results

    def _review_finalists(self, finalists: List[ScannedCandidate], progress_callback,
                          info, total_frames, current_frame_idx, frames_analyzed,
                          champion_score, champion_frame, champion_analysis) -> None:
        """Asks the judge about each finalist and folds the verdict into its ranking.

        A candidate the judge calls unusable is pushed below the threshold so it only
        surfaces if literally nothing else is available.
        """
        # Bypass the live-mode pacing: this is a batch of a known, small size.
        orig_interval = getattr(self.judge, "min_interval", 3.0)
        self.judge.min_interval = 0.0
        pm = getattr(getattr(self, "analyzer", None), "preference_model", None)
        fb_prompt = pm.get_judge_feedback_prompt() if pm is not None and hasattr(pm, "get_judge_feedback_prompt") else ""
        try:
            for i, cand in enumerate(finalists, 1):
                if progress_callback:
                    progress_callback(ScanProgress(
                        status="reviewing", progress_pct=100.0,
                        current_frame_idx=current_frame_idx, total_frames=total_frames,
                        current_time_sec=info.duration_sec, total_duration_sec=info.duration_sec,
                        fps_speed=0.0, estimated_time_remaining_sec=0.0,
                        total_candidates_analyzed=frames_analyzed, best_score_found=champion_score,
                        champion_frame=champion_frame, champion_analysis=champion_analysis,
                        error_message=f"AI Judge reviewing finalist {i}/{len(finalists)}...",
                    ))
                h, w = cand.frame.shape[:2]
                rect = subject_crop_rect(cand.analysis, w, h, self.config.subject_crop_padding)
                crop = cand.frame if rect is None else cand.frame[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
                notes = list(getattr(cand.analysis, "disqualified_reasons", []) or [])
                if getattr(cand.analysis, "is_prop_moment", False):
                    notes.append("speaker appears to be holding or presenting a stage prop")
                if fb_prompt:
                    try:
                        result = self.judge.judge(crop, cand.heuristic_score, notes, user_feedback_prompt=fb_prompt)
                    except TypeError:
                        result = self.judge.judge(crop, cand.heuristic_score, notes)
                else:
                    result = self.judge.judge(crop, cand.heuristic_score, notes)
                if result.verdict is None:
                    continue
                cand.verdict = result.verdict
                if getattr(result.verdict, "holding_prop", False) and getattr(self.config, "capture_prop_moments", True):
                    cand.analysis.is_prop_moment = True
                elif result.verdict.issues:
                    issue_text = " ".join(result.verdict.issues or []).lower()
                    if any(kw in issue_text for kw in (
                        "head bowed", "head slightly bowed", "bowed down", "looking down",
                        "gaze not engaged", "not engaged", "eyes closed", "microphone",
                    )):
                        cand.analysis.is_prop_moment = False
                if result.verdict.usable:
                    cand.overall_score = result.blended_score
                else:
                    cand.overall_score = min(cand.heuristic_score, 0.25)
                    cand.meets_min_score = False
                sharp = float(getattr(cand.analysis, "sharpness_score", 0.0) or 0.0)
                cand.rank_score = cand.overall_score * (0.50 + 0.50 * sharp)
                print(f"[AIJudge] {cand.timestamp_str}: {cand.heuristic_score:.0%} -> "
                      f"judge {result.verdict.score:.0%}"
                      f"{'' if result.verdict.usable else ' (unusable)'} — {result.verdict.summary}")
        finally:
            self.judge.min_interval = orig_interval

    def _select_diverse_top_candidates(
        self,
        candidates: List[ScannedCandidate],
        top_n: int = 5,
        min_interval_sec: float = 2.0
    ) -> List[ScannedCandidate]:
        """Applies greedy temporal non-maximum suppression (NMS) to select up to `top_n`
        highest-scoring candidate frames that are separated by at least `min_interval_sec`.

        Frames that cleared the quality threshold are preferred, but if too few did, the
        best of the rest fill the remaining slots rather than returning a short list —
        a recording where nothing scores 55% should still hand back its best moments.
        """
        if not candidates:
            return []

        ordered = sorted(candidates, key=lambda c: c.rank_score, reverse=True)
        preferred = [c for c in ordered if c.meets_min_score]
        fallback = [c for c in ordered if not c.meets_min_score]

        selected: List[ScannedCandidate] = []

        def _take(pool: List[ScannedCandidate], enforce_gap: bool) -> None:
            for cand in pool:
                if len(selected) >= top_n:
                    return
                if any(c.frame_idx == cand.frame_idx for c in selected):
                    continue
                if enforce_gap and any(
                    abs(cand.timestamp_sec - sel.timestamp_sec) < min_interval_sec
                    for sel in selected
                ):
                    continue
                selected.append(cand)

        # Spread-out frames that met the bar, then spread-out frames that did not,
        # then relax the spacing rule if there still aren't enough.
        if getattr(self.config, "capture_prop_moments", True) and top_n > 1:
            prop_cap = min(max(0, int(getattr(self.config, "max_prop_captures", 2))), max(0, top_n - 1))
            if prop_cap > 0:
                prop_pref = [c for c in preferred if getattr(c.analysis, "is_prop_moment", False)]
                for cand in prop_pref:
                    if len(selected) >= prop_cap:
                        break
                    if any(abs(cand.timestamp_sec - sel.timestamp_sec) < min_interval_sec for sel in selected):
                        continue
                    selected.append(cand)

        _take(preferred, enforce_gap=True)
        _take(fallback, enforce_gap=True)
        _take(preferred, enforce_gap=False)
        _take(fallback, enforce_gap=False)

        return sorted(selected, key=lambda c: c.overall_score, reverse=True)
