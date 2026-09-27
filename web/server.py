"""FastAPI web front-end for the Thumbnail Grabber.

Runs on the same LAN as the camera, serves the analyzed video as MJPEG and the
live telemetry over a WebSocket, so any browser on the network can drive it.

Run with:
    python -m web.server            (from the project root)
    python web/server.py --port 8000
"""
import os
import sys
import time
import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any, List

# Allow "python web/server.py" as well as "python -m web.server"
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import AppConfig
from analyzer import AnalysisResult
from stream_processor import StreamProcessor, CapturedScreenshot
from overlay import draw_overlay
from video_scanner import VideoRecordingScanner, ScanProgress, VideoInfo

STATIC_DIR = Path(__file__).resolve().parent / "static"


class AppState:
    """Holds the single shared StreamProcessor, view preferences, and video scanner state."""

    def __init__(self):
        self.config = AppConfig()
        self.processor: Optional[StreamProcessor] = None
        self.show_mesh: bool = True
        self.show_hud: bool = True
        self.is_capturing: bool = False
        self._lock = threading.Lock()

        # Video file scanner state
        self.is_scanning_video: bool = False
        self.scan_progress: Optional[ScanProgress] = None
        self.scan_cancel_event: Optional[threading.Event] = None
        self.scan_thread: Optional[threading.Thread] = None
        self.scan_results: List[CapturedScreenshot] = []
        self.scan_error: Optional[str] = None
        self.scan_video_path: Optional[str] = None

        # Latest frame + analysis pushed from the analysis thread
        self._frame: Optional[np.ndarray] = None
        self._analysis: Optional[AnalysisResult] = None
        self._frame_seq: int = 0

    # --- processor lifecycle -------------------------------------------------

    def start_processor(self):
        with self._lock:
            if self.processor:
                self.processor.stop()
            self.processor = StreamProcessor(
                config=self.config,
                on_frame_callback=self._on_frame,
            )
            self.processor.start()
            # Feed starts connected but capture stays paused, matching the desktop app
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
        """Newest frame off the camera, with the most recent analysis to draw over it.

        The overlay can trail the video by an analysis period; the boxes shifting a
        fraction of a second late is unnoticeable, whereas showing only scored frames
        drops the preview to the analysis rate and looks like a stuttering feed.
        """
        proc = self.processor
        if proc is not None:
            with proc.frame_lock:
                frame = proc.latest_frame
                seq = proc.frames_grabbed
            if frame is not None:
                return frame, proc.latest_analysis, seq
        return self._frame, self._analysis, self._frame_seq


state = AppState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state.start_processor()
    yield
    state.stop_processor()


app = FastAPI(title="Thumbnail Grabber", lifespan=lifespan)


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
    }


def _telemetry() -> Dict[str, Any]:
    proc = state.processor
    frame, analysis, _ = state.snapshot()
    cfg = state.config

    payload: Dict[str, Any] = {
        "status": proc.connection_status if proc else StreamProcessor.STATUS_DISCONNECTED,
        "reconnect_attempt": proc.reconnect_attempt if proc else 0,
        "fps": round(proc.current_fps, 1) if proc else 0.0,
        "frames_processed": proc.frames_processed if proc else 0,
        "is_capturing": state.is_capturing,
        "auto_capture": bool(proc.auto_capture_enabled) if proc else False,
        "single_winner": cfg.single_winner_mode,
        "show_mesh": state.show_mesh,
        "show_hud": state.show_hud,
        "source": cfg.stream_source,
        "output_dir": cfg.output_dir,
        "frame_size": [int(frame.shape[1]), int(frame.shape[0])] if frame is not None else None,
        "settings": {
            "min_score": int(cfg.thresholds.min_overall_score * 100),
            "max_yaw": int(cfg.thresholds.max_yaw_deg),
            "max_shoulder_tilt": int(cfg.thresholds.max_shoulder_tilt_deg),
            "burst_window": round(cfg.burst_window_seconds, 1),
            "cooldown": round(cfg.cooldown_seconds, 1),
            "people_target": cfg.max_people_target,
        },
        "selected_people": sorted(proc.analyzer.selected_person_ids) if proc else [],
        "face_detected": False,
    }

    if analysis is not None:
        pm = proc.analyzer.preference_model if proc else None
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
            "background_score": round(float(analysis.background_score), 3),
            "disqualified_reasons": list(getattr(analysis, "disqualified_reasons", []) or []),
            "hands_covering_face": bool(analysis.hands_covering_face),
            "is_hand_blurry": bool(analysis.is_hand_blurry),
            "faces": [_face_to_dict(f) for f in (analysis.faces or [])],
            "ml_trained": bool(pm.is_trained) if pm else False,
            "ml_confidence": round(float(pm.confidence), 3) if pm else 0.0,
            "ml_samples": len(pm.samples) if pm else 0,
            "ai_judge": proc.judge.status_line() if proc and getattr(proc, "judge", None) else "AI judge off",
        })

    return payload


def _capture_url(path: Optional[str]) -> Optional[str]:
    """Cache-busted /captures URL for a written file, or None if it isn't there."""
    if not path or not os.path.exists(path):
        return None
    return f"/captures/{os.path.basename(path)}?t={int(os.path.getmtime(path))}"


def _captures_payload() -> List[Dict[str, Any]]:
    proc = state.processor
    if not proc:
        return []
    caps = sorted(proc.saved_captures, key=lambda c: c.overall_score, reverse=True)
    if state.config.single_winner_mode:
        caps = caps[:state.config.max_retained_captures]
    out = []
    for idx, c in enumerate(caps):
        name = os.path.basename(c.filepath)
        a = c.analysis
        out.append({
            "rank": idx + 1,
            "filename": name,
            "url": f"/captures/{name}?t={int(os.path.getmtime(c.filepath)) if os.path.exists(c.filepath) else 0}",
            "crop_url": _capture_url(c.crop_filepath),
            "score": round(float(c.overall_score), 3),
            "timestamp": c.timestamp_str,
            "cutout_score": round(float(getattr(a, "cutout_score", 0.0)), 3),
            "heuristic_score": round(float(getattr(c, "heuristic_score", c.overall_score) or c.overall_score), 3),
            "judge_score": (round(float(c.judge_score), 3) if getattr(c, "judge_score", None) is not None else None),
            "judge_summary": getattr(c, "judge_summary", "") or "",
            "judge_issues": list(getattr(c, "judge_issues", []) or []),
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
        for c in proc.saved_captures:
            if os.path.basename(c.filepath) == filename:
                return c
    raise HTTPException(status_code=404, detail="Capture not found")


# ---------------------------------------------------------------------------
# Video + telemetry
# ---------------------------------------------------------------------------

def _mjpeg_generator(max_width: int = 1280, quality: int = 80):
    """Yields multipart MJPEG. Downscales the preview so a 1080p feed doesn't
    saturate Wi-Fi — analysis and saved screenshots stay at full resolution."""
    boundary = b"--frame\r\n"
    last_seq = -1
    target_fps = max(1, getattr(state.config, "preview_fps", 15))
    min_interval = 1.0 / target_fps

    while True:
        loop_start = time.time()
        frame, analysis, seq = state.snapshot()

        if frame is None:
            # Placeholder while connecting
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for camera feed...", (110, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (148, 163, 184), 2)
            ok, buf = cv2.imencode(".jpg", blank)
            if ok:
                yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(0.5)
            continue

        if seq != last_seq:
            last_seq = seq
            annotated = draw_overlay(frame, analysis, state.show_mesh, state.show_hud)

            h, w = annotated.shape[:2]
            if max_width and w > max_width:
                scale = max_width / float(w)
                annotated = cv2.resize(annotated, (max_width, int(h * scale)),
                                       interpolation=cv2.INTER_AREA)

            ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

        elapsed = time.time() - loop_start
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)


@app.get("/video.mjpg")
def video_feed(w: int = 1280, q: int = 80):
    """Live annotated preview. ?w=640&q=60 for phones on weak Wi-Fi."""
    return StreamingResponse(
        _mjpeg_generator(max_width=max(160, min(1920, w)), quality=max(30, min(95, q))),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


@app.websocket("/ws")
async def telemetry_socket(ws: WebSocket):
    await ws.accept()
    last_capture_count = -1
    try:
        while True:
            payload = _telemetry()
            proc = state.processor
            count = len(proc.saved_captures) if proc else 0
            # Only ship the gallery when it actually changed
            if count != last_capture_count:
                last_capture_count = count
                payload["captures"] = _captures_payload()
            await ws.send_json(payload)
            await asyncio.sleep(0.1)
    except (WebSocketDisconnect, RuntimeError):
        pass


# ---------------------------------------------------------------------------
# Control API
# ---------------------------------------------------------------------------

@app.post("/api/source")
def set_source(payload: Dict[str, Any] = Body(...)):
    src = str(payload.get("source", "")).strip()
    if not src:
        raise HTTPException(status_code=400, detail="source is required")
    state.config.stream_source = src
    if state.processor:
        state.processor.update_source(src)
    else:
        state.start_processor()
    return {"ok": True, "source": src}


@app.post("/api/capture/toggle")
def toggle_capture(payload: Dict[str, Any] = Body(...)):
    enabled = bool(payload.get("enabled", False))
    state.is_capturing = enabled
    if state.processor:
        state.processor.auto_capture_enabled = enabled
    return {"ok": True, "is_capturing": enabled}


@app.post("/api/capture/manual")
def manual_capture():
    proc = state.processor
    if not proc:
        raise HTTPException(status_code=409, detail="Stream not running")
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
    cfg = state.config
    if "min_score" in payload:
        cfg.thresholds.min_overall_score = float(payload["min_score"]) / 100.0
    if "max_yaw" in payload:
        cfg.thresholds.max_yaw_deg = float(payload["max_yaw"])
    if "max_shoulder_tilt" in payload:
        cfg.thresholds.max_shoulder_tilt_deg = float(payload["max_shoulder_tilt"])
    if "burst_window" in payload:
        cfg.burst_window_seconds = float(payload["burst_window"])
    if "cooldown" in payload:
        cfg.cooldown_seconds = float(payload["cooldown"])
    if "people_target" in payload:
        val = max(1, min(6, int(payload["people_target"])))
        cfg.max_people_target = val
        if state.processor:
            state.processor.analyzer.max_people_target = val
    return {"ok": True}


@app.post("/api/view")
def update_view(payload: Dict[str, Any] = Body(...)):
    if "show_mesh" in payload:
        state.show_mesh = bool(payload["show_mesh"])
    if "show_hud" in payload:
        state.show_hud = bool(payload["show_hud"])
    if "single_winner" in payload:
        state.config.single_winner_mode = bool(payload["single_winner"])
        if state.config.single_winner_mode and state.processor:
            state.processor.prune_to_best_n(state.config.max_retained_captures)
    return {"ok": True}


@app.post("/api/person/{person_id}")
def toggle_person(person_id: int, payload: Dict[str, Any] = Body(default={})):
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
    return {"ok": True, "selected_people": sorted(ids)}


@app.post("/api/click")
def click_video(payload: Dict[str, Any] = Body(...)):
    """Toggle a person by clicking their face box on the video."""
    proc = state.processor
    frame, analysis, _ = state.snapshot()
    if not proc or analysis is None or frame is None:
        raise HTTPException(status_code=409, detail="No active analysis")
    fx = float(payload.get("x_frac", -1)) * frame.shape[1]
    fy = float(payload.get("y_frac", -1)) * frame.shape[0]
    for f in (analysis.faces or []):
        bx, by, bw, bh = f.bbox
        if bx <= fx <= bx + bw and by <= fy <= by + bh:
            return toggle_person(f.person_id, {})
    return {"ok": False, "detail": "No face at that point"}


@app.get("/api/captures")
def list_captures():
    return {"captures": _captures_payload()}


@app.post("/api/captures/{filename}/feedback")
def capture_feedback(filename: str, payload: Dict[str, Any] = Body(...)):
    """Train the preference model from a thumbnail's star / thumbs-down."""
    proc = state.processor
    if not proc:
        raise HTTPException(status_code=409, detail="Stream not running")
    cap = _find_capture(filename)
    is_favorite = bool(payload.get("favorite", True))
    pm = proc.analyzer.preference_model
    pm.add_sample(cap.analysis, is_favorite=is_favorite)
    return {
        "ok": True,
        "favorite": is_favorite,
        "samples": len(pm.samples),
        "confidence": round(float(pm.confidence), 3),
    }


@app.get("/captures/{filename}")
def get_capture_image(filename: str):
    # Resolve against the output dir and refuse anything that escapes it
    base = Path(state.config.output_dir).resolve()
    target = (base / filename).resolve()
    if base not in target.parents or not target.exists():
        raise HTTPException(status_code=404, detail="Not found")
    media = "image/png" if target.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(str(target), media_type=media)


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


def _run_video_scan_worker(video_path: str, top_n: int, min_score: float,
                           sample_fps: float, min_interval_sec: float, out_dir: str):
    state.is_scanning_video = True
    state.scan_error = None
    state.scan_results = []
    state.scan_video_path = video_path

    def progress_cb(p: ScanProgress):
        state.scan_progress = p

    try:
        scanner = VideoRecordingScanner(state.config)
        results = scanner.scan_video(
            video_path=video_path,
            top_n=top_n,
            min_score=min_score,
            sample_fps=sample_fps,
            min_interval_sec=min_interval_sec,
            output_dir=out_dir,
            progress_callback=progress_cb,
            cancel_event=state.scan_cancel_event
        )
        state.scan_results = results
        if state.processor:
            for r in results:
                if not any(c.filepath == r.filepath for c in state.processor.saved_captures):
                    state.processor.saved_captures.append(r)
    except Exception as e:
        state.scan_error = str(e)
    finally:
        state.is_scanning_video = False


# ---------------------------------------------------------------------------
# Path safety
#
# None of these endpoints are authenticated — the design assumes a trusted LAN. That
# makes it all the more important that a request can't name an arbitrary path: an
# upload whose filename is "../../analyzer.py" would otherwise overwrite application
# code and run on the next restart.
# ---------------------------------------------------------------------------

def _sanitize_upload_name(raw: str) -> str:
    """Reduces a client-supplied filename to a bare, safe basename."""
    name = os.path.basename(str(raw or "").replace("\\", "/").strip())
    # Drop anything that isn't clearly filename material, and refuse dot-only names.
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
    roots = []
    for part in raw.split(os.pathsep):
        part = part.strip()
        if part:
            try:
                roots.append(Path(part).resolve())
            except Exception:
                continue
    # Uploads always land inside the output dir, so that has to stay reachable.
    roots.append((Path(state.config.output_dir) / "uploads").resolve())
    return roots


def _resolve_video_path(raw: str) -> Path:
    """Resolves a requested video path, enforcing the root allowlist if one is set."""
    if not raw:
        raise HTTPException(status_code=400, detail="No video path given")
    try:
        target = Path(raw).resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid video path")

    if not target.is_file():
        raise HTTPException(status_code=400, detail="Video file not found")

    roots = _allowed_video_roots()
    if roots and not any(root == target or root in target.parents for root in roots):
        raise HTTPException(
            status_code=403,
            detail="Path is outside the directories allowed by TG_VIDEO_ROOTS"
        )
    return target


def _resolve_output_dir(raw: str) -> str:
    """Keeps scan output inside the configured output directory.

    A subfolder is fine for organising scans; anywhere else on the filesystem is not.
    """
    base = Path(state.config.output_dir).resolve()
    if not raw:
        return str(base)
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        resolved = candidate.resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid output directory")
    if resolved != base and base not in resolved.parents:
        raise HTTPException(
            status_code=400,
            detail="Output directory must be inside the configured output directory"
        )
    return str(resolved)


# ---------------------------------------------------------------------------
# Output folder browsing
#
# A browser can't open a native folder dialog for a path on the machine doing the
# writing, and captures are written server-side. So the server lists directories and
# the UI walks them. Only directory names are ever returned - never file contents.
# ---------------------------------------------------------------------------

def _browse_roots() -> List[Path]:
    """Directories the picker may browse, or [] when unrestricted."""
    raw = (state.config.output_browse_roots or "").strip()
    roots = []
    for part in raw.split(os.pathsep):
        part = part.strip()
        if part:
            try:
                roots.append(Path(part).resolve())
            except Exception:
                continue
    return roots


def _check_browsable(target: Path) -> None:
    roots = _browse_roots()
    if roots and not any(r == target or r in target.parents for r in roots):
        raise HTTPException(status_code=403,
                            detail="Path is outside the directories allowed by TG_OUTPUT_ROOTS")


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


def _is_writable(path: Path) -> bool:
    return os.access(str(path), os.W_OK | os.X_OK)


@app.get("/api/browse")
def browse_folders(path: str = ""):
    """Lists subdirectories of `path` for the folder picker."""
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
                    continue  # unreadable entry, skip rather than fail the listing
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {target}")
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Could not read directory: {e}")

    entries.sort(key=lambda d: d["name"].lower())
    parent = str(target.parent) if target.parent != target else None

    home = Path.home()
    shortcuts = [{"name": "Home", "path": str(home)}]
    for sub in ("Pictures", "Videos", "Desktop", "Downloads"):
        q = home / sub
        if q.is_dir():
            shortcuts.append({"name": sub, "path": str(q)})
    shortcuts.append({"name": "Current output", "path": str(Path(state.config.output_dir))})

    return {
        "path": str(target),
        "parent": parent,
        "entries": entries[:1000],
        "drives": _drive_list(),
        "shortcuts": shortcuts,
        "writable": _is_writable(target),
        "is_current": str(target) == str(Path(state.config.output_dir).resolve()),
    }


@app.post("/api/output")
def set_output_dir(payload: Dict[str, Any] = Body(...)):
    """Points captures at a different folder, creating it on request.

    Takes effect immediately - save_screenshot reads config.output_dir per capture.
    It is not written back to .env, so a restart returns to TG_OUTPUT_DIR; set that
    to make a choice permanent.
    """
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

    state.config.output_dir = str(target)
    if state.processor:
        state.processor.config.output_dir = str(target)
    return {"ok": True, "output_dir": str(target)}


@app.post("/api/video/info")
def video_info(payload: Dict[str, Any] = Body(...)):
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
    out_dir = Path(state.config.output_dir) / "uploads"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Never trust the client-supplied name to form a path.
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
    if state.is_scanning_video:
        raise HTTPException(status_code=409, detail="A video scan is already running")

    v_path = str(_resolve_video_path(str(payload.get("path", "")).strip()))

    top_n = int(payload.get("top_n", 5))
    min_score = float(payload.get("min_score", state.config.thresholds.min_overall_score))
    sample_fps = float(payload.get("sample_fps", 5.0))
    min_interval_sec = float(payload.get("min_interval_sec", 2.0))
    out_dir = _resolve_output_dir(str(payload.get("output_dir", "")).strip())

    state.scan_cancel_event = threading.Event()
    state.scan_thread = threading.Thread(
        target=_run_video_scan_worker,
        args=(v_path, top_n, min_score, sample_fps, min_interval_sec, out_dir),
        daemon=True
    )
    state.scan_thread.start()

    return {"ok": True, "message": "Video scan started", "video": v_path}


@app.get("/api/video/scan/status")
def get_video_scan_status():
    p = state.scan_progress
    resp: Dict[str, Any] = {
        "is_scanning": state.is_scanning_video,
        "status": p.status if p else ("scanning" if state.is_scanning_video else "idle"),
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
        resp["champion_frame_preview"] = _frame_to_base64(p.champion_frame, max_w=400)

    if not state.is_scanning_video and state.scan_results:
        results_payload = []
        for idx, c in enumerate(state.scan_results):
            name = os.path.basename(c.filepath)
            results_payload.append({
                "rank": idx + 1,
                "filename": name,
                "url": f"/captures/{name}?t={int(os.path.getmtime(c.filepath)) if os.path.exists(c.filepath) else 0}",
                "crop_url": _capture_url(c.crop_filepath),
                "score": round(float(c.overall_score), 3),
                "timestamp": c.timestamp_str,
            })
        resp["results"] = results_payload

    return resp


@app.post("/api/video/scan/cancel")
def cancel_video_scan():
    if state.scan_cancel_event:
        state.scan_cancel_event.set()
    return {"ok": True, "message": "Cancellation requested"}


@app.get("/api/state")
def get_state():
    payload = _telemetry()
    payload["captures"] = _captures_payload()
    return payload


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(str(STATIC_DIR / "index.html"),
                        headers={"Cache-Control": "no-cache"})


class _RevalidatingStatics(StaticFiles):
    """Serves the UI with `Cache-Control: no-cache`.

    Without it a browser can hold app.js in its memory cache for the life of the tab
    and never revalidate, so after an upgrade it renders the new index.html against
    the old script: the new buttons appear but nothing is wired to them. `no-cache`
    means "revalidate", not "don't cache" - the ETag still turns an unchanged file
    into a 304, so this costs a round trip rather than a download.
    """

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
