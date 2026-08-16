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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import AppConfig
from analyzer import AnalysisResult
from stream_processor import StreamProcessor, CapturedScreenshot
from overlay import draw_overlay

STATIC_DIR = Path(__file__).resolve().parent / "static"


class AppState:
    """Holds the single shared StreamProcessor and view preferences."""

    def __init__(self):
        self.config = AppConfig()
        self.processor: Optional[StreamProcessor] = None
        self.show_mesh: bool = True
        self.show_hud: bool = True
        self.is_capturing: bool = False
        self._lock = threading.Lock()

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
            "hands_covering_face": bool(analysis.hands_covering_face),
            "is_hand_blurry": bool(analysis.is_hand_blurry),
            "faces": [_face_to_dict(f) for f in (analysis.faces or [])],
            "ml_trained": bool(pm.is_trained) if pm else False,
            "ml_confidence": round(float(pm.confidence), 3) if pm else 0.0,
            "ml_samples": len(pm.samples) if pm else 0,
        })

    return payload


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
            "score": round(float(c.overall_score), 3),
            "timestamp": c.timestamp_str,
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
    target_fps = max(1, state.config.analysis_fps)
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
    return FileResponse(str(target), media_type="image/jpeg")


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
    return FileResponse(str(STATIC_DIR / "index.html"))


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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
