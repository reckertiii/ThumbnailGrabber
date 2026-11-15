# Thumbnail Grabber

Watches a live camera feed (or scans pre-recorded video files), scores every frame for speaker thumbnail and slide-cutout quality, and automatically saves the best moments. Built for a Sony SRG-A40 over RTSP on a 24/7 Docker/TrueNAS server, and works with any RTSP stream, USB webcam, or recorded video file.

Scoring is tuned for producing crisp, engaged speaker frames whose background can be stripped cleanly for presentation slides or YouTube thumbnails. It combines:

- **3D Head Pose** (yaw/pitch/roll from MediaPipe's 4×4 facial transformation matrix)
- **Eye Openness & Horizontal Eye-Gaze** (blink suppression + conjugate `eyeLookIn/Out` blendshape scoring to catch sideways monitor/confidence-screen glances even when the head faces forward)
- **Expression & Temporal Mouth Velocity** (AU12 genuine smile vs mid-syllable speech distortion, tracking frame-to-frame `ΔjawOpen`, `ΔmouthPucker`, and `ΔmouthFunnel` to favor natural speech pauses)
- **Shoulder Level & Stage Prop / Candid Detection** (detects held props/objects without falsely penalizing torso hands, reserving dedicated prop slots)
- **Sharpness & Per-Channel RGB Exposure** (Laplacian focus + single-channel RGB highlight blowout and heavy stage LED color-wash detection)
- **Cutout Suitability** (framing headroom, torso clearance from lecterns/music stands, and background separability)
- **Expression & Temporal Diversity** (guarantees spread-out captures across a service while reserving a smile slot alongside serious/authoritative teaching poses)
- **Adaptive Preference Model & Optional Vision AI Judge** (learns from ⭐/👎 ratings + natural-language comments, with optional Claude or local Ollama vision verification)

Captures are written as **fast lossless PNG** by default (`capture_format = "png"`), preserving crisp hair and shoulder edges for background matting.

Two interfaces share one engine:

- **Web UI** (`python main.py` or Docker) — drive live capture, camera discovery, folder switching, gallery pinning, ZIP export, and offline video scanning from any browser on your LAN.
- **Headless CLI Scanner** (`python main.py --scan-video ...`) — batch-extract top thumbnails from recorded videos with optional `--start-sec` / `--end-sec` trimming.

## Install (Local Python)

Needs Python 3.10–3.13 and `ffmpeg` on `PATH` (for RTSP). The Docker image pins Python 3.11 with all system libraries and ML weights pre-baked.

```bash
git clone https://github.com/reckertiii/ThumbnailGrabber.git
cd ThumbnailGrabber
python -m venv .venv
```

Activate the virtual environment (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on Linux/macOS), then:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` (optional — you can also configure the camera directly from the Web UI):

```env
TG_STREAM_SOURCE=rtsp://user:password@192.168.1.100:554/media/video1
TG_OUTPUT_DIR=C:\Users\you\Pictures\ThumbnailGrabber
```

### Environment Variables

| Variable | Purpose |
|---|---|
| `TG_STREAM_SOURCE` | Initial stream URL, webcam index (`0`), file path, or `""` to configure in the Web UI |
| `TG_OUTPUT_DIR` | Default directory where captures are written |
| `TG_OUTPUT_ROOTS` | Restrict which directories the Web UI folder picker may browse (`:` on Linux, `;` on Windows) |
| `TG_VIDEO_ROOTS` | Restrict which directories the Web UI may browse/read video files from |
| `TG_MAX_UPLOAD_MB` | Cap on browser video upload size in MB (default `4096`) |
| `TG_WEB_PASSWORD` | Optional HTTP Basic Auth password for the Web UI & API (leave empty on trusted LAN; `/api/health` stays open for Docker healthchecks) |
| `TG_CAPTURE_MAX_MINUTES` | Auto-stop capture timer in minutes so capture never runs all week if forgotten (default `120`, `0` = unlimited) |
| `TG_IDLE_STANDBY` | Zero-CPU Idle Standby (`1` = default): pauses CV inference & frees DVR RAM when not capturing, not scanning, and no browser tab is open |
| `TG_MODELS_DIR` | Read-only directory for baked MediaPipe/YuNet weights (set by Docker image) |
| `TG_DATA_DIR` | Persistent directory for saved camera config, UI settings, folder gallery/pin manifests, and ⭐/👎 ML ratings |
| `ANTHROPIC_API_KEY` | Enables the AI Judge via the Anthropic Claude API |
| `TG_AI_JUDGE_MODEL` | Anthropic model ID (default `claude-sonnet-4-6`) |
| `OLLAMA_HOST` | Self-hosted Ollama server URL (e.g. `http://192.168.1.50:11434`) |
| `TG_AI_JUDGE_BACKEND` | `auto` (default), `anthropic`, `ollama`, or `off` |
| `TG_AI_JUDGE_OLLAMA_MODEL` | Ollama vision model tag (default `qwen3-vl:4b`; `qwen3-vl:30b` recommended on 24GB+ RAM) |
| `TG_AI_JUDGE_OLLAMA_NUM_THREAD` | Match to Ollama's container CPU limit to prevent throttling (`0` = auto) |
| `TG_AI_JUDGE_ACTIVE_HOURS` | Hours the AI Judge stays armed per session before unloading weights (default `2`, `0` = always) |

## Running the Web UI

```bash
python main.py
# or with overrides:
python main.py --port 8000 --host 0.0.0.0 --source 0
```

Open `http://localhost:8000` (or `http://<lan-ip>:8000` from any laptop, tablet, or phone on your network).

### Key Operational Features

- **Smart Start & Auto-Stop Capture Timer**: Clicking **▶ Start Capture** automatically wakes the AI Judge (and preloads Ollama weights if needed) for the duration of your capture window (`⏱ Max: 1h / 1.5h / 2h / 3h / 4h / No limit`), showing a live countdown on the Stop button and putting the AI Judge back to sleep automatically when the timer elapses.
- **Zero-CPU Idle Standby & Powered-Off Camera Standby**: Between services, when no browser tab is open and capture is stopped, MediaPipe inference pauses and the DVR RAM buffer is released (`0%` CPU), waking in `<50 ms` the moment you open the Web UI or start capture. If the PTZ camera is powered off on a PoE switch, RTSP enters quiet TCP socket standby polling instead of failing permanently.
- **Persistent Settings & Folder-Scoped Galleries**: Threshold sliders, view toggles, confirmed output folder, and per-folder gallery cards/scores/📌 Pinned flags persist in `TG_DATA_DIR` across container restarts and Watchtower updates. Switching folders in **Output Folder → Change Folder…** cleanly loads each campus/service folder's own gallery and pins.
- **One-Click Downloads & `.zip` Export**: Download individual full-resolution PNGs from any gallery card (`⬇`) or lightbox, or click **`⬇ Download ZIP`** to grab all visible gallery thumbnails at once.

### Web API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | Web UI |
| `GET /api/health` | Container healthcheck & standby/capture status |
| `GET /video.mjpg?w=1280&q=80` | Shared async MJPEG annotated preview stream |
| `WS /ws` | Live 10 Hz telemetry & gallery sync |
| `GET /api/state` | Full telemetry + gallery snapshot |
| `GET /api/camera`, `POST /api/camera/test`, `POST /api/camera/connect`, `POST /api/camera/forget` | Guided RTSP/webcam/file camera setup, stream-path discovery, and credential persistence |
| `POST /api/source` | Switch stream source directly |
| `POST /api/capture/toggle` | Start / stop auto-capture (with optional `max_minutes` window) |
| `POST /api/capture/manual` | Save current frame immediately |
| `POST /api/settings` | Update thresholds, burst window, tracked people, max retained, capture timer |
| `POST /api/view` | Toggle mesh, HUD, Top-N retention, and stage-prop capture |
| `POST /api/person/{id}` / `POST /api/click` | Include or ignore specific onstage people |
| `POST /api/buffer` | Resize the Q95 JPEG rolling DVR frame buffer |
| `POST /api/judge/wake` / `POST /api/judge/sleep` | Manually wake or unload the AI Judge |
| `GET /api/browse` / `POST /api/output` | Browse server folders & switch active output directory |
| `GET /api/browse/videos` | Browse server video recordings inside `TG_VIDEO_ROOTS` |
| `GET /api/captures` | Ranked gallery list |
| `GET /api/captures/{file}/download` | Download a single full-resolution capture |
| `GET /api/captures/download-zip` | Download visible/pinned gallery captures as a `.zip` archive |
| `POST /api/captures/{file}/pin` | Toggle 📌 Pinned protection against Top-N pruning |
| `POST /api/captures/{file}/feedback` | Submit ⭐/👎 rating + comment rules to train the preference model |
| `POST /api/captures/clear` | Clear unpinned shots between services while keeping 📌 Saved shots |
| `POST /api/video/info`, `POST /api/video/upload`, `POST /api/video/scan`, `GET /api/video/scan/status`, `POST /api/video/scan/cancel` | Offline video recording upload, server browsing, start/end trim scanning, and live preview |

## Offline Video Recording Scanner

Scan full sermon or event recordings (`.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, `.ts`) with an early-exit MediaPipe gate (`fast_reject`) that skips hand/pose/cutout passes on disqualified frames for ~2–3× faster CPU throughput:

- **Web UI**: Click **`📼 Scan Video File`** → **`📂 Browse Server…`** (or upload a video with live progress bar), optionally set **Start Time** / **End Time** (`MM:SS` or `HH:MM:SS`) to skip worship/announcements, choose an independent scan output folder if desired, and click **`▶ Start Scan`**. Refreshing the browser tab automatically reconnects to an active background scan.
- **CLI Mode**:
  ```bash
  python main.py --scan-video /path/to/recording.mp4 --top 5 --start-sec 20:00 --end-sec 58:30
  ```

## AI Judge (Optional)

Set `ANTHROPIC_API_KEY` (uses `claude-sonnet-4-6` by default) or point `OLLAMA_HOST` at a local [Ollama](https://ollama.com) instance with a vision model pulled (`qwen3-vl:4b`, `qwen3-vl:8b`, or `qwen3-vl:30b`).

- Reviews only candidates that have already cleared heuristic gates and would improve the kept gallery.
- Automatically wakes when you press **Start Capture** or **Start Scan**, and unloads when the capture timer finishes (or when you click **Sleep now**).

## Running with Docker (Recommended for 24/7 Server / TrueNAS)

See **[INSTALL.md](INSTALL.md)** for full TrueNAS SCALE setup, SMB sharing, and dataset mounting.

```bash
docker compose up -d
```

Open `http://localhost:8000` (or `http://<server-ip>:8000`). Watchtower (`nickfedor/watchtower`) automatically keeps the container up to date when new builds are published to GHCR, and all camera settings, UI sliders, output folder confirmation, and gallery pins survive container updates via `TG_DATA_DIR`.

## Rolling DVR Frame Buffer

Analysis runs at `analysis_fps` (default 8 FPS) while cameras stream at 30–60 FPS. A rolling DVR buffer stores recent full-resolution frames compressed as **Q95 JPEG in RAM** (~85% smaller than raw BGR — 5 seconds of 1080p at 30 FPS uses only ~120–160 MB instead of ~930 MB). When a peak moment triggers capture, the sharpest un-blinking frame in the surrounding window is decoded and re-scored so you always get the crispest micro-expression.

## Project Layout

| File | Role |
|---|---|
| `analyzer.py` | YuNet + MediaPipe Face/Hand/Pose scoring, eye-gaze, mouth velocity, RGB exposure, cutout suitability |
| `stream_processor.py` | RTSP grabber + standby polling, Q95 JPEG DVR buffer, peak burst picker, diversity retention |
| `video_scanner.py` | Offline video recording scanner with `fast_reject`, start/end time trimming, and diversity NMS |
| `ai_judge.py` | Optional Claude / Ollama vision-model reviewer with active-window lifecycle |
| `preference_model.py` | Online scikit-learn SGD preference learner + natural-language comment penalty rules |
| `settings_store.py` | Atomic persistence for UI settings and folder-scoped gallery/pin manifests in `TG_DATA_DIR` |
| `camera_probe.py`, `camera_store.py` | RTSP path discovery, diagnostics, and credential storage |
| `overlay.py` | Annotated HUD & face-mesh renderer for MJPEG preview |
| `config.py` | Weights, thresholds, and environment configuration |
| `web/server.py`, `web/static/` | FastAPI server, shared async MJPEG broadcaster, WebSocket telemetry, and browser UI |
| `main.py` | Entrypoint for the Web UI server and cross-platform CLI `--scan-video` |

## Licence

Proprietary — copyright (c) 2026 Robert Eckert, all rights reserved. See `LICENSE`.
Third-party dependencies carry their own permissive licences; see `THIRD-PARTY-NOTICES.md`.
