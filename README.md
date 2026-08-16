# Thumbnail Grabber

Watches a live camera feed, scores every frame for headshot quality, and automatically
saves the best moments. Built for a Sony SRG-A40 over RTSP, but works with any RTSP
stream, a USB webcam, or an MP4 file.

Scoring combines head pose (yaw/pitch/roll via `solvePnP`), eye openness, expression,
shoulder level, sharpness, exposure, and centering — plus an optional preference model
that learns which shots you actually like.

Two front-ends share one engine:

- **Web UI** — runs on the machine near the camera, drive it from any browser on the LAN.
- **Desktop UI** — the original PyQt6 app.

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your camera:

```
TG_STREAM_SOURCE=rtsp://user:password@192.168.50.93:554/media/video1
TG_OUTPUT_DIR=C:\Users\you\Pictures\ThumbnailGrabber
```

`.env` is gitignored — keep camera credentials there, never in `config.py`.
`TG_STREAM_SOURCE` also accepts a webcam index (`0`) or a path to an MP4.

Model weights (YuNet face detector, MediaPipe hand + pose landmarkers) download
automatically into `models/` on first run.

## Running the web UI

```bash
python -m web.server
```

Then open `http://localhost:8000`. To reach it from other devices on your network,
use the host machine's LAN IP — e.g. `http://192.168.50.20:8000`.

```bash
python -m web.server --port 8000 --host 0.0.0.0 --source 0
```

The browser gets everything the desktop app has: live annotated video, the full
telemetry readout, threshold sliders, per-person filtering (click a face to include
or ignore them), the ranked gallery, and ⭐/👎 buttons that train the preference model.

The preview is downscaled to 1280px wide by default. On a phone over weak Wi-Fi,
`http://<host>:8000/?w=640` is lighter — analysis and saved screenshots always stay
at full source resolution.

### Web endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | The UI |
| `GET /video.mjpg?w=1280&q=80` | Annotated MJPEG preview |
| `WS /ws` | Telemetry ~10×/sec |
| `GET /api/state` | Full state snapshot |
| `POST /api/source` | Switch stream source |
| `POST /api/capture/toggle` | Start / stop auto-capture |
| `POST /api/capture/manual` | Save current frame |
| `POST /api/settings` | Thresholds, burst window, people target |
| `POST /api/view` | Mesh / HUD / single-winner toggles |
| `POST /api/person/{id}` | Include or ignore a tracked person |
| `POST /api/click` | Toggle a person by clicking the video |
| `GET /api/captures` | Ranked gallery |
| `POST /api/captures/{file}/feedback` | Train the preference model |

## Running with Docker (Recommended for 24/7 Headless Server)

The entire application can be run in a lightweight Docker container with persistent volumes for captured screenshots and trained machine learning preference models:

### 1. Configure `.env`
Ensure your `.env` file contains your stream URL:
```env
TG_STREAM_SOURCE=rtsp://user:password@192.168.50.93:554/media/video1
```

### 2. Start with Docker Compose
```bash
docker compose up -d
```
Then open `http://localhost:8000` (or `http://<host-ip>:8000` on your LAN).

To view container logs or stop:
```bash
# View live logs
docker compose logs -f

# Stop the server
docker compose down
```

### 3. Build & Run Manually with Docker CLI
```bash
# Build the image
docker build -t thumbnail-grabber:latest .

# Run container with volume mounts
docker run -d \
  --name thumbnail-grabber \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/captures:/app/captures" \
  -v "$(pwd)/models:/app/models" \
  thumbnail-grabber:latest
```

## Running the desktop UI

```bash
python main.py
```

Optional overrides: `python main.py --source 0 --output D:\shots`

## Layout

| File | Role |
|---|---|
| `analyzer.py` | Frame scoring engine — detection, landmarks, pose, metrics |
| `stream_processor.py` | Grabber + analysis threads, burst peak picking, capture rules |
| `overlay.py` | Shared HUD renderer used by both front-ends |
| `preference_model.py` | scikit-learn model trained on ⭐/👎 feedback |
| `config.py` | Weights, thresholds, env-backed settings |
| `web/server.py` | FastAPI app |
| `web/static/` | Browser UI |
| `gui.py`, `main.py` | PyQt6 desktop app |

## Notes

- **Single Winner Mode deletes files.** With it on, anything outside the top
  `max_retained_captures` (default 5) is removed from disk as better shots arrive.
  Turn it off in the UI if you want to keep everything.
- **One camera per server process.** The FFMPEG capture options are set process-wide,
  and the analyzer keeps per-stream state, so run one process per camera.
- Each open browser tab gets its own MJPEG encode. Fine for a handful of viewers on a LAN.
