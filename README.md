# Thumbnail Grabber

Watches a live camera feed, scores every frame for headshot quality, and automatically
saves the best moments. Built for a Sony SRG-A40 over RTSP, but works with any RTSP
stream, a USB webcam, or an MP4 file.

Scoring is tuned for one job: producing frames whose background you can strip so the
speaker can be dropped onto a slide. It combines head pose (yaw/pitch/roll via
`solvePnP`), eye openness, expression, shoulder level, sharpness, exposure, and a
**cutout suitability** score — is the whole subject inside the frame, are they big
enough to enlarge, are their hands clear of their torso, and is the background behind
them plain enough to key cleanly. An optional preference model learns which shots you
actually pick.

Captures are written as **PNG** by default, alongside a **subject-tight crop**
(`..._subject.png`) ready to drop into a background remover. JPEG is available via
`capture_format` in `config.py`, but its compression artifacts land on hair and
shoulder edges — exactly where matting goes wrong.

Two front-ends share one engine:

- **Web UI** — runs on the machine near the camera, drive it from any browser on the LAN.
- **Desktop UI** — the original PyQt6 app.

## Install

Needs Python 3.10–3.13 and `ffmpeg` on PATH (for RTSP). The container image pins 3.11.

```bash
git clone https://github.com/reckertiii/ThumbnailGrabber.git
cd ThumbnailGrabber
python -m venv .venv
```

Activate it — `.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on
Linux/macOS — then:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your camera:

```
TG_STREAM_SOURCE=rtsp://user:password@192.168.1.100:554/media/video1
TG_OUTPUT_DIR=C:\Users\you\Pictures\ThumbnailGrabber
```

`.env` is gitignored — keep camera credentials there, never in `config.py`.
`TG_STREAM_SOURCE` also accepts a webcam index (`0`) or a path to an MP4.

Other settings, all optional — see `.env.example`:

| Variable | Purpose |
|---|---|
| `TG_VIDEO_ROOTS` | Restrict which directories the web UI may scan videos from |
| `TG_MAX_UPLOAD_MB` | Cap on web upload size (default 4096) |
| `TG_OUTPUT_ROOTS` | Restrict which directories the output-folder picker may browse |
| `TG_MODELS_DIR` | Where ML weights live (set by the Docker image) |
| `TG_DATA_DIR` | Where your ⭐/👎 ratings persist |
| `ANTHROPIC_API_KEY` | AI judge via the Claude API (see below) |
| `OLLAMA_HOST` | AI judge via a self-hosted Ollama vision model instead |
| `TG_AI_JUDGE_BACKEND` | `auto` (default), `anthropic`, `ollama`, or `off` |
| `TG_AI_JUDGE_ACTIVE_HOURS` | Hours the judge stays active after startup (default 2, 0 = always) |

Model weights (YuNet face detector, MediaPipe hand + pose landmarkers) download
automatically into `models/` on first run — about 17 MB, and the first start is slower
because of it.

### First run

```bash
python -m web.server
```

Open `http://localhost:8000`, press **Start Capture**, and point the camera at someone.
Captures land in the output folder, which you can change from the UI at any time
(**Output Folder → Change Folder…**) — including to a network share.

## AI judge (optional)

The heuristics score individual signals — jaw open, eyes shut, hands over the torso —
and each can pass while the photo as a whole is still wrong: arms folded, a finger
pointed at the floor, a face caught between expressions. The judge is a vision model
that looks at the finished crop the way a person picking a slide image does.

Set `ANTHROPIC_API_KEY` in `.env` and it turns on; leave it out and it stays off with
one line in the log saying so. It only reviews a candidate that has *already* cleared
the heuristics and would improve the kept set, so a service costs a few dozen calls,
not one per frame — roughly a cent each on the default `claude-opus-5`. The verdict is
blended into the capture's score (`ai_judge_weight`, default 50/50), a frame it calls
unusable is dropped, and every kept capture shows the judge's one-line summary and
flagged issues in the gallery. `ai_judge_max_calls_per_session` (default 300) is a hard
cost guard. For file scans it reviews the finalists only.

Live review runs on its own thread, so analysis never waits on the network; a failed
call keeps the heuristic score and carries on.

### Active window

The judge only runs for **2 hours after startup** (`TG_AI_JUDGE_ACTIVE_HOURS`). When
that elapses it stops reviewing and tells Ollama to unload the model, releasing ~20 GB
on the host — a machine left running all week shouldn't hold a vision model in memory
for a service that happens on Sunday. Captures carry on as normal on the heuristics
alone.

The web UI shows the countdown next to the AI model readout, with **Wake** to re-arm
for another full window (useful if you started early or the service ran long) and
**Sleep now** to release the model immediately. Stopping the app releases it too. Set
`TG_AI_JUDGE_ACTIVE_HOURS=0` for no window.

### Self-hosted judge

If you'd rather nothing leave your network — or just not pay per call — the judge can
use a local [Ollama](https://ollama.com) vision model instead. Install Ollama and pull
a model:

```bash
ollama pull qwen3-vl:4b
```

Then leave `ANTHROPIC_API_KEY` unset (or set `TG_AI_JUDGE_BACKEND=ollama`). With
`auto`, the Claude API is used when a key is present and Ollama otherwise, so you can
have both configured and switch by adding or removing the key.

Which model, set with `TG_AI_JUDGE_OLLAMA_MODEL`:

| Model | RAM | CPU speed | Notes |
|---|---|---|---|
| `qwen3-vl:4b` | ~4 GB | fast | the default; fine for a modest box |
| `qwen3-vl:8b` | ~6 GB | moderate | better judgment |
| `qwen3-vl:30b` | ~20 GB | **about as fast as the 4b** | mixture-of-experts: only ~3B parameters are active per token, so it runs like a small model and judges like a large one. The pick if you have the RAM. |
| `qwen3-vl:32b` | ~21 GB | slow | dense; only worth it with a GPU |

Avoid the `-thinking` tags for this — they reason out loud before answering, which is
slower and fights the JSON output. If RAM is dedicated to Ollama, set
`TG_AI_JUDGE_OLLAMA_KEEP_ALIVE=-1` so the model stays loaded between reviews instead
of paying a cold load at the start of each service.

**If Ollama runs in a container with a CPU limit** (a TrueNAS app, Docker with `--cpus`),
set `TG_AI_JUDGE_OLLAMA_NUM_THREAD` to that limit. Ollama otherwise spawns one thread per
physical core on the *host*, overruns the quota, and gets throttled — measured at 3×
slower. And give it real cores: the expensive part of a review is encoding the image,
which is compute-bound and scales with cores (unlike text generation, which is
memory-bound). On a 44-core box, 20–24 cores with the thread count matched gives about
30 s per review on `qwen3-vl:30b`. Slow is fine: a review that's still
running when the next candidate appears just means that candidate is skipped, not
queued, so a slow model reviews fewer frames and never stalls capture.

Honest expectation: the local model is good at the concrete things — mouth open, eyes
shut, arms folded, something across the body — and weaker than Claude at the finer
"does this read as awkward" call. For most services that's still a real improvement
over heuristics alone.

In Docker, either point `OLLAMA_HOST` at the TrueNAS Ollama app, or start the bundled
one:

```bash
docker compose --profile local-ai up -d
docker compose exec ollama ollama pull qwen3-vl:4b
```

## Running the web UI

```bash
python -m web.server
```

Then open `http://localhost:8000`. To reach it from other devices on your network,
use the host machine's LAN IP — e.g. `http://192.168.1.50:8000`.

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
| `GET /api/browse` | List server directories for the output-folder picker |
| `POST /api/output` | Set the output folder |
| `POST /api/judge/wake` | Re-arm the AI judge for another active window |
| `POST /api/judge/sleep` | End the window now and release the model |
| `POST /api/captures/{file}/feedback` | Train the preference model |
| `POST /api/video/info` | Inspect metadata of a video recording file |
| `POST /api/video/upload` | Upload a video recording for scanning |
| `POST /api/video/scan` | Start offline video recording scan |
| `GET /api/video/scan/status` | Live scan progress, ETA, and preview frame |
| `POST /api/video/scan/cancel` | Cancel an active video scan |

## Video Recording Scanning (Offline / File Mode)

In addition to live streams, Thumbnail Grabber can scan through entire pre-recorded video files (`.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`) and automatically extract the highest-quality thumbnails using **temporal Non-Maximum Suppression (NMS)**:

- **Desktop UI**: Click **"📼 Scan Video File..."** on the top bar to pick a video, configure top thumbnail count and sampling rate, and watch real-time dual preview and progress.
- **Web UI**: Click **"📼 Scan Video File"** to enter a server path (or upload a video) and monitor scanning progress in the browser.
- **CLI / Batch Mode**:
  ```bash
  python main.py --scan-video /path/to/recording.mp4 --top 5 --headless
  ```

## Running with Docker (Recommended for 24/7 Headless Server)

The entire application can be run in a lightweight Docker container with persistent volumes for captured screenshots and trained machine learning preference models:

### 1. Configure `.env`
Ensure your `.env` file contains your stream URL:
```env
TG_STREAM_SOURCE=rtsp://user:password@192.168.1.100:554/media/video1
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
| `ai_judge.py` | Optional vision-model review of candidate captures |
| `web/server.py` | FastAPI app |
| `web/static/` | Browser UI |
| `gui.py`, `main.py` | PyQt6 desktop app |

## Notes

- **Single Winner Mode deletes files.** With it on, anything outside the top
  `max_retained_captures` (default 5) is removed from disk as better shots arrive,
  including each shot's companion `_subject` crop. Turn it off in the UI if you want
  to keep everything.
- **The web UI has no authentication.** It is designed for a trusted LAN. Uploaded
  filenames are sanitised and scan output is confined to the output directory, but if
  the port is reachable more widely, set `TG_VIDEO_ROOTS` to limit what can be read.
- **Rating history is versioned.** The scoring metrics changed meaning in feature set
  v2, so ⭐/👎 ratings recorded before it are discarded on load rather than silently
  skewing scores. Rate a dozen or so captures to retrain.
- **Analysis and preview run at different rates.** Scoring a frame costs ~77 ms, so
  `analysis_fps` defaults to 8 while `preview_fps` defaults to 15 — the preview shows
  the newest camera frame with the most recent overlay drawn on it, so the video stays
  smooth even though scoring is slower. `pose_hand_interval` (default 2) re-detects
  hands and shoulders every Nth analysed frame; they move slowly enough that this
  halves their cost with no measurable change in output.
- **Capturing keeps going all service.** A new shot is saved when it would improve the
  kept set — beat the weakest keeper, or beat a nearby one it would replace. Keepers are
  spread at least `min_capture_spacing_sec` (20s) apart so five variations of one moment
  don't crowd out the rest of the service.
- **A high score doesn't always mean a capture.** The HUD and the web readout list what
  is holding it back — eyes closed, head turned, still settling. Only genuinely
  unusable frames are rejected outright; merely imperfect ones are scored down and left
  in the running, so a speaker who talks continuously still yields captures.
- **One camera per server process.** The FFMPEG capture options are set process-wide,
  and the analyzer keeps per-stream state, so run one process per camera.
- Each open browser tab gets its own MJPEG encode. Fine for a handful of viewers on a LAN.
