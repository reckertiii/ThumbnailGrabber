# Installing on TrueNAS SCALE

Deploys Thumbnail Grabber as a 24/7 container on TrueNAS SCALE, pairing with the TrueNAS Ollama app for local vision AI judging and exposing captured PNGs over SMB ( plus one-click `.zip` downloads directly in the browser).

Everything below assumes the NAS is at `192.168.32.104` with ZFS pool `Main` (`/mnt/Main`). Substitute your own IP and pool name.

---

## 1. Before You Start

You need:

- **TrueNAS SCALE** with the Apps service running.
- **The Ollama app** (optional, for local AI Judge), with a vision model pulled from its shell:
  ```bash
  ollama pull qwen3-vl:30b
  ```
  `qwen3-vl:30b` (MoE, ~20 GB RAM) runs fast on CPU with strong judgment; `qwen3-vl:4b` (~4 GB RAM) works well on smaller boxes.
- **Network reachability to your PTZ/RTSP camera.** Verify from a TrueNAS shell (especially if cameras sit on an AV VLAN):
  ```bash
  curl -v telnet://<camera-ip>:554
  ```
- **Confirm Ollama's node port** (usually in the `30000`s on TrueNAS):
  ```bash
  curl http://192.168.32.104:30068/api/tags
  ```

---

## 2. Create Datasets

Create persistent datasets under your pool (for example `/mnt/Main/apps`):

| Path | Holds | Notes |
|---|---|---|
| `/mnt/Main/apps/capture` (or campus shares) | Saved full-resolution PNG screenshots | Share over SMB or download via `.zip` in the Web UI |
| `/mnt/Main/apps/tg-models` | Saved camera credentials (`camera.json`), UI settings, per-folder gallery/pin manifests, and ⭐/👎 ML ratings | Small, survives every container update. (**Not** the ~17 MB MediaPipe weights — those are pre-baked at `/opt/tg-models` inside the image) |

---

## 3. Deploy the Container

Use **[deploy/truenas-compose.yml](deploy/truenas-compose.yml)** via **Apps → Discover Apps → Custom App → Install via YAML**, or configure the fields directly:

- **Image**: `ghcr.io/reckertiii/thumbnailgrabber:latest`
- **Container Port**: `8000` → **Host Port**: `9080` (TrueNAS requires `>= 9000`)
- **Bind Mounts** (mount each ZFS dataset at the **identical path** inside the container so paths match in the Web UI, TrueNAS, and SMB):
  - `/mnt/Main/apps:/mnt/Main/apps`
  - Add any additional campus/media datasets both under `volumes:` and in `TG_OUTPUT_ROOTS` / `TG_VIDEO_ROOTS`.

> **Important:** ZFS paths are **case-sensitive** (`/mnt/Main` vs `/mnt/main`), and Linux bind mounts are **not recursive** — mount each child ZFS dataset individually rather than mounting `/mnt/Main` alone. Check dataset mountpoints with `zfs list -o name,mountpoint`.

### Recommended Environment Variables

| Name | Example Value | Purpose |
|---|---|---|
| `TG_STREAM_SOURCE` | `""` | Leave blank to configure the camera cleanly in the Web UI without storing passwords in YAML |
| `TG_OUTPUT_DIR` | `/mnt/Main/apps/capture` | Default capture folder |
| `TG_DATA_DIR` | `/mnt/Main/apps/tg-models` | Persists camera config, UI settings, confirmed folder, gallery pins, and ML ratings |
| `TG_OUTPUT_ROOTS` | `/mnt/Main/apps:/mnt/Main/Grace Shared` | Colon-separated allowlist for the output folder picker |
| `TG_VIDEO_ROOTS` | `/mnt/Main/apps:/mnt/Main/Grace Shared` | Colon-separated allowlist for the server video browser & scanner |
| `TG_CAPTURE_MAX_MINUTES` | `120` | Automatically stops capture after 2 hours (adjustable in the top bar) so capture never runs all week if forgotten |
| `TG_IDLE_STANDBY` | `1` | Drops CPU to ~0% and frees DVR RAM when not capturing and no browser is open |
| `TG_WEB_PASSWORD` | `""` | Optional HTTP Basic Auth password if exposed beyond a trusted LAN |
| `OLLAMA_HOST` | `http://192.168.32.104:30068` | NAS LAN IP + Ollama port (never `localhost`) |
| `TG_AI_JUDGE_BACKEND` | `ollama` | Uses local Ollama vision model |
| `TG_AI_JUDGE_OLLAMA_MODEL` | `qwen3-vl:30b` | Must match pulled Ollama tag |
| `TG_AI_JUDGE_OLLAMA_NUM_THREAD` | `20` | Match your Ollama app's CPU core limit to avoid throttling |
| `TG_AI_JUDGE_ACTIVE_HOURS` | `2` | Unloads the Ollama model when idle |
| `TG_AI_JUDGE_OLLAMA_KEEP_ALIVE` | `-1` | Keeps `qwen3-vl:30b` pinned in RAM during active sessions on high-memory hosts |
| `TG_DVR_BUFFER_SECONDS` | `60` | Rolling Time-Machine RAM replay buffer (adjustable up to `180s` in the Web UI; click `⏪ Mine Last 60s` anytime) |
| `TG_MOOD_BUCKETS` | `1` | Guarantees expression diversity across smile, passionate, calm, and prop slots |
| `TG_BROADCAST_POLISH` | `1` | Strips H.264 macroblock artifacts and neutralizes harsh magenta/blue stage LED skin casts |
| `TG_SCAN_WORKERS` | `0` | Multi-worker parallel offline video scanning across CPU cores |
| `TZ` | `America/New_York` | Local timestamps on saved filenames |

### Memory & CPU Resources

Give the container **4–24 GB RAM** (depending on your Time-Machine buffer window and parallel video scan worker count) and **4–16 CPUs**. On a high-memory TrueNAS host (e.g. 500 GB RAM), setting `memory: 24G` in `deploy/truenas-compose.yml` easily accommodates a 60–180s 1080p DVR replay buffer (~1.5–4.5 GB Q95 JPEG), multi-worker parallel video scanning, `±15`-frame micro-scrub burst caches, and optional ONNX / `faster-whisper` models stored in `TG_DATA_DIR` (all DVR RAM still drops to 0 MB automatically during Idle Standby).

### Optional Studio Add-Ons (`TG_DATA_DIR`)

- **Neural Matting, Embeddings & Speaker Lock**: Click **Warm Studio Models** (or trigger your first cutout / speaker lock) to cache Apache-2.0 ONNX weights (`u2netp.onnx`, `vision_backbone_mnv2.onnx`, `face_recognition_sface_2021dec.onnx`) directly into `/mnt/Main/apps/tg-models` so they persist across container updates without bloating the base Docker image.
- **NDI Network Streams**: To ingest NDI sources alongside RTSP, drop `libndi.so.5` / `libndi.so.6` from the NewTek/Vizrt NDI SDK into `/mnt/Main/apps/tg-models/ndi/` and use `host` networking (or multicast routing) for mDNS discovery in **`📷 Camera… → NDI Network Source`**.
- **Sony SRG-A40 PTZ & AI Auto-Framing**: When connected to your Sony SRG-A40, open the **Sony SRG-A40 PTZ & AI Auto-Framing** drawer under the live monitor to nudge Pan/Tilt/Zoom, recall camera presets `1–6`, or toggle the camera's built-in AI Auto-Framing over HTTP CGI.

---

## 4. First Run

Open `http://192.168.32.104:9080`:

1. **Connect the Camera**: Click **`📷 Camera…`**, enter the camera's IP, username, and password, leave **Stream path** blank, and press **Test** — the app probes standard RTSP paths (including Sony SRG `/media/video1`), fills in the working path, and saves credentials to `TG_DATA_DIR/camera.json` (`0600`) when you click **Connect**.
2. **Confirm Output Folder**: Click **`Output Folder → Change Folder…`**, choose your target dataset folder (e.g. `/mnt/Main/apps/capture`), and click **Use This Folder**. Your confirmation is saved to `TG_DATA_DIR/ui_settings.json` so container updates won't force re-confirmation.
3. **Choose Auto-Stop Window & Start Capture**: Pick your desired capture timer in the top bar (default `⏱ Max: 2h`) and click **`▶ Start Capture`**. This automatically wakes the AI Judge, preloads the Ollama model in the background, counts down on the Stop button, and stops + unloads automatically when time expires.

---

## 5. Offline Video Scanning on TrueNAS

To pull thumbnails from a recorded service MP4/MKV already on a NAS share:

1. Click **`📼 Scan Video File`** → **`📂 Browse Server…`** to pick any video inside `TG_VIDEO_ROOTS` without typing Linux paths (or upload a file from your laptop with a live progress bar).
2. Optionally enter **Start Time** (e.g. `22:00` to skip worship/announcements) and **End Time** (`58:00`).
3. Optionally pick a separate **Scan Output Folder** without disturbing your live capture folder.
4. Click **`▶ Start Scan`**.

---

## 6. Accessing Captures (Browser `.zip` or SMB Share)

- **Directly in Browser**: Click **`⬇`** on any gallery card to download that PNG, or **`⬇ Download ZIP`** in the gallery header to download all top shots in one archive.
- **Over SMB (Windows / macOS)**:
  1. Set `user: "<uid>:<gid>"` in `truenas-compose.yml` matching your SMB dataset owner (`id <username>`) so Windows/macOS users can delete or rename files over SMB.
  2. Share the dataset under **Shares → Windows (SMB) Shares** and mount `\\192.168.32.104\<share>` on Windows or `smb://192.168.32.104/<share>` on macOS.

---

## 7. Troubleshooting

- **`Camera Standby (Power off)`**: Normal when your PTZ camera or PoE switch is powered off between services! The server quietly checks TCP port 554 every 15 seconds and reconnects automatically as soon as the camera powers on.
- **`0 fps · 💤 Idle Standby`**: Normal when capture is stopped and no browser had the feed open — waking happens automatically within milliseconds of viewing or starting capture.
- **Output Folder Warning ("not on a mounted volume")**: Means the chosen path is inside the container's ephemeral filesystem rather than a mounted ZFS dataset (often caused by `/mnt/main` vs `/mnt/Main` casing or mounting a parent pool instead of the child dataset).
- **Reviews take several minutes**: Set `TG_AI_JUDGE_OLLAMA_NUM_THREAD` to match the CPU limit configured on your TrueNAS Ollama app.
