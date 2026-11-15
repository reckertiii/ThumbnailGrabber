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
| `TZ` | `America/New_York` | Local timestamps on saved filenames |

### Memory & CPU Resources

Give the container **2–4 GB RAM** and **4+ CPUs**. Because the rolling DVR buffer stores frames as Q95 JPEG in RAM, a 5-second 1080p buffer at 30 FPS uses only ~130 MB (and releases to 0 MB automatically during Idle Standby).

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
