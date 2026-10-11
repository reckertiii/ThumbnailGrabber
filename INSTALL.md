# Installing on TrueNAS SCALE

Deploys the web UI as a container on TrueNAS, using the Ollama app already running on
the same box for the AI judge. The desktop app is not part of this — the container
ships without Qt.

Everything below assumes the NAS is at `192.168.32.104`. Substitute your own.

---

## 1. Before you start

You need:

- **TrueNAS SCALE** with the Apps service running.
- **The Ollama app**, with a vision model pulled. From its shell:
  ```bash
  ollama pull qwen3-vl:30b
  ```
  That is a 20 GB download and needs roughly 20 GB of RAM to run. `qwen3-vl:4b`
  (3.3 GB) works on a smaller box and judges less well.
- **The camera's RTSP URL**, reachable from the NAS. Check before deploying — the NAS
  and the camera are often on different VLANs:
  ```bash
  curl -v telnet://<camera-ip>:554
  ```
- **Note Ollama's port.** The catalog app usually remaps it to something in the 30000s
  rather than 11434. Confirm with:
  ```bash
  curl http://192.168.32.104:30068/api/tags
  ```
  It should list `qwen3-vl:30b`.

---

## 2. Create datasets

Two persistent locations are needed. Under **Datasets**, create:

| Dataset | Holds | Notes |
|---|---|---|
| `captures` | Saved screenshots and subject crops | Grows over time; put it where there is room |
| `tg-models` | Your ⭐/👎 ratings | Small. **Not** the ML weights — those are baked into the image |

The ML model weights (~17 MB) live at `/opt/tg-models` inside the image, deliberately
*not* on a mounted path. Mounting over them would hide them and force a re-download on
every start, which fails outright on a box with no outbound internet.

---

## 3. Deploy the container

**Apps → Discover Apps → Custom App.**

**Image**

| Field | Value |
|---|---|
| Repository | `ghcr.io/reckertiii/thumbnailgrabber` |
| Tag | `latest` |
| Pull policy | `IfNotPresent` — see §8 if you make the repo private |

**Port**

| Field | Value |
|---|---|
| Container port | `8000` |
| Node port | `9080` — TrueNAS requires 9000 or above; any free port in that range works |

**Storage**

| Host path | Mount path |
|---|---|
| your `captures` dataset | `/app/captures` |
| your `tg-models` dataset | `/app/models` |

**Environment variables**

| Name | Value | Why |
|---|---|---|
| `TG_STREAM_SOURCE` | `rtsp://user:pass@<camera-ip>:554/media/video1` | Your camera |
| `TG_OUTPUT_DIR` | `/app/captures` | Where captures land |
| `OLLAMA_HOST` | `http://192.168.32.104:30068` | The Ollama app. Use the NAS LAN IP, **not** `localhost` — inside a container that is the container itself |
| `TG_AI_JUDGE_OLLAMA_MODEL` | `qwen3-vl:30b` | Must match what you pulled |
| `TG_AI_JUDGE_OLLAMA_NUM_THREAD` | match the Ollama app's CPU limit | Ollama otherwise spawns one thread per host core, overruns its quota and is throttled — measured ~3× slower |
| `TG_AI_JUDGE_ACTIVE_HOURS` | `2` | Judge stops and releases the model after this |
| `TZ` | `America/New_York` | Capture filenames use local time |

Optional, worth setting if the port is reachable beyond your LAN — **the web UI has no
authentication**:

| Name | Value |
|---|---|
| `TG_VIDEO_ROOTS` | `/app/captures` — limits which directories the video scanner may read |
| `TG_OUTPUT_ROOTS` | `/app/captures` — limits where the output-folder picker may browse |

**Resources.** Give it 4 GB RAM and 4 CPUs to start. The frame buffer alone defaults to
about 0.9 GB — see §7.

Deploy.

---

## 4. First run

Open `http://192.168.32.104:9080`.

That is the node port from §3. The container still listens on 8000 internally;
only the outside port changed.

Check these three things in order:

1. **Stream status says Connected** with a frame rate. If not, the camera URL or network
   path is wrong — see §9.
2. **AI judge says `qwen3-vl:30b via local`** with minutes remaining. If it says "no
   Ollama at…", the host or port is wrong.
3. **Capture is refused until you confirm the output folder.** This is deliberate — on a
   hosted instance the default is whatever the image set, and a whole service written to
   the wrong place is only noticed afterwards.

   Click **Output Folder → Change Folder…**, navigate to `/app/captures`, and press
   **Use This Folder**. Choosing the folder already shown counts. Asked once per
   container start.

**Start Capture** is now available.

---

## 5. Verify it works end to end

With someone in frame, within a minute or two you should see in the app logs:

```
[Capture] Score=82% Sharp=0.94(141) Cutout=0.94 Subject=71%h …
[AIJudge] Kept (73% -> judge 90% = 82%, 31.2s): A strong frame with …
```

Each keeper writes two files to `/app/captures`: the full frame, and a `_subject.png`
crop framed to the speaker, ready to drop into a background remover.

If the gallery stays empty, the readout tells you why — the Score line ends with either
`✅ capturing` or `⏸ holding: <reason>`.

---

## 6. Reach the captures from Windows and Mac

The app writes to the NAS, because that is the only disk it has. To work on the
PNGs you share that dataset and mount it on each machine, where it behaves like any
other folder -- open a `_subject.png` straight from it into your background remover.

A browser cannot be given a folder on your own machine to write into, which is why
this is done at the filesystem level rather than in the app.

### Share it on TrueNAS

1. **Credentials -> Local Users.** Either note an existing user or add one for this.
   In a shell, `id <username>` prints the uid and gid.
2. **Set `user:` in the compose file to those numbers** (or the app's Run-As
   settings, if you deployed through the GUI). The container otherwise runs as root
   and writes files nobody else can delete -- you could open them from Windows but
   not clear them out. Redeploy after changing it.
3. **Datasets -> your `captures` dataset -> Edit Permissions.** Give that user
   Full Control, and apply recursively if captures already exist.
4. **Shares -> Windows (SMB) Shares -> Add.**

   | Field | Value |
   |---|---|
   | Path | your `captures` dataset |
   | Name | `tgcaptures` |
   | Purpose | Default share parameters |

   Save, and enable the SMB service when prompted.

### Mount it on Windows

File Explorer -> This PC -> **Map network drive**, pick a letter, and enter:

```
\\192.168.32.104\tgcaptures
```

Tick **Reconnect at sign-in** and **Connect using different credentials**, then give
the NAS username and password. Or from a terminal:

```
net use Z: \\192.168.32.104\tgcaptures /persistent:yes
```

### Mount it on Mac

Finder -> Go -> **Connect to Server** (Cmd-K):

```
smb://192.168.32.104/tgcaptures
```

Connect, authenticate, and tick **Remember this password in my keychain**. To have it
mount at login, drag the mounted volume into **System Settings -> General -> Login
Items**.

### Notes

- **Use the share, don't move it.** `TG_OUTPUT_DIR` must stay `/app/captures`. The
  gallery, the star and thumbs-down ratings, and the AI judge all read back from
  there, so pointing the app somewhere else to "save locally" breaks them.
- **Copy, don't cut, during a service.** The app keeps its own list of what it has
  saved; deleting files it is still ranking makes the gallery show gaps.
- **Both platforms, one share.** SMB is native to Windows and macOS, so the same
  share serves every machine with no per-machine browser setup.

---

## 7. Tune the frame buffer

The buffer keeps recent frames so a chosen moment can be re-examined at full frame rate
for its sharpest version. It is the largest allocation in the process, so size it
against the readout rather than a guess:

```
Buffer: 150/150 frames · 890 MB held · 2.5s covered · camera 60.4fps
```

**Watch the "covered" figure.** The buffer is sized in frames, so a camera running
faster than the configured rate gives less depth than asked for — a 60 fps camera with
the buffer set for 30 holds half the seconds. The readout names the value to set, and
raising it raises memory in proportion. Slider is in the UI; 0 turns the buffer off.

---

## 8. After you make the repo private

Repo visibility and **package** visibility are separate. A private repo publishes a
private image, and TrueNAS then needs a token to pull it.

Easiest fix: **GitHub → your profile → Packages → `thumbnailgrabber` → Package settings
→ change visibility to public.** Source stays private, pulls keep working, no token.

Alternatives: set the app's pull policy to `IfNotPresent` so it never re-pulls, or add a
GitHub personal access token as a registry credential in TrueNAS.

---

## 9. If something is wrong

**Stream says Disconnected.** Test the path from the NAS itself, not your desktop:
```bash
curl -v telnet://<camera-ip>:554
```
Camera and NAS are frequently on different VLANs.

**Judge says "no Ollama at…".** Check the port — the catalog app usually remaps 11434:
```bash
curl http://192.168.32.104:30068/api/tags
```

**Judge says "up but 'qwen3-vl:30b' is not pulled".** Exactly what it says; pull it in
the Ollama app's shell.

**Reviews take minutes.** Almost always `TG_AI_JUDGE_OLLAMA_NUM_THREAD` not matching the
Ollama app's CPU limit. Expect roughly 30 s per review on 20 cores.

**Judge says "asleep — 2.0h window elapsed".** Working as designed; it released the
model to free the RAM. Press **Wake** in the UI, or restart the app.

**Nothing captures though the score looks high.** Read the Score line's `⏸ holding:`
reason. Most often the output folder has not been confirmed since the last restart.

**The UI looks stale after an update.** It shouldn't — the server sends `no-cache` — but
a hard reload (Ctrl-Shift-R) settles it.

---

## 10. Updating

```bash
docker pull ghcr.io/reckertiii/thumbnailgrabber:latest
```
then restart the app. Captures and ratings survive — both are on mounted datasets.

Note that ratings are versioned against the scoring metrics. If an update changes what a
feature means, the old ratings are discarded on load with a message, because mixing them
would quietly skew every score. You then re-rate a dozen or so captures.
