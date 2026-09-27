"use strict";

const $ = (id) => document.getElementById(id);

// Tracks which controls the user is actively dragging, so incoming telemetry
// doesn't yank a slider out from under them.
const holding = new Set();
let isCapturing = false;
let sourceFocused = false;
let lastCapturesJSON = "";

// ---------------------------------------------------------------- helpers

async function post(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || res.statusText);
  }
  return res.json();
}

let toastTimer = null;
function toast(msg) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), 2600);
}

function setClass(el, cls, on) {
  el.classList.toggle(cls, !!on);
}

// ---------------------------------------------------------------- telemetry

const STATUS = {
  disconnected: ["#ef4444", "Disconnected"],
  connecting: ["#f59e0b", "Connecting…"],
  connected: ["#22c55e", "Connected"],
  reconnecting: ["#f59e0b", "Reconnecting"],
  failed: ["#ef4444", "Connection Failed"],
};

function render(d) {
  const [color, label] = STATUS[d.status] || ["#94a3b8", "Unknown"];
  $("statusDot").style.color = color;
  $("statusText").textContent =
    d.status === "reconnecting" ? `${label} (${d.reconnect_attempt})…` : label;

  $("fps").textContent = `${d.fps ?? 0} fps`;
  $("outputDir").textContent = d.output_dir || "—";
  $("scanOutputDir").textContent = d.output_dir || "—";
  if (!sourceFocused && $("src").value !== d.source) $("src").value = d.source || "";

  // Start / stop button
  isCapturing = !!d.is_capturing;
  const btn = $("btnStartStop");
  btn.textContent = isCapturing ? "⏹ Stop Capture" : "▶ Start Capture";
  setClass(btn, "btn-stop", isCapturing);
  setClass(btn, "btn-start", !isCapturing);
  $("chkAuto").checked = !!d.auto_capture;
  $("chkWinner").checked = !!d.single_winner;
  $("chkMesh").checked = !!d.show_mesh;
  $("chkHud").checked = !!d.show_hud;

  // Sliders (skip any the user is dragging)
  const s = d.settings || {};
  if (!holding.has("sldScore")) { $("sldScore").value = s.min_score; $("outScore").textContent = s.min_score + "%"; }
  if (!holding.has("sldYaw")) { $("sldYaw").value = s.max_yaw; $("outYaw").textContent = s.max_yaw + "°"; }
  if (!holding.has("sldShoulder")) { $("sldShoulder").value = s.max_shoulder_tilt; $("outShoulder").textContent = s.max_shoulder_tilt + "°"; }
  if (!holding.has("sldBurst")) { $("sldBurst").value = Math.round(s.burst_window * 10); $("outBurst").textContent = s.burst_window.toFixed(1) + "s"; }
  if (document.activeElement !== $("numPeople")) $("numPeople").value = s.people_target;
  $("outPeople").textContent = s.people_target === 1 ? "person" : "people";

  renderPeople(d.selected_people || [], s.people_target || 1);

  // Score bar
  const pct = Math.round((d.overall_score || 0) * 100);
  $("scoreVal").textContent = pct + "%";
  $("scoreBar").style.width = pct + "%";
  setClass($("scoreBar"), "qualified", d.is_qualified);

  if (d.face_detected) {
    $("mHead").textContent = `Head Yaw ${fmt(d.yaw)}° · Pitch ${fmt(d.pitch)}° · Roll ${fmt(d.roll)}°`;
    $("mEyes").textContent = `Eyes: ${d.is_blinking ? "CLOSED/BLINK" : "OPEN"} (EAR ${d.avg_ear.toFixed(2)})`;
    $("mExpr").textContent = `Expression: ${d.is_smiling ? "SMILE 😊" : d.is_awkward_expression ? "AWKWARD 😮" : "NEUTRAL"} (${d.smile_ratio.toFixed(2)})`;

    const sh = $("mShoulder");
    if (d.shoulder_detected) {
      sh.textContent = `Shoulders: ${d.is_shoulders_level ? "LEVEL ✅" : "TILTED ⚠️"} (${d.shoulder_tilt}°)`;
      setClass(sh, "good", d.is_shoulders_level);
      setClass(sh, "bad", !d.is_shoulders_level);
    } else {
      sh.textContent = "Shoulders: Headshot (out of frame)";
      sh.classList.remove("good", "bad");
    }

    $("mSharp").textContent = `Sharpness: ${d.sharpness} · EMA ${Math.round(d.smoothed_score * 100)}%`;
    const expIcon = d.brightness > 180 ? "☀️" : d.brightness < 60 ? "🌑" : "✅";
    $("mExposure").textContent = `Exposure: ${Math.round(d.exposure_score * 100)}% ${expIcon} (${d.brightness})`;
    const cutFlags = [d.is_subject_clipped ? "CLIPPED ⚠️" : null,
                      d.hands_on_torso ? "hands on body" : null].filter(Boolean).join(" · ");
    $("mCentering").textContent = `Cutout: ${Math.round(d.cutout_score * 100)}% · subject ${Math.round(d.subject_height_ratio * 100)}% of frame`
      + ` · bg ${Math.round(d.background_score * 100)}%` + (cutFlags ? ` · ${cutFlags}` : "");
    $("mHand").textContent = `Hand: ${d.hands_covering_face ? "OBSTRUCTING ⚠️" : d.is_hand_blurry ? "BLURRY 🔄" : "CLEAR ✅"}`;
    // Why nothing is saving. A high score with no captures otherwise looks like a
    // broken button rather than the scorer holding out for a better frame.
    const blockers = d.disqualified_reasons || [];
    const holdText = d.is_qualified
      ? "✅ capturing"
      : (blockers.length ? `⏸ holding: ${blockers.slice(0, 2).join(" · ")}` : "⏸ holding");
    $("mStability").textContent = `Score: ${d.is_stable ? "🟢 Stable" : "🟡 Unstable"} · People detected: ${d.people_detected} · ${holdText}`;
    $("mModel").textContent = d.ml_trained
      ? `AI Model: Trained (${Math.round(d.ml_confidence * 100)}% held-out accuracy · ${d.ml_samples} ratings · ML score ${Math.round(d.ml_score * 100)}%)`
      : "AI Model: Untrained — click ⭐ Best on a shot to train it";
  } else {
    ["mHead", "mEyes", "mExpr", "mShoulder", "mSharp", "mExposure", "mCentering", "mHand", "mStability"]
      .forEach((id) => { $(id).textContent = $(id).textContent.split(":")[0] + ": —"; });
    $("mShoulder").classList.remove("good", "bad");
  }

  if (d.captures) renderGallery(d.captures);
}

const fmt = (v) => (v >= 0 ? "+" : "") + v.toFixed(1);

function renderPeople(selected, target) {
  const wrap = $("people");
  const want = Math.max(target, 1);
  if (wrap.children.length !== want) {
    wrap.innerHTML = "";
    for (let pid = 1; pid <= want; pid++) {
      const b = document.createElement("button");
      b.className = "person-chip";
      b.textContent = "P" + pid;
      b.onclick = () => post(`/api/person/${pid}`, {}).catch((e) => toast(e.message));
      wrap.appendChild(b);
    }
  }
  [...wrap.children].forEach((b, i) => setClass(b, "on", selected.includes(i + 1)));
}

function renderGallery(caps) {
  const json = JSON.stringify(caps);
  if (json === lastCapturesJSON) return;  // avoid rebuilding / flicker
  lastCapturesJSON = json;

  const g = $("gallery");
  if (!caps.length) {
    g.innerHTML = '<p class="muted">No captures yet — press Start Capture.</p>';
    return;
  }
  g.innerHTML = "";
  caps.forEach((c) => {
    const card = document.createElement("div");
    card.className = "thumb" + (c.rank === 1 ? " rank1" : "");

    const img = document.createElement("img");
    img.src = c.url;
    img.alt = `Capture scoring ${Math.round(c.score * 100)} percent`;
    img.onclick = () => openLightbox(c);

    const info = document.createElement("div");
    info.className = "thumb-info";
    info.innerHTML = `<span class="rank-badge">🏆 #${c.rank}</span> | <b>${Math.round(c.score * 100)}%</b>`;

    const btns = document.createElement("div");
    btns.className = "thumb-btns";
    const fav = document.createElement("button");
    fav.className = "btn-fav";
    fav.textContent = "⭐ Best";
    fav.onclick = () => feedback(c.filename, true);
    const rej = document.createElement("button");
    rej.className = "btn-rej";
    rej.textContent = "👎 Bad";
    rej.onclick = () => feedback(c.filename, false);
    btns.append(fav, rej);

    card.append(img, info, btns);
    g.appendChild(card);
  });
}

async function feedback(filename, favorite) {
  try {
    const r = await post(`/api/captures/${encodeURIComponent(filename)}/feedback`, { favorite });
    toast(`${favorite ? "⭐ Learned as a winner" : "👎 Learned to penalize"} — ${r.samples} ratings` + (r.confidence ? `, ${Math.round(r.confidence * 100)}% held-out accuracy` : ", not enough ratings to train yet"));
  } catch (e) {
    toast("Training failed: " + e.message);
  }
}

function openLightbox(c) {
  $("lbImg").src = c.url;
  const shoulder = c.shoulder_detected ? `${c.shoulder_tilt}°` : "headshot / out of frame";
  $("lbInfo").innerHTML = `
    <b>Score:</b> ${Math.round(c.score * 100)}%<br>
    <b>Head yaw:</b> ${c.yaw}°<br>
    <b>Shoulder tilt:</b> ${shoulder}<br>
    <b>Eye aspect ratio:</b> ${c.avg_ear}<br>
    <b>Smile ratio:</b> ${c.smile_ratio}<br>
    <b>Sharpness:</b> ${c.sharpness}<br>
    <b>Cutout suitability:</b> ${Math.round((c.cutout_score || 0) * 100)}%<br>` +
    (c.judge_score != null
      ? `<b>AI judge:</b> ${Math.round(c.judge_score * 100)}% (analyzer ${Math.round((c.heuristic_score || 0) * 100)}%)<br>`
        + `<i>${c.judge_summary || ""}</i><br>`
        + ((c.judge_issues || []).length ? `<b>Flagged:</b> ${c.judge_issues.join("; ")}<br>` : "")
      : "") + `
    <b>File:</b> ${c.filename}` +
    (c.crop_url ? `<br><a href="${c.crop_url}" target="_blank" rel="noopener">Open subject crop ↗</a>` : "");
  $("lightbox").classList.remove("hidden");
}

// ---------------------------------------------------------------- websocket

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (ev) => {
    try { render(JSON.parse(ev.data)); } catch (e) { /* ignore malformed frame */ }
  };
  ws.onclose = () => {
    $("statusDot").style.color = "#ef4444";
    $("statusText").textContent = "Server offline";
    setTimeout(connect, 2000);
  };
}

// ---------------------------------------------------------------- wiring

function slider(id, outId, key, format) {
  const el = $(id);
  el.addEventListener("pointerdown", () => holding.add(id));
  el.addEventListener("input", () => {
    $(outId).textContent = format(+el.value);
  });
  const commit = () => {
    holding.delete(id);
    post("/api/settings", { [key]: key === "burst_window" ? +el.value / 10 : +el.value })
      .catch((e) => toast(e.message));
  };
  el.addEventListener("change", commit);
  el.addEventListener("pointerup", () => holding.delete(id));
}

slider("sldScore", "outScore", "min_score", (v) => v + "%");
slider("sldYaw", "outYaw", "max_yaw", (v) => v + "°");
slider("sldShoulder", "outShoulder", "max_shoulder_tilt", (v) => v + "°");
slider("sldBurst", "outBurst", "burst_window", (v) => (v / 10).toFixed(1) + "s");

$("numPeople").addEventListener("change", (e) =>
  post("/api/settings", { people_target: +e.target.value }).catch((err) => toast(err.message)));

$("btnStartStop").onclick = () =>
  post("/api/capture/toggle", { enabled: !isCapturing }).catch((e) => toast(e.message));

$("btnManual").onclick = async () => {
  try {
    await post("/api/capture/manual", {});
    toast("📸 Screenshot saved");
  } catch (e) {
    toast("Capture failed: " + e.message);
  }
};

$("btnSource").onclick = () => {
  const source = $("src").value.trim();
  if (!source) return toast("Enter a source first");
  post("/api/source", { source })
    .then(() => toast("Reconnecting to new source…"))
    .catch((e) => toast(e.message));
};

$("src").addEventListener("focus", () => { sourceFocused = true; });
$("src").addEventListener("blur", () => { sourceFocused = false; });

$("chkAuto").onchange = (e) => post("/api/capture/toggle", { enabled: e.target.checked }).catch(() => {});
$("chkMesh").onchange = (e) => post("/api/view", { show_mesh: e.target.checked }).catch(() => {});
$("chkHud").onchange = (e) => post("/api/view", { show_hud: e.target.checked }).catch(() => {});
$("chkWinner").onchange = (e) => post("/api/view", { single_winner: e.target.checked }).catch(() => {});

// Click a face on the video to include / ignore that person
$("video").onclick = (ev) => {
  const r = ev.target.getBoundingClientRect();
  post("/api/click", {
    x_frac: (ev.clientX - r.left) / r.width,
    y_frac: (ev.clientY - r.top) / r.height,
  }).catch(() => {});
};

// ---------------------------------------------------------------- video scanning modal

let scanPollTimer = null;
let isScanningVideo = false;

function openScanModal() {
  $("videoScanModal").classList.remove("hidden");
}

function closeScanModal() {
  $("videoScanModal").classList.add("hidden");
}

$("btnOpenScanModal").onclick = openScanModal;
$("btnModalCloseX").onclick = closeScanModal;
$("btnScanClose").onclick = closeScanModal;

$("scanMinScore").oninput = (e) => { $("outScanScore").textContent = e.target.value + "%"; };
$("scanInterval").oninput = (e) => { $("outScanInterval").textContent = (e.target.value / 10).toFixed(1) + "s"; };

$("btnCheckPath").onclick = async () => {
  const path = $("scanVideoPath").value.trim();
  if (!path) return toast("Enter a video file path first");
  try {
    const r = await post("/api/video/info", { path });
    const meta = $("scanVideoMeta");
    meta.innerHTML = `✓ <b>${r.filename}</b> — Duration: <b>${r.duration_str}</b> (${r.duration_sec}s) | Resolution: <b>${r.width}x${r.height}</b> | FPS: <b>${r.fps}</b> | Total Frames: <b>${r.total_frames.toLocaleString()}</b>`;
    meta.classList.remove("hidden");
  } catch (e) {
    toast("Inspect failed: " + e.message);
  }
};

$("btnUploadVideo").onclick = async () => {
  const fileInput = $("videoFileInput");
  if (!fileInput.files || !fileInput.files[0]) {
    return toast("Please select a video file first");
  }
  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);

  $("btnUploadVideo").disabled = true;
  $("btnUploadVideo").textContent = "Uploading…";
  try {
    const res = await fetch("/api/video/upload", {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    $("scanVideoPath").value = data.path;
    const meta = $("scanVideoMeta");
    meta.innerHTML = `✓ <b>${data.filename}</b> (Uploaded) — Duration: <b>${data.duration_str}</b> | Resolution: <b>${data.width}x${data.height}</b> | FPS: <b>${data.fps}</b>`;
    meta.classList.remove("hidden");
    toast("Video uploaded successfully!");
  } catch (e) {
    toast("Upload failed: " + e.message);
  } finally {
    $("btnUploadVideo").disabled = false;
    $("btnUploadVideo").textContent = "⬆ Upload & Scan";
  }
};

async function toggleWebScan() {
  if (isScanningVideo) {
    // Cancel
    try {
      await post("/api/video/scan/cancel", {});
      toast("Cancelling video scan…");
    } catch (e) {
      toast("Cancel failed: " + e.message);
    }
  } else {
    // Start scan
    const path = $("scanVideoPath").value.trim();
    if (!path) return toast("Specify or upload a video file first");

    const top_n = +$("scanTopN").value || 5;
    const sample_fps = +$("scanSpeed").value;
    const min_score = (+$("scanMinScore").value || 55) / 100;
    const min_interval_sec = (+$("scanInterval").value || 20) / 10;

    try {
      await post("/api/video/scan", {
        path,
        top_n,
        sample_fps,
        min_score,
        min_interval_sec,
      });

      isScanningVideo = true;
      $("btnStartWebScan").textContent = "⏹ Stop Scan";
      $("btnStartWebScan").classList.remove("btn-purple");
      $("btnStartWebScan").classList.add("btn-stop");
      $("scanProgBar").style.width = "0%";
      $("scanStatusText").textContent = "Initializing scanner…";
      startScanPolling();
    } catch (e) {
      toast("Start scan failed: " + e.message);
    }
  }
}

$("btnStartWebScan").onclick = toggleWebScan;

function startScanPolling() {
  clearInterval(scanPollTimer);
  scanPollTimer = setInterval(async () => {
    try {
      const res = await fetch("/api/video/scan/status");
      if (!res.ok) return;
      const data = await res.json();

      $("scanProgBar").style.width = Math.min(100, data.progress_pct) + "%";

      const bestText = data.best_score > 0 ? `Best Score: ${Math.round(data.best_score * 100)}%` : "No qualified face yet";
      $("scanStatusText").innerHTML = `Status: <b>${data.status}</b> | Progress: <b>${data.current_time_str} / ${data.total_duration_str}</b> (${data.progress_pct}%) | Speed: <b>${data.fps_speed} FPS</b> | ETA: <b>${data.eta_sec}s</b> | ${bestText}`;

      // Update current frame preview
      if (data.current_frame_preview) {
        $("scanCurrentFrame").src = data.current_frame_preview;
        $("scanCurrentFrame").classList.remove("hidden");
        $("scanCurrentPlaceholder").classList.add("hidden");
      }

      // Update champion preview
      if (data.champion_frame_preview) {
        $("scanChampFrame").src = data.champion_frame_preview;
        $("scanChampFrame").classList.remove("hidden");
        $("scanChampPlaceholder").classList.add("hidden");
      }

      if (!data.is_scanning) {
        clearInterval(scanPollTimer);
        isScanningVideo = false;
        $("btnStartWebScan").textContent = "▶ Start Scan";
        $("btnStartWebScan").classList.remove("btn-stop");
        $("btnStartWebScan").classList.add("btn-purple");

        if (data.status === "completed") {
          $("scanProgBar").style.width = "100%";
          const count = data.results_count || 0;
          toast(`✓ Scan Complete! Picked ${count} best thumbnails.`);
          $("scanStatusText").innerHTML = `✓ <b>Completed!</b> Extracted <b>${count}</b> best thumbnails into output folder.`;
          
          // Refresh main gallery
          fetch("/api/captures").then(r => r.json()).then(d => {
            if (d.captures) renderGallery(d.captures);
          }).catch(() => {});
        } else if (data.status === "cancelled") {
          toast("Video scan was cancelled.");
        } else if (data.error) {
          toast("Scan error: " + data.error);
        }
      }
    } catch (e) {
      // ignore network glitch
    }
  }, 350);
}


// ---------------------------------------------------------------- folder picker
// Captures are written by the server, so a native file dialog in the browser would
// pick a path on the wrong machine. The server lists its own directories instead and
// this walks them.

let folderCurrent = "";

async function loadFolder(path) {
  try {
    const res = await fetch("/api/browse?path=" + encodeURIComponent(path || ""));
    if (!res.ok) {
      const d = await res.json().catch(() => ({}));
      throw new Error(d.detail || res.statusText);
    }
    const d = await res.json();
    folderCurrent = d.path;
    $("folderPath").value = d.path;

    const sc = $("folderShortcuts");
    sc.innerHTML = "";
    [...(d.drives || []), ...(d.shortcuts || [])].forEach((q) => {
      const b = document.createElement("button");
      b.textContent = q.name;
      b.title = q.path;
      b.onclick = () => loadFolder(q.path);
      sc.appendChild(b);
    });

    const list = $("folderList");
    list.innerHTML = "";
    if (d.parent) {
      const up = document.createElement("button");
      up.className = "up";
      up.textContent = "\u2191 ..";
      up.onclick = () => loadFolder(d.parent);
      list.appendChild(up);
    }
    if (!d.entries.length) {
      const e = document.createElement("div");
      e.className = "empty";
      e.textContent = "No subfolders here \u2014 you can still use this folder.";
      list.appendChild(e);
    }
    d.entries.forEach((en) => {
      const b = document.createElement("button");
      b.textContent = "\u{1F4C1} " + en.name;
      b.onclick = () => loadFolder(en.path);
      list.appendChild(b);
    });

    $("folderNote").textContent = d.is_current
      ? "This is the current output folder."
      : (d.writable ? "Writable." : "\u26a0 Not writable by the server.");
  } catch (e) {
    $("folderNote").textContent = "\u26a0 " + e.message;
  }
}

function openFolderModal() {
  $("folderModal").classList.remove("hidden");
  $("folderCreate").checked = false;
  loadFolder("");
}
function closeFolderModal() { $("folderModal").classList.add("hidden"); }

$("btnPickFolder").onclick = openFolderModal;
$("btnScanPickFolder").onclick = openFolderModal;
$("btnFolderCloseX").onclick = closeFolderModal;
$("btnFolderClose").onclick = closeFolderModal;
$("btnFolderGo").onclick = () => loadFolder($("folderPath").value.trim());
$("folderPath").onkeydown = (e) => { if (e.key === "Enter") loadFolder($("folderPath").value.trim()); };

$("btnFolderUse").onclick = async () => {
  const path = $("folderPath").value.trim() || folderCurrent;
  try {
    const d = await post("/api/output", { path, create: $("folderCreate").checked });
    $("outputDir").textContent = d.output_dir;
    $("scanOutputDir").textContent = d.output_dir;
    toast("Saving captures to " + d.output_dir);
    closeFolderModal();
  } catch (e) {
    $("folderNote").textContent = "\u26a0 " + e.message;
  }
};

$("lbClose").onclick = () => $("lightbox").classList.add("hidden");
$("lightbox").onclick = (e) => { if (e.target.id === "lightbox") $("lightbox").classList.add("hidden"); };
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    $("lightbox").classList.add("hidden");
    $("videoScanModal").classList.add("hidden");
    $("folderModal").classList.add("hidden");
  }
});

connect();
