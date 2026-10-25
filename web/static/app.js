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
    const payload = await res.json().catch(() => ({}));
    const d = payload.detail;
    // FastAPI's detail can be a string or an object. An object passed straight
    // to Error() stringifies to "[object Object]", so pull the message out and
    // keep the rest on the error for callers that want the hint.
    const msg = typeof d === "string" ? d
      : (d && d.message) ? d.message
      : res.statusText || "Request failed (" + res.status + ")";
    const err = new Error(msg);
    if (d && typeof d === "object") err.detail = d;
    throw err;
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

  // Measured, not estimated from the configured resolution - so the number stays
  // honest and a NAS deployment can be tuned against what is really held.
  const buf = d.buffer;
  if (buf) {
    if (!holding.has("sldBuffer")) {
      $("sldBuffer").value = Math.round(buf.seconds);
      $("outBuffer").textContent = Math.round(buf.seconds) + "s";
    }
    $("mBuffer").textContent = buf.enabled
      ? `Buffer: ${buf.frames_held}/${buf.frames_capacity} frames · `
        + `${fmtMB(buf.mb_held)} held · ${fmtMB(buf.mb_capacity)} when full`
        + (buf.span_seconds ? ` · ${buf.span_seconds.toFixed(1)}s covered` : "")
        + (buf.measured_fps ? ` · camera ${buf.measured_fps}fps` : "")
        + (buf.seconds_short
            ? ` ⚠ only ${buf.span_seconds.toFixed(1)}s of the ${buf.seconds}s asked for`
              + ` — set buffer fps to ${Math.ceil(buf.measured_fps)}`
            : "")
      : "Buffer: off (no frame history, captures use the sampled frame)";
  }

  // Outside the face-detected branch: the judge's countdown has to keep ticking
  // while the stage is empty, which is exactly when you'd check how long is left.
  const judgeMins = d.ai_judge_minutes_left;
  $("mJudge").childNodes[0].nodeValue = d.ai_judge_active
    ? `AI judge: active${judgeMins == null ? "" : ` · ${judgeMins} min left`} `
    : `${d.ai_judge || "AI judge: off"} `;
  // The server reports the source with its password masked, which must never be
  // typed back into the box -- pressing Connect would send "***" as the password
  // and the camera would refuse it. A masked source goes in the placeholder, so
  // the operator can see what is connected while the box stays empty for a real
  // URL. Anything without credentials (a webcam index, a file path) still fills in.
  if (!sourceFocused) {
    const src = d.source || "";
    const masked = src.includes("://***@");
    if (masked) {
      $("src").placeholder = src;
      if ($("src").value.includes("://***@")) $("src").value = "";
    } else {
      $("src").placeholder = "rtsp://… , 0, or file path";
      if ($("src").value !== src) $("src").value = src;
    }
  }

  // Start / stop button
  isCapturing = !!d.is_capturing;
  const btn = $("btnStartStop");
  btn.textContent = isCapturing ? "⏹ Stop Capture" : "▶ Start Capture";
  setClass(btn, "btn-stop", isCapturing);
  setClass(btn, "btn-start", !isCapturing);
  // Disabled rather than failing on click, so the reason is visible up front instead
  // of arriving as an error after you have already pressed it.
  const needsFolder = !d.output_confirmed && !isCapturing;
  btn.disabled = needsFolder;
  btn.title = needsFolder ? "Confirm the output folder first" : "";
  $("btnManual").disabled = !d.output_confirmed;
  setClass($("outputCard"), "needs-attention", !d.output_confirmed);
  $("outputHint").textContent = d.output_confirmed
    ? ""
    : "Confirm where captures should go before starting — pick this folder or another.";
  $("chkAuto").checked = !!d.auto_capture;
  $("chkAuto").disabled = needsFolder;
  $("chkAuto").title = needsFolder ? "Confirm the output folder first" : "";
  $("chkWinner").checked = !!d.single_winner;
  $("chkMesh").checked = !!d.show_mesh;
  $("chkHud").checked = !!d.show_hud;
  if ($("chkProps")) $("chkProps").checked = d.capture_prop_moments !== false;

  // Sliders (skip any the user is dragging)
  const s = d.settings || {};
  if (!holding.has("sldScore")) { $("sldScore").value = s.min_score; $("outScore").textContent = s.min_score + "%"; }
  if (!holding.has("sldYaw")) { $("sldYaw").value = s.max_yaw; $("outYaw").textContent = s.max_yaw + "°"; }
  if (!holding.has("sldShoulder")) { $("sldShoulder").value = s.max_shoulder_tilt; $("outShoulder").textContent = s.max_shoulder_tilt + "°"; }
  if (!holding.has("sldBurst")) { $("sldBurst").value = Math.round(s.burst_window * 10); $("outBurst").textContent = s.burst_window.toFixed(1) + "s"; }
  if (document.activeElement !== $("numPeople")) $("numPeople").value = s.people_target;
  $("outPeople").textContent = s.people_target === 1 ? "person" : "people";
  if ($("numRetained") && document.activeElement !== $("numRetained") && s.max_retained != null) {
    $("numRetained").value = s.max_retained;
  }

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
    $("mHand").textContent = `Hand: ${d.hands_covering_face ? "OBSTRUCTING ⚠️" : d.is_hand_blurry ? "BLURRY 🔄" : d.is_prop_moment ? "🎭 STAGE PROP / CANDID" : "CLEAR ✅"}`;
    // Why nothing is saving. A high score with no captures otherwise looks like a
    // broken button rather than the scorer holding out for a better frame.
    const blockers = d.disqualified_reasons || [];
    const holdText = d.is_qualified
      ? "✅ capturing"
      : (blockers.length ? `⏸ holding: ${blockers.slice(0, 2).join(" · ")}` : "⏸ holding");
    $("mStability").textContent = `Score: ${d.is_stable ? "🟢 Stable" : "🟡 Unstable"} · People detected: ${d.people_detected} · ${holdText}`;
    const notesCount = d.ml_notes_count || 0;
    const notesBadge = notesCount > 0 ? ` · ${notesCount} comment rule${notesCount === 1 ? "" : "s"} active` : "";
    $("mModel").textContent = d.ml_trained
      ? `AI Model: Trained (${Math.round(d.ml_confidence * 100)}% balanced accuracy · ${d.ml_samples} ratings${notesBadge} · ML score ${Math.round(d.ml_score * 100)}%)`
      : (notesCount > 0
          ? `AI Model: Learning (${notesCount} comment rule${notesCount === 1 ? "" : "s"} active · ${d.ml_samples} ratings)`
          : "AI Model: Untrained — click ⭐ Best or 👎 Bad on a shot to teach it");
  } else {
    ["mHead", "mEyes", "mExpr", "mShoulder", "mSharp", "mExposure", "mCentering", "mHand", "mStability"]
      .forEach((id) => { $(id).textContent = $(id).textContent.split(":")[0] + ": —"; });
    $("mShoulder").classList.remove("good", "bad");
  }

  if (d.captures) renderGallery(d.captures);
}

const fmt = (v) => (v >= 0 ? "+" : "") + v.toFixed(1);
const fmtMB = (mb) => (mb >= 1024 ? (mb / 1024).toFixed(2) + " GB" : Math.round(mb) + " MB");

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
    card.className = "thumb" + (c.rank === 1 ? " rank1" : "") + (c.is_pinned ? " pinned" : "");

    const img = document.createElement("img");
    img.src = c.url;
    img.alt = `Capture scoring ${Math.round(c.score * 100)} percent`;
    img.onclick = () => openLightbox(c, false);

    const info = document.createElement("div");
    info.className = "thumb-info";
    const propBadge = c.is_prop_moment ? ` · <span title="Candid stage prop moment">🎭 Prop</span>` : "";
    const pinBadge = c.is_pinned ? ` · <span class="pin-badge" title="Protected from auto-deletion">📌 Saved</span>` : "";
    info.innerHTML = `<span class="rank-badge">🏆 #${c.rank}</span> | <b>${Math.round(c.score * 100)}%</b>${propBadge}${pinBadge}`;

    const btns = document.createElement("div");
    btns.className = "thumb-btns";

    const pin = document.createElement("button");
    pin.className = "btn-pin" + (c.is_pinned ? " on" : "");
    pin.textContent = c.is_pinned ? "📌 Saved" : "📌 Keep";
    pin.title = c.is_pinned
      ? "Protected from auto-deletion (click to unpin)"
      : "Flag this shot so it is never deleted when new Top-5 shots arrive";
    pin.onclick = () => pinCapture(c.filename);

    const fav = document.createElement("button");
    fav.className = "btn-fav";
    fav.textContent = "⭐ Best";
    fav.title = "Train AI as a winner and protect from deletion";
    fav.onclick = () => feedback(c.filename, true);

    const rej = document.createElement("button");
    rej.className = "btn-rej";
    rej.textContent = "👎 Bad…";
    rej.title = "Explain why this photo is bad so the AI learns to reject shots like it";
    rej.onclick = () => openLightbox(c, true);

    btns.append(pin, fav, rej);

    card.append(img, info, btns);
    g.appendChild(card);
  });
}

let currentLightboxCapture = null;

async function pinCapture(filename, desired = null) {
  try {
    const body = desired === null ? {} : { pinned: !!desired };
    const r = await post(`/api/captures/${encodeURIComponent(filename)}/pin`, body);
    if (r.captures) {
      renderGallery(r.captures);
    }
    if (currentLightboxCapture && currentLightboxCapture.filename === filename) {
      currentLightboxCapture.is_pinned = r.is_pinned;
      syncLightboxPinButton(r.is_pinned);
    }
    toast(r.is_pinned
      ? "📌 Flagged & protected — this photo will never be auto-deleted"
      : "📌 Unpinned — returned to normal Top-5 rotation");
  } catch (e) {
    toast("Pin failed: " + e.message);
  }
}

function syncLightboxPinButton(pinned) {
  const btn = $("lbBtnPin");
  if (!btn) return;
  btn.textContent = pinned ? "📌 Saved (Protected)" : "📌 Keep (Protect)";
  btn.classList.toggle("on", !!pinned);
}

async function clearCaptures(mode) {
  const isPinnedMode = mode === "pinned";
  const promptMsg = isPinnedMode
    ? "Delete all 📌 Saved (pinned) shots from the gallery and output folder?"
    : "Clear all unpinned top shots from the gallery and output folder to get ready for the next service?\n\n(Any 📌 Saved shots will be kept safe.)";
  if (!window.confirm(promptMsg)) return;
  try {
    const r = await post("/api/captures/clear", { mode });
    if (r.captures) {
      renderGallery(r.captures);
    }
    if (isPinnedMode) {
      toast(`🗑️ Deleted ${r.deleted_count} pinned shot${r.deleted_count === 1 ? "" : "s"} from folder`);
    } else {
      const keptNote = r.pinned_remaining ? ` (${r.pinned_remaining} 📌 Saved kept)` : "";
      toast(`🧹 Cleared ${r.deleted_count} unpinned shot${r.deleted_count === 1 ? "" : "s"} — ready for next service!${keptNote}`);
    }
  } catch (e) {
    toast("Clear failed: " + e.message);
  }
}

if ($("btnClearUnpinned")) {
  $("btnClearUnpinned").onclick = () => clearCaptures("unpinned");
}
if ($("btnClearPinned")) {
  $("btnClearPinned").onclick = () => clearCaptures("pinned");
}

async function feedback(filename, favorite, opts = {}) {
  try {
    const payload = {
      favorite,
      comment: opts.comment || "",
      reasons: opts.reasons || [],
      remove: !!opts.remove,
    };
    const r = await post(`/api/captures/${encodeURIComponent(filename)}/feedback`, payload);
    if (r.captures) {
      renderGallery(r.captures);
    }
    const rulesPart = r.notes_count ? ` · ${r.notes_count} comment rule${r.notes_count === 1 ? "" : "s"} active` : "";
    const accPart = r.confidence ? `, ${Math.round(r.confidence * 100)}% accuracy` : "";
    toast(
      `${favorite ? "⭐ Learned as winner & pinned safe" : (r.removed ? "👎 Learned & deleted bad shot" : "👎 Learned to penalize")}`
      + ` — ${r.samples} ratings${accPart}${rulesPart}`
    );
  } catch (e) {
    toast("Training failed: " + e.message);
  }
}

function getSelectedLightboxReasons() {
  const chips = document.querySelectorAll("#lbReasonChips .reason-chip.selected");
  return [...chips].map((el) => el.getAttribute("data-reason") || el.textContent.trim());
}

function openLightbox(c, focusBad = false) {
  currentLightboxCapture = c;
  $("lbImg").src = c.url;
  const shoulder = c.shoulder_detected ? `${c.shoulder_tilt}°` : "headshot / out of frame";
  const pinLine = c.is_pinned ? " · <b>📌 Protected from auto-deletion</b>" : "";
  $("lbInfo").innerHTML = `
    <b>Score:</b> ${Math.round(c.score * 100)}%${c.is_prop_moment ? " (🎭 Candid Stage Prop Moment)" : ""}${pinLine}<br>
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

  syncLightboxPinButton(!!c.is_pinned);

  // Reset and pre-highlight any reason chips that match AI judge issues
  const issueBlob = ((c.judge_issues || []).join(" ") + " " + (c.judge_summary || "")).toLowerCase();
  document.querySelectorAll("#lbReasonChips .reason-chip").forEach((chip) => {
    const r = (chip.getAttribute("data-reason") || "").toLowerCase();
    let autoSelect = false;
    if (focusBad) {
      if (r.includes("head bowed") && (issueBlob.includes("bowed") || issueBlob.includes("looking down") || issueBlob.includes("gaze"))) autoSelect = true;
      if (r.includes("eyes") && (issueBlob.includes("eye") || issueBlob.includes("blink"))) autoSelect = true;
      if (r.includes("mouth") && (issueBlob.includes("mouth") || issueBlob.includes("mid-word"))) autoSelect = true;
      if (r.includes("mic") && (issueBlob.includes("microphone") || issueBlob.includes("hand"))) autoSelect = true;
      if (r.includes("prop") && c.is_prop_moment) autoSelect = true;
      if (r.includes("blur") && issueBlob.includes("blur")) autoSelect = true;
    }
    chip.classList.toggle("selected", autoSelect);
  });

  if ($("lbComment")) {
    $("lbComment").value = "";
  }
  $("lightbox").classList.remove("hidden");
  if (focusBad && $("lbComment")) {
    setTimeout(() => $("lbComment").focus(), 40);
  }
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

$("chkAuto").onchange = (e) => post("/api/capture/toggle", { enabled: e.target.checked }).catch((err) => {
  e.target.checked = !e.target.checked;
  toast(err.message);
});
$("chkMesh").onchange = (e) => post("/api/view", { show_mesh: e.target.checked }).catch(() => {});
$("chkHud").onchange = (e) => post("/api/view", { show_hud: e.target.checked }).catch(() => {});
$("chkWinner").onchange = (e) => post("/api/view", { single_winner: e.target.checked }).catch(() => {});
if ($("numRetained")) {
  $("numRetained").onchange = (e) => {
    const val = Math.max(1, Math.min(200, parseInt(e.target.value, 10) || 5));
    e.target.value = val;
    post("/api/settings", { max_retained: val })
      .then(() => toast(`🏆 Keeping top ${val} rolling shot${val === 1 ? "" : "s"} (plus any 📌 Saved shots)`))
      .catch((err) => toast(err.message));
  };
}
if ($("chkProps")) {
  $("chkProps").onchange = (e) => post("/api/view", { capture_prop_moments: e.target.checked }).catch((err) => toast(err.message));
}

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
      const mins = Math.floor((data.eta_sec || 0) / 60);
      const secs = (data.eta_sec || 0) % 60;
      const etaFmt = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
      const mult = (data.total_duration_sec > 0 && data.total_frames > 0 && data.fps_speed > 0)
        ? ` (${(data.fps_speed / (data.total_frames / data.total_duration_sec)).toFixed(1)}x)` : "";
      if (data.status === "reviewing" && data.error) {
        $("scanStatusText").innerHTML = `🧠 <b>${data.error}</b> | Candidates Analyzed: <b>${data.candidates_analyzed}</b> | ${bestText}`;
      } else {
        $("scanStatusText").innerHTML = `Status: <b>${data.status}</b> | Progress: <b>${data.current_time_str} / ${data.total_duration_str}</b> (${data.progress_pct}%) | Speed: <b>${data.fps_speed} FPS${mult}</b> | ETA: <b>${etaFmt}</b> | ${bestText}`;
      }

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
    if (d.warning) {
      // Writable is not persistent. Keep the dialog open and say so, rather than
      // reporting success for a folder whose contents vanish on the next update.
      $("folderNote").textContent = "⚠ " + d.warning;
      toast("Folder set, but it is not on a mounted volume");
      return;
    }
    toast("Saving captures to " + d.output_dir);
    closeFolderModal();
  } catch (e) {
    $("folderNote").textContent = "\u26a0 " + e.message;
  }
};

// The judge only runs for a window after startup, then releases the model on the
// Ollama host. These re-arm or end that window without restarting the app.
// Slider is live while dragging but only committed on release, so a drag from 5s to
// 20s doesn't reallocate the buffer at every step in between.
$("sldBuffer").oninput = () => {
  holding.add("sldBuffer");
  $("outBuffer").textContent = $("sldBuffer").value + "s";
};
$("sldBuffer").onchange = async () => {
  try {
    const r = await post("/api/buffer", { seconds: Number($("sldBuffer").value) });
    toast(r.buffer.enabled
      ? `Buffer ${r.buffer.seconds}s — ${fmtMB(r.buffer.mb_capacity)} when full`
      : "Frame buffer off");
  } catch (e) { toast(e.message); }
  holding.delete("sldBuffer");
};

$("btnJudgeWake").onclick = async () => {
  try { const r = await post("/api/judge/wake", {}); toast(r.status); }
  catch (e) { toast(e.message); }
};
$("btnJudgeSleep").onclick = async () => {
  try { const r = await post("/api/judge/sleep", {}); toast(r.status); }
  catch (e) { toast(e.message); }
};

document.querySelectorAll("#lbReasonChips .reason-chip").forEach((chip) => {
  chip.onclick = () => chip.classList.toggle("selected");
});

async function submitLightboxFeedback(favorite) {
  if (!currentLightboxCapture) return;
  const comment = $("lbComment") ? $("lbComment").value.trim() : "";
  const reasons = getSelectedLightboxReasons();
  const remove = !favorite && $("lbRemoveBad") && $("lbRemoveBad").checked;
  const filename = currentLightboxCapture.filename;
  $("lightbox").classList.add("hidden");
  await feedback(filename, favorite, { comment, reasons, remove });
}

if ($("lbBtnPin")) {
  $("lbBtnPin").onclick = () => {
    if (!currentLightboxCapture) return;
    pinCapture(currentLightboxCapture.filename);
  };
}
if ($("lbBtnFav")) {
  $("lbBtnFav").onclick = () => submitLightboxFeedback(true);
}
if ($("lbBtnBad")) {
  $("lbBtnBad").onclick = () => submitLightboxFeedback(false);
}
if ($("lbComment")) {
  $("lbComment").onkeydown = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      submitLightboxFeedback(false);
    }
  };
}

$("lbClose").onclick = () => $("lightbox").classList.add("hidden");
$("lightbox").onclick = (e) => { if (e.target.id === "lightbox") $("lightbox").classList.add("hidden"); };
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    $("lightbox").classList.add("hidden");
    $("videoScanModal").classList.add("hidden");
    $("folderModal").classList.add("hidden");
    $("cameraModal").classList.add("hidden");
  }
});


/* ---------------------------------------------------------------------------
 * Camera connection
 *
 * Typing a whole RTSP URL means getting address, port, credentials and stream
 * path all right at once, and a failure used to report only that the source
 * would not open. This takes the parts separately and asks the camera which one
 * is wrong. The password is never sent back by the server, so a blank password
 * field means "keep the saved one" rather than "clear it".
 * ------------------------------------------------------------------------- */

function camKind() { return $("camKind").value; }

function camShowFields() {
  const k = camKind();
  $("camRtspFields").classList.toggle("hidden", k !== "rtsp");
  $("camWebcamFields").classList.toggle("hidden", k !== "webcam");
  $("camFileFields").classList.toggle("hidden", k !== "file");
}

function camPayload() {
  const k = camKind();
  const p = { kind: k };
  if (k === "rtsp") {
    p.host = $("camHost").value.trim();
    p.port = Number($("camPort").value) || 554;
    p.username = $("camUser").value.trim();
    p.path = $("camPath").value.trim();
    // Blank means unchanged; only send it when the operator actually typed one.
    const pw = $("camPass").value;
    if (pw) p.password = pw;
  } else if (k === "webcam") {
    p.webcam_index = Number($("camIndex").value) || 0;
  } else {
    p.file_path = $("camFile").value.trim();
  }
  return p;
}

function camResult(cls, message, hint, detail) {
  const el = $("camResult");
  el.classList.remove("hidden", "ok", "bad", "busy");
  if (cls) el.classList.add(cls);
  el.innerHTML = "";
  const main = document.createElement("b");
  main.textContent = message;
  el.appendChild(main);
  if (hint) {
    const h = document.createElement("span");
    h.className = "cam-hint";
    h.textContent = hint;
    el.appendChild(h);
  }
  if (detail) {
    const d = document.createElement("span");
    d.className = "cam-detail";
    d.textContent = detail;
    el.appendChild(d);
  }
}

function camDetailLine(r) {
  const bits = [];
  if (r.rtsp_status) bits.push("RTSP " + r.rtsp_status);
  if (r.server) bits.push(r.server);
  if (r.video) bits.push(r.video);
  if (r.tried_paths && r.tried_paths.length > 1) {
    bits.push(r.tried_paths.length + " paths tried");
  }
  if (r.elapsed_ms) bits.push(r.elapsed_ms + " ms");
  return bits.join(" \u00b7 ");
}

async function camOpen() {
  $("cameraModal").classList.remove("hidden");
  $("camResult").classList.add("hidden");
  try {
    const d = await fetch("/api/camera").then((r) => r.json());
    const c = d.settings || {};
    $("camKind").value = c.kind || "rtsp";
    $("camHost").value = c.host || "";
    $("camPort").value = c.port || 554;
    $("camUser").value = c.username || "";
    $("camPath").value = c.path || "";
    $("camIndex").value = c.webcam_index || 0;
    $("camFile").value = c.file_path || "";
    // Never prefilled -- the server does not hand it out. The placeholder says
    // whether one is already stored.
    $("camPass").value = "";
    $("camPass").placeholder = c.has_password ? "(unchanged)" : "camera password";
    $("camActive").textContent = d.active_source ? "Now: " + d.active_source : "";

    const dl = $("camPaths");
    dl.innerHTML = "";
    (d.common_paths || []).forEach((path) => {
      const o = document.createElement("option");
      o.value = path;
      dl.appendChild(o);
    });
    camShowFields();
  } catch (e) {
    camResult("bad", "Could not read the saved camera settings.", e.message);
  }
}

function camClose() { $("cameraModal").classList.add("hidden"); }

$("btnOpenCamera").onclick = camOpen;
$("btnCamCloseX").onclick = camClose;
$("btnCamClose").onclick = camClose;
$("camKind").onchange = () => { camShowFields(); $("camResult").classList.add("hidden"); };

$("btnCamTest").onclick = async () => {
  const btn = $("btnCamTest");
  btn.disabled = true;
  camResult("busy", "Asking the camera\u2026",
    "Trying the stream paths in turn; this can take a few seconds.");
  try {
    const r = await post("/api/camera/test", camPayload());
    if (r.ok) {
      camResult("ok", r.message, r.hint, camDetailLine(r));
      // Discovery found the path, so put it in the field ready to connect.
      if (r.working_path) $("camPath").value = r.working_path;
    } else {
      camResult("bad", r.message, r.hint, camDetailLine(r));
    }
  } catch (e) {
    camResult("bad", "The test could not run.", e.message);
  } finally {
    btn.disabled = false;
  }
};

$("btnCamConnect").onclick = async () => {
  const btn = $("btnCamConnect");
  btn.disabled = true;
  camResult("busy", "Connecting\u2026", "Checking the camera before switching over.");
  try {
    const r = await post("/api/camera/connect", camPayload());
    camResult("ok", "Connected. " + (r.source || ""),
      "Saved \u2014 this camera will be used after a restart too.");
    toast("Camera connected");
    setTimeout(camClose, 1200);
  } catch (e) {
    // The server refuses rather than tearing down a working stream, and says why.
    const tail = "Nothing changed \u2014 the previous source is still running.";
    const hint = (e.detail && e.detail.hint) ? e.detail.hint + " " + tail : tail;
    camResult("bad", e.message || "Could not connect.", hint);
  } finally {
    btn.disabled = false;
  }
};

connect();
