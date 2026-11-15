"use strict";

const $ = (id) => document.getElementById(id);

// Tracks which controls the user is actively dragging, so incoming telemetry
// doesn't yank a slider out from under them.
const holding = new Set();
let isCapturing = false;
let sourceFocused = false;
let lastCapturesJSON = "";
let visibleCapturesCount = 0;

// Separate custom scan output folder so 10 Hz live telemetry never overwrites it
let customScanOutputDir = "";
let folderPickerTarget = "live"; // "live" | "scan"
let videoBrowserTarget = "scan"; // "scan" | "camera"

// ---------------------------------------------------------------- helpers

function escHtml(v) {
  return String(v == null ? "" : v)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function fmtDurationShort(totalSec) {
  const s = Math.max(0, Math.floor(Number(totalSec) || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const rem = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${String(rem).padStart(2, "0")}s`;
  return `${rem}s`;
}

async function post(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) {
    const payload = await res.json().catch(() => ({}));
    const d = payload.detail;
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
  toastTimer = setTimeout(() => el.classList.add("hidden"), 2800);
}

function setClass(el, cls, on) {
  if (el) el.classList.toggle(cls, !!on);
}

function downloadCaptureFile(filename) {
  if (!filename) return;
  const a = document.createElement("a");
  a.href = `/api/captures/${encodeURIComponent(filename)}/download`;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

// ---------------------------------------------------------------- telemetry

const STATUS = {
  disconnected: ["#ef4444", "Disconnected"],
  connecting: ["#f59e0b", "Connecting…"],
  connected: ["#22c55e", "Connected"],
  reconnecting: ["#f59e0b", "Reconnecting"],
  standby: ["#38bdf8", "Camera Standby (Power off)"],
  unconfigured: ["#94a3b8", "No Camera Configured"],
  failed: ["#ef4444", "Connection Failed"],
};

function syncCaptureLimitSelect(maxMinutes) {
  const sel = $("selCaptureLimit");
  if (!sel || document.activeElement === sel || maxMinutes == null) return;
  const valStr = String(Math.round(Number(maxMinutes) || 0));
  let hasOpt = [...sel.options].some((o) => o.value === valStr);
  if (!hasOpt) {
    const opt = document.createElement("option");
    opt.value = valStr;
    opt.textContent = valStr === "0" ? "⏱ No limit" : `⏱ Max: ${valStr}m`;
    sel.appendChild(opt);
  }
  sel.value = valStr;
}

function render(d) {
  const [color, label] = STATUS[d.status] || ["#94a3b8", "Unknown"];
  $("statusDot").style.color = color;
  $("statusText").textContent =
    d.status === "reconnecting" ? `${label} (${d.reconnect_attempt})…` : label;

  if (d.idle_standby) {
    $("fps").textContent = "0 fps · 💤 Idle Standby";
  } else {
    $("fps").textContent = `${d.fps ?? 0} fps`;
  }

  $("outputDir").textContent = d.output_dir || "—";
  if (!customScanOutputDir) {
    $("scanOutputDir").textContent = d.output_dir || "—";
  }

  const src = d.source || "";
  if ($("activeSourcePill")) {
    $("activeSourcePill").textContent = src ? `📹 ${src}` : "📹 Set up camera…";
  }

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

  const judgeMins = d.ai_judge_minutes_left;
  $("mJudge").childNodes[0].nodeValue = d.ai_judge_active
    ? `AI judge: active${judgeMins == null ? "" : ` · ${judgeMins} min left`} `
    : `${d.ai_judge || "AI judge: off"} `;

  if (!sourceFocused && $("src")) {
    const masked = src.includes("://***@");
    if (masked) {
      $("src").placeholder = src;
      if ($("src").value.includes("://***@")) $("src").value = "";
    } else {
      $("src").placeholder = "rtsp://user:pass@host:554/stream";
      if ($("src").value !== src) $("src").value = src;
    }
  }

  syncCaptureLimitSelect(d.capture_max_minutes);

  // Start / stop button + live auto-stop countdown
  isCapturing = !!d.is_capturing;
  const btn = $("btnStartStop");
  if (isCapturing) {
    const rem = d.capture_seconds_remaining;
    btn.textContent = rem != null
      ? `⏹ Stop (${fmtDurationShort(rem)})`
      : "⏹ Stop Capture";
  } else {
    btn.textContent = "▶ Start Capture";
  }
  setClass(btn, "btn-stop", isCapturing);
  setClass(btn, "btn-start", !isCapturing);

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
  if (!holding.has("sldScore") && s.min_score != null) { $("sldScore").value = s.min_score; $("outScore").textContent = s.min_score + "%"; }
  if (!holding.has("sldYaw") && s.max_yaw != null) { $("sldYaw").value = s.max_yaw; $("outYaw").textContent = s.max_yaw + "°"; }
  if (!holding.has("sldShoulder") && s.max_shoulder_tilt != null) { $("sldShoulder").value = s.max_shoulder_tilt; $("outShoulder").textContent = s.max_shoulder_tilt + "°"; }
  if (!holding.has("sldBurst") && s.burst_window != null) { $("sldBurst").value = Math.round(s.burst_window * 10); $("outBurst").textContent = s.burst_window.toFixed(1) + "s"; }
  if (document.activeElement !== $("numPeople") && s.people_target != null) $("numPeople").value = s.people_target;
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
    const gazeSide = (d.eye_gaze_away_score || 0) >= 0.42 ? " · GAZE OFF 👀" : "";
    $("mEyes").textContent = `Eyes: ${d.is_blinking ? "CLOSED/BLINK" : "OPEN"} (EAR ${d.avg_ear.toFixed(2)})${gazeSide}`;
    const talkFlag = (d.mouth_velocity || 0) > 0.14 ? " · MID-SYLLABLE" : "";
    $("mExpr").textContent = `Expression: ${d.is_smiling ? "SMILE 😊" : d.is_awkward_expression ? "AWKWARD 😮" : "NEUTRAL"} (${d.smile_ratio.toFixed(2)})${talkFlag}`;

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
  visibleCapturesCount = caps.length;
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
    info.innerHTML = `<span class="rank-badge">🏆 #${Number(c.rank) || 1}</span> | <b>${Math.round(c.score * 100)}%</b>${propBadge}${pinBadge}`;

    const btns = document.createElement("div");
    btns.className = "thumb-btns";

    const pin = document.createElement("button");
    pin.className = "btn-pin" + (c.is_pinned ? " on" : "");
    pin.textContent = c.is_pinned ? "📌 Saved" : "📌 Keep";
    pin.title = c.is_pinned
      ? "Protected from auto-deletion (click to unpin)"
      : "Flag this shot so it is never deleted when new Top shots arrive";
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

    const dl = document.createElement("button");
    dl.className = "btn-dl";
    dl.textContent = "⬇";
    dl.title = "Download full-resolution file";
    dl.onclick = () => downloadCaptureFile(c.filename);

    btns.append(pin, fav, rej, dl);

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
      : "📌 Unpinned — returned to normal Top rotation");
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

if ($("btnDownloadZip")) {
  $("btnDownloadZip").onclick = () => {
    if (!visibleCapturesCount) {
      return toast("No captures in gallery yet to download");
    }
    window.location.href = "/api/captures/download-zip?mode=all";
  };
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
  const shoulder = c.shoulder_detected ? `${escHtml(c.shoulder_tilt)}°` : "headshot / out of frame";
  const pinLine = c.is_pinned ? " · <b>📌 Protected from auto-deletion</b>" : "";
  const issuesList = (c.judge_issues || []).map((i) => escHtml(i)).join("; ");

  $("lbInfo").innerHTML = `
    <b>Score:</b> ${Math.round(c.score * 100)}%${c.is_prop_moment ? " (🎭 Candid Stage Prop Moment)" : ""}${pinLine}<br>
    <b>Head yaw:</b> ${escHtml(c.yaw)}°<br>
    <b>Shoulder tilt:</b> ${shoulder}<br>
    <b>Eye aspect ratio:</b> ${escHtml(c.avg_ear)}<br>
    <b>Smile ratio:</b> ${escHtml(c.smile_ratio)}<br>
    <b>Sharpness:</b> ${escHtml(c.sharpness)}<br>
    <b>Cutout suitability:</b> ${Math.round((c.cutout_score || 0) * 100)}%<br>` +
    (c.judge_score != null
      ? `<b>AI judge:</b> ${Math.round(c.judge_score * 100)}% (analyzer ${Math.round((c.heuristic_score || 0) * 100)}%)<br>`
        + `<i>${escHtml(c.judge_summary || "")}</i><br>`
        + (issuesList ? `<b>Flagged:</b> ${issuesList}<br>` : "")
      : "") + `
    <b>File:</b> ${escHtml(c.filename)}` +
    (c.crop_url ? `<br><a href="${escHtml(c.crop_url)}" target="_blank" rel="noopener">Open subject crop ↗</a>` : "");

  syncLightboxPinButton(!!c.is_pinned);

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

if ($("selCaptureLimit")) {
  $("selCaptureLimit").onchange = (e) => {
    const mins = +e.target.value;
    post("/api/settings", { capture_max_minutes: mins })
      .then(() => {
        toast(mins > 0
          ? `⏱ Auto-stop capture window set to ${mins >= 60 ? (mins / 60) + "h" : mins + "m"}`
          : "⏱ Capture auto-stop timer disabled");
      })
      .catch((err) => toast(err.message));
  };
}

$("btnStartStop").onclick = () => {
  const maxMinutes = $("selCaptureLimit") ? +$("selCaptureLimit").value : undefined;
  post("/api/capture/toggle", { enabled: !isCapturing, max_minutes: maxMinutes })
    .catch((e) => toast(e.message));
};

$("btnManual").onclick = async () => {
  try {
    await post("/api/capture/manual", {});
    toast("📸 Screenshot saved");
  } catch (e) {
    toast("Capture failed: " + e.message);
  }
};

if ($("btnSource")) {
  $("btnSource").onclick = () => {
    const source = $("src").value.trim();
    if (!source) return toast("Enter a source URL first");
    post("/api/source", { source })
      .then(() => {
        toast("Reconnecting to new source…");
        camClose();
      })
      .catch((e) => toast(e.message));
  };
}

if ($("src")) {
  $("src").addEventListener("focus", () => { sourceFocused = true; });
  $("src").addEventListener("blur", () => { sourceFocused = false; });
}

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

// ---------------------------------------------------------------- server video browser

async function loadVideoBrowser(path) {
  try {
    const res = await fetch("/api/browse/videos?path=" + encodeURIComponent(path || ""));
    if (!res.ok) {
      const d = await res.json().catch(() => ({}));
      throw new Error(d.detail || res.statusText);
    }
    const d = await res.json();
    $("vidBrowsePath").value = d.path;

    const sc = $("vidBrowseShortcuts");
    sc.innerHTML = "";
    (d.shortcuts || []).forEach((q) => {
      const b = document.createElement("button");
      b.textContent = q.name;
      b.title = q.path;
      b.onclick = () => loadVideoBrowser(q.path);
      sc.appendChild(b);
    });

    const list = $("vidBrowseList");
    list.innerHTML = "";
    if (d.parent) {
      const up = document.createElement("button");
      up.className = "up";
      up.textContent = "↑ ..";
      up.onclick = () => loadVideoBrowser(d.parent);
      list.appendChild(up);
    }

    const entries = d.entries || [];
    if (!entries.length) {
      const empty = document.createElement("div");
      empty.className = "empty";
      empty.textContent = "No video files or subfolders found in this directory.";
      list.appendChild(empty);
    }

    entries.forEach((en) => {
      const b = document.createElement("button");
      if (en.kind === "dir") {
        const left = document.createElement("span");
        left.textContent = "📁 " + en.name;
        b.appendChild(left);
        b.onclick = () => loadVideoBrowser(en.path);
      } else {
        b.className = "video-item";
        const left = document.createElement("span");
        left.textContent = "🎬 " + en.name;
        const right = document.createElement("span");
        right.className = "file-meta";
        right.textContent = fmtMB(en.size_mb || 0);
        b.append(left, right);
        b.onclick = () => selectServerVideo(en.path);
      }
      list.appendChild(b);
    });

    $("vidBrowseNote").textContent = "Click a video file to select it.";
  } catch (e) {
    $("vidBrowseNote").textContent = "⚠ " + e.message;
  }
}

function openVideoBrowser(target) {
  videoBrowserTarget = target || "scan";
  $("videoBrowserModal").classList.remove("hidden");
  loadVideoBrowser("");
}

function closeVideoBrowser() {
  $("videoBrowserModal").classList.add("hidden");
}

function selectServerVideo(filePath) {
  closeVideoBrowser();
  if (videoBrowserTarget === "camera") {
    $("camFile").value = filePath;
  } else {
    $("scanVideoPath").value = filePath;
    inspectVideoPath(filePath);
  }
}

if ($("btnBrowseServerVideo")) {
  $("btnBrowseServerVideo").onclick = () => openVideoBrowser("scan");
}
if ($("btnCamBrowseFile")) {
  $("btnCamBrowseFile").onclick = () => openVideoBrowser("camera");
}
if ($("btnVidBrowseCloseX")) $("btnVidBrowseCloseX").onclick = closeVideoBrowser;
if ($("btnVidBrowseClose")) $("btnVidBrowseClose").onclick = closeVideoBrowser;
if ($("btnVidBrowseGo")) {
  $("btnVidBrowseGo").onclick = () => loadVideoBrowser($("vidBrowsePath").value.trim());
}
if ($("vidBrowsePath")) {
  $("vidBrowsePath").onkeydown = (e) => {
    if (e.key === "Enter") loadVideoBrowser($("vidBrowsePath").value.trim());
  };
}

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

async function inspectVideoPath(rawPath) {
  const path = (rawPath || $("scanVideoPath").value || "").trim();
  if (!path) return toast("Enter or browse to a video file first");
  try {
    const r = await post("/api/video/info", { path });
    const meta = $("scanVideoMeta");
    meta.innerHTML = `✓ <b>${escHtml(r.filename)}</b> — Duration: <b>${escHtml(r.duration_str)}</b> (${escHtml(r.duration_sec)}s) | Resolution: <b>${escHtml(r.width)}x${escHtml(r.height)}</b> | FPS: <b>${escHtml(r.fps)}</b> | Frames: <b>${Number(r.total_frames || 0).toLocaleString()}</b>`;
    meta.classList.remove("hidden");
  } catch (e) {
    toast("Inspect failed: " + e.message);
  }
}

$("btnCheckPath").onclick = () => inspectVideoPath();

$("btnUploadVideo").onclick = () => {
  const fileInput = $("videoFileInput");
  if (!fileInput.files || !fileInput.files[0]) {
    return toast("Please select a video file first");
  }
  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);

  const btn = $("btnUploadVideo");
  const progWrap = $("uploadProgWrap");
  const progBar = $("uploadProgBar");
  const progText = $("uploadProgText");

  btn.disabled = true;
  btn.textContent = "Uploading…";
  progWrap.classList.remove("hidden");
  progBar.style.width = "0%";
  progText.textContent = "Starting upload…";

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/video/upload");
  xhr.upload.onprogress = (ev) => {
    if (ev.lengthComputable && ev.total > 0) {
      const pct = Math.round((ev.loaded / ev.total) * 100);
      progBar.style.width = pct + "%";
      const loadedMB = (ev.loaded / (1024 * 1024)).toFixed(1);
      const totalMB = (ev.total / (1024 * 1024)).toFixed(1);
      progText.textContent = `Uploading ${pct}% (${loadedMB} / ${totalMB} MB)…`;
    }
  };
  xhr.onload = () => {
    btn.disabled = false;
    btn.textContent = "⬆ Upload Video";
    progWrap.classList.add("hidden");
    let data = {};
    try { data = JSON.parse(xhr.responseText || "{}"); } catch (_) {}
    if (xhr.status >= 200 && xhr.status < 300) {
      $("scanVideoPath").value = data.path;
      const meta = $("scanVideoMeta");
      meta.innerHTML = `✓ <b>${escHtml(data.filename)}</b> (Uploaded) — Duration: <b>${escHtml(data.duration_str)}</b> | Resolution: <b>${escHtml(data.width)}x${escHtml(data.height)}</b> | FPS: <b>${escHtml(data.fps)}</b>`;
      meta.classList.remove("hidden");
      toast("Video uploaded! Set trim/options if desired, then click Start Scan.");
    } else {
      toast("Upload failed: " + (data.detail || xhr.statusText || "Error"));
    }
  };
  xhr.onerror = () => {
    btn.disabled = false;
    btn.textContent = "⬆ Upload Video";
    progWrap.classList.add("hidden");
    toast("Network error while uploading video");
  };
  xhr.send(formData);
};

function setScanButtonRunning(running) {
  isScanningVideo = !!running;
  const btn = $("btnStartWebScan");
  if (!btn) return;
  btn.textContent = running ? "⏹ Stop Scan" : "▶ Start Scan";
  btn.classList.toggle("btn-stop", running);
  btn.classList.toggle("btn-purple", !running);
}

async function toggleWebScan() {
  if (isScanningVideo) {
    try {
      await post("/api/video/scan/cancel", {});
      toast("Cancelling video scan…");
    } catch (e) {
      toast("Cancel failed: " + e.message);
    }
  } else {
    const path = $("scanVideoPath").value.trim();
    if (!path) return toast("Specify, browse, or upload a video file first");

    const top_n = +$("scanTopN").value || 5;
    const sample_fps = +$("scanSpeed").value;
    const min_score = (+$("scanMinScore").value || 55) / 100;
    const min_interval_sec = (+$("scanInterval").value || 20) / 10;
    const start_time = $("scanStartTime") ? $("scanStartTime").value.trim() : "";
    const end_time = $("scanEndTime") ? $("scanEndTime").value.trim() : "";

    try {
      await post("/api/video/scan", {
        path,
        top_n,
        sample_fps,
        min_score,
        min_interval_sec,
        start_time,
        end_time,
        output_dir: customScanOutputDir || "",
      });

      setScanButtonRunning(true);
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
        $("scanStatusText").innerHTML = `🧠 <b>${escHtml(data.error)}</b> | Candidates Analyzed: <b>${Number(data.candidates_analyzed) || 0}</b> | ${escHtml(bestText)}`;
      } else {
        $("scanStatusText").innerHTML = `Status: <b>${escHtml(data.status)}</b> | Progress: <b>${escHtml(data.current_time_str)} / ${escHtml(data.total_duration_str)}</b> (${escHtml(data.progress_pct)}%) | Speed: <b>${escHtml(data.fps_speed)} FPS${escHtml(mult)}</b> | ETA: <b>${escHtml(etaFmt)}</b> | ${escHtml(bestText)}`;
      }

      if (data.current_frame_preview) {
        $("scanCurrentFrame").src = data.current_frame_preview;
        $("scanCurrentFrame").classList.remove("hidden");
        $("scanCurrentPlaceholder").classList.add("hidden");
      }

      if (data.champion_frame_preview) {
        $("scanChampFrame").src = data.champion_frame_preview;
        $("scanChampFrame").classList.remove("hidden");
        $("scanChampPlaceholder").classList.add("hidden");
      }

      if (!data.is_scanning) {
        clearInterval(scanPollTimer);
        setScanButtonRunning(false);

        if (data.status === "completed") {
          $("scanProgBar").style.width = "100%";
          const count = Number(data.results_count) || 0;
          toast(`✓ Scan Complete! Picked ${count} best thumbnails.`);
          $("scanStatusText").innerHTML = `✓ <b>Completed!</b> Extracted <b>${count}</b> best thumbnails into output folder.`;

          fetch("/api/captures").then((r) => r.json()).then((d) => {
            if (d.captures) renderGallery(d.captures);
          }).catch(() => {});
        } else if (data.status === "cancelled") {
          toast("Video scan was cancelled.");
        } else if (data.error) {
          toast("Scan error: " + data.error);
        }
      }
    } catch (e) {
      // ignore transient network glitch
    }
  }, 350);
}

async function checkInitialScanState() {
  try {
    const res = await fetch("/api/video/scan/status");
    if (!res.ok) return;
    const data = await res.json();
    if (data.is_scanning) {
      if (data.video_path && !$("scanVideoPath").value) {
        $("scanVideoPath").value = data.video_path;
      }
      if (data.scan_output_dir) {
        customScanOutputDir = data.scan_output_dir;
        $("scanOutputDir").textContent = data.scan_output_dir;
        if ($("btnScanResetFolder")) $("btnScanResetFolder").classList.remove("hidden");
      }
      setScanButtonRunning(true);
      startScanPolling();
      toast("📼 Reconnected to active background video scan");
    }
  } catch (_) {}
}

// ---------------------------------------------------------------- folder picker

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
      ? "This is the current live output folder."
      : (d.writable ? "Writable." : "\u26a0 Not writable by the server.");
  } catch (e) {
    $("folderNote").textContent = "\u26a0 " + e.message;
  }
}

function openFolderModal(target) {
  folderPickerTarget = target || "live";
  if ($("folderModalTitle")) {
    $("folderModalTitle").textContent = folderPickerTarget === "scan"
      ? "Choose Video Scan Output Folder"
      : "Choose Live Output Folder";
  }
  $("folderModal").classList.remove("hidden");
  $("folderCreate").checked = false;
  loadFolder(folderPickerTarget === "scan" && customScanOutputDir ? customScanOutputDir : "");
}
function closeFolderModal() { $("folderModal").classList.add("hidden"); }

$("btnPickFolder").onclick = () => openFolderModal("live");
$("btnScanPickFolder").onclick = () => openFolderModal("scan");
if ($("btnScanResetFolder")) {
  $("btnScanResetFolder").onclick = () => {
    customScanOutputDir = "";
    $("scanOutputDir").textContent = $("outputDir").textContent || "—";
    $("btnScanResetFolder").classList.add("hidden");
    toast("Scan output will use the live output folder");
  };
}
$("btnFolderCloseX").onclick = closeFolderModal;
$("btnFolderClose").onclick = closeFolderModal;
$("btnFolderGo").onclick = () => loadFolder($("folderPath").value.trim());
$("folderPath").onkeydown = (e) => { if (e.key === "Enter") loadFolder($("folderPath").value.trim()); };

$("btnFolderUse").onclick = async () => {
  const path = $("folderPath").value.trim() || folderCurrent;
  if (folderPickerTarget === "scan") {
    customScanOutputDir = path;
    $("scanOutputDir").textContent = path;
    if ($("btnScanResetFolder")) $("btnScanResetFolder").classList.remove("hidden");
    toast("Video scan output folder set to " + path);
    closeFolderModal();
    return;
  }

  try {
    const d = await post("/api/output", { path, create: $("folderCreate").checked });
    $("outputDir").textContent = d.output_dir;
    if (!customScanOutputDir) {
      $("scanOutputDir").textContent = d.output_dir;
    }
    if (d.captures) {
      renderGallery(d.captures);
    }
    if (d.warning) {
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

// Frame buffer & Judge controls
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

if ($("lbBtnDownload")) {
  $("lbBtnDownload").onclick = () => {
    if (currentLightboxCapture) downloadCaptureFile(currentLightboxCapture.filename);
  };
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
    $("videoBrowserModal").classList.add("hidden");
    $("videoScanModal").classList.add("hidden");
    $("folderModal").classList.add("hidden");
    $("cameraModal").classList.add("hidden");
  }
});


/* ---------------------------------------------------------------------------
 * Camera connection
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
    $("camPass").value = "";
    $("camPass").placeholder = c.has_password ? "(unchanged)" : "camera password";
    $("camActive").textContent = d.active_source ? "Now: " + d.active_source : "No active source";
    if ($("btnCamForget")) {
      $("btnCamForget").disabled = !d.configured;
    }

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
if ($("activeSourcePill")) $("activeSourcePill").onclick = camOpen;
$("btnCamCloseX").onclick = camClose;
$("btnCamClose").onclick = camClose;
$("camKind").onchange = () => { camShowFields(); $("camResult").classList.add("hidden"); };

if ($("btnCamForget")) {
  $("btnCamForget").onclick = async () => {
    if (!window.confirm("Forget the saved camera settings and revert to the server environment default?")) return;
    try {
      const r = await post("/api/camera/forget", {});
      $("btnCamForget").disabled = true;
      $("camActive").textContent = r.active_source ? "Now: " + r.active_source : "No active source";
      camResult("ok", "Saved camera forgotten.", "Reverted to environment default.");
      toast("Saved camera removed");
    } catch (e) {
      camResult("bad", "Could not forget camera.", e.message);
    }
  };
}

$("btnCamTest").onclick = async () => {
  const btn = $("btnCamTest");
  btn.disabled = true;
  camResult("busy", "Asking the camera\u2026",
    "Trying the stream paths in turn; this can take a few seconds.");
  try {
    const r = await post("/api/camera/test", camPayload());
    if (r.ok) {
      camResult("ok", r.message, r.hint, camDetailLine(r));
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
    const tail = "Nothing changed \u2014 the previous source is still running.";
    const hint = (e.detail && e.detail.hint) ? e.detail.hint + " " + tail : tail;
    camResult("bad", e.message || "Could not connect.", hint);
  } finally {
    btn.disabled = false;
  }
};

connect();
checkInitialScanState();
