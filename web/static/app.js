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
    $("mCentering").textContent = `Centering: ${Math.round(d.centering_score * 100)}%`;
    $("mHand").textContent = `Hand: ${d.hands_covering_face ? "OBSTRUCTING ⚠️" : d.is_hand_blurry ? "BLURRY 🔄" : "CLEAR ✅"}`;
    $("mStability").textContent = `Score: ${d.is_stable ? "🟢 Stable" : "🟡 Unstable"} · People detected: ${d.people_detected}`;
    $("mModel").textContent = d.ml_trained
      ? `AI Model: Trained (${Math.round(d.ml_confidence * 100)}% fit · ${d.ml_samples} samples · ML score ${Math.round(d.ml_score * 100)}%)`
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
    toast(`${favorite ? "⭐ Learned as a winner" : "👎 Learned to penalize"} — ${r.samples} samples, ${Math.round(r.confidence * 100)}% fit`);
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
    <b>File:</b> ${c.filename}`;
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

$("lbClose").onclick = () => $("lightbox").classList.add("hidden");
$("lightbox").onclick = (e) => { if (e.target.id === "lightbox") $("lightbox").classList.add("hidden"); };
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") $("lightbox").classList.add("hidden");
});

connect();
