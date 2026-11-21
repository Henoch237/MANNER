"use strict";

const transcript = document.getElementById("transcript");
const welcome = document.getElementById("welcome");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const micBtn = document.getElementById("mic");
const phaseEl = document.getElementById("phase");
const phaseText = document.getElementById("phase-text");
const logoutBtn = document.getElementById("logout");
const logs = new LogPanel(document.getElementById("log-stream"), document.getElementById("log-jump"));

const YES_NO = new Set(["yes", "y", "yeah", "yep", "confirm", "no", "n", "nope", "cancel"]);

let phase = "idle";          // server phase: idle | planning | executing
let sending = false;         // a /api/message call is in flight
let cursor = 0;              // next log line to fetch
let openPlan = null;         // plan card waiting for Run / Discard
let runningPlan = null;      // plan card being executed
let openStore = null;        // "remember this speed?" card waiting for an answer

// ------------------------------------------------------------ plumbing --
async function api(method, url, body) {
  const res = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (res.status === 401) {          // session closed or replaced: back to sign-in
    window.location.reload();
    throw new Error("unauthenticated");
  }
  let data = {};
  try { data = await res.json(); } catch (_) { /* not JSON */ }
  return { ok: res.ok, status: res.status, data };
}

function scrollChat() {
  transcript.scrollTop = transcript.scrollHeight;
}

function refreshControls() {
  const busy = sending || phase !== "idle";
  input.disabled = busy;
  sendBtn.disabled = busy;
  micBtn.disabled = busy || micBusy;
  document.querySelectorAll(".chip, .suggestion, .msg .actions .btn").forEach((b) => {
    if (!b.dataset.locked) b.disabled = busy;
  });
  logoutBtn.disabled = phase === "executing";
  const label = phase === "executing" ? "Running on the robot…"
    : sending || phase === "planning" ? "Thinking…" : "Ready";
  phaseText.textContent = label;
  phaseEl.dataset.phase = phase === "executing" ? "executing" : busy ? "planning" : "idle";
  if (!busy && document.activeElement === document.body) input.focus();
}

// --------------------------------------------------------- chat pieces --
function addUser(text) {
  if (welcome) welcome.hidden = true;
  transcript.append(el("div", "msg msg-user", text));
  scrollChat();
}

function addAssistant() {
  const msg = el("div", "msg msg-bot");
  msg.append(el("div", "bot-avatar", "AI"));
  const body = el("div", "bot-body");
  const typing = el("div", "typing");
  typing.setAttribute("aria-label", "Thinking");
  typing.append(el("span"), el("span"), el("span"));
  body.append(typing);
  msg.append(body);
  transcript.append(msg);
  scrollChat();
  return body;
}

function notice(level, text) {
  const n = el("div", `notice notice-${level}`);
  n.append(el("span", "notice-icon", { error: "!", warn: "!", success: "✓", info: "i" }[level] || "i"));
  n.append(el("span", null, text));
  return n;
}

function stateSummary(s) {
  if (!s) return "Not connected to the robot yet — it connects on the first run.";
  const bits = [`at ${humanize(s.location || "unknown")}`,
    s.holding ? `holding the ${humanize(s.holding)}` : "holding nothing",
    `gripper ${s.gripper || "unknown"}`];
  return `Robot ${bits.join(" · ")}. Last speed ${s.last_speed}% / accel ${s.last_accel}%.`;
}

function rawToggle(raw) {
  const d = el("details", "raw");
  d.append(el("summary", null, "Qwen response"));
  d.append(rawView(raw));
  return d;
}

function planCard(plan, intent, raw) {
  const card = el("div", "plan");
  const who = intent && intent.recipient && intent.recipient !== "none" ? ` for ${cap(intent.recipient)}` : "";
  card.append(el("p", "plan-intro", `Here is the plan${who}. Check it before running it on the robot.`));

  const chips = intentChips(intent);
  if (chips.length) {
    const row = el("div", "intent");
    chips.forEach((c) => row.append(el("span", `ichip ichip-${c.tone}`, c.label)));
    card.append(row);
  }

  const ol = el("ol", "steps");
  plan.forEach((step) => {
    const li = el("li", "step");
    li.append(el("span", "step-title", stepTitle(step)));
    const meta = stepMeta(step);
    if (meta.length) li.append(el("span", "step-meta", meta.join(" · ")));
    ol.append(li);
  });
  card.append(ol);

  const actions = el("div", "actions");
  const run = el("button", "btn btn-run", "Run on robot");
  const skip = el("button", "btn btn-ghost", "Discard");
  run.addEventListener("click", () => execute(card));
  skip.addEventListener("click", () => discard(card));
  actions.append(run, skip);
  card.append(actions);
  if (raw) card.append(rawToggle(raw));
  return card;
}

function setPlanStatus(card, label, tone) {
  const actions = card.querySelector(".actions");
  const pill = el("span", `status status-${tone}`, label);
  actions.replaceChildren(pill);
  card.dataset.status = tone;
  if (openPlan === card) openPlan = null;
}

function storeCard(pending) {
  const card = el("div", "question");
  card.append(el("p", null,
    `Remember this speed as your default? Your speed multiplier would become ×${Number(pending.would_be).toFixed(2)}.`));
  const actions = el("div", "actions");
  const yes = el("button", "btn btn-primary", "Yes, remember it");
  const no = el("button", "btn btn-ghost", "No");
  yes.addEventListener("click", () => send("yes", "Yes, remember it"));
  no.addEventListener("click", () => send("no", "No"));
  actions.append(yes, no);
  card.append(actions);
  return card;
}

function closeStore(answer) {
  if (!openStore) return;
  const a = openStore.querySelector(".actions");
  a.replaceChildren(el("span", "status status-muted", answer));
  openStore = null;
}

function render(body, r) {
  body.replaceChildren();
  let shown = 0;
  const add = (node) => { body.append(node); shown += 1; };

  if (r.error) add(notice("error", r.error));
  (r.notices || []).forEach((n) => add(notice(n.level, n.text)));

  const d = r.drift || {};
  if (d.after !== undefined && Math.abs(d.after - d.before) > 1e-9) {
    const dir = d.after < d.before ? "slower" : "faster";
    add(notice("info", `Going ${dir}: speed ×${d.after.toFixed(2)} for this session.`));
  }

  if (r._cmd === "state") add(notice("info", stateSummary(r.robot_state)));

  if (r.kind === "plan" && r.plan) {
    const card = planCard(r.plan, r.intent, r.raw);
    add(card);
    openPlan = card;
  }

  if (r.pending_store) {
    const card = storeCard(r.pending_store);
    add(card);
    openStore = card;
  }

  if (!shown) {
    if (r.kind === "no_plan") {
      add(notice("warn", "I couldn't turn that into robot actions. Try rephrasing the request."));
      if (r.raw) add(rawToggle(r.raw));
    } else {
      add(notice("success", "Done."));
    }
  }
}

// ------------------------------------------------------------- actions --
async function send(text, shownAs) {
  if (!text || sending || phase !== "idle") return;
  // A yes/no to the open "remember this speed?" question keeps the plan;
  // any other message discards it (the server applies the same rule).
  const answersStore = openStore !== null && YES_NO.has(text.toLowerCase().replace(/[.!,]/g, ""));
  if (openStore) closeStore(answersStore ? (shownAs || text) : "Not answered");
  if (openPlan && !answersStore) setPlanStatus(openPlan, "Discarded", "muted");
  addUser(shownAs || text);
  const body = addAssistant();
  sending = true;
  refreshControls();
  try {
    const r = await api("POST", "/api/message", { text });
    if (r.status === 409) render(body, { error: "The robot is busy. Try again in a moment." });
    else if (!r.ok) render(body, { error: `Server error (${r.status}).` });
    else render(body, { ...r.data, _cmd: text.toLowerCase().trim() });
  } catch (e) {
    if (e.message !== "unauthenticated") render(body, { error: "The server is unreachable." });
  }
  sending = false;
  refreshControls();
  scrollChat();
}

async function execute(card) {
  if (sending || phase !== "idle") return;
  const r = await api("POST", "/api/execute");
  if (r.status !== 202) {
    setPlanStatus(card, r.data.error || "Could not start", "bad");
    return;
  }
  setPlanStatus(card, "Running…", "run");
  runningPlan = card;
  phase = "executing";
  refreshControls();
  poll();
}

async function discard(card) {
  if (sending || phase !== "idle") return;
  await api("POST", "/api/discard");
  setPlanStatus(card, "Discarded", "muted");
}

function finishRun(lastRun, robotState) {
  const card = runningPlan;
  runningPlan = null;
  if (!card) return;
  const ok = lastRun && lastRun.ok;
  setPlanStatus(card, ok ? "Completed" : "Failed", ok ? "ok" : "bad");
  card.append(notice(ok ? "success" : "error",
    ok ? stateSummary(robotState) : "The run did not finish. Check the logs on the right."));
  scrollChat();
}

// ---------------------------------------------------------------- poll --
let pollTimer = null;
let restored = false;

async function poll() {
  clearTimeout(pollTimer);
  let next = 1200;
  try {
    const r = await api("GET", `/api/status?since=${cursor}`);
    if (r.ok) {
      const s = r.data;
      logs.append(s.logs);
      cursor = s.next;
      phase = s.phase;
      if (!restored) restore(s);
      if (runningPlan && s.phase === "idle" && s.last_run) finishRun(s.last_run, s.robot_state);
      refreshControls();
      next = s.phase === "idle" ? 1200 : 400;
    }
  } catch (e) {
    if (e.message === "unauthenticated") return;
  }
  pollTimer = setTimeout(poll, next);
}

// After a page reload: a plan still waiting, or a run in progress.
function restore(s) {
  restored = true;
  if (s.phase === "executing") {
    if (welcome) welcome.hidden = true;
    const body = addAssistant();
    body.replaceChildren();
    const card = el("div", "plan");
    card.append(el("p", "plan-intro", "A plan is running on the robot."));
    card.append(el("div", "actions"));
    body.append(card);
    setPlanStatus(card, "Running…", "run");
    runningPlan = card;
  } else if (s.pending_plan) {
    if (welcome) welcome.hidden = true;
    const body = addAssistant();
    body.replaceChildren(planCard(s.pending_plan, null, null));
    openPlan = body.firstChild;
  }
}

// -------------------------------------------------------------- wiring --
form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  input.value = "";
  send(text);
});

document.querySelectorAll(".chip").forEach((c) =>
  c.addEventListener("click", () => send(c.dataset.cmd, c.textContent)));
document.querySelectorAll(".suggestion").forEach((c) =>
  c.addEventListener("click", () => send(c.dataset.text)));

document.querySelectorAll(".filter").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".filter").forEach((x) => x.classList.toggle("active", x === b));
  logs.setFilter(b.dataset.filter);
}));
document.getElementById("log-clear").addEventListener("click", () => logs.clear());

const logsToggle = document.getElementById("logs-toggle");
function showLogs(open) {
  document.body.classList.toggle("logs-open", open);
  logsToggle.setAttribute("aria-expanded", String(open));
  if (open) logs.scrollToEnd();
}
logsToggle.addEventListener("click", () => showLogs(!document.body.classList.contains("logs-open")));
document.getElementById("log-close").addEventListener("click", () => showLogs(false));

logoutBtn.addEventListener("click", async () => {
  const r = await api("POST", "/api/auth/logout");
  if (r.ok) window.location.reload();
  else {
    const body = addAssistant();
    render(body, { error: r.data.error || "You can't sign out right now." });
  }
});

// ------------------------------------------------------- push-to-talk --
// Hold the mic button: record from the browser, send a 16 kHz mono WAV to
// /api/asr, and put the transcript into the input bar. The normal Send
// flow takes it from there - nothing else changes.
const MIC_TARGET_SR = 16000;     // what the ASR endpoint expects
const MIC_MAX_MS = 30000;        // safety cap on one hold
const MIC_INPUT_PLACEHOLDER = input.placeholder;
let micBusy = false;             // a transcription is in flight
let micStream = null, micContext = null, micProcessor = null;
let micChunks = null, micCancel = false, micTimer = null;

function micFlashPlaceholder(text) {
  input.placeholder = text;
  setTimeout(() => { input.placeholder = MIC_INPUT_PLACEHOLDER; }, 2500);
}

function encodeWavBlob(samples, sampleRate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
  writeStr(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); writeStr(8, "WAVE");
  writeStr(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, 1, true); v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  writeStr(36, "data"); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

function downsampleTo16k(chunks, srcRate) {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const merged = new Float32Array(total);
  let off = 0;
  for (const c of chunks) { merged.set(c, off); off += c.length; }
  if (srcRate === MIC_TARGET_SR) return merged;
  const ratio = srcRate / MIC_TARGET_SR;
  const out = new Float32Array(Math.floor(total / ratio));
  for (let i = 0; i < out.length; i++) out[i] = merged[Math.floor(i * ratio)];
  return out;
}

async function micStart(e) {
  e.preventDefault();
  if (micBtn.disabled || micStream) return;
  micCancel = false;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (_) {
    micFlashPlaceholder("Microphone unavailable");
    return;
  }
  if (micCancel) {                    // released while the prompt was up
    stream.getTracks().forEach((t) => t.stop());
    return;
  }
  micStream = stream;
  micChunks = [];
  micContext = new (window.AudioContext || window.webkitAudioContext)();
  const src = micContext.createMediaStreamSource(micStream);
  micProcessor = micContext.createScriptProcessor(4096, 1, 1);
  micProcessor.onaudioprocess = (ev) =>
    micChunks.push(new Float32Array(ev.inputBuffer.getChannelData(0)));
  src.connect(micProcessor);
  micProcessor.connect(micContext.destination);
  micBtn.classList.add("recording");
  micTimer = setTimeout(micStop, MIC_MAX_MS);
}

async function micStop() {
  if (!micStream) return;
  clearTimeout(micTimer);
  const chunks = micChunks, srcRate = micContext.sampleRate;
  micProcessor.disconnect();
  micProcessor.onaudioprocess = null;
  micStream.getTracks().forEach((t) => t.stop());
  micStream = null;
  micChunks = null;
  await micContext.close();
  micContext = null;
  micProcessor = null;
  micBtn.classList.remove("recording");
  if (micCancel || !chunks.reduce((n, c) => n + c.length, 0)) return;

  micBusy = true;
  refreshControls();
  try {
    const fd = new FormData();
    fd.append("audio", encodeWavBlob(downsampleTo16k(chunks, srcRate), MIC_TARGET_SR), "speech.wav");
    const res = await fetch("/api/asr", { method: "POST", body: fd });
    if (res.status === 401) { window.location.reload(); return; }
    const data = await res.json().catch(() => ({}));
    if (res.ok && data.ok && data.text) {
      input.value = data.text;
      input.focus();
    } else if (res.ok) {
      micFlashPlaceholder("Didn't catch that - try again");
    } else {
      micFlashPlaceholder("Transcription failed - see logs");
    }
  } catch (_) {
    micFlashPlaceholder("The server is unreachable");
  }
  micBusy = false;
  refreshControls();
}

micBtn.addEventListener("pointerdown", micStart);
micBtn.addEventListener("pointerup", () => {
  if (micStream) { micCancel = false; micStop(); }  // deliberate release: send
  else micCancel = true;                            // released during the prompt: abort
});
micBtn.addEventListener("pointercancel", () => { micCancel = true; micStop(); });
micBtn.addEventListener("pointerleave", () => { micCancel = true; micStop(); });
micBtn.addEventListener("contextmenu", (e) => e.preventDefault());

refreshControls();
poll();
