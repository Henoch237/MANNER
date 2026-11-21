"use strict";

const video = document.getElementById("video");
const camera = document.getElementById("camera");
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("start");
const bars = document.querySelectorAll(".progress span");

const MESSAGES = {
  no_face: "No face detected — look at the camera.",
  multiple: "Several faces detected — only one person in front of the camera.",
  unknown: "Face not recognised.",
  no_enrollment: "Nobody is enrolled yet. Use the enrollment link shown in the terminal.",
  busy: "The robot is finishing a run, please wait…",
};
const MAX_UNKNOWN = 15;   // frames before pausing on an unrecognised face

let stream = null;
let running = false;

function setProgress(n) {
  bars.forEach((b, i) => b.classList.toggle("on", i < n));
}

function setState(cls, text) {
  camera.classList.remove("scanning", "ok", "bad");
  if (cls) camera.classList.add(cls);
  statusEl.textContent = text;
}

async function scan() {
  running = true;
  startBtn.hidden = true;
  setProgress(0);
  setState("scanning", "Looking for a face…");
  let unknown = 0;

  while (running) {
    let r;
    try {
      r = await postJSON("/api/auth/identify", { image: grabFrame(video) });
    } catch (_) {
      setState("bad", "The server is unreachable.");
      break;
    }
    const s = r.data.status;
    if (s === "recognized") {
      setProgress(bars.length);
      setState("ok", `Welcome ${r.data.name} — opening your session…`);
      stopCamera(stream);
      setTimeout(() => window.location.reload(), 600);
      return;
    }
    if (s === "verifying") {
      setProgress(r.data.progress);
      setState("scanning", "Face detected, verifying…");
      unknown = 0;
    } else {
      setProgress(0);
      if (s === "no_enrollment") { setState("bad", MESSAGES[s]); break; }
      if (s === "unknown") unknown += 1;
      if (unknown >= MAX_UNKNOWN) {
        setState("bad", "Face not recognised. You may not be enrolled, or the light is too low.");
        break;
      }
      setState("scanning", MESSAGES[s] || "Looking for a face…");
    }
    await new Promise((res) => setTimeout(res, 250));
  }
  running = false;
  startBtn.hidden = false;
}

startBtn.addEventListener("click", scan);

(async () => {
  try {
    stream = await startCamera(video);
    scan();
  } catch (err) {
    setState("bad", cameraErrorMessage(err));
  }
})();
