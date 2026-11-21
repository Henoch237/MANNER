"use strict";

// Live perception view: MJPEG from /api/camera.mjpg (RealSense + YOLO boxes),
// shown on top of the log panel. Retries while the camera is unavailable.
(function () {
  const box = document.getElementById("cam");
  const img = document.getElementById("cam-img");
  const msg = document.getElementById("cam-msg");
  const stateEl = document.getElementById("cam-state");
  const toggle = document.getElementById("cam-toggle");
  const RETRY_MS = 4000;
  let collapsed = false;
  let retry = null;

  try { collapsed = localStorage.getItem("camCollapsed") === "1"; } catch (_) { /* no storage */ }

  function show(text) {
    msg.textContent = text;
    msg.hidden = !text;
  }

  function stop() {
    img.removeAttribute("src");          // closes the stream connection
    clearTimeout(retry);
  }

  async function start() {
    stop();
    if (collapsed) return;
    let st;
    try {
      const res = await fetch("/api/camera/status");
      if (res.status === 401) return;    // app.js handles the sign-out
      st = await res.json();
    } catch (_) {
      st = { available: false, error: "server unreachable" };
    }
    if (!st.available) {
      box.classList.add("off");
      stateEl.textContent = "unavailable";
      show(`Camera unavailable${st.error ? ": " + st.error : ""}`);
      retry = setTimeout(start, RETRY_MS);
      return;
    }
    box.classList.remove("off");
    stateEl.textContent = st.error ? "error" : "live";
    show(st.error ? st.error : "Connecting…");
    img.src = `/api/camera.mjpg?t=${Date.now()}`;
  }

  img.addEventListener("load", () => show(""));          // first frame arrived
  img.addEventListener("error", () => {
    if (collapsed) return;
    box.classList.add("off");
    stateEl.textContent = "reconnecting";
    show("Camera stream interrupted - reconnecting…");
    stop();
    retry = setTimeout(start, RETRY_MS);
  });

  function applyCollapsed() {
    box.classList.toggle("collapsed", collapsed);
    toggle.textContent = collapsed ? "Show" : "Hide";
    toggle.setAttribute("aria-expanded", String(!collapsed));
    try { localStorage.setItem("camCollapsed", collapsed ? "1" : "0"); } catch (_) { /* no storage */ }
  }

  toggle.addEventListener("click", () => {
    collapsed = !collapsed;
    applyCollapsed();
    if (collapsed) { stop(); stateEl.textContent = "hidden"; } else start();
  });

  applyCollapsed();
  if (collapsed) stateEl.textContent = "hidden"; else start();
})();
