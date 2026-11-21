// Shared webcam helpers for the login and enrollment pages.
"use strict";

async function startCamera(video) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("This browser cannot access the camera.");
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((resolve) => {
    if (video.readyState >= 2) resolve();
    else video.onloadeddata = () => resolve();
  });
  return stream;
}

function stopCamera(stream) {
  if (stream) stream.getTracks().forEach((t) => t.stop());
}

// Current frame as a JPEG data URL, at most maxWidth px wide.
function grabFrame(video, maxWidth = 640) {
  const scale = Math.min(1, maxWidth / video.videoWidth);
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.85);
}

function cameraErrorMessage(err) {
  if (err && err.name === "NotAllowedError") {
    return "Camera access was denied. Allow it in the browser, then reload the page.";
  }
  if (err && err.name === "NotFoundError") return "No camera found.";
  return (err && err.message) || "Camera unavailable.";
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let data = {};
  try { data = await res.json(); } catch (_) { /* non-JSON error page */ }
  return { ok: res.ok, status: res.status, data };
}
