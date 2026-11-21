"use strict";

const token = document.querySelector("[data-token]").dataset.token;
const video = document.getElementById("video");
const statusEl = document.getElementById("status");
const captureBtn = document.getElementById("capture");
const nameSel = document.getElementById("name");

const PROBLEMS = {
  no_face: "No face in the photo.",
  multiple: "Several faces in the photo — one person only.",
};

captureBtn.addEventListener("click", async () => {
  captureBtn.disabled = true;
  statusEl.textContent = "Checking the photo…";
  try {
    const r = await postJSON("/api/enroll", {
      token, name: nameSel.value, image: grabFrame(video, 800),
    });
    if (r.ok && r.data.ok) {
      statusEl.textContent = `Photo saved — ${r.data.photos} valid photo(s) for ${nameSel.value}.`;
    } else {
      statusEl.textContent = PROBLEMS[r.data.error] || r.data.error || `Error ${r.status}`;
    }
  } catch (_) {
    statusEl.textContent = "The server is unreachable.";
  }
  captureBtn.disabled = false;
});

(async () => {
  try {
    await startCamera(video);
    statusEl.textContent = "Camera ready.";
  } catch (err) {
    statusEl.textContent = cameraErrorMessage(err);
    captureBtn.disabled = true;
  }
})();
