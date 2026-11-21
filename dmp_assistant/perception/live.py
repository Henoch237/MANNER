"""One background thread owns the RealSense and runs YOLO on every frame.

It feeds both the web live view (annotated JPEG) and the pick: it offers
the same read() / deproject() / detect() interface as RealSenseCamera +
ObbDetector, so ObjectLocator(camera=live, detector=live) works unchanged,
and the camera is never opened twice.
"""
import threading
import time

import numpy as np

from dmp_assistant import paths


class LiveCamera(threading.Thread):
    def __init__(self, config, max_fps=10.0, camera=None, detector=None):
        super().__init__(name="live-camera", daemon=True)
        c, d = config["camera"], config["detector"]
        if camera is None:
            from dmp_assistant.perception.camera import RealSenseCamera
            camera = RealSenseCamera(c["width"], c["height"], c["fps"], c["warmup_frames"])
        if detector is None:
            from dmp_assistant.perception.detector import ObbDetector
            detector = ObbDetector(paths.YOLO_WEIGHTS, d["conf"], d["imgsz"])
        self.camera, self.detector = camera, detector
        self.period = 1.0 / max_fps

        self._cond = threading.Condition()
        self._halt = threading.Event()   # not _stop: Thread uses that name
        self._seq = 0
        self._bgr = self._depth = None
        self._dets = []
        self._read_dets = []
        self._jpeg = None
        self.error = None
        # Set by the locator while it measures, drawn on the live view.
        self.focus = None          # class being located
        self.note = None           # last measurement, e.g. "bottle (91, -567, 59) mm"

    # -------------------------------------------------------- thread --
    def run(self):
        while not self._halt.is_set():
            t0 = time.time()
            try:
                bgr, depth = self.camera.read()
                dets = self.detector.detect(bgr)
                jpeg = self._render(bgr, dets)
                self.error = None
            except Exception as e:                   # keep the thread alive
                self.error = f"{type(e).__name__}: {e}"
                time.sleep(0.5)
                continue
            with self._cond:
                self._bgr, self._depth, self._dets, self._jpeg = bgr, depth, dets, jpeg
                self._seq += 1
                self._cond.notify_all()
            time.sleep(max(0.0, self.period - (time.time() - t0)))

    def stop(self):
        self._halt.set()
        self.join(timeout=2)
        self.camera.close()

    # -------------------------------------- camera / detector interface --
    def read(self, timeout_s=5.0):
        """A frame grabbed AFTER this call (the robot has just stopped, so
        an older frame may still show it moving). -> (bgr, depth_mm)."""
        with self._cond:
            target = self._seq + 2
            if not self._cond.wait_for(lambda: self._seq >= target, timeout_s):
                raise RuntimeError(f"live camera: no new frame in {timeout_s:.0f} s ({self.error})")
            self._read_dets = self._dets
            return self._bgr, self._depth

    def detect(self, bgr, cls=None):
        """Detections of the frame last returned by read() (already computed)."""
        return [d for d in self._read_dets if cls is None or d.cls == cls]

    def deproject(self, u, v, depth_mm):
        return self.camera.deproject(u, v, depth_mm)

    # ------------------------------------------------------- web view --
    def jpeg(self, after_seq=0, timeout_s=2.0):
        """-> (seq, jpeg bytes) of a frame newer than after_seq, or the last one."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > after_seq, timeout_s)
            return self._seq, self._jpeg

    def _render(self, bgr, dets):
        import cv2
        view = bgr.copy()
        for d in dets:
            hot = self.focus is not None and d.cls == self.focus
            color = (0, 200, 255) if hot else (80, 220, 120)
            pts = np.round(d.corners).astype(np.int32)
            cv2.polylines(view, [pts], True, color, 3 if hot else 2)
            label = f"{d.cls} {d.conf:.2f}"
            x, y = int(pts[:, 0].min()), int(pts[:, 1].min()) - 6
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y = max(y, h + 4)
            cv2.rectangle(view, (x, y - h - 4), (x + w + 6, y + 2), color, -1)
            cv2.putText(view, label, (x + 3, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)
        status = f"locating {self.focus}" if self.focus else "watching"
        if self.note:
            status += f"  |  {self.note}"
        cv2.rectangle(view, (0, 0), (view.shape[1], 24), (20, 20, 20), -1)
        cv2.putText(view, status, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (230, 230, 230), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes() if ok else None
