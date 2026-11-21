"""Web front-end: face login, then the robot session in the browser.

Run from the project root:
  export DASHSCOPE_API_KEY=...
  python -m dmp_assistant.web.server            # opens http://127.0.0.1:5000

Flow: the login page streams webcam frames to /api/auth/identify; after
REQUIRED_MATCHES consecutive frames recognised as the same enrolled person,
a Session is opened for that person (planner.current_user = them) and the
browser switches to the session page. One active session at a time - there
is one robot. The server listens on 127.0.0.1 only.

Enrollment page (webcam capture of new faces): the URL with a one-time
token is printed in the terminal at start-up.
"""
import argparse
import base64
import io
import os
import secrets
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

if __name__ == "__main__":      # direct run: make the project root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import yaml
from flask import (Flask, Response, abort, jsonify, redirect, render_template, request,
                   session, url_for)
from PIL import Image

from dmp_assistant import paths
from dmp_assistant.auth.face_auth import FaceDB, encode_single_face
from dmp_assistant.session import Session
from dmp_assistant.web.asr import transcribe_wav
from dmp_assistant.web.logbuffer import LogBuffer
from dmp_assistant.web.present import notices

ROBOT_IP = "192.168.58.4"       # real robot (sim: 192.168.85.130)
REQUIRED_MATCHES = 3            # consecutive frames agreeing on one person
MAX_IMAGE_BYTES = 4 * 1024 * 1024

WEB_DIR = Path(__file__).resolve().parent


def _load_roster():
    with open(paths.PEOPLE_YAML, "r", encoding="utf-8") as f:
        return list((yaml.safe_load(f) or {}).get("people", {}))


def _decode_image(data_url):
    """'data:image/jpeg;base64,...' (or bare base64) -> RGB uint8 array."""
    if not isinstance(data_url, str) or not data_url:
        abort(400, "missing image")
    b64 = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
    try:
        raw = base64.b64decode(b64, validate=True)
    except ValueError:
        abort(400, "image is not valid base64")
    if len(raw) > MAX_IMAGE_BYTES:
        abort(413, "image too large")
    try:
        return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception:
        abort(400, "unreadable image")


class AppState:
    """Server-side state shared by all requests (guarded by self.lock)."""

    def __init__(self, faces, logs, planner_factory, executor_factory, roster, live=None,
                 live_error=None):
        self.lock = threading.Lock()
        self.live = live                  # perception LiveCamera, or None
        self.live_error = live_error      # why there is no live camera
        self.locator = None               # ObjectLocator on the live camera
        self.faces = faces
        self.logs = logs
        self.roster = roster
        self.planner_factory = planner_factory
        self._executor_factory = executor_factory
        self._executor = None             # one robot connection for all sessions
        self.enroll_token = secrets.token_urlsafe(16)
        self.active = None                # {"sid", "user", "session", "log_start"}
        self.phase = "idle"               # idle | planning | executing
        self.last_run = None              # {"ok": bool} after an execution

    def shared_executor(self):
        if self._executor is None:
            self._executor = self._executor_factory()
        return self._executor


LIVE_RETRY_S = 10.0


def keep_live_camera(state):
    """Open the RealSense + YOLO thread, retrying until it works (camera
    plugged in late, or stuck after a killed process). Once it runs, the
    shared executor's picks use it, so the camera is never opened twice."""
    from dmp_assistant.perception.live import LiveCamera
    from dmp_assistant.perception.locator import ObjectLocator, load_config
    cfg = load_config()
    last = None
    while state.live is None:
        try:
            live = LiveCamera(cfg)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            state.live_error = err
            if err != last:
                print(f"[PERCEPTION WARN] no live camera: {err} (retrying every "
                      f"{LIVE_RETRY_S:.0f} s)")
                last = err
            time.sleep(LIVE_RETRY_S)
            continue
        live.start()
        with state.lock:
            state.locator = ObjectLocator(config=cfg, camera=live, detector=live)
            state.live, state.live_error = live, None
            if state._executor is not None and state._executor._locator is None:
                state._executor._locator = state.locator
        print("[PERCEPTION] live camera started (RealSense + YOLO)")


def create_app(state):
    app = Flask(__name__, template_folder=str(WEB_DIR / "templates"),
                static_folder=str(WEB_DIR / "static"))
    app.secret_key = secrets.token_bytes(32)     # new each run: old cookies die
    app.config["MAX_CONTENT_LENGTH"] = 2 * MAX_IMAGE_BYTES
    logo = next((f"img/{p.name}" for p in (WEB_DIR / "static" / "img").glob("robot_logo.*")), None)

    def current():
        """The active session if this browser owns it, else None."""
        a = state.active
        if a is None or session.get("sid") != a["sid"]:
            return None
        return a

    def require_session():
        a = current()
        if a is None:
            abort(401)
        return a

    # ------------------------------------------------------------ pages --
    @app.get("/")
    def index():
        a = current()
        if a is None:
            return render_template("login.html", logo=logo)
        return render_template("app.html", logo=logo, user=a["user"])

    @app.get("/enroll")
    def enroll_page():
        if request.args.get("token") != state.enroll_token:
            abort(403)
        return render_template("enroll.html", logo=logo, roster=sorted(state.roster),
                               token=state.enroll_token)

    # ------------------------------------------------------------- auth --
    @app.post("/api/auth/identify")
    def identify():
        body = request.get_json(silent=True) or {}
        rgb = _decode_image(body.get("image"))
        status, name, dist = state.faces.identify(rgb)
        if status != "match":
            session.pop("streak_name", None)
            session["streak_n"] = 0
            return jsonify(status=status)

        if session.get("streak_name") == name:
            session["streak_n"] = session.get("streak_n", 0) + 1
        else:
            session["streak_name"], session["streak_n"] = name, 1
        if session["streak_n"] < REQUIRED_MATCHES:
            return jsonify(status="verifying", progress=session["streak_n"],
                           required=REQUIRED_MATCHES)

        with state.lock:
            if state.phase != "idle":
                return jsonify(status="busy"), 409
            planner = state.planner_factory()
            planner.set_current_user(name)
            sid = secrets.token_urlsafe(16)
            state.active = {
                "sid": sid, "user": name,
                "session": Session(planner, state.shared_executor,
                                   allow_user_switch=False),
                "log_start": state.logs.next_seq,
            }
            state.last_run = None
        session.clear()
        session["sid"] = sid
        print(f"[AUTH] session opened for '{name}' (distance {dist:.3f})")
        return jsonify(status="recognized", name=name)

    @app.post("/api/auth/logout")
    def logout():
        with state.lock:
            a = current()
            if a is not None:
                if state.phase == "executing":
                    return jsonify(error="the robot is executing - wait for the end"), 409
                print(f"[AUTH] session closed for '{a['user']}'")
                state.active = None
        session.clear()
        return jsonify(ok=True)

    # ---------------------------------------------------------- session --
    @app.post("/api/message")
    def message():
        a = require_session()
        text = ((request.get_json(silent=True) or {}).get("text") or "").strip()
        if not text:
            abort(400, "empty message")
        with state.lock:
            if state.phase != "idle":
                return jsonify(error="busy"), 409
            state.phase = "planning"
        s = a["session"]
        planner = s.planner
        drift_before = planner.session_drift
        planner.last_intent = planner.last_raw = None
        start = state.logs.next_seq
        error = None
        try:
            print(f"\n[{a['user']}] > {text}")
            reply = s.handle(text)
        except Exception as e:                      # e.g. network error to DashScope
            print(f"[SESSION ERROR] {type(e).__name__}: {e}")
            reply, error = None, "The request failed unexpectedly - see the logs."
        finally:
            with state.lock:
                state.phase = "idle"
        kind = reply.kind if reply else "error"
        if kind == "quit":
            kind = "command"
            error = None
            print("[SESSION] use the Sign out button to close the web session.")
        return jsonify(
            kind=kind,
            plan=reply.plan if reply else None,
            intent=planner.last_intent,            # Qwen's intent header, if any
            raw=planner.last_raw,                  # Qwen's raw answer, if any
            notices=notices(state.logs.since(start)),
            drift={"before": drift_before, "after": planner.session_drift},
            pending_store=planner.pending_store,   # "remember this speed?" question
            robot_state=s.state() if text.lower() == "state" else None,
            connected=s.state() is not None,
            error=error)

    @app.post("/api/execute")
    def execute():
        a = require_session()
        with state.lock:
            if state.phase != "idle":
                return jsonify(error="busy"), 409
            if a["session"].pending_plan is None:
                return jsonify(error="no plan waiting for confirmation"), 409
            state.phase = "executing"
            state.last_run = None

        def run():
            ok = False
            try:
                ok = a["session"].execute()
            except Exception as e:
                print(f"[EXECUTOR ERROR] {type(e).__name__}: {e}")
            finally:
                with state.lock:
                    state.last_run = {"ok": bool(ok)}
                    state.phase = "idle"

        threading.Thread(target=run, name="robot-execution", daemon=True).start()
        return jsonify(started=True), 202

    @app.post("/api/discard")
    def discard():
        a = require_session()
        with state.lock:
            if state.phase != "idle":
                return jsonify(error="busy"), 409
            a["session"].discard()
        return jsonify(ok=True)

    @app.post("/api/asr")
    def asr():
        """Speech-to-text for the composer: the browser posts a 16 kHz mono
        WAV recorded while the mic button was held."""
        a = require_session()
        upload = request.files.get("audio")
        if upload is None or not upload.filename:
            abort(400, "missing audio file")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            upload.save(tmp_path)
            print(f"[ASR] transcribing {os.path.getsize(tmp_path)} bytes "
                  f"of audio for '{a['user']}'...")
            text = transcribe_wav(tmp_path)
        except Exception as e:
            print(f"[ASR ERROR] {type(e).__name__}: {e}")
            return jsonify(ok=False, error="transcription failed - see the logs"), 502
        finally:
            os.unlink(tmp_path)
        print(f"[ASR] '{a['user']}' said: {text!r}")
        return jsonify(ok=True, text=text)

    # ----------------------------------------------------- live camera --
    @app.get("/api/camera/status")
    def camera_status():
        require_session()
        live = state.live
        return jsonify(available=live is not None,
                       error=state.live_error if live is None else live.error)

    @app.get("/api/camera.mjpg")
    def camera_stream():
        """MJPEG of the perception camera with the YOLO boxes. Ends when the
        session that opened it is closed."""
        sid = require_session()["sid"]
        live = state.live
        if live is None:
            return jsonify(error=state.live_error or "no camera"), 503

        def frames():
            seq = 0
            while state.active is not None and state.active["sid"] == sid:
                seq, jpg = live.jpeg(seq)
                if jpg is None:
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")

        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    def status():
        a = require_session()
        since = max(request.args.get("since", 0, type=int), a["log_start"])
        s = a["session"]
        return jsonify(phase=state.phase, last_run=state.last_run,
                       pending_plan=s.pending_plan, robot_state=s.state(),
                       logs=state.logs.since(since), next=state.logs.next_seq)

    # ------------------------------------------------------- enrollment --
    @app.post("/api/enroll")
    def enroll():
        body = request.get_json(silent=True) or {}
        if body.get("token") != state.enroll_token:
            abort(403)
        name = body.get("name")
        if name not in state.roster:
            return jsonify(ok=False, error="unknown person (not in people.yaml)"), 400
        rgb = _decode_image(body.get("image"))
        _, problem = encode_single_face(rgb)
        if problem:
            return jsonify(ok=False, error=problem)
        person_dir = paths.FACES_DIR / name
        person_dir.mkdir(parents=True, exist_ok=True)
        n = len(list(person_dir.glob("*.jpg")))
        Image.fromarray(rgb).save(person_dir / f"{name}_{n + 1:02d}.jpg", quality=92)
        with state.lock:
            counts = state.faces.load()
        return jsonify(ok=True, photos=counts.get(name, 0))

    @app.errorhandler(401)
    def unauthorized(_):
        return jsonify(error="not authenticated"), 401

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--robot-ip", default=ROBOT_IP)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-camera", action="store_true",
                    help="no live view; the RealSense is then opened only for a pick")
    args = ap.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        sys.exit("DASHSCOPE_API_KEY is not set. Run: export DASHSCOPE_API_KEY=<your key>")

    logs = LogBuffer.install()               # tee stdout into the web log

    from dmp_assistant.planning.planner import Planner
    from dmp_assistant.execution.executor import Executor

    roster = _load_roster()
    faces = FaceDB(paths.FACES_DIR, roster)
    try:
        faces.load()
    except RuntimeError as e:
        sys.exit(f"Face authentication unavailable: {e}")
    state = AppState(faces, logs, Planner,
                     lambda: Executor(args.robot_ip, locator=state.locator), roster)
    if not args.no_camera:
        threading.Thread(target=keep_live_camera, args=(state,), name="live-camera-start",
                         daemon=True).start()
    app = create_app(state)

    base = f"http://127.0.0.1:{args.port}"
    print("=" * 60)
    print(f"Session page : {base}/")
    print(f"Enrollment   : {base}/enroll?token={state.enroll_token}")
    if faces.empty:
        print("No face enrolled yet - open the enrollment page first.")
    print("=" * 60)
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(base + "/",)).start()
    try:
        app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)
    finally:
        if state.live is not None:
            state.live.stop()


if __name__ == "__main__":
    main()
