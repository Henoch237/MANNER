"""Offline check of the web flow: face login -> plan -> execute -> logout.
dashscope, the face matcher and the robot are all faked - no network,
camera or arm. Run: python tests/test_web_session.py  (or pytest)."""
import base64
import io
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

# ---- stub dashscope BEFORE the planner is imported ----
fake = types.ModuleType("dashscope")
fake.api_key = None


class _Gen:
    queue = []
    last_messages = None

    @staticmethod
    def call(**kw):
        _Gen.last_messages = kw["messages"]
        content = _Gen.queue.pop(0)
        return types.SimpleNamespace(status_code=200, output=types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]))


fake.Generation = _Gen
sys.modules["dashscope"] = fake

from PIL import Image

from dmp_assistant.planning.planner import Planner
from dmp_assistant.web.logbuffer import LogBuffer
from dmp_assistant.web.server import AppState, REQUIRED_MATCHES, create_app


class FakeFaces:
    empty = False

    def __init__(self, name):
        self.name = name

    def identify(self, rgb):
        assert rgb.ndim == 3 and rgb.shape[2] == 3
        return "match", self.name, 0.31

    def load(self):
        return {}


class FakeExecutor:
    def __init__(self):
        self.state = {"location": "initial_pose", "holding": None,
                      "gripper": "unknown", "last_speed": 30, "last_accel": 100}
        self.plans = []

    def run_plan(self, plan):
        self.plans.append(plan)
        print("[EXECUTOR BT] Plan completed successfully.")
        return True


def jpeg_data_url():
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (120, 90, 60)).save(buf, "JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def envelope(plan, recipient="none", adjustment="none", store=False):
    return json.dumps({"intent": {"manner": "none", "adjustment": adjustment,
                                  "store": store, "recipient": recipient},
                       "plan": plan})


def test_web_flow(tmp_path=None):
    import tempfile
    tmp = Path(tmp_path or tempfile.mkdtemp())
    prefs = tmp / "prefs.yaml"
    prefs.write_text("stored: {}\n")

    orig_stdout = sys.stdout
    logs = LogBuffer.install()
    try:
        robot = FakeExecutor()
        state = AppState(FakeFaces("sarah"), logs,
                         lambda: Planner(api_key="fake", preferences_path=prefs),
                         lambda: robot, roster=["john", "sarah"])
        app = create_app(state)
        c = app.test_client()

        # Not logged in: login page, APIs refused
        assert "Sign in" in c.get("/").get_data(as_text=True)
        assert c.post("/api/message", json={"text": "hi"}).status_code == 401
        assert c.get("/api/status").status_code == 401

        # Face login needs REQUIRED_MATCHES consecutive matches
        for i in range(1, REQUIRED_MATCHES):
            r = c.post("/api/auth/identify", json={"image": jpeg_data_url()}).get_json()
            assert r == {"status": "verifying", "progress": i, "required": REQUIRED_MATCHES}, r
        r = c.post("/api/auth/identify", json={"image": jpeg_data_url()}).get_json()
        assert r == {"status": "recognized", "name": "sarah"}, r

        page = c.get("/").get_data(as_text=True)
        assert "Sarah" in page and "ROBOT" in page and "Sign out" in page and "Logs" in page

        # Garbage image is rejected
        assert c.post("/api/auth/identify", json={"image": "not-base64!"}).status_code == 400

        # A request -> plan with sarah's (faster profile) register
        _Gen.queue = [envelope([{"action": "stir", "cup_id": "coffee_cup"}])]
        r = c.post("/api/message", json={"text": "stir the coffee"}).get_json()
        assert r["kind"] == "plan", r
        step = r["plan"][0]
        assert (step["speed"], step["accel"], step["omega_scale"]) == (45, 45, 1.0), step
        # The LLM is told the authenticated user, not people.yaml's default
        assert "CURRENT USER (default recipient when the request names no one):\n'sarah'" \
            in _Gen.last_messages[0]["content"]
        # Structured answer for the chat; raw lines only go to the log panel
        assert r["intent"]["recipient"] == "none" and r["raw"].startswith("{"), r
        assert r["notices"] == [] and r["pending_store"] is None, r
        logs_now = c.get("/api/status").get_json()["logs"]
        assert any(l["text"] == "[PLANNER] Final plan:" and l["t"] for l in logs_now)

        # Execute in the background, poll until idle
        assert c.post("/api/execute").status_code == 202
        for _ in range(100):
            s = c.get("/api/status").get_json()
            if s["phase"] == "idle":
                break
            time.sleep(0.02)
        assert s["last_run"] == {"ok": True}, s
        assert s["robot_state"]["location"] == "initial_pose"
        assert robot.plans == [[step]]
        assert c.post("/api/execute").status_code == 409     # nothing pending anymore

        # Session commands; identity switching is disabled on the web
        r = c.post("/api/message", json={"text": "user john"}).get_json()
        assert r["kind"] == "command"
        assert r["notices"][0]["text"].startswith("Switching user is disabled"), r
        r = c.post("/api/message", json={"text": "state"}).get_json()
        assert r["robot_state"]["location"] == "initial_pose", r

        # Unknown person -> friendly question, no API call spent
        r = c.post("/api/message", json={"text": "stir for gertrude"}).get_json()
        assert r["kind"] == "no_plan" and r["notices"][0]["level"] == "error", r
        assert "gertrude" in r["notices"][0]["text"], r

        # Slower + remember: drift, store question, and the plan survives the "yes"
        _Gen.queue = [envelope([{"action": "pour"}], adjustment="down", store=True)]
        r = c.post("/api/message", json={"text": "pour, slower, and remember it"}).get_json()
        assert r["kind"] == "plan" and r["drift"] == {"before": 1.0, "after": 0.8}, r
        assert abs(r["pending_store"]["would_be"] - 0.8) < 1e-9, r
        r = c.post("/api/message", json={"text": "yes"}).get_json()
        assert r["notices"] == [{"level": "success",
                                 "text": "Saved as your default speed (multiplier 1.00 → 0.80)."}], r
        assert c.get("/api/status").get_json()["pending_plan"] is not None
        assert c.post("/api/discard").status_code == 200

        # Plan then discard
        _Gen.queue = [envelope([{"action": "pour"}])]
        assert c.post("/api/message", json={"text": "pour"}).get_json()["kind"] == "plan"
        assert c.post("/api/discard").status_code == 200
        assert c.get("/api/status").get_json()["pending_plan"] is None

        # Enrollment is token-protected
        assert c.get("/enroll?token=wrong").status_code == 403
        assert c.get(f"/enroll?token={state.enroll_token}").status_code == 200
        assert c.post("/api/enroll", json={"token": "wrong"}).status_code == 403

        # Logout closes the session
        assert c.post("/api/auth/logout").status_code == 200
        assert c.get("/api/status").status_code == 401
        assert "Sign in" in c.get("/").get_data(as_text=True)
    finally:
        sys.stdout = orig_stdout


class FakeLive:
    """Stand-in for perception.live.LiveCamera: numbered fake JPEG frames."""
    error = None

    def __init__(self):
        self.seq = 0

    def jpeg(self, after_seq=0, timeout_s=2.0):
        self.seq = max(self.seq, after_seq) + 1
        return self.seq, b"\xff\xd8frame%d\xff\xd9" % self.seq


def login(c):
    for _ in range(REQUIRED_MATCHES):
        r = c.post("/api/auth/identify", json={"image": jpeg_data_url()}).get_json()
    assert r["status"] == "recognized", r


def test_web_camera():
    orig_stdout = sys.stdout
    logs = LogBuffer.install()
    try:
        # Live camera present: status + MJPEG stream, both behind the login
        state = AppState(FakeFaces("sarah"), logs, lambda: Planner(api_key="fake"),
                         FakeExecutor, roster=["sarah"], live=FakeLive())
        c = create_app(state).test_client()
        assert c.get("/api/camera/status").status_code == 401
        assert c.get("/api/camera.mjpg").status_code == 401
        login(c)
        assert 'id="cam-img"' in c.get("/").get_data(as_text=True)
        assert c.get("/api/camera/status").get_json() == {"available": True, "error": None}
        r = c.get("/api/camera.mjpg", buffered=False)
        assert r.status_code == 200 and r.mimetype == "multipart/x-mixed-replace"
        chunks = r.response
        first, second = next(chunks), next(chunks)
        assert first.startswith(b"--frame\r\nContent-Type: image/jpeg") and b"frame1" in first
        assert b"frame2" in second
        state.active = None                    # session closed -> the stream ends
        assert list(chunks) == []
        r.close()

        # No camera: the page still works, the view reports why
        state = AppState(FakeFaces("sarah"), logs, lambda: Planner(api_key="fake"),
                         FakeExecutor, roster=["sarah"], live=None,
                         live_error="RuntimeError: No RealSense camera detected.")
        c = create_app(state).test_client()
        login(c)
        st = c.get("/api/camera/status").get_json()
        assert st["available"] is False and "RealSense" in st["error"], st
        assert c.get("/api/camera.mjpg").status_code == 503
    finally:
        sys.stdout = orig_stdout


if __name__ == "__main__":
    test_web_flow()
    test_web_camera()
    print("web session flow: all checks green")
