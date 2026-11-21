"""Offline checks of the camera pick: frame math, hand-eye solver, locator
fusion and the executor's detected-pick sequence. No camera, model or arm.
Run: python tests/test_perception.py  (or pytest)."""
import math
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

import numpy as np

from dmp_assistant.perception import geometry as geo
from dmp_assistant.perception.detector import Detection
from dmp_assistant.perception.locator import (LocateError, ObjectLocator, fuse,
                                              load_config, load_hand_eye, measure)

RNG = np.random.default_rng(0)
SCAN = [102.26, -383.20, 493.60, 179.21, 4.76, 91.51]   # scene.yaml initial_pose


def base_config():
    """perception.yaml with the site tuning that would move the expected
    numbers neutralised, so the tests do not depend on today's tuning."""
    cfg = load_config()
    cfg["locate"]["offset_mm"] = [0.0, 0.0, 0.0]
    return cfg


def T_FC():
    """Fixed camera mount for the tests (grasping-v6 T_gc, mm), so they do
    not depend on whatever config/hand_eye.yaml currently holds."""
    T = np.array([[0.01629, 0.95060, 0.30999, -96.14],
                  [-0.99983, 0.01268, 0.01364, 28.99],
                  [0.00903, -0.31015, 0.95064, 52.59],
                  [0, 0, 0, 1.0]])
    U, _, Vt = np.linalg.svd(T[:3, :3])
    T[:3, :3] = U @ Vt
    return T


def random_T(max_deg=30, max_mm=200):
    axis = RNG.normal(size=3)
    R = geo.euler_to_R(*RNG.uniform(-max_deg, max_deg, 3))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = RNG.uniform(-max_mm, max_mm, 3)
    return T


# ---------------------------------------------------------------- geometry --
def test_euler_roundtrip():
    for _ in range(50):
        e = list(RNG.uniform([-179, -80, -179], [179, 80, 179]))
        assert np.allclose(geo.R_to_euler(geo.euler_to_R(*e)), e, atol=1e-6)


def test_grasp_rz_folds_to_nearest():
    # long axis along base X, tool X must follow it (jaw_axis y)
    assert math.isclose(geo.grasp_rz(0.0, 91.5), 0.0, abs_tol=1e-9) or \
        math.isclose(geo.grasp_rz(0.0, 91.5), 180.0, abs_tol=1e-9)
    for ref in (-170, -30, 0, 45, 91.5, 170):
        for yaw in (-120, -45, 0, 30, 89, 179):
            rz = geo.grasp_rz(yaw, ref)
            assert abs(rz - ref) <= 90 + 1e-9                     # smallest wrist turn
            assert math.isclose(geo.wrap180(2 * (rz - yaw)), 0, abs_tol=1e-9)  # same axis
            rz_x = geo.grasp_rz(yaw, ref, jaw_axis="x")
            assert math.isclose(abs(geo.wrap180(2 * (rz_x - yaw))), 180, abs_tol=1e-9)


def test_fingertip_to_flange_top_down():
    # rx=180: tool Z points down, the flange is gripper_length ABOVE the tip
    fl = geo.fingertip_to_flange([100, -500, 10], [180, 0, 90], 220)
    assert np.allclose(fl[:3], [100, -500, 230])


def test_obb_long_axis():
    a, asp = geo.obb_long_axis(100, 20, 0.3)
    assert math.isclose(a, 0.3) and math.isclose(asp, 5)
    a, asp = geo.obb_long_axis(20, 100, 0.3)
    assert math.isclose(a, 0.3 + math.pi / 2) and math.isclose(asp, 5)


def test_axis_mean_handles_wrap():
    assert abs(geo.wrap180(2 * (geo.axis_mean_deg([89, -89, 88]) - 89.3))) < 2


# ---------------------------------------------------------------- hand-eye --
def synthetic_captures(X, n=15, noise_mm=0.0, noise_deg=0.0):
    board = random_T(10, 50)
    board[:3, 3] += [0, -500, 0]
    F, C = [], []
    for _ in range(n):
        f = geo.pose_to_T([RNG.uniform(-100, 100), RNG.uniform(-450, -300),
                           RNG.uniform(350, 500), 180 + RNG.uniform(-25, 25),
                           RNG.uniform(-25, 25), RNG.uniform(60, 120)])
        c = np.linalg.inv(f @ X) @ board                       # board in camera
        if noise_mm or noise_deg:
            c[:3, :3] = c[:3, :3] @ geo.euler_to_R(*RNG.normal(0, noise_deg, 3))
            c[:3, 3] += RNG.normal(0, noise_mm, 3)
        F.append(f)
        C.append(c)
    return F, C


def test_hand_eye_yaml_loads():
    T = load_hand_eye()
    assert T.shape == (4, 4) and np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-9)


def test_hand_eye_park_exact():
    X = T_FC()
    F, C = synthetic_captures(X)
    est = geo.hand_eye_park(F, C)
    assert np.allclose(est[:3, :3], X[:3, :3], atol=1e-6)
    assert np.allclose(est[:3, 3], X[:3, 3], atol=1e-3)
    pos, ang = geo.target_in_base_spread(F, C, est)
    assert pos < 1e-3 and ang < 1e-3


def test_hand_eye_park_noisy():
    X = random_T(40, 100)
    F, C = synthetic_captures(X, n=20, noise_mm=0.5, noise_deg=0.1)
    est = geo.hand_eye_park(F, C)
    assert np.linalg.norm(est[:3, 3] - X[:3, 3]) < 5.0
    pos, _ = geo.target_in_base_spread(F, C, est)
    assert pos < 3.0


def test_calibrate_tool_solve_all():
    from dmp_assistant.tools.calibrate_hand_eye import solve_all
    X = random_T(40, 100)
    F, C = synthetic_captures(X, n=12)
    poses = [geo.T_to_pose(f) for f in F]
    res = solve_all(poses, C)
    best = min(res.values(), key=lambda r: r[1])
    assert np.allclose(best[0][:3, 3], X[:3, 3], atol=0.1)


def test_calibrate_flipped_board_numbering():
    """Some captures number the 9x6 board from the other end (180 deg):
    canonical_board must bring them back to one board frame."""
    from dmp_assistant.tools.calibrate_hand_eye import canonical_board, compute_and_save
    flip = np.eye(4)
    flip[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    flip[:3, 3] = [200.0, 125.0, 0.0]
    for trial in range(5):                       # any board orientation in the image
        X = random_T(40, 100)
        F, C = synthetic_captures(X, n=12)
        mixed = [c @ flip if RNG.random() < 0.5 else c for c in C]
        poses = [geo.T_to_pose(f) for f in F]
        est = compute_and_save(poses, mixed, (9, 6, 25.0), save=False)
        assert est is not None and np.allclose(est[:3, 3], X[:3, 3], atol=0.1), trial
    # the per-capture image rule alone is idempotent
    assert np.allclose(canonical_board(canonical_board(C[0], 9, 6, 25.0), 9, 6, 25.0),
                       canonical_board(C[0], 9, 6, 25.0))


def test_calibrate_ignores_repeated_poses():
    from dmp_assistant.tools.calibrate_hand_eye import compute_and_save, distinct_captures
    X = random_T(40, 100)
    F, C = synthetic_captures(X, n=3)
    poses = [geo.T_to_pose(f) for f in F]
    rep_p, rep_c = poses[:1] * 10 + poses[1:] * 10, C[:1] * 10 + C[1:] * 10
    assert len(distinct_captures(rep_p, rep_c)[0]) == 3
    assert compute_and_save(rep_p, rep_c, save=False) is None      # 3 poses < minimum


# ---------------------------------------------------------------- locator --
class Pinhole:
    """Stand-in for RealSense deproject: ideal pinhole, 640x480."""
    fx = fy = 600.0
    cx, cy = 320.0, 240.0

    def deproject(self, u, v, z):
        return np.array([(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z])

    def project(self, p):
        return np.array([p[0] * self.fx / p[2] + self.cx, p[1] * self.fy / p[2] + self.cy])


def box(cx, cy, w, h, r, cls="bottle", conf=0.9):
    c, s = math.cos(r), math.sin(r)
    ex, ey = np.array([c, s]) * w / 2, np.array([-s, c]) * h / 2
    pts = np.array([[cx, cy]]) + np.array([ex + ey, -ex + ey, -ex - ey, ex - ey])
    return Detection(cls, conf, cx, cy, w, h, r, pts)


def scene_for(obj_base, T_fc, flange, cam=Pinhole()):
    """Pixel of a base point seen from `flange`, and its camera depth."""
    T_cam_base = np.linalg.inv(geo.pose_to_T(flange) @ T_fc)
    p = (T_cam_base @ np.append(obj_base, 1))[:3]
    return cam.project(p), p[2]


def test_measure_recovers_base_point_and_yaw():
    cfg = base_config()
    T_fc = T_FC()
    cam = Pinhole()
    obj = np.array([150.0, -420.0, 80.0])                        # top of the object
    (u, v), z = scene_for(obj, T_fc, SCAN, cam)
    assert 0 < u < 640 and 0 < v < 480, "test object must be in view"

    depth = np.full((480, 640), z + 80.0, np.float32)           # table further away
    det = box(u, v, 120, 40, 0.4)
    yy, xx = np.mgrid[0:480, 0:640]
    depth[(abs(xx - u) < 50) & (abs(yy - v) < 30)] = z          # object top
    top, yaw = measure(det, depth, SCAN, T_fc, cam.deproject, cfg)
    assert np.linalg.norm(top - obj) < 1.0
    assert math.isclose(yaw, geo.image_axis_to_base_yaw(0.4, SCAN, T_fc), abs_tol=1e-9)

    round_det = box(u, v, 50, 48, 0.4)
    assert measure(round_det, depth, SCAN, T_fc, cam.deproject, cfg)[1] is None

    assert measure(det, np.zeros_like(depth), SCAN, T_fc, cam.deproject, cfg) is None


def test_fuse_rules():
    s = [(np.array([0.0, -500, 50]), 10.0, 0.9)] * 3
    t = fuse(s, "bottle", 15, 3)
    assert t.hits == 3 and math.isclose(t.yaw, 10.0, abs_tol=1e-9)
    try:
        fuse(s[:2], "bottle", 15, 3)
        assert False, "too few hits must fail"
    except LocateError:
        pass
    far = s[:2] + [(np.array([40.0, -500, 50]), 10.0, 0.9)]
    try:
        fuse(far, "bottle", 15, 3)
        assert False, "unstable position must fail"
    except LocateError:
        pass
    mostly_round = [(np.array([0.0, -500, 50]), None, 0.9)] * 2 + [s[0]]
    assert fuse(mostly_round, "cylinder", 15, 3).yaw is None


class FakeWorld:
    """One object on the table, rendered from wherever the flange is."""

    def __init__(self, obj_base, T_fc, flange=lambda: SCAN):
        self.obj, self.T_fc, self.flange = obj_base, T_fc, flange
        self.cam = Pinhole()
        self.deproject = self.cam.deproject
        self._dets = []

    def read(self):
        (u, v), z = scene_for(self.obj, self.T_fc, self.flange(), self.cam)
        depth = np.full((480, 640), z + 80.0, np.float32)       # table further away
        yy, xx = np.mgrid[0:480, 0:640]
        depth[(abs(xx - u) < 50) & (abs(yy - v) < 30)] = z      # object top
        self._dets = [box(u, v, 120, 40, 0.4)]
        return np.zeros((480, 640, 3), np.uint8), depth

    def detect(self, bgr, cls=None):
        return [d for d in self._dets if cls is None or d.cls == cls]


def make_locator(obj_base, flange=lambda: SCAN):
    T_fc = T_FC()
    world = FakeWorld(obj_base, T_fc, flange)
    return ObjectLocator(config=base_config(), T_flange_cam=T_fc, camera=world, detector=world)


def test_live_camera_feeds_locator_and_view():
    """LiveCamera (one thread: camera + YOLO) behind ObjectLocator: the pick
    reads fresh frames and reuses the thread's detections; the web view gets
    JPEGs with the status line."""
    from dmp_assistant.perception.live import LiveCamera
    obj = np.array([150.0, -420.0, 80.0])
    world = FakeWorld(obj, T_FC())
    world.close = lambda: None
    calls = {"detect": 0}

    class CountingDetector:
        def detect(self, bgr, cls=None):
            calls["detect"] += 1
            return world.detect(bgr, cls)

    cfg = base_config()
    live = LiveCamera(cfg, max_fps=200, camera=world, detector=CountingDetector())
    live.start()
    try:
        loc = ObjectLocator(config=cfg, T_flange_cam=T_FC(), camera=live, detector=live)
        n_before = calls["detect"]
        t = loc.locate("bottle", SCAN)
        assert np.linalg.norm(t.top - obj) < 1.0
        assert calls["detect"] - n_before >= cfg["locate"]["frames"]   # thread ran YOLO
        assert live.focus is None and live.note.startswith("bottle at (150, -420, 80)")
        seq, jpg = live.jpeg()
        assert seq > 0 and jpg[:2] == b"\xff\xd8"
        try:
            loc.locate("hollow", SCAN)
            assert False
        except LocateError:
            assert live.note.startswith("hollow:")
    finally:
        live.stop()


def test_locate_and_grasp_poses():
    obj = np.array([150.0, -420.0, 80.0])
    loc = make_locator(obj)
    t = loc.locate("bottle", SCAN)
    assert np.linalg.norm(t.top - obj) < 1.0 and t.hits == loc.cfg["locate"]["frames"]
    hover, grasp = loc.grasp_poses(t, SCAN)
    g = loc.cfg["grasp"]
    depth = loc.cfg["classes"]["bottle"]["depth_mm"]
    R = geo.euler_to_R(*grasp[3:6])
    tip = np.array(grasp[:3]) + R @ [0, 0, g["gripper_length_mm"]]
    assert np.allclose(tip, [obj[0], obj[1], obj[2] - depth], atol=1.0)
    assert hover[2] - grasp[2] > 0 and np.allclose(hover[:2], grasp[:2])
    assert grasp[3:5] == SCAN[3:5] and abs(grasp[5] - SCAN[5]) <= 90

    try:
        loc.locate("hollow", SCAN)                          # not on the table
        assert False
    except LocateError:
        pass


def test_offset_mm_shifts_the_located_point():
    obj = np.array([150.0, -420.0, 80.0])
    loc = make_locator(obj)
    loc.cfg["locate"]["offset_mm"] = [-8.0, 5.0, 0.0]
    t = loc.locate("bottle", SCAN)
    assert np.linalg.norm(t.top - (obj + [-8.0, 5.0, 0.0])) < 1.0


def test_perception_yaml_is_usable():
    cfg = load_config()
    assert len(cfg["locate"]["offset_mm"]) == 3
    assert cfg["grasp"]["table_z_mm"] < cfg["workspace"]["z"][0]


def test_grasp_never_below_table():
    loc = make_locator(np.array([150.0, -420.0, 10.0]))      # flat object
    t = loc.locate("bottle", SCAN)
    _, grasp = loc.grasp_poses(t, SCAN)
    tip_z = grasp[2] + (geo.euler_to_R(*grasp[3:6]) @ [0, 0, 220.0])[2]
    assert tip_z >= loc.cfg["grasp"]["table_z_mm"] - 1e-6


def test_workspace_guard():
    loc = make_locator(np.array([150.0, -420.0, 80.0]))
    t = loc.locate("bottle", SCAN)
    t.top = np.array([900.0, -420.0, 80.0])
    try:
        loc.grasp_poses(t, SCAN)
        assert False
    except LocateError:
        pass


# ---------------------------------------------------------------- executor --
class FakeRobot:
    def __init__(self, gripper_pos=40):
        self.moves = []
        self.flange = list(SCAN)
        self.gripper_pos = gripper_pos

    def GetActualToolFlangePose(self, flag=0):
        return 0, list(self.flange)

    def GetGripperCurPosition(self):
        return 0, 0, self.gripper_pos


_restore = []


def restore_executor_module():
    while _restore:
        _restore.pop()()


def make_executor(calls, gripper_pos=40, verify=True):
    import dmp_assistant.execution.executor as ex
    robot = FakeRobot(gripper_pos)

    def movel(r, pose, vel=30, acc=100):
        calls.append(("movel", [round(v, 1) for v in pose]))
        r.flange = list(pose)
        return 0

    patches = {"movel": movel,
               "gripper_open": lambda r: calls.append(("open",)),
               "gripper_close": lambda r: calls.append(("close",)),
               "time": types.SimpleNamespace(sleep=lambda s: None)}
    saved = {k: getattr(ex, k) for k in patches}
    for k, v in patches.items():
        setattr(ex, k, v)
    _restore.append(lambda: [setattr(ex, k, v) for k, v in saved.items()])

    e = ex.Executor.__new__(ex.Executor)
    e.robot = robot
    e.state = {"location": "initial_pose", "holding": None, "gripper": "unknown",
               "last_speed": 30, "last_accel": 100}
    e._locator = make_locator(np.array([150.0, -420.0, 80.0]), lambda: robot.flange)
    e._locator.cfg = {**e._locator.cfg,
                      "grasp": {**e._locator.cfg["grasp"], "verify_grasp": verify}}
    return e


def test_executor_detected_pick_sequence():
    calls = []
    e = make_executor(calls)
    try:
        e._pick("bottle", 25, 25)
    finally:
        restore_executor_module()
    kinds = [c[0] for c in calls]
    # scan pose, open, hover, grasp, close, retract
    assert kinds == ["movel", "open", "movel", "movel", "close", "movel"], kinds
    assert calls[1 + 1][1] == calls[5][1]                       # retract == hover
    assert e.state["holding"] == "bottle"


def test_executor_empty_grasp_fails():
    calls = []
    e = make_executor(calls, gripper_pos=0)
    try:
        e._pick("bottle", 25, 25)
        assert False, "closing on nothing must fail the pick"
    except RuntimeError as err:
        assert "not grasped" in str(err)
    finally:
        restore_executor_module()
    assert e.state["holding"] is None
    assert [c[0] for c in calls][-2:] == ["open", "movel"]      # release, back to hover


def test_pick_without_grasp_check_keeps_object():
    """verify_grasp off (this robot reads gripper position 0 even when
    holding): a 0 reading must not release the object."""
    calls = []
    e = make_executor(calls, gripper_pos=0, verify=False)
    try:
        e._pick("bottle", 25, 25)
    finally:
        restore_executor_module()
    assert e.state["holding"] == "bottle"
    assert [c[0] for c in calls].count("open") == 1            # only before the grasp


def test_pick_is_camera_only():
    import dmp_assistant.execution.executor as ex
    e = ex.Executor.__new__(ex.Executor)
    assert e._is_detected("bottle") and not e._is_detected("red_block")
    try:
        e._pick("red_block")
        assert False, "a stored-pose object must not be picked"
    except RuntimeError as err:
        assert "camera" in str(err)


def test_schema_and_validator_only_allow_camera_picks():
    from dmp_assistant.planning.schema_builder import build_schema
    items = build_schema()["properties"]["plan"]["items"]["oneOf"]
    by_action = {a["properties"]["action"]["const"]: a for a in items}
    camera = ["bottle", "cylinder", "hollow", "sanitizer"]
    assert sorted(by_action["pick"]["properties"]["object_id"]["enum"]) == camera
    assert sorted(by_action["pick_and_place"]["properties"]["object_id"]["enum"]) == camera
    assert "bin" in by_action["pick_and_place"]["properties"]["destination_id"]["enum"]

    from dmp_assistant.planning.planner import Planner
    p = Planner(api_key="offline")
    intent = {"manner": "none", "adjustment": "none", "store": False, "recipient": "none"}
    assert p._validate([{"action": "pick", "object_id": "bottle"}], intent)
    assert not p._validate([{"action": "pick", "object_id": "red_block"}], intent)
    assert not p._validate([{"action": "pick_and_place", "object_id": "red_block",
                             "destination_id": "bin"}], intent)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:                                  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
