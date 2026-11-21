"""Offline checks of the pour start-configuration guard and joint-6
fallback (skills/pour.py). Fake robot: no arm, no waiting.
Run: python tests/test_pour_guard.py  (or pytest)."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

import numpy as np

import dmp_assistant.skills.pour as pour_mod

POSE = [100.0, -500.0, 300.0, 90.0, 0.0, 90.0]
REF = pour_mod.REF_WRIST


class FakeClock:
    """time.time()/sleep() without waiting: sleep just advances the clock."""

    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def sleep(self, s):
        self.now += s


class FakeRobot:
    """Moves instantly. movej_error: {call index: error code}."""

    def __init__(self, joints, movej_error=None, stuck=False):
        self.j = list(map(float, joints))
        self.pose = list(POSE)
        self.movej = []
        self.movel = []
        self.movej_error = movej_error or {}
        self.stuck = stuck                    # MoveJ accepted but the arm never arrives

    def GetActualJointPosDegree(self):
        return 0, list(self.j)

    def GetActualTCPPose(self, flag=1):
        return 0, list(self.pose)

    def MoveJ(self, joint_pos, tool, user, vel, acc):
        err = self.movej_error.get(len(self.movej), 0)
        self.movej.append((list(joint_pos), vel))
        if err == 0 and not self.stuck:
            self.j = list(joint_pos)
        return err

    def MoveL(self, desc_pos, tool, user, vel, acc, ovl, blendR, blendMode):
        self.movel.append(list(desc_pos))
        self.pose = list(desc_pos)
        return 0


def run(robot, **kw):
    real_time = pour_mod.time
    pour_mod.time = FakeClock()
    try:
        return pour_mod.pour(robot, confirm=False, **kw)
    finally:
        pour_mod.time = real_time


def joints(j4, j5, j6):
    return [10.0, -80.0, 100.0, j4, j5, j6]


def test_wrist_in_tolerance_uses_dmp_without_realign():
    r = FakeRobot(joints(REF[0] + 10, REF[1] - 5, REF[2] + 20))
    dev = run(r)
    assert r.movej == [], "no realign inside the tolerance"
    assert len(r.movel) == pour_mod.N_WP + 1          # waypoints + return to start
    assert dev is not None and dev < 1e-6
    assert r.movel[-1] == POSE


def test_wrist_outside_tolerance_is_realigned_then_dmp():
    start = joints(REF[0] + 60, REF[1], REF[2])
    r = FakeRobot(start)
    dev = run(r)
    assert len(r.movej) == 1
    target, vel = r.movej[0]
    assert target[:3] == start[:3], "J1-3 must not move"
    assert target[3:] == REF and vel == pour_mod.REALIGN_VEL
    assert len(r.movel) == pour_mod.N_WP + 1 and dev is not None


def test_realign_refused_falls_back_to_joint6():
    start = joints(REF[0] + 60, REF[1], REF[2])
    r = FakeRobot(start, movej_error={0: 14})          # realign MoveJ refused
    dev = run(r, speed=0.7)
    assert r.movel == [], "no DMP waypoints in the fallback"
    tilt, back = r.movej[1], r.movej[2]
    assert np.isclose(tilt[0][5], start[5] + pour_mod.FALLBACK_J6_DEG)
    assert tilt[0][:5] == start[:5], "only joint 6 turns"
    assert back[0] == start, "returns to the start joints"
    assert np.isclose(tilt[1], pour_mod.FALLBACK_VEL * 0.7)   # tempo follows speed
    assert dev is not None, "a fallback pour is a pour, not an abort"


def test_realign_that_never_settles_falls_back_and_reports_failure():
    r = FakeRobot(joints(REF[0] + 60, REF[1], REF[2]), stuck=True)
    assert run(r) is None                              # arm did not move: no pour
    assert r.movel == []


def test_fallback_tilt_refused_is_an_abort():
    start = joints(REF[0] + 60, REF[1], REF[2])
    r = FakeRobot(start, movej_error={0: 14, 1: 14})   # realign and tilt refused
    assert run(r) is None
    assert r.j == start


def test_unreadable_joints_do_not_pour():
    r = FakeRobot(joints(*REF))
    r.GetActualJointPosDegree = lambda: (1, [])
    assert run(r) is None
    assert r.movel == [] and r.movej == []


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
