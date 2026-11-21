"""
hand_over.py — Robot-to-human handover skill (simple sequencing, no DMP).

Sequence:
    1. Move to a fixed OBSERVATION pose (robot "sits back", wrist camera
       facing up, able to analyze user posture).
    2. Short observation pause.
    3. Move to the handover PRESENT pose with wrist orientation FROZEN
       (guardrail: orientation is force-copied from the observation pose,
       so the gripper cannot rotate in transit -> no slip, camera stays up).
    4. Verified full stop (pose-settling poll), then confirmation dwell.
    5. Gripper open (instant release; safe because queue already drained).
    6. Retreat back to the observation pose.

Scene additions required (config/scene.yaml):
    handover:
      observation_pose:
        value: [x, y, z, rx, ry, rz]   # sitting-back, camera-up wrist pose
      present_pose:
        value: [x, y, z, rx, ry, rz]   # where the object is offered
        # NOTE: only the POSITION of present_pose is used at runtime.
        # Orientation is overridden by the guardrail (frozen to observation).
"""

import sys
import time
from pathlib import Path

if __name__ == "__main__":      # direct run: make the project root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
import numpy as np

from dmp_assistant import paths
from dmp_assistant.skills.motion import movel
from dmp_assistant.skills.gripper import gripper_open


# ------------------------------------------------------------------
# Tunables (fixed policy — deliberately NOT planner parameters)
# ------------------------------------------------------------------
PRESENT_FACTOR = 0.6       # presentation vel = register speed * this factor
SETTLE_TOL_MM = 2.0        # TCP position tolerance for "stopped"
SETTLE_TOL_DEG = 1.0       # TCP orientation tolerance for "stopped"
SETTLE_WINDOW_S = 0.5      # must hold tolerance this long consecutively
ORIENT_WARN_DEG = 2.0      # log a warning if scene poses disagree beyond this


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _resolve(scene, ref):
    """Resolve 'handover.observation_pose' style refs against scene dict."""
    node = scene
    for p in ref.split("."):
        node = node[p]
    while isinstance(node, dict) and "ref" in node:
        node = _resolve(scene, node["ref"])
    return list(node["value"])


def _orient_dist_deg(a, b):
    """Wrap-aware per-channel Euler distance (deg). Euler angles wrap at
    +/-180 deg (e.g. -161.8 vs +178.2 is physically ~20 deg, not 340),
    so a naive abs(a-b) false-fires near the wrist envelope."""
    return max(min(abs(x - y), 360.0 - abs(x - y))
               for x, y in zip(a[3:6], b[3:6]))


def _wait_settled(robot, target, timeout_s=10.0):
    """Poll TCP pose until it holds within tolerance of target.
    Returns True if settled, False on timeout. This replaces any trust in
    GetRobotMotionDone() (flickers between queued segments)."""
    t0 = time.time()
    ok_since = None
    while time.time() - t0 < timeout_s:
        # Poll the SDK directly (NOT get_current_pose) — this loop runs at
        # ~20 Hz and get_state prints on every call, which would flood the
        # planner/executor logs.
        err, pose = robot.GetActualTCPPose(0)
        if err == 0 and pose is not None:
            pos_ok = all(abs(p - t) < SETTLE_TOL_MM
                         for p, t in zip(pose[:3], target[:3]))
            ori_ok = _orient_dist_deg(pose, target) < SETTLE_TOL_DEG
            if pos_ok and ori_ok:
                if ok_since is None:
                    ok_since = time.time()
                elif time.time() - ok_since >= SETTLE_WINDOW_S:
                    return True
            else:
                ok_since = None
        time.sleep(0.05)
    print("[HANDOVER WARN] Pose did not settle within timeout — proceeding anyway.")
    return False


def _frozen_present_pose(observation, present):
    """GUARDRAIL: build the handover target as present POSITION + observation
    ORIENTATION. The wrist physically cannot rotate during the transit, so a
    held object cannot swing/slip and the camera stays up by construction."""
    mismatch = _orient_dist_deg(observation, present)
    if mismatch > ORIENT_WARN_DEG:
        print(f"[HANDOVER GUARDRAIL] present_pose orientation differs from "
              f"observation by {mismatch:.2f} deg — overriding with observation "
              f"orientation (camera-up policy).")
    return list(present[:3]) + list(observation[3:6])


# ------------------------------------------------------------------
# Main skill
# ------------------------------------------------------------------
def hand_over(robot, scene_path=paths.SCENE_YAML,
              observe_s=1.5,           # pause at observation pose (posture read)
              confirm_dwell_s=3.0,     # verified-still hold before release
              speed=30, accel=100):    # register speed/accel (all moves)
    """Present the held object to a human. Returns True on success."""
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = yaml.safe_load(f)["poses"]

    observation = _resolve(scene, "handover.observation_pose")
    present_raw = _resolve(scene, "handover.present_pose")
    present = _frozen_present_pose(observation, present_raw)

    vel = int(speed * PRESENT_FACTOR)       # presentation: 60% of register speed
    print(f"[HANDOVER] transit={speed}/{accel}%, present vel={vel}%, "
          f"observe={observe_s}s, dwell={confirm_dwell_s}s")

    # 1) Sit back to the observation pose (camera up, view of the user)
    print("[HANDOVER] 1/5 Moving to observation pose...")
    movel(robot, observation, vel=speed, acc=accel)
    _wait_settled(robot, observation)

    # 2) Observation pause — future posture-analysis hook goes here
    print(f"[HANDOVER] 2/5 Observing user for {observe_s}s...")
    time.sleep(observe_s)

    # 3) Present the object — frozen orientation transit (guardrail)
    print("[HANDOVER] 3/5 Moving to present pose (orientation frozen)...")
    movel(robot, present, vel=vel, acc=accel)
    stopped = _wait_settled(robot, present)

    # 4) Confirmation dwell — timer starts only after a VERIFIED full stop
    if not stopped:
        print("[HANDOVER WARN] Proceeding to dwell without confirmed stop.")
    print(f"[HANDOVER] 4/5 Confirmation dwell {confirm_dwell_s}s...")
    time.sleep(confirm_dwell_s)

    # 5) Release — instant open; safe because the queue is fully drained
    print("[HANDOVER] 5/5 Releasing object...")
    gripper_open(robot)                      # sleeps internally, resets errors

    # 6) Retreat — sit back out of the human's reaching space
    print("[HANDOVER] Retreating to observation pose...")
    movel(robot, observation, vel=speed, acc=accel)
    _wait_settled(robot, observation)

    print("[HANDOVER] Complete.")
    return True


if __name__ == "__main__":
    from fairino import Robot
    rob = Robot.RPC("192.168.58.4")
    hand_over(rob, speed=50, accel=50, confirm_dwell_s=4.0)
