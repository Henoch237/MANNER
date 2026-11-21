"""
run_pour.py - the pour skill. Standalone script AND importable function.

  python -m dmp_assistant.skills.pour           # recorded demo
  python -m dmp_assistant.skills.pour 80 6.0 0.7 2  # [tilt_deg] [dwell_s] [speed] [revs]

  from dmp_assistant.skills.pour import pour     # used by testers / the executor
  pour(robot, tilt_deg=80, dwell_s=6.0, speed=0.7, revs=2)

Blocks:  1+2 connect + current pose
         2.5 start-configuration guard: joints 4-6 must be within TOL_DEG
             of REF_WRIST (the recorded pour start). If not, MoveJ the wrist
             to the reference and re-check. If that fails, fall back to a
             pure joint-space pour: rotate joint 6 by FALLBACK_J6_DEG, hold,
             return.
         3 DMP + parameters -> theta(t)
         4 theta(t) -> poses (matrices, position pinned)
         5 waypoints -> MoveL queue (async)
         6 wait for settle -> return to start
"""

import sys
import time
from pathlib import Path

if __name__ == "__main__":      # direct run: make the project root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from fairino import Robot
from dmp_assistant import paths

ROBOT_IP = "192.168.58.4"          # real robot (sim: 192.168.85.130)
DMP_FILE = paths.POURING_DMP
VEL = 30
BLEND_R = 15.0                     # blend corners (smooth, no hard stops)
N_WP = 12                          # waypoints per revolution

# -------------------------------------------------- start-configuration guard
# Recorded wrist configuration (deg) at pour start, REAL robot (192.168.58.4).
# Verified from two working pours at different heights: J4-6 were identical
# to within 0.2 deg while J3 moved 19 deg, so only the wrist needs checking.
REF_WRIST = [-147.70, -89.10, 88.40]   # J4, J5, J6
CHECK_J = (3, 4, 5)        # only joints 4-6 matter (0-based indices)
TOL_DEG = 25.0             # allowed deviation per checked joint
REALIGN_VEL = 20
FALLBACK_J6_DEG = -100.0   # clockwise pour rotation about joint 6
FALLBACK_VEL = 15
FALLBACK_DWELL_S = 3.0     # hold at the tilted pose when dwell_s is not given

# ---------------------------------------------------------------- helpers --
def euler_to_R(rx, ry, rz):
    """FAIRINO [Rx,Ry,Rz] -> rotation matrix (R = Rz @ Ry @ Rx)."""
    rx, ry, rz = np.deg2rad([rx, ry, rz])
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx

def R_to_euler(R):
    """rotation matrix -> [Rx,Ry,Rz] radians (exact inverse of euler_to_R)."""
    ry = np.arcsin(np.clip(-R[2,0], -1, 1))
    rx = np.arctan2(R[2,1], R[2,2])
    rz = np.arctan2(R[1,0], R[0,0])
    return np.array([rx, ry, rz])

def axis_rotation(axis, deg):
    """rotation matrix: `deg` degrees about `axis` (Rodrigues)."""
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    t = np.deg2rad(deg)
    K = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return np.eye(3) + np.sin(t)*K + (1-np.cos(t))*(K@K)


# ------------------------------------------------- guard + fallback helpers --
def joints_now(robot):
    err, j = robot.GetActualJointPosDegree()
    if err != 0:
        return None
    return np.array(j, dtype=float)

def joint_deviation(j):
    """Max deviation of J4-6 from the recorded wrist configuration."""
    return max(abs(j[i] - REF_WRIST[k]) for k, i in enumerate(CHECK_J))

def realign_target(j):
    """Keep the current J1-J3 (arm stays at the pour location), drive only
    J4-6 to the recorded wrist configuration."""
    t = list(map(float, j))
    for k, i in enumerate(CHECK_J):
        t[i] = REF_WRIST[k]
    return t

def wait_joints_settled(robot, target, tol=1.5, timeout=40):
    t0 = time.time()
    settled = 0
    while time.time() - t0 < timeout:
        j = joints_now(robot)
        if j is not None:
            if np.max(np.abs(j - np.array(target, float))) < tol:
                settled += 1
                if settled >= 8:
                    return True
            else:
                settled = 0
        time.sleep(0.05)
    return False

def ensure_start_config(robot):
    """True if joints 4-6 are (or were brought) within TOL_DEG of REF_WRIST."""
    j = joints_now(robot)
    if j is None:
        print("guard: cannot read joints -> guard failed")
        return False
    dev = joint_deviation(j)
    print(f"guard: J4-6 deviation from reference = {dev:.1f} deg (tol {TOL_DEG})")
    if dev <= TOL_DEG:
        return True
    target = realign_target(j)
    print("guard: outside tolerance -> realigning wrist "
          f"(J4-6 -> {REF_WRIST}, J1-3 unchanged)")
    e = robot.MoveJ(joint_pos=target, tool=0, user=0,
                    vel=REALIGN_VEL, acc=0)
    if e != 0:
        print(f"guard: MoveJ returned {e} -> guard failed")
        return False
    if not wait_joints_settled(robot, target):
        print("guard: realign did not settle -> guard failed")
        return False
    j = joints_now(robot)
    dev = joint_deviation(j) if j is not None else float("inf")
    print(f"guard: after realign deviation = {dev:.1f} deg")
    return dev <= TOL_DEG

def fallback_pour_j6(robot, dwell_s=FALLBACK_DWELL_S, angle=FALLBACK_J6_DEG,
                     speed=1.0):
    """Pure joint-space pour: rotate J6 clockwise, hold, rotate back.
    `speed` is the same tempo factor as the DMP pour (0.7 = gentle).
    Returns True when the arm poured and is back at its start joints."""
    j = joints_now(robot)
    if j is None:
        print("fallback: cannot read joints, nothing done")
        return False
    vel = float(np.clip(FALLBACK_VEL * speed, 1, 100))
    target = j.copy()
    target[5] += angle
    print(f"fallback: J6 {j[5]:.1f} -> {target[5]:.1f} deg, dwell {dwell_s:.1f}s, vel {vel:.0f}")
    e = robot.MoveJ(joint_pos=list(map(float, target)), tool=0, user=0,
                    vel=vel, acc=0)
    if e != 0:
        print(f"fallback: MoveJ returned {e}, aborted")
        return False
    tilted = wait_joints_settled(robot, target)
    if not tilted:
        print("fallback: tilt did not settle - returning to start anyway")
    else:
        time.sleep(dwell_s)
    back = list(map(float, j))
    e = robot.MoveJ(joint_pos=back, tool=0, user=0, vel=vel, acc=0)
    if e != 0:
        print(f"fallback: return MoveJ returned {e} - arm is still TILTED")
        return False
    if not wait_joints_settled(robot, back):
        print("fallback: return did not settle")
        return False
    print("fallback: done, back at start")
    return tilted


# ---------------------------------------------------------------- the pour --
def pour(robot, tilt_deg=None, dwell_s=None, speed=1.0, revs=1,
         confirm=True):
    """Pour from the current pose. Parameters default to the recorded demo.
    Returns the final position deviation (mm) - also after a joint-6
    fallback pour - or None if aborted / nothing was poured."""
    # ---- BLOCK 2: current pose (pour happens FROM HERE, in place) ----
    err, pose = robot.GetActualTCPPose()
    if err != 0:
        raise RuntimeError(f"GetActualTCPPose failed: {err}")
    pose0 = np.array(pose, dtype=float)
    print("current pose:", np.round(pose0, 1))

    # ---- BLOCK 2.5: start-configuration guard --------------------------
    # The recorded DMP assumes joints 4-6 start near the configuration they
    # had during recording. If the arm arrives from a very different posture
    # the orientation trajectory cannot be followed. Check first; if the arm
    # cannot be brought into range, pour by rotating joint 6 directly.
    if not ensure_start_config(robot):
        print("DMP not usable from this configuration -> joint-6 fallback")
        if confirm and input("\nEnter to pour (joint-6 fallback): ").strip().lower() \
                not in ("", "y"):
            print("aborted.")
            return None
        if not fallback_pour_j6(robot, dwell_s=dwell_s if dwell_s else FALLBACK_DWELL_S,
                                speed=speed):
            return None
        err, p1 = robot.GetActualTCPPose()
        if err != 0:
            return None
        dev = float(np.max(np.abs(np.array(p1, float)[:3] - pose0[:3])))
        print("fallback pour: back at start, deviation", f"{dev:.2f} mm")
        return dev
    # re-read pose: the realign MoveJ may have moved the TCP
    err, pose = robot.GetActualTCPPose()
    if err == 0:
        pose0 = np.array(pose, dtype=float)
        print("pour from pose:", np.round(pose0, 1))

    # ---- BLOCK 3: DMP + parameters -> theta(t) ----
    dmp = np.load(DMP_FILE, allow_pickle=True).item()
    T = 2*np.pi/(dmp["omega"] * speed)       # one cycle (speed scales it)
    t = np.arange(0, T, 0.01)
    th = np.zeros(len(t))
    y = float(dmp["theta0"])                 # warm start: recorded start
    z = float(dmp["theta_dot0"]) * dmp["tau"]
    g, r = float(dmp["g"][0]), float(dmp["r"][0])
    for i in range(len(t)):
        ph = dmp["omega"] * speed * t[i]
        dd = np.minimum(np.abs(ph - dmp["c"]), 2*np.pi - np.abs(ph - dmp["c"]))
        psi = np.exp(-dmp["h"] * dd**2)
        f = float(dmp["w"][0] @ (psi / psi.sum())) * r
        z += (dmp["alpha_z"]*(dmp["beta_z"]*(g - y) - z) + f) / dmp["tau"] * 0.01
        y += (z / dmp["tau"]) * 0.01
        th[i] = y

    rec_peak = th.max()
    if tilt_deg is not None:                 # parameter 1: tilt scaling
        th = th[0] + (th - th[0]) * (tilt_deg / rec_peak)

    if dwell_s is not None:                  # parameter 2: dwell stretch
        idx = np.where(th > 0.85 * th.max())[0]
        if len(idx) >= 3:
            t1, t2 = t[idx[0]], t[idx[-1]]
            shift = dwell_s - (t2 - t1)
            t_new = np.interp(t, [0, t1, t2, t[-1]],
                                 [0, t1, t2 + shift, t[-1] + shift])
            t = np.arange(0, t_new[-1], 0.01)
            th = np.interp(t, t_new, th)
            T = t[-1]

    dwell_now = float((th > 0.85 * th.max()).sum() * 0.01)
    print(f"theta(t): {th.min():.0f} .. {th.max():.0f} deg, "
          f"dwell {dwell_now:.1f}s, cycle {T:.1f}s, revs {revs}")

    th_full = np.tile(th, revs)              # parameter 4: tile cycles
    t_full = np.arange(0, T * revs, 0.01)[:len(th_full)]
    th_full = th_full[:len(t_full)]

    # ---- BLOCK 4: theta(t) -> poses (matrices, position pinned) ----
    pos = pose0[:3]
    R_up = euler_to_R(*pose0[3:6])
    axis = dmp["axis"]
    eulers = np.degrees(np.unwrap(np.stack(
        [R_to_euler(axis_rotation(axis, a) @ R_up) for a in th_full]),
        axis=0))
    poses = np.hstack([np.tile(pos, (len(t_full), 1)), eulers])

    # ---- BLOCK 5: waypoints -> MoveL queue (async) ----
    idx = np.linspace(0, len(t_full) - 1, N_WP * revs, dtype=int)
    wps = poses[idx]
    wps[0] = pose0
    print("waypoints:", len(wps))
    if confirm and input("\nEnter to pour: ").strip().lower() not in ("", "y"):
        print("aborted.")
        return None
    t0 = time.time()
    errors = []
    for i, wp in enumerate(wps):
        e = robot.MoveL(desc_pos=list(map(float, wp)), tool=0, user=0,
                        vel=VEL, acc=0, ovl=50,
                        blendR=BLEND_R, blendMode=0)
        errors.append(e)
        if e != 0:
            print(f"  waypoint {i}: MoveL returned {e}  <-- NOTE THIS")
    print("fed all waypoints (MoveL is async - arm still moving)")

    # ---- BLOCK 6: wait for settle, return to start ----
    target = wps[-1]
    settled = 0
    while time.time() - t0 < 90:
        err, p = robot.GetActualTCPPose()
        if err == 0:
            if np.max(np.abs(np.array(p, float) - target)) < 2.0:
                settled += 1
                if settled >= 10:
                    break
            else:
                settled = 0
        time.sleep(0.05)
    print(f"motion done in {time.time()-t0:.1f}s")

    robot.MoveL(desc_pos=list(map(float, pose0)), tool=0, user=0,
                vel=20, acc=0, ovl=50, blendR=-1.0, blendMode=0)
    # The return MoveL is also async - drain the queue before reading the
    # pose (and before any caller fires its next action, e.g. a gripper
    # command whose ResetAllError would kill this still-draining segment).
    t0r = time.time()
    settled = 0
    while time.time() - t0r < 30:
        err, p = robot.GetActualTCPPose()
        if err == 0:
            if np.max(np.abs(np.array(p, float) - pose0)) < 2.0:
                settled += 1
                if settled >= 10:
                    break
            else:
                settled = 0
        time.sleep(0.05)
    print(f"return done in {time.time()-t0r:.1f}s")
    err, p1 = robot.GetActualTCPPose()
    dev = float(np.max(np.abs(np.array(p1, float)[:3] - pose0[:3])))
    print("back at start, deviation", f"{dev:.2f} mm")
    if any(errors):
        print("MoveL errors at waypoints:",
              [i for i, e in enumerate(errors) if e != 0])
    return dev


# ------------------------------------------------------------ standalone --
if __name__ == "__main__":
    robot = Robot.RPC(ROBOT_IP)
    robot.LoggerInit(output_model=0)
    robot.SetLoggerLevel(1)
    print("RobotEnable ->", robot.RobotEnable(1))
    time.sleep(1.0)
    tilt = float(sys.argv[1]) if len(sys.argv) > 1 else None
    dwell = float(sys.argv[2]) if len(sys.argv) > 2 else None
    speed = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    revs = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    pour(robot, tilt_deg=tilt, dwell_s=dwell, speed=speed, revs=revs)
    print("DONE")