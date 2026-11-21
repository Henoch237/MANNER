"""
run_pour.py - pour runner with parameters (defaults = the recorded demo).

  BLOCK 1: connect to the robot
  BLOCK 2: read the current pose (the pour happens FROM HERE, in place)
  BLOCK 3: load the DMP, apply parameters, replay -> theta(t)
  BLOCK 4: turn theta(t) into robot poses (matrices, position fixed)
  BLOCK 5: waypoints + MoveL (async queue)
  BLOCK 6: wait for the arm to stop, return to start

Run:
  python run_pour.py                          # exactly the recorded pour
  python run_pour.py 80                       # tilt to 80 deg instead of 104
  python run_pour.py 80 6.0                   # ... and hang 6 s at the peak
  python run_pour.py 80 6.0 0.7 2             # ... gently (0.7x), 2 pours
                                              # args: [tilt_deg] [dwell_s]
                                              #       [speed] [revolutions]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from dmp_assistant import paths
import sys
import time
import numpy as np
from fairino import Robot

ROBOT_IP = "192.168.58.4"          # real robot (sim: 192.168.85.130)
DMP_FILE = paths.POURING_DMP
VEL = 30
BLEND_R = 15.0                     # blend corners (smooth, no hard stops)
N_WP = 12                          # waypoints per revolution

# ---- parameters: None / defaults reproduce the recorded demonstration ----
TILT_DEG = float(sys.argv[1]) if len(sys.argv) > 1 else None   # default 104
DWELL_S  = float(sys.argv[2]) if len(sys.argv) > 2 else None   # default ~3.4
SPEED    = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0    # 0.7 = gentle
REVS     = int(sys.argv[4])   if len(sys.argv) > 4 else 1

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

# ------------------------------------------------------------- BLOCK 1+2 --
robot = Robot.RPC(ROBOT_IP)
robot.LoggerInit(output_model=0)
robot.SetLoggerLevel(1)
print("RobotEnable ->", robot.RobotEnable(1))
time.sleep(1.0)

err, pose = robot.GetActualTCPPose()
pose0 = np.array(pose, dtype=float)
print("current pose:", np.round(pose0, 1))

# ------------------------------------------------------------- BLOCK 3 --
dmp = np.load(DMP_FILE, allow_pickle=True).item()
T = 2*np.pi/(dmp["omega"] * SPEED)           # one cycle (speed scales it)
t = np.arange(0, T, 0.01)
th = np.zeros(len(t))                        # the tilt angle, degrees
y = float(dmp["theta0"])                     # warm start: recorded start
z = float(dmp["theta_dot0"]) * dmp["tau"]    # warm start: recorded velocity
g, r = float(dmp["g"][0]), float(dmp["r"][0])
for i in range(len(t)):
    ph = dmp["omega"] * SPEED * t[i]         # phase (speed-scaled)
    dd = np.minimum(np.abs(ph - dmp["c"]), 2*np.pi - np.abs(ph - dmp["c"]))
    psi = np.exp(-dmp["h"] * dd**2)
    f = float(dmp["w"][0] @ (psi / psi.sum())) * r
    z += (dmp["alpha_z"]*(dmp["beta_z"]*(g - y) - z) + f) / dmp["tau"] * 0.01
    y += (z / dmp["tau"]) * 0.01
    th[i] = y

# parameter 1: tilt_deg - linear scale around the start angle
rec_peak = th.max()
if TILT_DEG is not None:
    th = th[0] + (th - th[0]) * (TILT_DEG / rec_peak)

# parameter 2: dwell_s - stretch the hang time at the peak.
# Time-warp: everything below/above the dwell keeps its pace, the dwell
# segment expands/contracts to the requested length.
if DWELL_S is not None:
    idx = np.where(th > 0.85 * th.max())[0]
    if len(idx) >= 3:
        t1, t2 = t[idx[0]], t[idx[-1]]
        shift = DWELL_S - (t2 - t1)
        t_new = np.interp(t, [0, t1, t2, t[-1]],
                             [0, t1, t2 + shift, t[-1] + shift])
        t = np.arange(0, t_new[-1], 0.01)
        th = np.interp(t, t_new, th)
        T = t[-1]

# parameter 3/4 already applied (speed in the rollout; revs = tiling below)
dwell_now = float((th > 0.85 * th.max()).sum() * 0.01)
print(f"theta(t): {th.min():.0f} .. {th.max():.0f} deg, "
      f"dwell {dwell_now:.1f}s, cycle {T:.1f}s, revs {REVS}")

th_full = np.tile(th, REVS)                  # tile the cycles
t_full = np.arange(0, T * REVS, 0.01)[:len(th_full)]
th_full = th_full[:len(t_full)]

# ------------------------------------------------------------- BLOCK 4 --
pos = pose0[:3]                              # pinned TCP position (mm)
R_up = euler_to_R(*pose0[3:6])               # current orientation
axis = dmp["axis"]                           # learned tilt axis (base frame)
eulers = np.degrees(np.unwrap(np.stack(
    [R_to_euler(axis_rotation(axis, a) @ R_up) for a in th_full]), axis=0))
poses = np.hstack([np.tile(pos, (len(t_full), 1)), eulers])

# ------------------------------------------------------------- BLOCK 5 --
idx = np.linspace(0, len(t_full) - 1, N_WP * REVS, dtype=int)
wps = poses[idx]
wps[0] = pose0                               # first wp = exactly here
print("waypoints:", len(wps))
input("\nEnter to pour: ")
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

# ------------------------------------------------------------- BLOCK 6 --
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
err, p1 = robot.GetActualTCPPose()
print("back at start, deviation",
      f"{np.max(np.abs(np.array(p1, float)[:3] - pose0[:3])):.2f} mm")
if any(errors):
    print("MoveL errors at waypoints:",
          [i for i, e in enumerate(errors) if e != 0])
print("DONE")
