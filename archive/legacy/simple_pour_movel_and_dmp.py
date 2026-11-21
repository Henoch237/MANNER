"""
run_pour.py - the simplest possible pour runner. Block by block:

  BLOCK 1: connect to the robot
  BLOCK 2: read the current pose (the pour happens FROM HERE, in place)
  BLOCK 3: load the DMP file and replay it -> theta(t), the tilt angle
  BLOCK 4: turn theta(t) into robot poses (rotation matrices, position fixed)
  BLOCK 5: pick 12 waypoints and MoveL through them
  BLOCK 6: wait for the arm to stop, return to the start pose

Run:  python run_pour.py
Needs: data/dmp/learned_pouring_dmp.npy.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from dmp_assistant import paths
import time
import numpy as np
from fairino import Robot

ROBOT_IP = "192.168.58.4"          # real robot (sim: 192.168.85.130)
DMP_FILE = paths.POURING_DMP
VEL = 30
BLEND_R = 15.0                     # blend corners (smooth, no hard stops)

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
T = 2*np.pi/dmp["omega"]                     # one pour cycle, seconds
t = np.arange(0, T, 0.01)                    # 1640 samples
th = np.zeros(len(t))                        # the tilt angle, degrees
y = float(dmp["theta0"])                     # warm start: recorded start
z = float(dmp["theta_dot0"]) * dmp["tau"]    # warm start: recorded velocity
g, r = float(dmp["g"][0]), float(dmp["r"][0])
for i in range(len(t)):
    ph = dmp["omega"] * t[i]                 # phase of the rhythmic cycle
    dd = np.minimum(np.abs(ph - dmp["c"]), 2*np.pi - np.abs(ph - dmp["c"]))
    psi = np.exp(-dmp["h"] * dd**2)          # basis functions
    f = float(dmp["w"][0] @ (psi / psi.sum())) * r   # learned forcing
    z += (dmp["alpha_z"]*(dmp["beta_z"]*(g - y) - z) + f) / dmp["tau"] * 0.01
    y += (z / dmp["tau"]) * 0.01
    th[i] = y
print(f"theta(t): {th.min():.0f} .. {th.max():.0f} deg, period {T:.1f}s")

# ------------------------------------------------------------- BLOCK 4 --
# The whole pour is ONE rotation about the learned axis, starting from the
# CURRENT orientation. Position stays exactly where the arm is now.
pos = pose0[:3]                              # pinned TCP position (mm)
R_up = euler_to_R(*pose0[3:6])               # current orientation
axis = dmp["axis"]                           # learned tilt axis (base frame)
eulers = np.degrees(np.unwrap(np.stack(
    [R_to_euler(axis_rotation(axis, a) @ R_up) for a in th]), axis=0))
poses = np.hstack([np.tile(pos, (len(t), 1)), eulers])   # (1640, 6)

# ------------------------------------------------------------- BLOCK 5 --
# 12 waypoints, evenly spaced in time; the controller blends between them.
idx = np.linspace(0, len(t) - 1, 12, dtype=int)
wps = poses[idx]
wps[0] = pose0                               # first wp = exactly here
print("waypoints:", len(wps),
      "| tilt reaches", f"{th.max():.0f} deg")
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
# wait until the arm physically reaches the LAST waypoint and stops there
target = wps[-1]
settled = 0
while time.time() - t0 < 60:
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

# back to the exact start pose (blocking, gentle)
robot.MoveL(desc_pos=list(map(float, pose0)), tool=0, user=0,
            vel=20, acc=0, ovl=50, blendR=-1.0, blendMode=0)
err, p1 = robot.GetActualTCPPose()
print("back at start, deviation",
      f"{np.max(np.abs(np.array(p1, float)[:3] - pose0[:3])):.2f} mm")
if any(errors):
    print("MoveL errors seen at waypoints:",
          [i for i, e in enumerate(errors) if e != 0])
print("DONE")