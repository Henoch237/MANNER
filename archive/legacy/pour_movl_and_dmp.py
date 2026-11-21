"""
test_pour_sim.py - run the 1-D tilt pouring DMP on the SIMULATION robot.

IP points at the sim (192.168.85.130). No gripper, no LLM, no scene.yaml.
The pour happens IN PLACE at the current pose: position held fixed,
orientation composed in rotation matrices (singularity-free).

Usage:
  python test_pour_sim.py            # 1 pour revolution
  python test_pour_sim.py 3          # 3 revolutions (pour more)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from dmp_assistant import paths
import os
import sys
import time
import numpy as np

from fairino import Robot

# ============================ CONFIG ============================
SIM_IP = "192.168.85.130"             # <-- simulation, NOT the real robot
POUR_DMP_FILE = paths.POURING_DMP

try:
    POUR_REVOLUTIONS = max(1, int(sys.argv[1])) if len(sys.argv) > 1 else 1
except (ValueError, IndexError):
    POUR_REVOLUTIONS = 1
N_WAYPOINTS = 12                      # arc-length sampled per revolution
VEL, ACC, OVL = 30, 100, 50
BLEND_R, BLEND_MODE = 0.0, 0
VERIFY = True                         # poll actual pose after each waypoint
VERIFY_SETTLE_S = 0.4                 # let blend settle before measuring
SLOW_MODE = False                     # True -> vel=12 for calm observation

# ================== rotation helpers (round-trip verified) ==================
def euler_to_R(rx, ry, rz):
    """FAIRINO [Rx,Ry,Rz]: R = Rz @ Ry @ Rx."""
    rx, ry, rz = np.deg2rad([rx, ry, rz])
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx

def R_to_ZYX_euler(R):
    """Exact inverse -> [Rx,Ry,Rz] radians. Unwrap across time afterwards."""
    theta = np.arcsin(np.clip(-R[2,0], -1, 1))
    phi = np.arctan2(R[2,1], R[2,2])
    psi = np.arctan2(R[1,0], R[0,0])
    return np.array([phi, theta, psi])

def axisang_R(axis, ang):
    ax = np.asarray(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    K = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3) + np.sin(ang)*K + (1-np.cos(ang))*(K@K)

# ============================ TILT DMP ============================
def load_tilt_dmp(path=POUR_DMP_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run the recorder refit first")
    d = np.load(path, allow_pickle=True).item()
    assert d.get("n_dims") == 1, f"expected n_dims=1 tilt DMP, got {d.get('n_dims')}"
    for k in ("axis", "r_up_euler", "pos_fixed_mm", "theta0", "theta_dot0"):
        assert k in d, f"tilt DMP missing '{k}' - re-run the recorder refit"
    return d

def rollout_theta(dmp, n_periods, dt=0.01):
    """Warm-started rollout (theta0, theta_dot0 from the recording)."""
    T = 2 * np.pi / dmp["omega"]
    tt = np.arange(0, n_periods * T, dt)
    yy = np.zeros(len(tt))
    g, r = float(dmp["g"][0]), float(dmp["r"][0])
    y = float(dmp["theta0"])
    zz = float(dmp["theta_dot0"]) * dmp["tau"]
    for i in range(len(tt)):
        th = dmp["omega"] * tt[i]
        d = np.minimum(np.abs(th - dmp["c"]), 2*np.pi - np.abs(th - dmp["c"]))
        p = np.exp(-dmp["h"] * d ** 2)
        f = float(dmp["w"][0] @ (p / (p.sum() + 1e-10))) * r
        zz += (dmp["alpha_z"] * (dmp["beta_z"] * (g - y) - zz) + f) \
              / dmp["tau"] * dt
        y += (zz / dmp["tau"]) * dt
        yy[i] = y
    return tt, yy

def pour_waypoints(dmp, pose0, revolutions, n_wp=N_WAYPOINTS):
    """Matrix-composed waypoints: R(t) = axisang(axis, theta(t)) @ R_up_current.
    TCP position HELD at pose0[:3] - the tilt pivots in place."""
    pos_mm = np.asarray(pose0[:3], dtype=float)
    R_up = euler_to_R(*pose0[3:6])
    axis = dmp["axis"]

    tt, th = rollout_theta(dmp, n_periods=revolutions)
    eulers = np.degrees(np.unwrap(np.stack(
        [R_to_ZYX_euler(axisang_R(axis, np.radians(a)) @ R_up) for a in th]
    ), axis=0))

    # arc-length sampling: uniform in tilt angle (dwell gets few, flanks many)
    T = 2 * np.pi / dmp["omega"]
    n_per = int(round(T / 0.01))
    wps = []
    for rev in range(revolutions):
        seg = eulers[rev*n_per:(rev+1)*n_per]
        seg_th = th[rev*n_per:(rev+1)*n_per]
        arc = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(seg_th)))])
        targets = np.linspace(0, arc[-1], n_wp)
        idx = np.clip(np.searchsorted(arc, targets), 0, len(seg) - 1)
        idx[0] = 0
        for i in idx:
            wps.append(np.concatenate([pos_mm, seg[i]]).tolist())
    return wps, th

# ============================ MAIN ============================
def main():
    print("=" * 55)
    print(f"POUR DMP SIM TEST  ->  {SIM_IP}")
    print(f"revolutions: {POUR_REVOLUTIONS}")
    print("=" * 55)

    dmp = load_tilt_dmp()
    robot = Robot.RPC(SIM_IP)
    robot.LoggerInit(output_model=0)
    robot.SetLoggerLevel(1)
    err = robot.RobotEnable(1)
    print(f"RobotEnable -> {err}")
    time.sleep(1.0)

    err, pose = robot.GetActualTCPPose()
    assert err == 0, f"GetActualTCPPose failed: {err}"
    pose0 = np.array(pose, dtype=float)
    print(f"current pose: pos={np.round(pose0[:3],1)} "
          f"orient={np.round(pose0[3:6],1)}")

    wps, th = pour_waypoints(dmp, pose0, POUR_REVOLUTIONS)
    print(f"axis={np.round(dmp['axis'],3)}  period={2*np.pi/dmp['omega']:.2f}s  "
          f"theta: {th.min():.0f}..{th.max():.0f} deg")
    print(f"first wp: {np.round(wps[0],1)}")
    print(f"peak wp : {np.round(wps[N_WAYPOINTS//2],1)}")

    if input("\nExecute pour on SIM? (y/n): ").strip().lower() != "y":
        print("Aborted.")
        return

    vel = 12 if SLOW_MODE else VEL
    print("Pouring...")
    log_t, log_pose = [], []
    t_start = time.time()
    try:
        for k, wp in enumerate(wps):
            robot.MoveL(desc_pos=wp, tool=0, user=0, vel=vel, acc=ACC, ovl=OVL,
                        blendR=BLEND_R, blendMode=BLEND_MODE)
            if VERIFY:
                time.sleep(VERIFY_SETTLE_S)
                err, p_act = robot.GetActualTCPPose()
                if err == 0:
                    log_t.append(time.time() - t_start)
                    log_pose.append(list(p_act))
                    R_cmd = euler_to_R(*wp[3:6])
                    R_act = euler_to_R(*np.array(p_act)[3:6])
                    dR = R_act @ R_cmd.T
                    c = np.clip((np.trace(dR) - 1) / 2, -1, 1)
                    err_deg = np.degrees(np.arccos(c))
                    print(f"  wp{k+1:2d}: rotation err {err_deg:5.2f} deg")
            if (k + 1) % N_WAYPOINTS == 0:
                print(f"  revolution {(k + 1) // N_WAYPOINTS} done")
        # gentle return to exact start pose (upright), no blend
        robot.MoveL(desc_pos=pose0.tolist(), tool=0, user=0, vel=VEL,
                    acc=30, ovl=OVL)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] returning to start pose...")
        robot.MoveL(desc_pos=pose0.tolist(), tool=0, user=0, vel=VEL,
                    acc=30, ovl=OVL)

    err, p1 = robot.GetActualTCPPose()
    dev = np.max(np.abs(np.array(p1[:3]) - pose0[:3]))
    print(f"back at start, position deviation {dev:.2f} mm")
    if VERIFY and log_pose:
        np.savez(paths.DEMOS_DIR / "pour_verify_log.npz",
                 t=np.array(log_t), pose=np.array(log_pose),
                 commanded=np.array(wps))
        print("[SAVED] pour_verify_log.npz - upload it for analysis if odd")
    print("POUR SIM TEST COMPLETE")


if __name__ == "__main__":
    main()