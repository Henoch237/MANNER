"""
record_dmp.py  (v3.1 — RHYTHMIC pour cycle, bug-fixed)

Record ONE pour cycle on the FAIRINO F5:
    upright -> tilt & pour -> HANG (dwell) -> return to upright
and fit a 6-DOF rhythmic DMP to it (positions + orientation).

The cycle is CLOSED: start pose == end pose. Looping the DMP replays
the pour; `revolutions` at execution time = how much to pour.

Saved learned_pouring_dmp.npy uses the exact schema of
learned_stirring_dmp.npy -> loadable by QwenStirr.RhythmicDMP as-is.

Modes:
  MODE = "drag"      hand-guide the arm (drag-teach)
  MODE = "keyframe"  jog to poses, Enter captures each; capture the
                     tilted pose TWICE to bake in the dwell

Robot conventions follow DMP_record_REAL.py (IP 192.168.58.4,
alpha_z=48, beta_z=12). 60 BFs (vs 40 for stirring) because the
dwell plateau needs finer phase resolution than a smooth circle.

Usage:
  python -m dmp_assistant.tools.record_dmp           # record + fit + save
  python -m dmp_assistant.tools.record_dmp inspect
"""

import sys
import time
import threading
from pathlib import Path

if __name__ == "__main__":      # direct run: make the project root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from fairino import Robot
from dmp_assistant import paths
from dmp_assistant.skills.gripper import gripper_open, gripper_close
from dmp_assistant.skills.motion import movel

# ============================ CONFIG ============================
ROBOT_IP = "192.168.58.4"          # REAL robot (sim = 192.168.85.130)

OUTPUT_DMP = paths.POURING_DMP
OUTPUT_RAW = paths.DEMOS_DIR / "pouring_demo_raw.npz"
OUTPUT_PLOT = paths.DEMOS_DIR / "pouring_demo_fit.png"

MODE = "drag"                      # "drag" or "keyframe"
EXECUTE_PREVIEW = False            # True = replay 1 revolution after fitting

# --- drag mode ---
POLL_SLEEP_S = 0.01
MAX_RECORD_S = 120.0
TRIM_EDGE_S = 0.15
HEARTBEAT_S = 5.0                  # print elapsed time every N seconds

# --- keyframe mode ---
KEYFRAME_SEGMENT_S = 3.0
MIN_KEYFRAMES = 4

# --- DMP fit ---
N_BFS = 60
ALPHA_Z = 48.0
BETA_Z = 12.0
SMOOTH_SIGMA = 2.0

CLOSURE_WARN_MM = 5.0
CLOSURE_WARN_DEG = 3.0
TILT_WARN_DEG = 110.0
SINGULARITY_WARN_DEG = 15.0    # keep Ry at least this far from +/-90
MIN_CYCLE_S = 8.0                  # soft warning below this
MIN_SAMPLES = 100
DO_BOTTLE_SETUP = True


# ============================ SMOOTHING ============================
def gaussian_smooth(x, sigma):
    """Smooth over time; works for 1-D (N,) or 2-D (N, D) arrays. Shape-preserving."""
    x = np.asarray(x, dtype=float)
    if sigma <= 0:
        return x.copy()
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    if x.ndim == 1:
        xp = np.pad(x, r, mode="edge")
        return np.convolve(xp, k, mode="valid")
    xp = np.pad(x, ((r, r), (0, 0)), mode="edge")
    out = np.empty_like(x)
    for d in range(x.shape[1]):
        out[:, d] = np.convolve(xp[:, d], k, mode="valid")
    return out


# ============================ ROBOT I/O ============================
def _read_pose(robot):
    err, pose = robot.GetActualTCPPose()
    if err == 0 and pose is not None:
        return np.array(pose, dtype=float)
    return None


def _drag_switch(robot, on):
    fn = None
    for name in ("DragTeachSwitch", "SetDragTeachSwitch"):
        if hasattr(robot, name):
            fn = getattr(robot, name)
            break
    if fn is None:
        print("[WARN] No drag-teach API found.")
        return False
    try:
        err = fn(1 if on else 0)
        print(f"  drag-teach {'ON' if on else 'OFF'} -> err {err}")
        return err == 0
    except Exception as e:
        print(f"[WARN] Drag-teach failed: {e} "
              f"(firmware may need RobotEnable(0), or use MODE='keyframe')")
        return False


def _bottle_setup(robot):
    if not DO_BOTTLE_SETUP:
        return
    gripper_open(robot)
    time.sleep(1.0)
    input(">>> Place the bottle in the gripper, then press Enter: ")
    gripper_close(robot)
    time.sleep(1.5)


def _print_choreography():
    print("\n ONE pour cycle, performed in place at the pour station:")
    print("   1. UPRIGHT (start pose - you MUST return to it at the end)")
    print("   2. Tilt and pour")
    print("   3. HANG in the tilted pose 2-3 s (the dwell)")
    print("   4. Return to UPRIGHT - same pose as step 1\n")


# ============================ RECORDING ============================
def _preflight_pose_check(robot, label="start"):
    """Read current pose and report the Euler singularity budget."""
    pose = _read_pose(robot)
    if pose is None:
        print("[WARN] could not read pose for pre-flight check")
        return None
    rx, ry, rz = pose[3], pose[4], pose[5]
    print(f">>> {label} pose: X={pose[0]:.1f} Y={pose[1]:.1f} Z={pose[2]:.1f}")
    print(f">>> orientation: Rx={rx:.1f}  Ry={ry:.1f}  Rz={rz:.1f}")
    for tilt_max, tag in ((60, "moderate"), (75, "max acceptable")):
        peak = abs(ry) + tilt_max
        margin = 90 - peak
        flag = "OK" if margin >= 15 else "  <-- TOO CLOSE to +/-90!"
        print(f">>>   {tag:15s} tilt {tilt_max} deg -> Ry peaks ~{peak:.0f} "
              f"(margin {margin:.0f} deg) {flag}")
    print(">>> Bottle upright, camera up? Then proceed. "
          "If Ry is large, re-orient the wrist BEFORE recording.")
    return pose


def record_drag(robot):
    print("\n=== DRAG MODE ===")
    _print_choreography()
    if not _drag_switch(robot, True):
        return None, None

    stop_flag = threading.Event()

    def _wait_enter():
        input()
        stop_flag.set()

    input(">>> GRAB THE ARM at the upright pour-station pose,\n"
          ">>> then press Enter to START recording. ")
    threading.Thread(target=_wait_enter, daemon=True).start()

    print(">>> RECORDING NOW - perform the cycle slowly.")
    print(f">>> Press Enter AGAIN when you are back at the upright pose "
          f"(max {MAX_RECORD_S:.0f} s).")
    t0 = time.time()
    raw = []
    last_beat = 0.0
    while not stop_flag.is_set() and (time.time() - t0) < MAX_RECORD_S:
        t = time.time() - t0
        pose = _read_pose(robot)
        if pose is not None:
            raw.append([t] + pose.tolist())
        if t - last_beat >= HEARTBEAT_S:
            print(f"    ... recording, {t:.0f} s elapsed "
                  f"(tilt -> dwell -> upright -> Enter)")
            last_beat = t
        time.sleep(POLL_SLEEP_S)

    _drag_switch(robot, False)
    data = np.array(raw, dtype=float)
    print(f"\n>>> RECORDING STOPPED - teaching finished, "
          f"{len(data)} samples over {data[-1, 0]:.1f} s.")
    return data[:, 0], data[:, 1:7]


def record_keyframe(robot):
    print("\n=== KEYFRAME MODE ===")
    print(" Jog to each key pose, Enter captures it.")
    print(" Capture the tilted pose TWICE to bake in the dwell.")
    print(" First and last keyframe MUST be the same upright pose.")
    print(" Type 'd' + Enter when done.\n")
    _print_choreography()

    keyframes = []
    while True:
        ans = input(f">>> Keyframe {len(keyframes) + 1} "
                    f"(Enter to capture, 'd' to finish): ").strip().lower()
        if ans == "d":
            break
        pose = _read_pose(robot)
        if pose is None:
            print("    read failed, try again")
            continue
        keyframes.append(pose)
        print(f"    captured: {np.round(pose, 1)}")

    if len(keyframes) < MIN_KEYFRAMES:
        print(f"[ERROR] Need at least {MIN_KEYFRAMES} keyframes.")
        return None, None

    n_seg = len(keyframes) - 1
    t = np.arange(0, n_seg * KEYFRAME_SEGMENT_S + 1e-9, 0.05)
    K = np.stack(keyframes)
    Y = np.stack([np.interp(t, np.linspace(0, t[-1], len(K)), K[:, d])
                  for d in range(6)], axis=1)
    print(f"\n[RECORD] {len(K)} keyframes -> {len(t)} samples, {t[-1]:.1f} s")
    return t, Y


# ==================== FIT (1-DOF tilt DMP, singularity-free) ====================
# The pour is ONE physical rotation about ONE fixed base axis. Learning in
# Euler channels scrambles near Ry=+/-90 (gimbal lock) - the physical system
# lives there by design. So we learn the scalar tilt theta(t) via the matrix
# log (exact, singularity-free) and replay by composing rotation matrices;
# Euler angles are only a serialization format for MoveL.
def euler_to_R(rx, ry, rz):
    """FAIRINO [Rx,Ry,Rz] convention: R = Rz @ Ry @ Rx (verified by
    exact round-trip against recorded TCP poses)."""
    rx, ry, rz = np.deg2rad([rx, ry, rz])
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx

def R_to_ZYX_euler(R):
    """Exact inverse of euler_to_R -> [Rx,Ry,Rz] radians. Use np.unwrap on
    the time sequence to keep branch continuity across +/-90 crossings."""
    theta = np.arcsin(np.clip(-R[2,0], -1, 1))
    phi = np.arctan2(R[2,1], R[2,2])
    psi = np.arctan2(R[1,0], R[0,0])
    return np.array([phi, theta, psi])

def axisang_R(axis, ang):
    ax = np.asarray(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    K = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3) + np.sin(ang)*K + (1-np.cos(ang))*(K@K)

def preprocess(t, Y):
    """Trim edges / resample uniform. ALWAYS returns matched (t, Y)."""
    t = np.asarray(t, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if len(t) > MIN_SAMPLES and (t[1] - t[0]) < 0.5:   # dense drag recording
        keep = (t > TRIM_EDGE_S) & (t < t[-1] - TRIM_EDGE_S)
        t, Y = t[keep], Y[keep]
        dt = float(np.median(np.diff(t)))
        t_u = np.arange(t[0], t[-1], dt)
        Y = np.stack([np.interp(t_u, t, Y[:, d]) for d in range(6)], axis=1)
        t = t_u
    t = t - t[0]
    Y = Y.copy()
    Y[:, 3:6] = np.rad2deg(np.unwrap(np.deg2rad(Y[:, 3:6]), axis=0))
    return t, Y

def extract_tilt(t, Y):
    """
    From a 6-DOF demo extract the scalar tilt theta(t) about the dominant
    fixed axis:  theta(t) = angle of R(t) @ R_up^T  (matrix log).
    Returns dict with theta (deg), axis (3,), R_up, spread (axis consistency).
    """
    Rs = np.stack([euler_to_R(*o) for o in Y[:, 3:6]])
    R_up = Rs[0]

    n = len(t)
    axes = np.zeros((n, 3)); angs = np.zeros(n)
    for i, R in enumerate(Rs):
        dR = R @ R_up.T
        c = np.clip((np.trace(dR) - 1) / 2, -1, 1)
        angs[i] = np.arccos(c)
        s = 2 * np.sin(angs[i])
        if s > 1e-9:
            axes[i] = np.array([dR[2,1]-dR[1,2], dR[0,2]-dR[2,0],
                                dR[1,0]-dR[0,1]]) / s
    # dominant axis: sign-aligned, angle-weighted mean
    ref = axes[np.argmax(angs)]
    a = axes.copy()
    for i in range(n):
        if np.dot(a[i], ref) < 0:
            a[i] = -a[i]
    w = angs ** 2
    axis = (a * w[:, None]).sum(axis=0)
    axis /= np.linalg.norm(axis)

    # signed theta about the dominant axis
    theta = np.zeros(n)
    for i in range(n):
        if 2*np.sin(angs[i]) > 1e-9:
            theta[i] = angs[i] * np.sign(np.dot(axes[i], axis))
    theta = np.rad2deg(np.unwrap(theta))   # angs was already in radians

    # axis consistency where the motion matters
    sig = np.where(np.abs(theta) > 20)[0]
    if len(sig) > 0:
        al = np.degrees(np.arccos(np.clip(np.abs(a[sig] @ axis), -1, 1)))
        spread = float(np.max(al))
    else:
        spread = 180.0
    return {"theta": theta, "axis": axis, "R_up": R_up, "spread": spread,
            "Rs": Rs}

def fit_tilt_dmp(t, theta):
    """Fit 1-D rhythmic DMP to the tilt profile. Same conventions as the
    stirring fit (alpha_z=48, beta_z=12). Schema-compatible with
    QwenStirr.RhythmicDMP (n_dims=1)."""
    T = float(t[-1])
    dt = float(np.mean(np.diff(t)))

    th = gaussian_smooth(theta, SMOOTH_SIGMA)
    dth = np.gradient(th, dt)
    ddth = np.gradient(dth, dt)

    g1 = float(th.mean()); r1 = float(np.max(np.abs(th - g1)))
    omega = 2 * np.pi / T
    tau = 1.0 / omega
    th_ph = omega * (t - t[0])

    z = tau * dth
    dz = tau * ddth
    f_t = dz - ALPHA_Z * (BETA_Z * (g1 - th) - z)
    f_n = f_t / r1 if r1 > 1e-6 else np.zeros_like(f_t)

    c = np.linspace(0, 2 * np.pi, N_BFS, endpoint=False)
    h = np.ones(N_BFS) * (N_BFS ** 1.5) / (2 * np.pi)
    Phi = np.zeros((len(t), N_BFS))
    for i in range(len(t)):
        d = np.minimum(np.abs(th_ph[i] - c), 2*np.pi - np.abs(th_ph[i] - c))
        p = np.exp(-h * d ** 2)
        Phi[i, :] = p / (p.sum() + 1e-10)
    w1, *_ = np.linalg.lstsq(Phi, f_n, rcond=None)

    return {
        "n_dims": 1, "n_bfs": N_BFS,
        "alpha_z": ALPHA_Z, "beta_z": BETA_Z,
        "c": c, "h": h, "w": w1.reshape(1, N_BFS),
        "g": np.array([g1]), "r": np.array([r1]),
        "tau": tau, "omega": omega,
        # --- tilt-DMP extras (executor needs these) ---
        "axis": None,            # filled by caller (extract_tilt)
        "r_up_euler": None,      # recorded start orientation [deg]
        "pos_fixed_mm": None,    # recorded TCP position [mm]
        "theta0": float(th[0]),  # warm-start state
        "theta_dot0": float(dth[0]),
        "duration": T,
        "units": "theta=deg, axis=base-frame unit vector",
    }

def rollout_tilt(dmp, n_periods=2, dt=0.01, y0=None, z0=None):
    """Roll out the fitted 1-D tilt DMP. Warm-start via theta0/theta_dot0."""
    T = 2 * np.pi / dmp["omega"]
    tt = np.arange(0, n_periods * T, dt)
    yy = np.zeros(len(tt))
    g, r = float(dmp["g"][0]), float(dmp["r"][0])
    y = float(dmp["theta0"]) if y0 is None else y0
    zz = float(dmp["theta_dot0"]) * dmp["tau"] if z0 is None else z0
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

def report_tilt(dmp, ext, t, theta):
    T = 2 * np.pi / dmp["omega"]
    tt, yy = rollout_tilt(dmp, n_periods=2)
    n1 = int(round(T / 0.01))
    p1, p2 = yy[:n1], yy[n1:2*n1]

    th_demo_p2 = np.interp(tt[n1:2*n1] - T, t, theta)
    recon = np.abs(p2 - th_demo_p2)
    settle = np.abs(p2 - p1)

    th_s = gaussian_smooth(theta, SMOOTH_SIGMA)
    t_max = float(th_s.max()); t_min = float(th_s.min())
    amp = t_max - t_min
    dwell_frac = float(np.mean(th_s > t_min + 0.85 * amp)) * 100
    closure = abs(float(theta[-1] - theta[0]))

    print("\n[FIT] 1-D tilt DMP summary:")
    print(f"  period      : {T:.2f} s (omega={dmp['omega']:.2f} rad/s)")
    print(f"  tilt axis   : {np.round(ext['axis'], 3)} (base frame)")
    print(f"  axis spread : {ext['spread']:.2f} deg "
          f"(max deviation where |theta|>20 deg; want < 5)")
    print(f"  theta range : {t_min:.1f} .. {t_max:.1f} deg "
          f"(amplitude {amp:.1f}; want 30-110)")
    print(f"  dwell       : {dwell_frac:.0f}% of cycle above 85% of amplitude")
    print(f"  closure     : {closure:.1f} deg (start vs end; want < 10)")
    print(f"  recon err   : {recon.max():.2f} deg max / {recon.mean():.2f} mean")
    print(f"  settle err  : {settle.max():.2f} deg (period2 vs period1,"
          f" informational)")
    ok = True
    if ext["spread"] > 5.0:
        print("  [WARN] rotation not single-axis - pour about ONE axis only.")
        ok = False
    if not (30 <= amp <= 110):
        print("  [WARN] tilt amplitude outside 30-110 deg.")
        ok = False
    if closure > 10:
        print("  [WARN] cycle not closed - return to the exact upright pose.")
        ok = False
    if recon.max() > 8:
        print("  [WARN] fit error high - slower demo, or raise N_BFS.")
        ok = False
    print(f"  QUALITY GATE: {'PASS' if ok else 'FAIL (see warnings above)'}")
    return ok

def maybe_plot(dmp, ext, t, theta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    T = 2 * np.pi / dmp["omega"]
    tt, yy = rollout_tilt(dmp, n_periods=2)
    n1 = int(round(T / 0.01))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, theta, "--", alpha=0.6, label="demo")
    ax.plot(tt[n1:2*n1] - T, yy[n1:2*n1], label="fit (settled)")
    ax.set_xlabel("t (s)"); ax.set_ylabel("tilt theta (deg)")
    ax.set_title("1-D tilt DMP fit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=110)
    print(f"  plot saved: {OUTPUT_PLOT}")

def preview_execute(robot, dmp):
    """Replay 1 revolution in place at the CURRENT position (matrix-composed
    orientation; position held fixed). blendR on all intermediate waypoints."""
    pose0 = _read_pose(robot)
    if pose0 is None:
        print("[PREVIEW] could not read start pose; aborting preview.")
        return
    pos_mm = pose0[:3]                       # TCP fixed at current position
    R_up = euler_to_R(*pose0[3:6])           # current upright orientation
    axis = dmp["axis"]

    tt, yy = rollout_tilt(dmp, n_periods=2, dt=0.01)
    T = 2 * np.pi / dmp["omega"]
    n1 = int(round(T / 0.01))
    second = yy[n1:]
    idx = np.linspace(0, len(second) - 1, 8, dtype=int)
    eulers = np.degrees(np.unwrap(np.stack(
        [R_to_ZYX_euler(axisang_R(axis, np.radians(second[i])) @ R_up)
         for i in idx]), axis=0))

    wps = [np.concatenate([pos_mm, eulers[k]]).tolist() for k in range(8)]
    print("\n[PREVIEW] 1 revolution, 8 waypoints, position fixed...")
    for wp in wps:
        robot.MoveL(desc_pos=wp, tool=0, user=0, vel=30, acc=100, ovl=50,
                    blendR=15.0, blendMode=0)
    robot.MoveL(desc_pos=wps[0], tool=0, user=0, vel=30, acc=100, ovl=50,
                blendR=15.0, blendMode=0)
    robot.MoveL(desc_pos=pose0.tolist(), tool=0, user=0, vel=30, acc=30,
                ovl=50)
    print("[PREVIEW] done.")

def inspect(path=OUTPUT_DMP):
    d = np.load(path, allow_pickle=True).item()
    print(f"[INSPECT] {path}")
    for k in ("n_dims", "n_bfs", "tau", "omega", "duration"):
        print(f"  {k:10s}: {d[k]}")
    print(f"  period   : {2*np.pi/d['omega']:.2f} s")
    print(f"  axis     : {np.round(d['axis'], 3)} (base frame)")
    print(f"  r_up     : {np.round(d['r_up_euler'], 1)} (deg)")
    print(f"  pos_fixed: {np.round(d['pos_fixed_mm'], 1)} (mm)")
    print(f"  theta    : 0 -> g={float(d['g'][0]):.1f} deg, "
          f"r={float(d['r'][0]):.1f} deg")


# ============================ MAIN ============================
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        inspect()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "refit":
        z = np.load(OUTPUT_RAW)
        t, Y = z["t"], z["Y"]
        t, Y = preprocess(t, Y)
        ext = extract_tilt(t, Y)
        dmp = fit_tilt_dmp(t, ext["theta"])
        dmp["axis"] = ext["axis"]
        dmp["r_up_euler"] = Y[0, 3:6]
        dmp["pos_fixed_mm"] = Y[0, :3]
        ok = report_tilt(dmp, ext, t, ext["theta"])
        maybe_plot(dmp, ext, t, ext["theta"])
        np.save(OUTPUT_DMP, dmp)
        print(f"[SAVED] {OUTPUT_DMP}  gate: {'PASS' if ok else 'FAIL'}")
        return

    print("SAFETY: e-stop reachable. Arm goes compliant while recording.")
    robot = Robot.RPC(ROBOT_IP)
    robot.LoggerInit(output_model=0)
    robot.SetLoggerLevel(1)
    err = robot.RobotEnable(1)
    print(f"RobotEnable -> {err}")
    time.sleep(1.0)

    _bottle_setup(robot)

    print(">>> Pre-position the arm AT the pour station: bottle UPRIGHT,")
    print(">>> camera UP as in the physical system.")
    print(">>> Recording starts from the CURRENT pose - no automatic motion.")
    input(">>> Press Enter when ready: ")

    if MODE == "drag":
        t, Y = record_drag(robot)
    else:
        t, Y = record_keyframe(robot)
    if t is None:
        return

    t, Y = preprocess(t, Y)
    if len(t) < MIN_SAMPLES:
        print(f"[ERROR] Only {len(t)} samples after preprocessing - "
              f"recording too short. Re-record.")
        return
    print(f"[PREPROCESS] {len(t)} samples, {t[-1]:.1f} s cycle")

    ext = extract_tilt(t, Y)
    dmp = fit_tilt_dmp(t, ext["theta"])
    dmp["axis"] = ext["axis"]
    dmp["r_up_euler"] = Y[0, 3:6]
    dmp["pos_fixed_mm"] = Y[0, :3]
    ok = report_tilt(dmp, ext, t, ext["theta"])
    maybe_plot(dmp, ext, t, ext["theta"])

    np.save(OUTPUT_DMP, dmp)
    np.savez(OUTPUT_RAW, t=t, Y=Y, mode=MODE)
    print(f"\n[SAVED] {OUTPUT_DMP}")
    print(f"[SAVED] {OUTPUT_RAW}")
    if not ok:
        print("[NOTE] Quality gate FAILED - file saved anyway, but consider "
              "re-recording before using it.")

    if EXECUTE_PREVIEW:
        preview_execute(robot, dmp)


if __name__ == "__main__":
    main()