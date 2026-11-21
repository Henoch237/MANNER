"""
calibrate_hand_eye.py - eye-in-hand calibration of the wrist RealSense.

Fix a printed chessboard on the table, then capture it from 12-20 robot
poses that differ in POSITION and in TILT (rotate the wrist 15-30 deg
around different axes between captures; pure translations do not
constrain the rotation). The result goes to config/hand_eye.yaml
(T_flange_cam, mm), the old file to hand_eye.yaml.bak.

  python -m dmp_assistant.tools.calibrate_hand_eye                       # live
  python -m dmp_assistant.tools.calibrate_hand_eye --cols 9 --rows 6 --square-mm 25
  python -m dmp_assistant.tools.calibrate_hand_eye --from-samples data/calibration/hand_eye_<stamp>.npz

Live keys (in the camera window):
  d      toggle drag-teach (move the arm by hand), turn it OFF before capturing
  space  capture: board pose (solvePnP) + flange pose
  u      drop the last capture
  c      compute, compare methods, save the best to config/hand_eye.yaml
  q      quit (samples are saved on every capture)

--cols/--rows count INNER corners (a 10x7-square board has 9x6).
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

if __name__ == "__main__":      # direct run: make the project root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import yaml

from dmp_assistant import paths
from dmp_assistant.perception import geometry as geo

ROBOT_IP = "192.168.58.4"          # real robot (sim: 192.168.85.130)
MIN_SAMPLES = 8
MAX_RMS_PX = 2.0                   # reprojection limit per capture (640x480)
CAPTURE_FRAMES = 5                 # corners averaged per capture (less noise)
MAX_SPREAD_MM = 5.0                # board spread above this -> result not saved
SAME_POSE_MM, SAME_POSE_DEG = 10.0, 3.0   # closer than this = duplicate capture
STILL_MM, STILL_DEG = 0.5, 0.2     # flange drift during a capture = arm moving


# ---------------------------------------------------------------- board --
def board_points(cols, rows, square_mm):
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return obj


def find_board(gray, cols, rows):
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), cv2.CALIB_CB_EXHAUSTIVE)
        return ok, corners
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows))
    if ok:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
    return ok, corners


def canonical_board(T, cols, rows, square_mm):
    """A 9x6 board looks the same turned by 180 deg, so the detector may
    number its corners from either end. Pick the numbering whose X axis
    points towards image +u, so every capture uses the same board frame
    (keep the board's long side roughly left-right in the image)."""
    if T[0, 0] >= 0:
        return T
    flip = np.eye(4)
    flip[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    flip[:3, 3] = [(cols - 1) * square_mm, (rows - 1) * square_mm, 0.0]
    return T @ flip


def _rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def consistent_board_numbering(flange_poses, boards, cols, rows, square_mm):
    """Make every capture use the same board numbering as the first one,
    whatever the image orientation. Between captures 0 and j the flange
    turns by A and the camera sees the board turn by B; A @ X = X @ B, so
    both rotations have the SAME angle for any X. Of the two numberings of
    capture j, keep the one whose B angle matches A's."""
    if not boards:
        return boards
    flip = np.eye(4)
    flip[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    flip[:3, 3] = [(cols - 1) * square_mm, (rows - 1) * square_mm, 0.0]
    F0, C0 = geo.pose_to_T(flange_poses[0]), boards[0]
    out = [C0]
    for p, C in zip(flange_poses[1:], boards[1:]):
        a = _rot_angle_deg((np.linalg.inv(F0) @ geo.pose_to_T(p))[:3, :3])
        keep = abs(_rot_angle_deg((C0 @ np.linalg.inv(C))[:3, :3]) - a)
        flipped = abs(_rot_angle_deg((C0 @ np.linalg.inv(C @ flip))[:3, :3]) - a)
        out.append(C if keep <= flipped else C @ flip)
    return out


def pose_delta(a, b):
    """-> (translation mm, rotation deg) between two flange poses."""
    Ra, Rb = geo.euler_to_R(*a[3:6]), geo.euler_to_R(*b[3:6])
    c = np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)
    return float(np.linalg.norm(np.subtract(a[:3], b[:3]))), float(np.degrees(np.arccos(c)))


def distinct_captures(flange_poses, boards):
    """Drop captures taken from (almost) the same robot pose: repeats add
    no information and weigh that pose too much."""
    keep_p, keep_b = [], []
    for p, b in zip(flange_poses, boards):
        if all(pose_delta(p, q)[0] > SAME_POSE_MM or pose_delta(p, q)[1] > SAME_POSE_DEG
               for q in keep_p):
            keep_p.append(p)
            keep_b.append(b)
    return keep_p, keep_b


def capture_board(cam, obj, cols, rows, n=CAPTURE_FRAMES):
    """Fresh frames with the robot still -> board pose from the corners
    averaged over n frames. -> (T, rms) or (None, reason)."""
    found = []
    for _ in range(n):
        bgr, _ = cam.read()
        ok, corners = find_board(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cols, rows)
        if ok:
            found.append(corners.reshape(-1, 2))
    if len(found) < (n + 1) // 2:
        return None, f"board seen in {len(found)}/{n} frames"
    corners = np.mean(found, axis=0).astype(np.float32).reshape(-1, 1, 2)
    T, rms = board_pose(corners, obj, cam.K, cam.dist)
    return (T, rms) if T is not None else (None, "solvePnP failed")


def board_pose(corners, obj, K, dist):
    """-> (4x4 T_cam_board in mm, reprojection RMS px) or (None, None)."""
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, 1))))
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T, rms


# ---------------------------------------------------------------- solve --
def solve_all(flange_poses, T_cam_board):
    """Every available method -> {name: (T_flange_cam, pos_std_mm, ang_dev_deg)}."""
    F = [geo.pose_to_T(p) for p in flange_poses]
    results = {"park (numpy)": geo.hand_eye_park(F, T_cam_board)}

    if hasattr(cv2, "calibrateHandEye"):
        R_g2b = [T[:3, :3] for T in F]
        t_g2b = [T[:3, 3].reshape(3, 1) for T in F]
        R_t2c = [T[:3, :3] for T in T_cam_board]
        t_t2c = [T[:3, 3].reshape(3, 1) for T in T_cam_board]
        for name in ("TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"):
            try:
                R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c,
                                            method=getattr(cv2, f"CALIB_HAND_EYE_{name}"))
            except cv2.error as e:
                print(f"  opencv {name}: failed ({e})")
                continue
            X = np.eye(4)
            X[:3, :3] = R
            X[:3, 3] = t.ravel()
            results[f"opencv {name.lower()}"] = X
    else:
        print("  (this OpenCV build has no calibrateHandEye; numpy Park-Martin only)")

    return {k: (X, *geo.target_in_base_spread(F, T_cam_board, X)) for k, X in results.items()}


def rotation_diversity_deg(flange_poses):
    """Largest rotation between any two captures (needs to be >> 10 deg)."""
    Rs = [geo.euler_to_R(*p[3:6]) for p in flange_poses]
    best = 0.0
    for i in range(len(Rs)):
        for j in range(i + 1, len(Rs)):
            c = np.clip((np.trace(Rs[i].T @ Rs[j]) - 1) / 2, -1, 1)
            best = max(best, float(np.degrees(np.arccos(c))))
    return best


def save_result(X, method, n, pos_std, ang_dev, path=paths.HAND_EYE_YAML):
    if Path(path).exists():
        shutil.copy(path, str(path) + ".bak")
    data = {
        "version": "1.0",
        "source": f"calibrate_hand_eye.py, {time.strftime('%Y-%m-%d %H:%M')}",
        "method": method,
        "n_samples": int(n),
        "board_spread_mm": round(pos_std, 2),
        "board_spread_deg": round(ang_dev, 2),
        "T_flange_cam": [[round(float(v), 6) for v in row] for row in X],
    }
    header = ("# hand_eye.yaml\n"
              "# Eye-in-hand calibration: p_flange = T_flange_cam @ p_cam (mm).\n"
              "# Written by: python -m dmp_assistant.tools.calibrate_hand_eye\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None)
    print(f"Saved {path} (previous kept as {Path(path).name}.bak)")


def compute_and_save(flange_poses, T_cam_board, board=(9, 6, 25.0), save=True,
                     force=False):
    T_cam_board = [canonical_board(T, *board) for T in T_cam_board]
    total = len(flange_poses)
    flange_poses, T_cam_board = distinct_captures(flange_poses, T_cam_board)
    T_cam_board = consistent_board_numbering(flange_poses, T_cam_board, *board)
    n = len(flange_poses)
    if n < total:
        print(f"{total - n} of {total} captures repeat an earlier robot pose: ignored.")
    if n < MIN_SAMPLES:
        print(f"Need at least {MIN_SAMPLES} DIFFERENT robot poses, have {n}.")
        return None
    div = rotation_diversity_deg(flange_poses)
    print(f"\n{n} captures, largest rotation between two of them: {div:.1f} deg")
    if div < 20:
        print("  WARNING: too little rotation - tilt the wrist more between captures.")

    results = solve_all(flange_poses, T_cam_board)
    print("\nBoard pose in base across captures (lower = better):")
    for name, (X, p, a) in sorted(results.items(), key=lambda kv: kv[1][1]):
        t = X[:3, 3]
        print(f"  {name:20s} spread {p:6.2f} mm  {a:5.2f} deg   "
              f"t = ({t[0]:7.1f}, {t[1]:7.1f}, {t[2]:7.1f}) mm")
    name, (X, p, a) = min(results.items(), key=lambda kv: kv[1][1])
    print(f"\nBest: {name}")
    if p > MAX_SPREAD_MM and not force:
        print(f"  NOT SAVED: spread {p:.1f} mm > {MAX_SPREAD_MM} mm. Check --square-mm, the "
              f"board flatness, and that the arm was still at each capture. "
              f"config/hand_eye.yaml is unchanged.")
        return None
    if save:
        save_result(X, name, n, p, a)
    return X


# ---------------------------------------------------------------- live --
def live(args):
    try:                                            # errors only: no drawFrameAxes spam
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass
    from fairino import Robot
    from dmp_assistant.perception.camera import RealSenseCamera

    robot = Robot.RPC(args.robot_ip)
    from dmp_assistant.skills.robot_state import wait_for_state
    wait_for_state(robot)
    cam = RealSenseCamera()
    obj = board_points(args.cols, args.rows, args.square_mm)
    paths.CALIB_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.CALIB_DIR / f"hand_eye_{time.strftime('%Y%m%d_%H%M%S')}.npz"

    flange_poses, boards = [], []
    drag = False
    print(__doc__)
    try:
        while True:
            bgr, _ = cam.read()
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            ok, corners = find_board(gray, args.cols, args.rows)
            view = bgr.copy()
            T, rms = (None, None)
            if ok:
                cv2.drawChessboardCorners(view, (args.cols, args.rows), corners, ok)
                T, rms = board_pose(corners, obj, cam.K, cam.dist)
                if T is not None:
                    cv2.drawFrameAxes(view, cam.K, cam.dist, cv2.Rodrigues(T[:3, :3])[0],
                                      T[:3, 3], args.square_mm * 2)
            good = T is not None and rms <= args.max_rms and not drag
            status = (f"captures: {len(flange_poses)}  board: "
                      f"{'rms %.2f px' % rms if T is not None else 'not found'}"
                      f"  drag: {'ON' if drag else 'off'}")
            cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if good else (0, 0, 255), 2)
            cv2.imshow("hand-eye calibration", view)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("d"):
                drag = not drag
                robot.DragTeachSwitch(1 if drag else 0)
                print(f"drag-teach {'ON' if drag else 'off'}")
            elif key == ord("u") and flange_poses:
                flange_poses.pop()
                boards.pop()
                print(f"dropped last capture, {len(flange_poses)} left")
            elif key == ord(" "):
                if drag:
                    print("turn drag-teach off (d) before capturing: the arm must be still")
                    continue
                time.sleep(0.5)                         # let the arm settle
                err, before = robot.GetActualToolFlangePose(0)
                T, rms = capture_board(cam, obj, args.cols, args.rows)
                if T is None:
                    print(f"not captured: {rms} (whole board in view? glare?)")
                    continue
                if rms > args.max_rms:
                    print(f"reprojection {rms:.2f} px > {args.max_rms}, not captured "
                          f"(glare, blur, or the board is not flat)")
                    continue
                err2, flange = robot.GetActualToolFlangePose(0)
                if err or err2:
                    print(f"GetActualToolFlangePose failed ({err or err2}), not captured")
                    continue
                moved = pose_delta(before, flange)
                if moved[0] > STILL_MM or moved[1] > STILL_DEG:
                    print(f"arm moved during the capture ({moved[0]:.1f} mm, {moved[1]:.1f} deg),"
                          f" not captured: is drag mode still on?")
                    continue
                dup = [i + 1 for i, q in enumerate(flange_poses)
                       if pose_delta(flange, q)[0] <= SAME_POSE_MM
                       and pose_delta(flange, q)[1] <= SAME_POSE_DEG]
                if dup:
                    print(f"same robot pose as capture {dup[0]}, not captured: "
                          f"move the arm to a NEW pose (change position AND tilt)")
                    continue
                flange_poses.append(list(flange))
                boards.append(canonical_board(T, args.cols, args.rows, args.square_mm))
                np.savez(out, flange_poses=np.array(flange_poses), T_cam_board=np.array(boards),
                         K=cam.K, dist=cam.dist, cols=args.cols, rows=args.rows,
                         square_mm=args.square_mm)
                print(f"capture {len(flange_poses)}: flange={np.round(flange, 1).tolist()}"
                      f"  rms={rms:.2f} px  -> {out.name}")
            elif key == ord("c"):
                compute_and_save(flange_poses, boards, (args.cols, args.rows, args.square_mm),
                                 force=args.force)
    finally:
        if drag:
            robot.DragTeachSwitch(0)
        cam.close()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Eye-in-hand calibration (chessboard).")
    ap.add_argument("--robot-ip", default=ROBOT_IP)
    ap.add_argument("--cols", type=int, default=9, help="inner corners per row")
    ap.add_argument("--rows", type=int, default=6, help="inner corners per column")
    ap.add_argument("--square-mm", type=float, default=25.0, help="printed square size")
    ap.add_argument("--max-rms", type=float, default=MAX_RMS_PX,
                    help="reject captures above this reprojection error (px)")
    ap.add_argument("--from-samples", type=Path, default=None,
                    help="recompute from a saved .npz instead of running live")
    ap.add_argument("--force", action="store_true",
                    help=f"save even when the board spread exceeds {MAX_SPREAD_MM} mm")
    ap.add_argument("--dry-run", action="store_true", help="with --from-samples: do not save")
    args = ap.parse_args()

    if args.from_samples:
        d = np.load(args.from_samples)
        board = (int(d["cols"]), int(d["rows"]), float(d["square_mm"]))
        compute_and_save(list(d["flange_poses"]), list(d["T_cam_board"]), board,
                         save=not args.dry_run, force=args.force)
    else:
        live(args)


if __name__ == "__main__":
    main()
