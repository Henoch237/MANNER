"""Frame math for the camera pick. Pure numpy: no camera, robot or model.

Conventions (same as skills/pour.py and grasping-v6):
  - FAIRINO pose [x, y, z, rx, ry, rz], mm / deg, R = Rz @ Ry @ Rx.
  - Robot poses are FLANGE poses (tool 0).
  - T_flange_cam maps camera-frame points (mm) into the flange frame.
  - Image angles are measured from +u towards +v (image x right, y down),
    which is the camera frame's +x towards +y.
"""
import math

import numpy as np


def euler_to_R(rx, ry, rz):
    """FAIRINO [Rx,Ry,Rz] (deg) -> rotation matrix (R = Rz @ Ry @ Rx)."""
    rx, ry, rz = np.deg2rad([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_euler(R):
    """rotation matrix -> FAIRINO [Rx,Ry,Rz] in degrees."""
    sy = math.hypot(R[0, 0], R[1, 0])
    if sy < 1e-6:                                   # gimbal lock
        return [math.degrees(math.atan2(-R[1, 2], R[1, 1])),
                math.degrees(math.atan2(-R[2, 0], sy)), 0.0]
    return [math.degrees(math.atan2(R[2, 1], R[2, 2])),
            math.degrees(math.atan2(-R[2, 0], sy)),
            math.degrees(math.atan2(R[1, 0], R[0, 0]))]


def pose_to_T(pose):
    """[x,y,z,rx,ry,rz] -> 4x4 homogeneous transform (mm)."""
    T = np.eye(4)
    T[:3, :3] = euler_to_R(*pose[3:6])
    T[:3, 3] = pose[:3]
    return T


def T_to_pose(T):
    return [float(v) for v in T[:3, 3]] + [float(v) for v in R_to_euler(T[:3, :3])]


def cam_point_to_base(p_cam_mm, flange_pose, T_flange_cam):
    """Camera-frame point (mm) -> base-frame point (mm)."""
    p = np.append(np.asarray(p_cam_mm, float), 1.0)
    return (pose_to_T(flange_pose) @ T_flange_cam @ p)[:3]


def image_axis_to_base_yaw(angle_rad, flange_pose, T_flange_cam):
    """Direction of an image-plane axis (angle from +u towards +v) ->
    yaw (deg) of that axis projected on the base XY plane."""
    d_cam = np.array([math.cos(angle_rad), math.sin(angle_rad), 0.0])
    d_base = euler_to_R(*flange_pose[3:6]) @ T_flange_cam[:3, :3] @ d_cam
    return math.degrees(math.atan2(d_base[1], d_base[0]))


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def axis_mean_deg(angles):
    """Mean of undirected axes (period 180 deg), via doubled-angle averaging."""
    a = np.deg2rad(np.asarray(angles, float) * 2.0)
    return math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) / 2.0


def grasp_rz(object_yaw_deg, ref_rz_deg, jaw_axis="y"):
    """Tool rz that closes the jaws across the object's long axis, chosen
    within +-90 deg of ref_rz (the gripper is symmetric, so the smaller
    wrist rotation always exists).

    Top-down tool (rx ~ 180): rz is the base yaw of tool X. With jaw_axis
    "y" the jaws close along tool Y, so tool X goes along the long axis;
    with "x" the jaws close along tool X, so tool X goes across it."""
    target = object_yaw_deg if jaw_axis == "y" else object_yaw_deg + 90.0
    return ref_rz_deg + wrap180(2.0 * (target - ref_rz_deg)) / 2.0


def fingertip_to_flange(tip_xyz, rpy, gripper_length_mm):
    """Flange pose that puts the fingertip at tip_xyz with orientation rpy."""
    R = euler_to_R(*rpy)
    flange = np.asarray(tip_xyz, float) - R @ np.array([0.0, 0.0, gripper_length_mm])
    return [float(v) for v in flange] + [float(v) for v in rpy]


def obb_long_axis(w, h, r):
    """YOLO-OBB xywhr -> (long-axis angle in rad, long/short aspect)."""
    long_side, short_side = (w, h) if w >= h else (h, w)
    angle = r if w >= h else r + math.pi / 2.0
    return angle, long_side / max(short_side, 1e-6)


# ------------------------------------------------------------------
# Hand-eye (AX = XB), Park & Martin 1994. Used when the installed
# OpenCV has no calibrateHandEye (OpenCV 5 Python wheels).
# ------------------------------------------------------------------
def _log_rot(R):
    """Rotation matrix -> axis-angle vector."""
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = math.acos(cos)
    if theta < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta * w / (2.0 * math.sin(theta))


def hand_eye_park(T_base_flange, T_cam_target):
    """Eye-in-hand calibration. Inputs: lists of 4x4 transforms, one pair
    per capture (flange in base, board in camera). Returns T_flange_cam.

    Relative motions between captures i and j:
      A = inv(F_i) @ F_j   (flange motion)
      B = C_i @ inv(C_j)   (camera motion seen through the fixed board)
    and A @ X = X @ B with X = T_flange_cam."""
    n = len(T_base_flange)
    As, Bs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            As.append(np.linalg.inv(T_base_flange[i]) @ T_base_flange[j])
            Bs.append(T_cam_target[i] @ np.linalg.inv(T_cam_target[j]))

    M = np.zeros((3, 3))
    for A, B in zip(As, Bs):
        M += np.outer(_log_rot(B[:3, :3]), _log_rot(A[:3, :3]))
    # R = (M^T M)^(-1/2) M^T
    w, V = np.linalg.eigh(M.T @ M)
    R = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T @ M.T
    U, _, Vt = np.linalg.svd(R)                     # re-orthonormalise
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt

    C = np.vstack([A[:3, :3] - np.eye(3) for A in As])
    d = np.concatenate([R @ B[:3, 3] - A[:3, 3] for A, B in zip(As, Bs)])
    t, *_ = np.linalg.lstsq(C, d, rcond=None)

    X = np.eye(4)
    X[:3, :3] = R
    X[:3, 3] = t
    return X


def target_in_base_spread(T_base_flange, T_cam_target, T_flange_cam):
    """Consistency check: the board is fixed, so base->board must be the
    same for every capture. Returns (position std mm, max angle dev deg)."""
    Ts = [F @ T_flange_cam @ C for F, C in zip(T_base_flange, T_cam_target)]
    pos = np.array([T[:3, 3] for T in Ts])
    R0 = Ts[0][:3, :3]
    ang = [math.degrees(np.linalg.norm(_log_rot(R0.T @ T[:3, :3]))) for T in Ts]
    return float(np.linalg.norm(pos.std(axis=0))), float(max(ang))
