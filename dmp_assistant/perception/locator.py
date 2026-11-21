"""Find an object with the camera and turn it into a top-down grasp.

  locator = ObjectLocator()                      # camera + YOLO, lazily
  target = locator.locate("bottle", flange_pose) # base-frame top centre + yaw
  hover, grasp = locator.grasp_poses(target, flange_pose)

Run alone for a live view (no robot motion; with --robot-ip it reads the
flange pose and prints base-frame coordinates):
  python -m dmp_assistant.perception.locator [--robot-ip 192.168.58.4]
"""
import sys
from dataclasses import dataclass
from pathlib import Path

if __name__ == "__main__":      # direct run: make the project root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import yaml

from dmp_assistant import paths
from dmp_assistant.perception import geometry as geo


class LocateError(RuntimeError):
    """The object could not be located reliably: the pick must not move."""


@dataclass
class Target:
    cls: str
    top: np.ndarray        # base-frame point on the object's top surface, mm
    yaw: float | None      # base yaw of the long axis (deg), None = round
    conf: float
    hits: int
    spread_mm: float


def load_hand_eye(path=paths.HAND_EYE_YAML):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    T = np.array(data["T_flange_cam"], float)
    if T.shape != (4, 4):
        raise ValueError(f"{path}: T_flange_cam must be 4x4")
    U, _, Vt = np.linalg.svd(T[:3, :3])            # rounded values -> exact rotation
    T[:3, :3] = U @ Vt
    return T


def load_config(path=paths.PERCEPTION_YAML):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def shrunk_box_mask(det, shape, shrink):
    import cv2
    c = np.array([det.cx, det.cy])
    pts = c + (np.asarray(det.corners, float) - c) * shrink
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask.astype(bool)


def measure(det, depth_mm, flange_pose, T_flange_cam, deproject, cfg):
    """One detection in one frame -> (top point in base, yaw or None).
    Returns None when the box has no usable depth."""
    loc = cfg["locate"]
    lo, hi = loc["depth_range_mm"]
    mask = shrunk_box_mask(det, depth_mm.shape, loc["box_shrink"])
    d = depth_mm[mask]
    d = d[(d > lo) & (d < hi)]
    if d.size < 20:
        return None
    # Nearest depths in the box = the object's top (a hollow object's
    # centre pixel would see the bottom, so the centre pixel is not used).
    z = float(np.percentile(d, loc["depth_percentile"]))
    top = geo.cam_point_to_base(deproject(det.cx, det.cy, z), flange_pose, T_flange_cam)

    angle, aspect = geo.obb_long_axis(det.w, det.h, det.r)
    yaw = None
    if aspect >= loc["min_aspect_for_yaw"]:
        yaw = geo.image_axis_to_base_yaw(angle, flange_pose, T_flange_cam)
    return top, yaw


def fuse(samples, cls, max_spread_mm, min_hits):
    """samples: list of (top, yaw, conf) from successive frames."""
    if len(samples) < min_hits:
        raise LocateError(f"'{cls}' seen in {len(samples)} frame(s), need {min_hits}.")
    tops = np.array([s[0] for s in samples])
    top = np.median(tops, axis=0)
    spread = float(np.max(np.linalg.norm(tops[:, :2] - top[:2], axis=1)))
    if spread > max_spread_mm:
        raise LocateError(f"'{cls}' position unstable across frames "
                          f"({spread:.1f} mm > {max_spread_mm} mm).")
    yaws = [s[1] for s in samples if s[1] is not None]
    # Elongated in most frames -> trust the axis; otherwise treat as round.
    yaw = geo.axis_mean_deg(yaws) if len(yaws) * 2 > len(samples) else None
    conf = float(np.mean([s[2] for s in samples]))
    return Target(cls, top, yaw, conf, len(samples), spread)


class ObjectLocator:
    def __init__(self, config=None, T_flange_cam=None, camera=None, detector=None):
        self.cfg = config or load_config()
        self.T_flange_cam = load_hand_eye() if T_flange_cam is None else T_flange_cam
        self._camera = camera
        self._detector = detector

    # Hardware is opened on first use, so building an Executor does not
    # need the camera or torch.
    @property
    def camera(self):
        if self._camera is None:
            from dmp_assistant.perception.camera import RealSenseCamera
            c = self.cfg["camera"]
            self._camera = RealSenseCamera(c["width"], c["height"], c["fps"],
                                           c["warmup_frames"])
        return self._camera

    @property
    def detector(self):
        if self._detector is None:
            from dmp_assistant.perception.detector import ObbDetector
            d = self.cfg["detector"]
            self._detector = ObbDetector(paths.YOLO_WEIGHTS, d["conf"], d["imgsz"])
        return self._detector

    def _valid_box(self, det, shape):
        f = self.cfg["detector"]["max_box_fraction"]
        return max(det.w, det.h) <= f * max(shape[0], shape[1])

    def locate(self, cls, flange_pose, near=None):
        """Median top point + long-axis yaw of `cls` over several frames.
        The robot must be still. `near` (base xy) picks the instance closest
        to it; otherwise the most confident one in the first frame."""
        loc = self.cfg["locate"]
        samples = []
        anchor = None if near is None else np.asarray(near[:2], float)
        live = hasattr(self.camera, "focus")             # LiveCamera: show it on the view
        if live:
            self.camera.focus, self.camera.note = cls, None
        try:
            target = self._locate(cls, flange_pose, loc, samples, anchor)
        except LocateError as e:
            if live:
                self.camera.note = f"{cls}: {e}"
            raise
        finally:
            if live:
                self.camera.focus = None
        if live:
            t = target.top
            self.camera.note = f"{cls} at ({t[0]:.0f}, {t[1]:.0f}, {t[2]:.0f}) mm"
        return target

    def _locate(self, cls, flange_pose, loc, samples, anchor):
        for _ in range(loc["frames"]):
            bgr, depth = self.camera.read()
            cands = []
            for det in self.detector.detect(bgr, cls):
                if not self._valid_box(det, bgr.shape):
                    continue
                m = measure(det, depth, flange_pose, self.T_flange_cam,
                            self.camera.deproject, self.cfg)
                if m is not None:
                    cands.append((m[0], m[1], det.conf))
            if not cands:
                continue
            if anchor is None:
                best = cands[0]                         # highest confidence
                anchor = best[0][:2]
            else:
                best = min(cands, key=lambda c: np.linalg.norm(c[0][:2] - anchor))
            samples.append(best)
        target = fuse(samples, cls, loc["max_spread_mm"], loc["min_hits"])
        target.top = target.top + np.asarray(loc.get("offset_mm", [0, 0, 0]), float)
        yaw = "round" if target.yaw is None else f"{target.yaw:.1f} deg"
        print(f"[PERCEPTION] {cls}: top=({target.top[0]:.1f}, {target.top[1]:.1f}, "
              f"{target.top[2]:.1f}) mm  yaw={yaw}  conf={target.conf:.2f}  "
              f"hits={target.hits}  spread={target.spread_mm:.1f} mm")
        return target

    def grasp_poses(self, target, ref_pose):
        """-> (hover flange pose, grasp flange pose). Orientation: ref_pose's
        rx/ry (top-down), rz turned so the jaws close across the long axis."""
        g = self.cfg["grasp"]
        depth = self.cfg.get("classes", {}).get(target.cls, {}) \
            .get("depth_mm", g["default_depth_mm"])
        rz = ref_pose[5] if target.yaw is None else \
            geo.grasp_rz(target.yaw, ref_pose[5], g["jaw_axis"])
        rpy = [ref_pose[3], ref_pose[4], rz]

        x, y, z_top = target.top
        tip_grasp = max(z_top - depth, g["table_z_mm"])
        clamp = " (clamped to table_z_mm)" if tip_grasp > z_top - depth else ""
        print(f"[PICK] fingertip target ({x:.1f}, {y:.1f}, {tip_grasp:.1f}) mm{clamp}, "
              f"object top {z_top:.1f} mm, tool rz {rz:.1f} deg")
        hover = geo.fingertip_to_flange([x, y, z_top + g["hover_mm"]], rpy,
                                        g["gripper_length_mm"])
        grasp = geo.fingertip_to_flange([x, y, tip_grasp], rpy,
                                        g["gripper_length_mm"])
        self.check_workspace(hover, "hover")
        self.check_workspace(grasp, "grasp")
        return hover, grasp

    def check_workspace(self, pose, label):
        ws = self.cfg["workspace"]
        for i, axis in enumerate("xyz"):
            lo, hi = ws[axis]
            if not lo <= pose[i] <= hi:
                raise LocateError(f"{label} pose {axis}={pose[i]:.1f} mm outside "
                                  f"workspace [{lo}, {hi}].")

    def close(self):
        if self._camera is not None:
            self._camera.close()
            self._camera = None


# ------------------------------------------------------------------
# Live view
# ------------------------------------------------------------------
def _draw(bgr, dets, labels):
    import cv2
    for det, label in zip(dets, labels):
        pts = np.round(det.corners).astype(np.int32)
        cv2.polylines(bgr, [pts], True, (0, 255, 255), 2)
        cv2.putText(bgr, label, (int(det.cx) - 60, int(det.cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return bgr


def main():
    import argparse
    import cv2

    ap = argparse.ArgumentParser(description="Live YOLO-OBB view with base-frame coordinates.")
    ap.add_argument("--robot-ip", default=None,
                    help="read the flange pose from this robot (no motion)")
    args = ap.parse_args()

    robot = None
    if args.robot_ip:
        from fairino import Robot
        robot = Robot.RPC(args.robot_ip)
        from dmp_assistant.skills.robot_state import wait_for_state
        wait_for_state(robot)

    loc = ObjectLocator()
    print(f"Classes: {loc.detector.classes}. Press q to quit.")
    try:
        while True:
            bgr, depth = loc.camera.read()
            dets = [d for d in loc.detector.detect(bgr) if loc._valid_box(d, bgr.shape)]
            flange = None
            if robot is not None:
                err, flange = robot.GetActualToolFlangePose(0)
                flange = flange if err == 0 else None
            labels = []
            for d in dets:
                label = f"{d.cls} {d.conf:.2f}"
                if flange is not None:
                    m = measure(d, depth, flange, loc.T_flange_cam,
                                loc.camera.deproject, loc.cfg)
                    if m is not None:
                        label += f" ({m[0][0]:.0f},{m[0][1]:.0f},{m[0][2]:.0f})"
                labels.append(label)
            cv2.imshow("perception", _draw(bgr, dets, labels))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        loc.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
