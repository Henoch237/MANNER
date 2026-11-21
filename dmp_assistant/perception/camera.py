"""RealSense colour + depth, depth aligned to colour.

Frames are grabbed on demand (no background thread): the robot is stopped
while the pick looks, so the latest frame is all we need.
"""
import time

import numpy as np


class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30, warmup_frames=15, retries=1):
        import pyrealsense2 as rs
        self._rs = rs

        if len(rs.context().query_devices()) == 0:
            raise RuntimeError("No RealSense camera detected. Check the USB connection.")

        # A process killed while streaming (Ctrl+C, kill) can leave the
        # camera stuck: "Frame didn't arrive within 5000". A hardware reset
        # (same as unplugging it) usually brings it back.
        for attempt in range(retries + 1):
            try:
                self._start(width, height, fps, warmup_frames)
                return
            except RuntimeError as e:
                if attempt == retries:
                    raise
                print(f"[CAMERA] RealSense start failed ({e}); hardware reset and retry...")
                self._hardware_reset()

    def _start(self, width, height, fps, warmup_frames):
        rs = self._rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(cfg)
        try:
            self.depth_scale_mm = profile.get_device().first_depth_sensor() \
                .get_depth_scale() * 1000.0
            # Depth is aligned INTO the colour image, so the colour intrinsics
            # are the ones that deproject both.
            self.intr = profile.get_stream(rs.stream.color) \
                .as_video_stream_profile().get_intrinsics()
            self.K = np.array([[self.intr.fx, 0, self.intr.ppx],
                               [0, self.intr.fy, self.intr.ppy],
                               [0, 0, 1]])
            self.dist = np.array(self.intr.coeffs, float)

            self._align = rs.align(rs.stream.color)
            # Stateful filters: created once, reused on every frame. No hole
            # filling - it invents depth at object edges, where we measure.
            self._spatial = rs.spatial_filter()
            self._temporal = rs.temporal_filter()

            for _ in range(warmup_frames):
                self.pipeline.wait_for_frames()
        except RuntimeError:
            self.close()
            raise

    def _hardware_reset(self, settle_s=6.0):
        for dev in self._rs.context().query_devices():
            try:
                dev.hardware_reset()
            except RuntimeError as e:
                print(f"[CAMERA] hardware reset failed: {e}")
        time.sleep(settle_s)              # the camera re-enumerates on USB

    def read(self):
        """-> (bgr uint8 HxWx3, depth_mm float32 HxW, 0 = no reading)."""
        frames = self._align.process(self.pipeline.wait_for_frames())
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("RealSense returned an incomplete frame.")
        depth = self._temporal.process(self._spatial.process(depth))
        depth_mm = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale_mm
        return np.asanyarray(color.get_data()).copy(), depth_mm

    def deproject(self, u, v, depth_mm):
        """Pixel + depth (mm) -> camera-frame point (mm), lens model included."""
        return np.array(self._rs.rs2_deproject_pixel_to_point(
            self.intr, [float(u), float(v)], float(depth_mm)))

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
