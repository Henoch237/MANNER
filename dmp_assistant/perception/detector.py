"""YOLOv8-OBB detector (model/best.pt): oriented boxes for the table objects."""
from dataclasses import dataclass

import numpy as np

from dmp_assistant import paths


@dataclass
class Detection:
    cls: str
    conf: float
    cx: float            # box centre, pixels
    cy: float
    w: float             # box sides, pixels
    h: float
    r: float             # rotation of the w side, rad (from +u towards +v)
    corners: np.ndarray  # 4x2 pixel corners


class ObbDetector:
    def __init__(self, weights=paths.YOLO_WEIGHTS, conf=0.5, imgsz=640):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        if self.model.task != "obb":
            raise ValueError(f"{weights} is a '{self.model.task}' model, expected 'obb'.")
        self.names = dict(self.model.names)
        self.conf = conf
        self.imgsz = imgsz

    @property
    def classes(self):
        return list(self.names.values())

    def detect(self, bgr, cls=None):
        """All detections (optionally one class), best confidence first."""
        res = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        if res.obb is None or len(res.obb) == 0:
            return []
        xywhr = res.obb.xywhr.cpu().numpy()
        confs = res.obb.conf.cpu().numpy()
        ids = res.obb.cls.cpu().numpy().astype(int)
        corners = res.obb.xyxyxyxy.cpu().numpy()
        out = []
        for (cx, cy, w, h, r), c, k, pts in zip(xywhr, confs, ids, corners):
            name = self.names[k]
            if cls is not None and name != cls:
                continue
            out.append(Detection(name, float(c), float(cx), float(cy),
                                 float(w), float(h), float(r), pts))
        return sorted(out, key=lambda d: d.conf, reverse=True)
