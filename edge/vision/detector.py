"""YOLOv8 wrapper. Loads on first use so the server starts fast."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    label: str
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.w * self.h


class YoloDetector:
    def __init__(self, weights: str = "yolov8n.pt", device: str = "cpu", conf: float = 0.35) -> None:
        self.weights = weights
        self.device = device
        self.conf = conf
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from ultralytics import YOLO  # heavy import, deferred
            self._model = YOLO(self.weights)
        return self._model

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        model = self._ensure_loaded()
        res = model.predict(frame_bgr, device=self.device, conf=self.conf, verbose=False)[0]
        names = res.names
        out: List[Detection] = []
        if res.boxes is None:
            return out
        for b in res.boxes:
            cls = int(b.cls[0].item())
            xyxy = [int(v) for v in b.xyxy[0].tolist()]
            out.append(Detection(
                label=names[cls],
                conf=float(b.conf[0].item()),
                x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3],
            ))
        return out

    def best(self, dets: List[Detection], label_contains: str) -> Optional[Detection]:
        lc = label_contains.lower()
        cands = [d for d in dets if lc in d.label.lower()]
        if not cands:
            return None
        return max(cands, key=lambda d: d.conf)
