"""MediaPipe Tasks Hands wrapper (new API, forward-compatible).

Auto-downloads `hand_landmarker.task` on first use. Returns normalized
[0, 1] coords in frame space.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("hands")

_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/latest/"
              "hand_landmarker.task")
_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
_MODEL_PATH = _MODEL_DIR / "hand_landmarker.task"


def _ensure_model() -> Path:
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("downloading hand_landmarker.task to %s ...", _MODEL_PATH)
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    log.info("downloaded (%d bytes)", _MODEL_PATH.stat().st_size)
    return _MODEL_PATH


@dataclass
class HandResult:
    tip_x: float
    tip_y: float
    palm_x: float
    palm_y: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bbox_norm(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class HandsDetector:
    # 調低預設的信心門檻到 0.3，讓模糊的手也更容易被抓到
    def __init__(self, max_num_hands: int = 1, min_det: float = 0.3) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        model_path = _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        
        # 重點修改：從 VIDEO 改成 IMAGE 模式。
        # 因為 ESP32 傳輸會有掉幀和延遲，VIDEO 模式的追蹤器會錯亂，IMAGE 模式每幀獨立算反而更準！
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            running_mode=mp_vision.RunningMode.IMAGE,
            min_hand_detection_confidence=min_det,
            min_hand_presence_confidence=min_det,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> Optional[HandResult]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        
        # IMAGE 模式不需要算時間戳記了，直接推論
        res = self._detector.detect(mp_img)
        
        if not res.hand_landmarks:
            return None
            
        lms = res.hand_landmarks[0]
        xs = [lm.x for lm in lms]
        ys = [lm.y for lm in lms]
        return HandResult(
            tip_x=float(lms[8].x), tip_y=float(lms[8].y),
            palm_x=float(lms[9].x), palm_y=float(lms[9].y),
            x1=float(min(xs)), y1=float(min(ys)),
            x2=float(max(xs)), y2=float(max(ys)),
        )

    def close(self) -> None:
        try:
            self._detector.close()
        except Exception:
            pass