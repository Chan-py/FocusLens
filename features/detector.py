"""Face landmark detection abstraction.

FaceDetector is the boundary between raw camera frames and the typed
FrameFeatures used by the rest of the pipeline.  Swap implementations
to change backends without touching any downstream code.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import cv2
import numpy as np

from models import FrameFeatures, GazePosition
from .ear import compute_ear
from .gaze import compute_gaze
from .head_pose import compute_head_pose


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> FrameFeatures: ...

    @abstractmethod
    def close(self) -> None: ...


class MediaPipeFaceDetector(FaceDetector):
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence:  float = 0.5,
    ) -> None:
        import mediapipe as mp
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> FrameFeatures:
        h, w = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        res  = self._mesh.process(rgb)

        if not res.multi_face_landmarks:
            return FrameFeatures(
                face_detected=False,
                ear=None, gaze=None,
                pitch=None, yaw=None, roll=None,
            )

        lm = res.multi_face_landmarks[0].landmark
        pitch, yaw, roll = compute_head_pose(lm, w, h)
        return FrameFeatures(
            face_detected=True,
            ear=compute_ear(lm, w, h),
            gaze=compute_gaze(gray, lm, w, h),
            pitch=pitch, yaw=yaw, roll=roll,
        )

    def close(self) -> None:
        self._mesh.close()


class StubFaceDetector(FaceDetector):
    """Fixed-output detector for development without camera or MediaPipe."""

    def __init__(self, features: FrameFeatures | None = None) -> None:
        self._features = features or FrameFeatures(
            face_detected=True, ear=0.25, gaze=GazePosition.CENTER,
            pitch=-5.0, yaw=2.0, roll=0.0,
        )

    def detect(self, frame_bgr: np.ndarray) -> FrameFeatures:
        return self._features

    def close(self) -> None:
        pass
