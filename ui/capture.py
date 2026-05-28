"""Camera capture abstraction.

CaptureSource is the structural Protocol that cv2.VideoCapture and StubCapture
both satisfy.  Use it as the type wherever cap is passed or stored.
"""

import cv2
import numpy as np
from typing import Protocol


class CaptureSource(Protocol):
    """Minimal interface shared by cv2.VideoCapture and StubCapture."""
    def isOpened(self) -> bool: ...
    def read(self)     -> tuple[bool, np.ndarray]: ...
    def release(self)  -> None: ...


class StubCapture:
    """Synthetic camera: returns blank frames so the UI loop keeps running."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self._w    = width
        self._h    = height
        self._open = True

    def isOpened(self) -> bool:
        return self._open

    def read(self) -> tuple[bool, np.ndarray]:
        frame = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        cv2.putText(frame, "STUB CAPTURE (dev mode)", (10, self._h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 200), 2)
        return True, frame

    def release(self) -> None:
        self._open = False
