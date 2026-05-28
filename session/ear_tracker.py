"""EAR sliding-window tracker: dynamic threshold + blink counting.

Separated from SessionState so it can be tested and reasoned about
independently.  All blink-related state lives here.
"""

from collections import deque
import numpy as np

from config import CLOSED_EYE_FRAMES, EAR_WINDOW_MIN_SAMPLES, EAR_WINDOW_SIZE
from models import DetectionThresholds


class EarTracker:
    """
    Maintains a sliding window of EAR values to compute a person-adaptive
    blink threshold.  Also counts blinks at two granularities:
      - total_blinks : entire session
      - unit_blinks  : current DRUT unit (reset via reset_unit())
    """

    def __init__(self, thresholds: DetectionThresholds) -> None:
        self._window:   deque[float] = deque(maxlen=EAR_WINDOW_SIZE)
        self._cef:      int          = 0          # consecutive sub-threshold frames
        self._baseline: float        = thresholds.ear_baseline
        self._offset:   float        = thresholds.ear_offset
        self._cached_threshold: float = thresholds.ear_baseline - thresholds.ear_offset
        self.total_blinks: int = 0
        self.unit_blinks:  int = 0

    @property
    def threshold(self) -> float:
        """Last computed threshold — updated on every update() call."""
        return self._cached_threshold

    def _compute_threshold(self) -> float:
        if len(self._window) < EAR_WINDOW_MIN_SAMPLES:
            return self._baseline - self._offset
        return float(np.mean(self._window)) - self._offset

    def update(self, ear: float) -> None:
        """Ingest one frame's EAR value; update blink state."""
        self._window.append(ear)
        self._cached_threshold = self._compute_threshold()
        if ear < self._cached_threshold:
            self._cef += 1
        else:
            if self._cef >= CLOSED_EYE_FRAMES:
                self.total_blinks += 1
                self.unit_blinks  += 1
            self._cef = 0

    def reset_unit(self) -> int:
        """Call at each DRUT unit boundary.  Returns unit count then resets."""
        count = self.unit_blinks
        self.unit_blinks = 0
        return count
