"""Session-level accumulation: DRUT history, timestamps, distraction reasons.

SessionState is the only significantly stateful component in the pipeline.
It receives one DistractionResult per frame and accumulates per-unit statistics.

  update()     — ingest one frame; returns True when a unit boundary is reached.
  close_unit() — must be called by the orchestrator immediately after update()
                 returns True; takes the unit blink count from EarTracker.reset_unit().
"""

import time

import numpy as np

from models import DetectionThresholds, DistractionReason, DistractionResult


class SessionState:
    def __init__(self, thresholds: DetectionThresholds) -> None:
        self._unit_frames: int = thresholds.drut_unit_frames
        self.session_start: float = time.time()

        # per-unit accumulators (reset in close_unit)
        self._frame_count:   int                          = 0
        self._d_count:       int                          = 0
        self._reason_counts: dict[DistractionReason, int] = {r: 0 for r in DistractionReason}

        # session history
        self.drut_history:        list[float]             = []
        self.timestamps:          list[float]             = []
        self.blink_rate_per_unit: list[int]               = []
        self.distraction_reasons: list[DistractionReason] = []
        self.blink_baseline:      float | None            = None
        self.total_blinks:        int                     = 0

        # monotonic counter for logging (does not reset)
        self._total_frames: int = 0

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def update(self, result: DistractionResult, total_blinks: int) -> bool:
        """Ingest one frame. Returns True when a unit boundary is reached.

        When True is returned, the caller must immediately call close_unit()
        with the blink count for this unit (typically ear.reset_unit()).
        """
        self._total_frames += 1
        self._frame_count  += 1
        self.total_blinks   = total_blinks
        if result.distracted:
            self._d_count += 1
            self._reason_counts[result.reason] += 1
        return self._frame_count >= self._unit_frames

    def close_unit(self, unit_blinks: int) -> float:
        """Finalize the current unit. Returns the DRUT for this unit.

        Call ear.reset_unit() and pass the result as unit_blinks so the
        reset is explicit in the orchestration layer.
        """
        drut    = self._d_count / self._frame_count
        elapsed = time.time() - self.session_start

        self.drut_history.append(drut)
        self.timestamps.append(elapsed)
        self.blink_rate_per_unit.append(unit_blinks)

        top = max(self._reason_counts, key=self._reason_counts.get)
        self.distraction_reasons.append(
            top if self._reason_counts[top] > 0 else DistractionReason.NONE
        )

        if self.blink_baseline is None and len(self.blink_rate_per_unit) >= 3:
            self.blink_baseline = float(np.mean(self.blink_rate_per_unit[:3]))

        self._frame_count   = 0
        self._d_count       = 0
        self._reason_counts = {k: 0 for k in self._reason_counts}

        return drut
