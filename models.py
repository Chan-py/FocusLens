"""Pipeline boundary types — the explicit contracts between stages.

Reading this file gives a complete picture of what data flows
between modules and in what direction.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


# ── Stage 0: User input ───────────────────────────────────────────────────────

class Scenario(Enum):
    SCREEN = "screen"
    PAPER  = "paper"
    BOOK   = "book"

    @property
    def label(self) -> str:
        return {"screen": "화면 작업", "paper": "종이 공부", "book": "독서"}[self.value]


@dataclass(frozen=True)
class UserIntent:
    scenario: Scenario


# ── Stage 1: Calibration measurements ─────────────────────────────────────────

class GazePosition(Enum):
    LEFT       = "left"
    CENTER     = "center"
    RIGHT      = "right"
    OFF_CENTER = "off-center"
    UNKNOWN    = "unknown"


@dataclass(frozen=True)
class CalibrationData:
    """Raw measurements from the calibration phase. Immutable after collection."""
    pitch_baseline: float
    yaw_baseline:   float
    ear_baseline:   float
    focus_gaze:     GazePosition | None   # measured focus-state gaze direction, if any

    @classmethod
    def defaults(cls) -> CalibrationData:
        """Bypass calibration for development/testing."""
        return cls(pitch_baseline=0.0, yaw_baseline=0.0,
                   ear_baseline=0.22, focus_gaze=None)


# ── Stage 2: Detection thresholds (derived, never mutated) ────────────────────

@dataclass(frozen=True)
class DetectionThresholds:
    """
    The contract between calibration and tracking.
    Produced once by derive_thresholds(); consumed by EarTracker and
    assess_distraction(). Neither consumer needs to know about scenarios.
    """
    pitch_min:                float
    pitch_max:                float
    yaw_threshold:            float
    yaw_baseline:             float          # calibrated neutral yaw position
    ear_offset:               float          # threshold = mean(window) − ear_offset
    ear_baseline:             float          # fallback before window fills
    allowed_gazes:            frozenset[GazePosition]
    drut_unit_frames:         int
    drut_focus_threshold:     float
    fatigue_blink_multiplier: float


# ── Stage 3: Per-frame sensor output ─────────────────────────────────────────

@dataclass(frozen=True)
class FrameFeatures:
    """All features extracted from a single frame. None = not computable."""
    face_detected: bool
    ear:           float        | None
    gaze:          GazePosition | None
    pitch:         float        | None
    yaw:           float        | None
    roll:          float        | None


# ── Stage 4: Per-frame distraction decision ───────────────────────────────────

class DistractionReason(Enum):
    NONE           = "none"
    NO_FACE        = "no_face"
    HEAD_DEVIATION = "head_deviation"
    EYES_CLOSED    = "eyes_closed"
    GAZE_DEVIATION = "gaze_deviation"


@dataclass(frozen=True)
class DistractionResult:
    reason:   DistractionReason
    features: FrameFeatures
    reasons:  frozenset[DistractionReason] = field(default_factory=frozenset)

    @property
    def distracted(self) -> bool:
        return self.reason is not DistractionReason.NONE


# ── Stage 5→6: Session analytics output ──────────────────────────────────────

@dataclass(frozen=True)
class DropEvent:
    time_min: float
    trigger:  DistractionReason


class FocusPattern(Enum):
    FRONT_LOADED      = "front_loaded"
    BACK_LOADED       = "back_loaded"
    CONSISTENT        = "consistent"
    INSUFFICIENT_DATA = "insufficient_data"


class BlinkTrend(Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE     = "stable"


@dataclass(frozen=True)
class SessionSummary:
    """Immutable analytics snapshot passed to the LLM reporter."""
    scenario:                Scenario
    session_duration_min:    float
    effective_focus_ratio:   float
    total_blinks:            int
    golden_hour:             tuple[float, float] | None   # (start_min, end_min)
    drop_events:             tuple[DropEvent, ...]
    fatigue_onset_min:       float | None
    focus_pattern:           FocusPattern
    blink_trend:             BlinkTrend
    top_distraction:         DistractionReason
    blink_baseline_per_unit: float | None
    drut_history:            tuple[float, ...]
    calibration_summary:     dict
