"""Shared test fixtures and helpers.

run_pipeline() is the key helper: it feeds a list of FrameFeatures through
the detection → session accumulation stages and returns the resulting
SessionState and EarTracker.  Tests use it to set up known states without
touching main() or the camera.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable regardless of how pytest is invoked.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from analytics import compute_summary
from detection import assess_distraction, derive_thresholds
from models import (
    CalibrationData,
    DetectionThresholds,
    DistractionResult,
    FrameFeatures,
    GazePosition,
    Scenario,
    UserIntent,
)
from session import EarTracker, SessionState


# ── shared constants ──────────────────────────────────────────────────────────

UNIT = 10   # small drut_unit_frames so tests don't need 800 iterations


# ── factory helpers ───────────────────────────────────────────────────────────

def make_thresholds(unit_frames: int = UNIT) -> DetectionThresholds:
    return DetectionThresholds(
        pitch_min=-20.0,
        pitch_max=20.0,
        yaw_threshold=20.0,
        ear_offset=0.02,
        ear_baseline=0.22,
        allowed_gazes=frozenset({GazePosition.CENTER}),
        drut_unit_frames=unit_frames,
        drut_focus_threshold=0.2,
        fatigue_blink_multiplier=1.5,
    )


def make_intent(scenario: Scenario = Scenario.SCREEN) -> UserIntent:
    return UserIntent(scenario=scenario)


def make_cal(**overrides) -> CalibrationData:
    defaults = dict(pitch_baseline=0.0, yaw_baseline=0.0,
                    ear_baseline=0.22, focus_gaze=None)
    return CalibrationData(**{**defaults, **overrides})


def focused_frame(ear: float = 0.28) -> FrameFeatures:
    return FrameFeatures(
        face_detected=True, ear=ear, gaze=GazePosition.CENTER,
        pitch=-5.0, yaw=2.0, roll=0.0,
    )


def no_face_frame() -> FrameFeatures:
    return FrameFeatures(
        face_detected=False, ear=None, gaze=None,
        pitch=None, yaw=None, roll=None,
    )


def gaze_deviated_frame(gaze: GazePosition = GazePosition.LEFT) -> FrameFeatures:
    return FrameFeatures(
        face_detected=True, ear=0.28, gaze=gaze,
        pitch=-5.0, yaw=2.0, roll=0.0,
    )


def head_deviated_frame(yaw: float = 30.0) -> FrameFeatures:
    return FrameFeatures(
        face_detected=True, ear=0.28, gaze=GazePosition.LEFT,
        pitch=-5.0, yaw=yaw, roll=0.0,
    )


def eyes_closed_frame() -> FrameFeatures:
    return FrameFeatures(
        face_detected=True, ear=0.05, gaze=GazePosition.CENTER,
        pitch=-5.0, yaw=2.0, roll=0.0,
    )


# ── pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline(
    frames:     list[FrameFeatures],
    thresholds: DetectionThresholds | None = None,
    intent:     UserIntent | None = None,
    cal:        CalibrationData | None = None,
) -> tuple[SessionState, EarTracker]:
    """Feed frames through detection + session stages. Returns (state, ear).

    Use this to set up known SessionState for analytics tests, or to run
    integration checks with ReplayDetector-equivalent behaviour.
    """
    if thresholds is None:
        thresholds = make_thresholds()
    ear   = EarTracker(thresholds)
    state = SessionState(thresholds)

    for f in frames:
        result = assess_distraction(f, thresholds, ear.threshold)
        if f.ear is not None:
            ear.update(f.ear)
        if state.update(result, ear.total_blinks):
            state.close_unit(ear.reset_unit())

    return state, ear
