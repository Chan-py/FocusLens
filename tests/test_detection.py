"""Tests for detection.py — assess_distraction() and derive_thresholds()."""

import pytest
from tests.conftest import (
    eyes_closed_frame, focused_frame, gaze_deviated_frame,
    head_deviated_frame, make_cal, make_intent, make_thresholds, no_face_frame,
)

from detection import assess_distraction, derive_thresholds
from models import DistractionReason, GazePosition, Scenario, UserIntent


# ── assess_distraction: sequential check order ────────────────────────────────

def test_no_face_returns_no_face():
    result = assess_distraction(no_face_frame(), make_thresholds(), 0.2)
    assert result.reason == DistractionReason.NO_FACE
    assert result.distracted


def test_head_deviation_checked_before_eyes_and_gaze():
    # yaw well beyond threshold; also has deviated gaze — HEAD_DEVIATION must win
    frame  = head_deviated_frame(yaw=30.0)
    result = assess_distraction(frame, make_thresholds(), 0.2)
    assert result.reason == DistractionReason.HEAD_DEVIATION


def test_eyes_closed_checked_before_gaze():
    # ear below dynamic threshold; gaze is also deviated
    frame  = eyes_closed_frame()
    result = assess_distraction(frame, make_thresholds(), ear_dynamic=0.2)
    assert result.reason == DistractionReason.EYES_CLOSED


def test_gaze_deviation_detected_when_only_distraction():
    frame  = gaze_deviated_frame(GazePosition.RIGHT)
    result = assess_distraction(frame, make_thresholds(), ear_dynamic=0.2)
    assert result.reason == DistractionReason.GAZE_DEVIATION


def test_focused_frame_returns_none():
    result = assess_distraction(focused_frame(), make_thresholds(), ear_dynamic=0.2)
    assert result.reason == DistractionReason.NONE
    assert not result.distracted


# ── UNKNOWN gaze must not be flagged as distracted ───────────────────────────

def test_unknown_gaze_not_distracted():
    frame  = gaze_deviated_frame(GazePosition.UNKNOWN)
    result = assess_distraction(frame, make_thresholds(), ear_dynamic=0.2)
    assert result.reason == DistractionReason.NONE


# ── distracted property is consistent with reason ────────────────────────────

@pytest.mark.parametrize("frame,expect_distracted", [
    (no_face_frame(),               True),
    (head_deviated_frame(),         True),
    (eyes_closed_frame(),           True),
    (gaze_deviated_frame(),         True),
    (focused_frame(),               False),
    (gaze_deviated_frame(GazePosition.UNKNOWN), False),
])
def test_distracted_property_matches_reason(frame, expect_distracted):
    result = assess_distraction(frame, make_thresholds(), ear_dynamic=0.2)
    assert result.distracted == expect_distracted


# ── derive_thresholds: scenario mapping ──────────────────────────────────────

def test_screen_uses_default_yaw_tolerance():
    from config import DEFAULT_YAW_TOLERANCE
    thresholds = derive_thresholds(make_intent(Scenario.SCREEN), make_cal())
    assert thresholds.yaw_threshold == DEFAULT_YAW_TOLERANCE


def test_paper_uses_tighter_yaw_tolerance():
    from config import DEFAULT_YAW_TOLERANCE, PAPER_YAW_TOLERANCE
    thresholds = derive_thresholds(make_intent(Scenario.PAPER), make_cal())
    assert thresholds.yaw_threshold == PAPER_YAW_TOLERANCE
    assert thresholds.yaw_threshold < DEFAULT_YAW_TOLERANCE


def test_pitch_range_derived_from_calibration_baseline():
    from config import DEFAULT_PITCH_TOLERANCE
    cal        = make_cal(pitch_baseline=-10.0)
    thresholds = derive_thresholds(make_intent(), cal)
    assert thresholds.pitch_min == pytest.approx(-10.0 - DEFAULT_PITCH_TOLERANCE)
    assert thresholds.pitch_max == pytest.approx(-10.0 + DEFAULT_PITCH_TOLERANCE)


def test_focus_gaze_added_to_allowed_gazes_for_paper():
    cal        = make_cal(focus_gaze=GazePosition.RIGHT)
    thresholds = derive_thresholds(make_intent(Scenario.PAPER), cal)
    assert GazePosition.RIGHT in thresholds.allowed_gazes
    assert GazePosition.CENTER in thresholds.allowed_gazes
