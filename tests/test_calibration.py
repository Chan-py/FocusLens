"""Tests for calibration/calibrator.py — compute_calibration() pure function."""

import pytest
from tests.conftest import focused_frame, make_cal

from calibration import compute_calibration
from models import FrameFeatures, GazePosition


def _frame(pitch=0.0, yaw=0.0, ear=0.25) -> FrameFeatures:
    return FrameFeatures(
        face_detected=True, ear=ear, gaze=GazePosition.CENTER,
        pitch=pitch, yaw=yaw, roll=0.0,
    )


# ── baseline averaging ────────────────────────────────────────────────────────

def test_pitch_baseline_is_mean_of_forward_frames():
    frames = [_frame(pitch=-10.0), _frame(pitch=-8.0), _frame(pitch=-12.0)]
    cal = compute_calibration(forward_frames=frames, blink_frames=frames)
    assert cal.pitch_baseline == pytest.approx(-10.0)


def test_yaw_baseline_is_mean_of_forward_frames():
    frames = [_frame(yaw=2.0), _frame(yaw=4.0)]
    cal = compute_calibration(forward_frames=frames, blink_frames=frames)
    assert cal.yaw_baseline == pytest.approx(3.0)


def test_ear_baseline_is_mean_of_blink_frames():
    fwd    = [_frame(ear=0.30)]
    blinks = [_frame(ear=0.28), _frame(ear=0.32)]
    cal = compute_calibration(forward_frames=fwd, blink_frames=blinks)
    assert cal.ear_baseline == pytest.approx(0.30)


# ── study posture overrides pitch_baseline ────────────────────────────────────

def test_study_frames_override_pitch_for_paper_scenario():
    fwd   = [_frame(pitch=0.0)]
    blink = [_frame(ear=0.25)]
    study = [_frame(pitch=-20.0), _frame(pitch=-18.0)]
    cal = compute_calibration(forward_frames=fwd, blink_frames=blink,
                              study_frames=study)
    assert cal.pitch_baseline == pytest.approx(-19.0)


def test_no_study_frames_keeps_forward_pitch():
    frames = [_frame(pitch=-5.0)]
    cal = compute_calibration(forward_frames=frames, blink_frames=frames,
                              study_frames=None)
    assert cal.pitch_baseline == pytest.approx(-5.0)


# ── fallbacks when frames are empty or have no valid values ──────────────────

def test_fallback_pitch_zero_when_no_detected_faces():
    empty = []
    cal = compute_calibration(forward_frames=empty, blink_frames=empty)
    assert cal.pitch_baseline == pytest.approx(0.0)
    assert cal.yaw_baseline   == pytest.approx(0.0)


def test_fallback_ear_baseline_when_no_valid_ears():
    no_ear = [FrameFeatures(face_detected=True, ear=None, gaze=None,
                            pitch=0.0, yaw=0.0, roll=0.0)]
    cal = compute_calibration(forward_frames=no_ear, blink_frames=no_ear)
    assert cal.ear_baseline == pytest.approx(0.22)


def test_zero_ear_values_are_excluded_from_baseline():
    frames = [_frame(ear=0.0), _frame(ear=0.30)]
    cal = compute_calibration(forward_frames=frames, blink_frames=frames)
    assert cal.ear_baseline == pytest.approx(0.30)


# ── focus_gaze is not yet collected (stub) ────────────────────────────────────

def test_focus_gaze_is_none():
    frames = [_frame()]
    cal = compute_calibration(forward_frames=frames, blink_frames=frames)
    assert cal.focus_gaze is None
