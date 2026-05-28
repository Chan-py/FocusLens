"""Tests for session/ear_tracker.py — EarTracker."""

import pytest
from tests.conftest import make_thresholds

from config import CLOSED_EYE_FRAMES, EAR_WINDOW_MIN_SAMPLES
from session.ear_tracker import EarTracker


def _tracker() -> EarTracker:
    return EarTracker(make_thresholds())


# ── threshold fallback before window fills ────────────────────────────────────

def test_initial_threshold_uses_baseline_minus_offset():
    t = _tracker()
    assert t.threshold == pytest.approx(0.22 - 0.02)


def test_threshold_unchanged_below_min_samples():
    t = _tracker()
    for _ in range(EAR_WINDOW_MIN_SAMPLES - 1):
        t.update(0.30)
    assert t.threshold == pytest.approx(0.22 - 0.02)


def test_threshold_adapts_after_window_fills():
    t = _tracker()
    high_ear = 0.35
    for _ in range(EAR_WINDOW_MIN_SAMPLES):
        t.update(high_ear)
    assert t.threshold == pytest.approx(high_ear - 0.02)


# ── blink counting ────────────────────────────────────────────────────────────

def test_blink_counted_after_closed_eye_frames():
    t = _tracker()
    for _ in range(CLOSED_EYE_FRAMES):
        t.update(0.05)          # below threshold → eyes closed
    t.update(0.30)              # back above → blink registered
    assert t.total_blinks == 1


def test_no_blink_if_fewer_than_required_closed_frames():
    t = _tracker()
    for _ in range(CLOSED_EYE_FRAMES - 1):
        t.update(0.05)
    t.update(0.30)
    assert t.total_blinks == 0


def test_multiple_blinks_accumulated():
    t = _tracker()
    for _ in range(3):
        for _ in range(CLOSED_EYE_FRAMES):
            t.update(0.05)
        t.update(0.30)
    assert t.total_blinks == 3


# ── unit-boundary reset ───────────────────────────────────────────────────────

def test_reset_unit_returns_unit_count_and_clears_it():
    t = _tracker()
    for _ in range(CLOSED_EYE_FRAMES):
        t.update(0.05)
    t.update(0.30)              # 1 blink in this unit

    count = t.reset_unit()
    assert count == 1
    assert t.unit_blinks == 0


def test_reset_unit_does_not_affect_total_blinks():
    t = _tracker()
    for _ in range(CLOSED_EYE_FRAMES):
        t.update(0.05)
    t.update(0.30)
    t.reset_unit()
    assert t.total_blinks == 1


# ── threshold is cached (not recomputed on every read) ────────────────────────

def test_threshold_consistent_within_same_frame():
    t = _tracker()
    for _ in range(EAR_WINDOW_MIN_SAMPLES):
        t.update(0.30)
    thr1 = t.threshold
    thr2 = t.threshold
    assert thr1 == thr2
