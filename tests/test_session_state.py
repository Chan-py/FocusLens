"""Tests for session/session_state.py — update() / close_unit() interface."""

import pytest
from tests.conftest import UNIT, focused_frame, make_thresholds, no_face_frame, run_pipeline

from detection import assess_distraction
from models import DistractionReason
from session import EarTracker, SessionState


def _setup():
    thr   = make_thresholds(unit_frames=UNIT)
    ear   = EarTracker(thr)
    state = SessionState(thr)
    return state, ear, thr


# ── update() signals unit boundary ───────────────────────────────────────────

def test_update_returns_false_before_unit_boundary():
    state, ear, thr = _setup()
    result = assess_distraction(focused_frame(), thr, ear.threshold)
    ready  = state.update(result, ear.total_blinks)
    assert ready is False


def test_update_returns_true_at_unit_boundary():
    state, ear, thr = _setup()
    result = assess_distraction(focused_frame(), thr, ear.threshold)
    ready  = False
    for _ in range(UNIT):
        ready = state.update(result, ear.total_blinks)
    assert ready is True


def test_close_unit_resets_frame_counter():
    state, ear, thr = _setup()
    result = assess_distraction(focused_frame(), thr, ear.threshold)
    for _ in range(UNIT):
        state.update(result, ear.total_blinks)
    state.close_unit(ear.reset_unit())
    # one more frame should NOT trigger another boundary yet
    assert state.update(result, ear.total_blinks) is False


# ── DRUT calculation ──────────────────────────────────────────────────────────

def test_all_distracted_frames_give_drut_one():
    state, ear, thr = _setup()
    result = assess_distraction(no_face_frame(), thr, ear.threshold)
    for _ in range(UNIT):
        state.update(result, ear.total_blinks)
    drut = state.close_unit(ear.reset_unit())
    assert drut == pytest.approx(1.0)


def test_all_focused_frames_give_drut_zero():
    state, ear, thr = _setup()
    result = assess_distraction(focused_frame(), thr, ear.threshold)
    for _ in range(UNIT):
        state.update(result, ear.total_blinks)
    drut = state.close_unit(ear.reset_unit())
    assert drut == pytest.approx(0.0)


def test_half_distracted_gives_drut_half():
    state, ear, thr = _setup()
    r_dist    = assess_distraction(no_face_frame(),    thr, ear.threshold)
    r_focused = assess_distraction(focused_frame(), thr, ear.threshold)
    for i in range(UNIT):
        state.update(r_dist if i < UNIT // 2 else r_focused, ear.total_blinks)
    drut = state.close_unit(ear.reset_unit())
    assert drut == pytest.approx(0.5)


# ── distraction reason tracking ───────────────────────────────────────────────

def test_dominant_reason_recorded_per_unit():
    state, ear, thr = _setup()
    r_no_face = assess_distraction(no_face_frame(), thr, ear.threshold)
    r_focused = assess_distraction(focused_frame(), thr, ear.threshold)
    for i in range(UNIT):
        # 7 no_face vs 3 focused → NO_FACE is dominant
        state.update(r_no_face if i < 7 else r_focused, ear.total_blinks)
    state.close_unit(ear.reset_unit())
    assert state.distraction_reasons[-1] == DistractionReason.NO_FACE


def test_no_distraction_reason_is_none():
    state, ear, thr = _setup()
    result = assess_distraction(focused_frame(), thr, ear.threshold)
    for _ in range(UNIT):
        state.update(result, ear.total_blinks)
    state.close_unit(ear.reset_unit())
    assert state.distraction_reasons[-1] == DistractionReason.NONE


# ── blink baseline set after 3 units ─────────────────────────────────────────

def test_blink_baseline_set_after_three_units():
    state, ear = run_pipeline([focused_frame()] * (UNIT * 3))
    assert state.blink_baseline is not None


def test_blink_baseline_none_before_three_units():
    state, ear = run_pipeline([focused_frame()] * (UNIT * 2))
    assert state.blink_baseline is None


# ── total_frames counter is monotonic ────────────────────────────────────────

def test_total_frames_increments_across_units():
    state, ear, thr = _setup()
    result = assess_distraction(focused_frame(), thr, ear.threshold)
    for i in range(UNIT * 2):
        state.update(result, ear.total_blinks)
        if state._frame_count == 0:   # just reset by close_unit
            state.close_unit(0)
    assert state.total_frames == UNIT * 2
