"""Tests for analytics.py — compute_summary() and its helper functions."""

import pytest
from tests.conftest import (
    UNIT, focused_frame, make_cal, make_intent, make_thresholds,
    no_face_frame, run_pipeline,
)

from analytics import compute_summary
from models import (
    BlinkTrend, DistractionReason, FocusPattern, Scenario,
)


def _summary(frames, *, thr=None, intent=None, cal=None):
    thr    = thr    or make_thresholds()
    intent = intent or make_intent()
    cal    = cal    or make_cal()
    state, _ = run_pipeline(frames, thr, intent, cal)
    return compute_summary(state, intent, cal)


# ── empty session ─────────────────────────────────────────────────────────────

def test_empty_session_returns_zero_focus_ratio():
    s = _summary([])
    assert s.effective_focus_ratio == 0.0
    assert s.drut_history == ()


# ── effective focus ratio ─────────────────────────────────────────────────────

def test_all_focused_units_give_ratio_one():
    s = _summary([focused_frame()] * (UNIT * 4))
    assert s.effective_focus_ratio == pytest.approx(1.0)


def test_all_distracted_units_give_ratio_zero():
    s = _summary([no_face_frame()] * (UNIT * 4))
    assert s.effective_focus_ratio == pytest.approx(0.0)


# ── golden hour ───────────────────────────────────────────────────────────────

def test_golden_hour_not_none_when_focused_periods_exist():
    frames = (
        [focused_frame()]  * (UNIT * 2) +
        [no_face_frame()]  * UNIT       +
        [focused_frame()]  * (UNIT * 3)
    )
    s = _summary(frames)
    assert s.golden_hour is not None
    start_min, end_min = s.golden_hour
    assert end_min >= start_min   # single-unit golden hours have start == end


def test_golden_hour_selects_longest_streak():
    # Test _golden_hour() directly with fake timestamps so we don't depend on
    # wall-clock speed.  drut: [F, F, D, F, F, F] — streak of 3 beats streak of 2.
    from analytics import _golden_hour
    drut = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    ts   = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]   # seconds, monotonic
    gh   = _golden_hour(drut, ts, focus_thr=0.2)
    assert gh is not None
    start_min, end_min = gh
    # Longest streak starts at ts[3]=40s, ends at ts[5]=60s
    assert start_min == pytest.approx(round(40.0 / 60, 1))
    assert end_min   == pytest.approx(round(60.0 / 60, 1))
    assert end_min   >  start_min


def test_golden_hour_none_when_no_focused_units():
    s = _summary([no_face_frame()] * (UNIT * 3))
    assert s.golden_hour is None


# ── drop events ───────────────────────────────────────────────────────────────

def test_drop_event_detected_on_focused_to_distracted_transition():
    frames = (
        [focused_frame()]  * (UNIT * 2) +
        [no_face_frame()]  * (UNIT * 2)
    )
    s = _summary(frames)
    assert len(s.drop_events) == 1
    assert s.drop_events[0].trigger == DistractionReason.NO_FACE


def test_no_drop_event_when_already_distracted_from_start():
    s = _summary([no_face_frame()] * (UNIT * 3))
    assert len(s.drop_events) == 0


# ── focus pattern ─────────────────────────────────────────────────────────────

def test_front_loaded_pattern_detected():
    frames = (
        [focused_frame()]  * (UNIT * 4) +   # focused first half
        [no_face_frame()]  * (UNIT * 4)     # distracted second half
    )
    s = _summary(frames)
    assert s.focus_pattern == FocusPattern.FRONT_LOADED


def test_back_loaded_pattern_detected():
    frames = (
        [no_face_frame()]  * (UNIT * 4) +
        [focused_frame()]  * (UNIT * 4)
    )
    s = _summary(frames)
    assert s.focus_pattern == FocusPattern.BACK_LOADED


def test_consistent_pattern_when_similar_halves():
    frames = [focused_frame()] * (UNIT * 4)
    s = _summary(frames)
    assert s.focus_pattern == FocusPattern.CONSISTENT


# ── top distraction ───────────────────────────────────────────────────────────

def test_top_distraction_is_no_face_for_no_face_session():
    s = _summary([no_face_frame()] * (UNIT * 2))
    assert s.top_distraction == DistractionReason.NO_FACE


def test_top_distraction_is_none_for_focused_session():
    s = _summary([focused_frame()] * (UNIT * 2))
    assert s.top_distraction == DistractionReason.NONE


# ── blink trend ───────────────────────────────────────────────────────────────

def test_blink_trend_stable_when_insufficient_data():
    s = _summary([focused_frame()] * UNIT)   # only 1 unit — below min
    assert s.blink_trend == BlinkTrend.STABLE
