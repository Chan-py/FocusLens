"""Session analytics — pure function: SessionState → SessionSummary.

No side effects, no I/O.  Threshold constants are imported from config
so analytics behaviour stays consistent with detection behaviour.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

from config import DRUT_FOCUS_THRESHOLD, FATIGUE_BLINK_MULTIPLIER
from models import (
    BlinkTrend,
    CalibrationData,
    DistractionReason,
    DropEvent,
    FocusPattern,
    SessionSummary,
    UserIntent,
)
from session.session_state import SessionState


def compute_summary(
    state:  SessionState,
    intent: UserIntent,
    cal:    CalibrationData,
) -> SessionSummary:
    """Derive analytics summary from accumulated session data."""
    drut    = state.drut_history
    ts      = state.timestamps
    blinks  = state.blink_rate_per_unit
    reasons = state.distraction_reasons

    if not drut:
        return _empty_summary(state, intent, cal)

    session_min = (time.time() - state.session_start) / 60
    focus_thr   = DRUT_FOCUS_THRESHOLD

    # ── effective focus ratio ─────────────────────────────────────────────────
    focused_units         = [d for d in drut if d < focus_thr]
    effective_focus_ratio = len(focused_units) / len(drut)

    # ── golden hour: longest consecutive focused streak ───────────────────────
    golden_hour = _golden_hour(drut, ts, focus_thr)

    # ── drop events: focused → distracted transitions ────────────────────────
    drop_events = tuple(
        DropEvent(
            time_min=round(ts[i] / 60, 1),
            trigger=reasons[i] if i < len(reasons) else DistractionReason.NONE,
        )
        for i in range(1, len(drut))
        if drut[i - 1] < focus_thr and drut[i] >= focus_thr
    )

    # ── fatigue onset: blink rate ≥ 1.5× baseline ────────────────────────────
    fatigue_onset_min = _fatigue_onset(blinks, ts, state.blink_baseline)

    # ── focus pattern ─────────────────────────────────────────────────────────
    focus_pattern = _focus_pattern(drut, focus_thr)

    # ── blink trend ───────────────────────────────────────────────────────────
    blink_trend = _blink_trend(blinks)

    # ── dominant distraction cause ────────────────────────────────────────────
    top_distraction = (
        Counter(reasons).most_common(1)[0][0] if reasons else DistractionReason.NONE
    )

    return SessionSummary(
        scenario=intent.scenario,
        session_duration_min=round(session_min, 1),
        effective_focus_ratio=round(effective_focus_ratio, 2),
        total_blinks=state.total_blinks,
        golden_hour=golden_hour,
        drop_events=drop_events,
        fatigue_onset_min=fatigue_onset_min,
        focus_pattern=focus_pattern,
        blink_trend=blink_trend,
        top_distraction=top_distraction,
        blink_baseline_per_unit=(
            round(state.blink_baseline, 1) if state.blink_baseline else None
        ),
        drut_history=tuple(round(d, 3) for d in drut),
        calibration_summary={
            "scenario":       intent.scenario.value,
            "pitch_baseline": round(cal.pitch_baseline, 2),
            "yaw_baseline":   round(cal.yaw_baseline, 2),
            "ear_baseline":   round(cal.ear_baseline, 3),
            "focus_gaze":     cal.focus_gaze.value if cal.focus_gaze else None,
        },
    )


# ── helpers (pure) ────────────────────────────────────────────────────────────

def _golden_hour(
    drut: list[float], ts: list[float], focus_thr: float
) -> tuple[float, float] | None:
    max_streak = cur_streak = 0
    cur_start = golden_start = golden_end = 0
    for i, d in enumerate(drut):
        if d < focus_thr:
            if cur_streak == 0:
                cur_start = i
            cur_streak += 1
            if cur_streak > max_streak:
                max_streak   = cur_streak
                golden_start = cur_start
                golden_end   = i
        else:
            cur_streak = 0

    if max_streak == 0 or golden_start >= len(ts) or golden_end >= len(ts):
        return None
    return (round(ts[golden_start] / 60, 1), round(ts[golden_end] / 60, 1))


def _fatigue_onset(
    blinks: list[int], ts: list[float], baseline: float | None
) -> float | None:
    if not baseline or not blinks:
        return None
    threshold = baseline * FATIGUE_BLINK_MULTIPLIER
    for i, br in enumerate(blinks):
        if br >= threshold and i < len(ts):
            return round(ts[i] / 60, 1)
    return None


def _focus_pattern(drut: list[float], focus_thr: float) -> FocusPattern:
    mid = len(drut) // 2
    if mid == 0:
        return FocusPattern.INSUFFICIENT_DATA
    first  = float(np.mean([d < focus_thr for d in drut[:mid]]))
    second = float(np.mean([d < focus_thr for d in drut[mid:]]))
    if first > second + 0.2:
        return FocusPattern.FRONT_LOADED
    if second > first + 0.2:
        return FocusPattern.BACK_LOADED
    return FocusPattern.CONSISTENT


def _blink_trend(blinks: list[int]) -> BlinkTrend:
    if len(blinks) < 4:
        return BlinkTrend.STABLE
    half      = len(blinks) // 2
    first_avg = float(np.mean(blinks[:half]))
    last_avg  = float(np.mean(blinks[half:]))
    if first_avg == 0:
        return BlinkTrend.STABLE
    if last_avg > first_avg * 1.3:
        return BlinkTrend.INCREASING
    if last_avg < first_avg * 0.7:
        return BlinkTrend.DECREASING
    return BlinkTrend.STABLE


def _empty_summary(
    state: SessionState, intent: UserIntent, cal: CalibrationData
) -> SessionSummary:
    return SessionSummary(
        scenario=intent.scenario,
        session_duration_min=round((time.time() - state.session_start) / 60, 1),
        effective_focus_ratio=0.0,
        total_blinks=state.total_blinks,
        golden_hour=None,
        drop_events=(),
        fatigue_onset_min=None,
        focus_pattern=FocusPattern.INSUFFICIENT_DATA,
        blink_trend=BlinkTrend.STABLE,
        top_distraction=DistractionReason.NONE,
        blink_baseline_per_unit=None,
        drut_history=(),
        calibration_summary={
            "scenario":       intent.scenario.value,
            "pitch_baseline": round(cal.pitch_baseline, 2),
            "yaw_baseline":   round(cal.yaw_baseline, 2),
            "ear_baseline":   round(cal.ear_baseline, 3),
            "focus_gaze":     cal.focus_gaze.value if cal.focus_gaze else None,
        },
    )
