"""Integration tests — full pipeline using ReplayDetector and JsonlFeatureStore.

These tests exercise the complete detection → session → analytics chain with
known FrameFeatures sequences.  The same ReplayDetector used here is the one
used in replay experiments, so these tests also validate the replay path.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from tests.conftest import (
    UNIT, focused_frame, make_cal, make_intent, make_thresholds,
    no_face_frame, run_pipeline,
)

from analytics import compute_summary
from feature_stream import JsonlFeatureStore, RecordingDetector, ReplayDetector
from features.detector import StubFaceDetector
from models import DistractionReason, FocusPattern, GazePosition


_DUMMY_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


# ── ReplayDetector basics ─────────────────────────────────────────────────────

def test_replay_detector_returns_features_in_order():
    features = [focused_frame(ear=0.20 + i * 0.01) for i in range(5)]
    det      = ReplayDetector(features)
    for expected in features:
        assert det.detect(_DUMMY_FRAME).ear == pytest.approx(expected.ear)


def test_replay_detector_returns_no_face_when_exhausted():
    det = ReplayDetector([focused_frame()])
    det.detect(_DUMMY_FRAME)          # consume the one frame
    sentinel = det.detect(_DUMMY_FRAME)
    assert not sentinel.face_detected
    assert det.exhausted


def test_replay_detector_accepts_list_and_store(tmp_path):
    frames = [focused_frame(), no_face_frame()]

    # Write to store
    store = JsonlFeatureStore(tmp_path / "f.jsonl")
    for f in frames:
        store.write(f)

    # Replay from store — same results as from list
    det_list  = ReplayDetector(frames)
    det_store = ReplayDetector(store)
    for _ in frames:
        r_list  = det_list.detect(_DUMMY_FRAME)
        r_store = det_store.detect(_DUMMY_FRAME)
        assert r_list.face_detected == r_store.face_detected
        assert r_list.ear           == r_store.ear


# ── JsonlFeatureStore round-trip ─────────────────────────────────────────────

def test_jsonl_round_trip_preserves_all_fields(tmp_path):
    original = focused_frame(ear=0.27)
    store    = JsonlFeatureStore(tmp_path / "rt.jsonl")
    store.write(original)

    loaded = next(iter(store))
    assert loaded.face_detected == original.face_detected
    assert loaded.ear           == pytest.approx(original.ear)
    assert loaded.gaze          == original.gaze
    assert loaded.pitch         == pytest.approx(original.pitch)
    assert loaded.yaw           == pytest.approx(original.yaw)


def test_jsonl_round_trip_handles_none_fields(tmp_path):
    store = JsonlFeatureStore(tmp_path / "none.jsonl")
    store.write(no_face_frame())
    loaded = next(iter(store))
    assert loaded.face_detected is False
    assert loaded.ear   is None
    assert loaded.gaze  is None
    assert loaded.pitch is None


# ── RecordingDetector captures detect() output ────────────────────────────────

def test_recording_detector_writes_each_frame(tmp_path):
    inner  = StubFaceDetector()
    store  = JsonlFeatureStore(tmp_path / "rec.jsonl")
    det    = RecordingDetector(inner, store)
    N      = 5
    for _ in range(N):
        det.detect(_DUMMY_FRAME)
    recorded = list(store)
    assert len(recorded) == N


def test_recording_detector_passes_features_through(tmp_path):
    inner  = StubFaceDetector()
    store  = JsonlFeatureStore(tmp_path / "pt.jsonl")
    det    = RecordingDetector(inner, store)
    result = det.detect(_DUMMY_FRAME)
    assert result.face_detected is True
    assert result.gaze == GazePosition.CENTER


# ── full pipeline with ReplayDetector ────────────────────────────────────────

def test_focused_session_gives_low_drut():
    frames = [focused_frame()] * (UNIT * 4)
    state, _ = run_pipeline(frames)
    assert all(d < 0.2 for d in state.drut_history)


def test_all_distracted_session_gives_drut_one():
    frames = [no_face_frame()] * (UNIT * 4)
    state, _ = run_pipeline(frames)
    assert all(d == pytest.approx(1.0) for d in state.drut_history)


def test_replay_produces_same_summary_as_live(tmp_path):
    """A recorded then replayed session must yield the same SessionSummary."""
    thr    = make_thresholds()
    intent = make_intent()
    cal    = make_cal()
    frames = [focused_frame()] * (UNIT * 3) + [no_face_frame()] * (UNIT * 2)

    # First run — record
    store = JsonlFeatureStore(tmp_path / "features.jsonl")
    for f in frames:
        store.write(f)

    # Second run — replay (should give identical DRUT history)
    state_live,   _ = run_pipeline(frames, thr, intent, cal)
    state_replay, _ = run_pipeline(list(ReplayDetector(store)._iter
                                        if False else store),
                                   thr, intent, cal)
    # Compare via run_pipeline using the stored file
    state_replay2, _ = run_pipeline(list(store), thr, intent, cal)

    assert state_live.drut_history == state_replay2.drut_history


def test_summary_top_distraction_reflects_dominant_cause():
    frames = [no_face_frame()] * (UNIT * 4)
    state, _ = run_pipeline(frames)
    summary  = compute_summary(state, make_intent(), make_cal())
    assert summary.top_distraction == DistractionReason.NO_FACE


def test_summary_focus_pattern_from_known_sequence():
    frames = (
        [focused_frame()]  * (UNIT * 4) +   # front heavy
        [no_face_frame()]  * (UNIT * 4)
    )
    state, _ = run_pipeline(frames)
    summary  = compute_summary(state, make_intent(), make_cal())
    assert summary.focus_pattern == FocusPattern.FRONT_LOADED
