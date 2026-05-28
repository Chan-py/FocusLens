"""FrameFeatures stream recording and replay.

RecordingDetector  — wraps any FaceDetector; writes each FrameFeatures to a store
                     while passing it through unchanged.
ReplayDetector     — reads FrameFeatures from any Iterable[FrameFeatures]; the
                     camera frame argument is ignored.

Both implement FaceDetector, so they are drop-in replacements in the pipeline.
The same ReplayDetector works for file-based replay experiments and in-memory
test fixtures:

    # Experiment: replay from file
    detector = ReplayDetector(JsonlFeatureStore("run_01/features.jsonl"))

    # Unit test: replay from an in-memory list
    detector = ReplayDetector([known_feature_a, known_feature_b])

File format (JSONL, one object per line):
  {"face_detected": true, "ear": 0.28, "gaze": "center", "pitch": -4.1, ...}
  {"face_detected": false, "ear": null, "gaze": null, ...}
  ...

Session metadata (UserIntent + CalibrationData) is stored separately as
run_XX/meta.json so that replay experiments can load the original calibration.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from features.detector import FaceDetector
from models import CalibrationData, FrameFeatures, GazePosition, Scenario, UserIntent


# ── serialisation helpers ─────────────────────────────────────────────────────

def _features_to_dict(f: FrameFeatures) -> dict:
    return {
        "face_detected": f.face_detected,
        "ear":           f.ear,
        "gaze":          f.gaze.value if f.gaze is not None else None,
        "pitch":         f.pitch,
        "yaw":           f.yaw,
        "roll":          f.roll,
    }


def _dict_to_features(d: dict) -> FrameFeatures:
    gaze_raw = d.get("gaze")
    return FrameFeatures(
        face_detected=d["face_detected"],
        ear=d["ear"],
        gaze=GazePosition(gaze_raw) if gaze_raw is not None else None,
        pitch=d["pitch"],
        yaw=d["yaw"],
        roll=d["roll"],
    )


# ── storage ───────────────────────────────────────────────────────────────────

class JsonlFeatureStore:
    """Append-write and sequential-read for a FrameFeatures JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def write(self, features: FrameFeatures) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_features_to_dict(features), ensure_ascii=False) + "\n")

    def __iter__(self) -> Iterator[FrameFeatures]:
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield _dict_to_features(json.loads(line))

    def __len__(self) -> int:
        """Count frames (reads whole file — use sparingly)."""
        return sum(1 for _ in self)


# ── session metadata ──────────────────────────────────────────────────────────

def save_session_meta(
    directory: Path,
    intent:    UserIntent,
    cal:       CalibrationData,
) -> None:
    """Save UserIntent + CalibrationData alongside a recording."""
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        "scenario":       intent.scenario.value,
        "pitch_baseline": cal.pitch_baseline,
        "yaw_baseline":   cal.yaw_baseline,
        "ear_baseline":   cal.ear_baseline,
        "focus_gaze":     cal.focus_gaze.value if cal.focus_gaze else None,
    }
    (directory / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_session_meta(directory: Path) -> tuple[UserIntent, CalibrationData]:
    """Load UserIntent + CalibrationData from a recording directory."""
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    intent = UserIntent(scenario=Scenario(meta["scenario"]))
    fg_raw = meta.get("focus_gaze")
    cal = CalibrationData(
        pitch_baseline=meta["pitch_baseline"],
        yaw_baseline=meta["yaw_baseline"],
        ear_baseline=meta["ear_baseline"],
        focus_gaze=GazePosition(fg_raw) if fg_raw else None,
    )
    return intent, cal


# ── detector wrappers ─────────────────────────────────────────────────────────

_NO_FACE = FrameFeatures(
    face_detected=False, ear=None, gaze=None,
    pitch=None, yaw=None, roll=None,
)


class RecordingDetector(FaceDetector):
    """Transparent wrapper: detects normally and records each FrameFeatures."""

    def __init__(self, inner: FaceDetector, store: JsonlFeatureStore) -> None:
        self._inner = inner
        self._store = store

    def detect(self, frame: np.ndarray) -> FrameFeatures:
        features = self._inner.detect(frame)
        self._store.write(features)
        return features

    def close(self) -> None:
        self._inner.close()


class ReplayDetector(FaceDetector):
    """Replays FrameFeatures from any Iterable; ignores the camera frame.

    When the source is exhausted, returns a no-face sentinel so the session
    can be closed cleanly (the main loop will either reach its frame limit
    or the user will press 'q').
    """

    def __init__(self, source: Iterable[FrameFeatures]) -> None:
        self._iter:     Iterator[FrameFeatures] = iter(source)
        self.exhausted: bool                    = False

    def detect(self, frame: np.ndarray) -> FrameFeatures:
        if self.exhausted:
            return _NO_FACE
        try:
            return next(self._iter)
        except StopIteration:
            self.exhausted = True
            return _NO_FACE

    def close(self) -> None:
        pass
