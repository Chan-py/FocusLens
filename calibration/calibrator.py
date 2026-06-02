"""Calibration phase: scenario selection + few-shot posture measurement.

Produces (UserIntent, CalibrationData) — the two frozen inputs that
drive the rest of the pipeline.  Nothing downstream needs to inspect
scenario values again; derive_thresholds() translates them into
DetectionThresholds once.

compute_calibration() is extracted as a pure function so that the baseline
computation logic can be tested independently of the camera I/O.
UI rendering (_render_*) is deliberately separated from data collection
so the collection logic is readable without cv2 knowledge.
"""

from __future__ import annotations

import json
import time

import cv2
import numpy as np

from features.detector import FaceDetector
from models import CalibrationData, FrameFeatures, GazePosition, Scenario, UserIntent
from ui.capture import CaptureSource

_WIN = "FocusLens Calibration"


class CalibrationAborted(Exception):
    """Raised when the calibration window is closed by the user."""


def _window_closed() -> bool:
    return cv2.getWindowProperty(_WIN, cv2.WND_PROP_VISIBLE) < 1


# ── pure computation (testable without camera) ────────────────────────────────

def compute_calibration(
    forward_frames: list[FrameFeatures],
    blink_frames:   list[FrameFeatures],
    study_frames:   list[FrameFeatures] | None = None,
) -> CalibrationData:
    """Derive CalibrationData from collected FrameFeatures.

    forward_frames — frames collected while looking straight ahead
    blink_frames   — frames collected while blinking naturally (for EAR baseline)
    study_frames   — frames collected in study posture (PAPER/BOOK only);
                     when provided, overrides pitch_baseline
    """
    pitches = [f.pitch for f in forward_frames if f.pitch is not None]
    yaws    = [f.yaw   for f in forward_frames if f.yaw   is not None]
    ears    = [f.ear   for f in (blink_frames or forward_frames)
               if f.ear is not None and f.ear > 0]

    pitch_baseline = float(np.mean(pitches)) if pitches else 0.0
    yaw_baseline   = float(np.mean(yaws))    if yaws    else 0.0
    ear_baseline   = float(np.mean(ears))    if ears    else 0.22

    if study_frames:
        study_pitches = [f.pitch for f in study_frames if f.pitch is not None]
        if study_pitches:
            pitch_baseline = float(np.mean(study_pitches))
        # EAR is foreshortened when head is tilted — use study-posture EAR as baseline
        study_ears = [f.ear for f in study_frames if f.ear is not None and f.ear > 0]
        if study_ears:
            ear_baseline = float(np.mean(study_ears))

    # focus_gaze: not yet collected; see _doc/todo_unimplemented.md §1
    focus_gaze: GazePosition | None = None

    return CalibrationData(
        pitch_baseline=pitch_baseline,
        yaw_baseline=yaw_baseline,
        ear_baseline=ear_baseline,
        focus_gaze=focus_gaze,
    )


# ── interactive calibration (camera I/O) ─────────────────────────────────────

class Calibrator:
    def __init__(self, detector: FaceDetector) -> None:
        self._detector = detector

    def run(self, cap: CaptureSource) -> tuple[UserIntent, CalibrationData]:
        """Run calibration. Raises CalibrationAborted if the window is closed."""
        scenario = self._select_scenario(cap)
        intent   = UserIntent(scenario=scenario)

        forward_frames = self._collect(cap, 5, "Look straight forward",
                                       "Keep your natural posture")
        blink_frames   = self._collect(cap, 5, "Blink naturally", "Relax your eyes")
        study_frames: list[FrameFeatures] | None = None
        if scenario in (Scenario.PAPER, Scenario.BOOK):
            study_frames = self._collect(
                cap, 5,
                f"Take your {scenario.label} posture",
                "This becomes your FOCUS baseline",
            )

        cal = compute_calibration(forward_frames, blink_frames, study_frames)

        self._show_complete(cap)

        print("\n=== Calibration 완료 ===")
        print(json.dumps({
            "scenario":       intent.scenario.value,
            "pitch_baseline": round(cal.pitch_baseline, 2),
            "yaw_baseline":   round(cal.yaw_baseline, 2),
            "ear_baseline":   round(cal.ear_baseline, 3),
            "focus_gaze":     cal.focus_gaze.value if cal.focus_gaze else None,
        }, indent=2, ensure_ascii=False))

        return intent, cal

    def _select_scenario(self, cap: CaptureSource) -> Scenario:
        print("\n" + "=" * 50)
        print("FocusLens Calibration")
        print("=" * 50)
        print("지금 무엇을 할 건가요?")
        print("  [1] 화면 작업 (Screen Work)")
        print("  [2] 종이 공부 (Paper Study)")
        print("  [3] 독서     (Book Reading)")
        print("=" * 50)

        key_map = {"1": Scenario.SCREEN, "2": Scenario.PAPER, "3": Scenario.BOOK}
        while True:
            ret, frame = cap.read()
            if not ret:
                raise CalibrationAborted()
            self._render_scenario_prompt(frame)
            cv2.imshow(_WIN, frame)
            key = cv2.waitKey(1) & 0xFF
            if _window_closed():
                raise CalibrationAborted()
            ch = chr(key) if key < 128 else ""
            if ch in key_map:
                print(f"\n선택: {key_map[ch].label}")
                return key_map[ch]

    def _collect(
        self,
        cap: CaptureSource,
        duration_sec: float,
        message: str,
        sub: str = "",
    ) -> list[FrameFeatures]:
        collected: list[FrameFeatures] = []
        start = time.time()

        while time.time() - start < duration_sec:
            ret, frame = cap.read()
            if not ret:
                raise CalibrationAborted()
            features = self._detector.detect(frame)
            if features.face_detected:
                collected.append(features)
            remaining = int(duration_sec - (time.time() - start)) + 1
            progress  = (time.time() - start) / duration_sec
            self._render_collecting(frame, message, sub, remaining, progress)
            cv2.imshow(_WIN, frame)
            cv2.waitKey(1)
            if _window_closed():
                raise CalibrationAborted()

        return collected

    @staticmethod
    def _render_scenario_prompt(frame: np.ndarray) -> None:
        cv2.putText(frame, "Select scenario:", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, "[1] Screen  [2] Paper  [3] Book", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    @staticmethod
    def _render_collecting(
        frame: np.ndarray,
        message: str,
        sub: str,
        remaining: int,
        progress: float,
    ) -> None:
        h, w = frame.shape[:2]
        cv2.putText(frame, message, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if sub:
            cv2.putText(frame, sub, (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, f"Collecting... {remaining}s", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        bar_w = int(w * min(progress, 1.0))
        cv2.rectangle(frame, (0, h - 10), (bar_w, h), (0, 255, 0), -1)

    @staticmethod
    def _show_complete(cap: CaptureSource) -> None:
        start = time.time()
        while time.time() - start < 2:
            ret, frame = cap.read()
            if not ret or _window_closed():
                break
            cv2.putText(frame, "Calibration Complete!", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.putText(frame, "Starting session...", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.imshow(_WIN, frame)
            cv2.waitKey(1)
        cv2.destroyWindow(_WIN)
