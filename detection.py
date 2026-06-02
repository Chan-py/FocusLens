"""Pipeline stages 2 and 3 — both are pure functions with no side effects.

  derive_thresholds(intent, cal) → DetectionThresholds
      Translates scenario choice + measured baselines into concrete numeric
      thresholds.  This is the single place where scenario knowledge influences
      detection behavior; tracker and session code remain scenario-agnostic.

  assess_distraction(features, thresholds, ear_dynamic) → DistractionResult
      Wang et al. sequential check: face → head pose → eyes closed → gaze.
      Depends only on its arguments; safe to call from any context.
"""

from models import (
    CalibrationData,
    DetectionThresholds,
    DistractionReason,
    DistractionResult,
    FrameFeatures,
    GazePosition,
    Scenario,
    UserIntent,
)
from config import (
    DEFAULT_PITCH_TOLERANCE,
    DEFAULT_YAW_TOLERANCE,
    DRUT_FOCUS_THRESHOLD,
    DRUT_UNIT_FRAMES,
    EAR_ALPHA,
    FATIGUE_BLINK_MULTIPLIER,
    PAPER_YAW_TOLERANCE,
)


def derive_thresholds(intent: UserIntent, cal: CalibrationData) -> DetectionThresholds:
    pitch_tol = DEFAULT_PITCH_TOLERANCE

    match intent.scenario:
        case Scenario.PAPER | Scenario.BOOK:
            yaw_tol = PAPER_YAW_TOLERANCE
            extra_gazes: set[GazePosition] = {cal.focus_gaze} if cal.focus_gaze else set()
            allowed_gazes = frozenset({GazePosition.CENTER} | extra_gazes)
        case Scenario.SCREEN:
            yaw_tol = DEFAULT_YAW_TOLERANCE
            allowed_gazes = frozenset({GazePosition.CENTER})
        case _:
            raise ValueError(f"Unhandled scenario: {intent.scenario}")

    return DetectionThresholds(
        pitch_min=cal.pitch_baseline - pitch_tol,
        pitch_max=cal.pitch_baseline + pitch_tol,
        yaw_threshold=yaw_tol,
        yaw_baseline=cal.yaw_baseline,
        ear_offset=EAR_ALPHA,
        ear_baseline=cal.ear_baseline,
        allowed_gazes=allowed_gazes,
        drut_unit_frames=DRUT_UNIT_FRAMES,
        drut_focus_threshold=DRUT_FOCUS_THRESHOLD,
        fatigue_blink_multiplier=FATIGUE_BLINK_MULTIPLIER,
    )


def assess_distraction(
    features:    FrameFeatures,
    thresholds:  DetectionThresholds,
    ear_dynamic: float,
) -> DistractionResult:
    if not features.face_detected:
        r = DistractionReason.NO_FACE
        return DistractionResult(reason=r, features=features, reasons=frozenset({r}))

    violations: set[DistractionReason] = set()

    if features.pitch is not None and features.yaw is not None:
        if (abs(features.yaw - thresholds.yaw_baseline) > thresholds.yaw_threshold
                or features.pitch > thresholds.pitch_max
                or features.pitch < thresholds.pitch_min):
            violations.add(DistractionReason.HEAD_DEVIATION)

    if features.ear is not None and features.ear < ear_dynamic:
        violations.add(DistractionReason.EYES_CLOSED)

    if (features.gaze is not None
            and features.gaze is not GazePosition.UNKNOWN
            and features.gaze not in thresholds.allowed_gazes):
        violations.add(DistractionReason.GAZE_DEVIATION)

    if not violations:
        return DistractionResult(reason=DistractionReason.NONE, features=features)

    _priority = [DistractionReason.HEAD_DEVIATION, DistractionReason.EYES_CLOSED, DistractionReason.GAZE_DEVIATION]
    primary = next(r for r in _priority if r in violations)
    return DistractionResult(reason=primary, features=features, reasons=frozenset(violations))
