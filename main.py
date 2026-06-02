"""FocusLens — main entry point.

Orchestrates the pipeline stages in order:
  1. Calibration  → (UserIntent, CalibrationData)
  2. Thresholds   → DetectionThresholds        [derive_thresholds]
  3. Tracking     → SessionState               [main loop]
  4. Analytics    → SessionSummary             [compute_summary]
  5. Report       → str                        [LLMReporter]
"""

import csv
import dataclasses
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
from dotenv import load_dotenv
load_dotenv()

from analytics import compute_summary
from calibration import CalibrationAborted, Calibrator
from detection import assess_distraction, derive_thresholds
from feature_stream import (
    JsonlFeatureStore, RecordingDetector, ReplayDetector,
    load_session_meta, save_session_meta,
)
from features import FaceDetector, MediaPipeFaceDetector, StubFaceDetector
from models import CalibrationData, DistractionResult, Scenario, UserIntent
from report import LLMReporter, LocalSLMReporter, OpenAIReporter, StubLLMReporter, summary_to_dict
from session import EarTracker, SessionState
from ui import CaptureSource, StubCapture, draw_overlay


# ── session mode — pick exactly one ──────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class LiveMode:
    """Standard live session — camera input, no recording."""

@dataclasses.dataclass(frozen=True)
class RecordMode:
    """Live session that records FrameFeatures + calibration to disk."""
    path: str

@dataclasses.dataclass(frozen=True)
class ReplayMode:
    """Replay a previously recorded session from disk; no camera needed."""
    path: str

SessionMode = LiveMode | RecordMode | ReplayMode


# ── configuration ─────────────────────────────────────────────────────────────

DEV_MODE = False           # True → skip camera / MediaPipe; run with stubs
CAMERA_INDEX   = 0
LLM_MODE       = "slm"                     # "openai" | "slm" | "stub"
OPENAI_KEY     = os.getenv("OPENAI_API_KEY", "")
SLM_MODEL_PATH: str | None = "models/qwen2.5-3b-instruct-q4_k_m.gguf"

SESSION: SessionMode = LiveMode()
# SESSION: SessionMode = RecordMode("runs/test_01")
# SESSION: SessionMode = ReplayMode("runs/test_01")

LOG_INTERVAL = 30   # terminal log row every N frames


# ── factories ─────────────────────────────────────────────────────────────────

def _make_reporter() -> LLMReporter:
    if LLM_MODE == "stub" or DEV_MODE:
        return StubLLMReporter()
    if LLM_MODE == "openai" and OPENAI_KEY:
        return OpenAIReporter(api_key=OPENAI_KEY)
    if LLM_MODE == "slm" and SLM_MODEL_PATH:
        return LocalSLMReporter(model_path=SLM_MODEL_PATH)
    print("[안내] LLM 설정 없음 → StubLLMReporter로 대체합니다.")
    print("  OPENAI_API_KEY 환경변수 설정 또는 SLM_MODEL_PATH 지정으로 실제 리포트 생성 가능.")
    return StubLLMReporter()


def _make_capture_and_detector(
) -> tuple[CaptureSource, FaceDetector, tuple[UserIntent, CalibrationData] | None]:
    """Return (cap, detector, pre_calibration).

    pre_calibration is non-None when calibration should be skipped
    (DEV_MODE or ReplayMode — both have no live camera input).
    """
    match SESSION:
        case ReplayMode(path=p):
            run_dir     = Path(p)
            intent, cal = load_session_meta(run_dir)
            store       = JsonlFeatureStore(run_dir / "features.jsonl")
            print(f"[replay] {p} — {len(store)} frames")
            return StubCapture(), ReplayDetector(store), (intent, cal)

        case LiveMode() | RecordMode():
            if DEV_MODE:
                return (StubCapture(), StubFaceDetector(),
                        (UserIntent(scenario=Scenario.SCREEN), CalibrationData.defaults()))

            cap = cv2.VideoCapture(CAMERA_INDEX)
            if not cap.isOpened():
                raise RuntimeError(
                    f"카메라 {CAMERA_INDEX}를 열 수 없습니다. "
                    "CAMERA_INDEX를 변경하거나 DEV_MODE = True로 실행하세요."
                )
            detector: FaceDetector = MediaPipeFaceDetector()
            match SESSION:
                case RecordMode(path=p):
                    run_dir = Path(p)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    detector = RecordingDetector(
                        detector, JsonlFeatureStore(run_dir / "features.jsonl")
                    )
                    print(f"[record] FrameFeatures → {run_dir / 'features.jsonl'}")
            return cap, detector, None


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        cap, detector, pre_cal = _make_capture_and_detector()
    except RuntimeError as e:
        print(e)
        return

    _print_banner()

    # ── Step 1: calibration ───────────────────────────────────────────────────
    if pre_cal is not None:
        intent, cal = pre_cal
    else:
        try:
            intent, cal = Calibrator(detector).run(cap)
        except CalibrationAborted:
            print("\n캘리브레이션이 취소됐습니다.")
            detector.close()
            cap.release()
            cv2.destroyAllWindows()
            return
        match SESSION:
            case RecordMode(path=p):
                save_session_meta(Path(p), intent, cal)

    # ── Step 2: thresholds + session objects ──────────────────────────────────
    thresholds = derive_thresholds(intent, cal)
    ear        = EarTracker(thresholds)
    state      = SessionState(thresholds)
    debug_log: list[dict] = []

    _print_session_header(intent, thresholds)

    # ── Step 3: tracking loop ─────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        features = detector.detect(frame)
        if isinstance(detector, ReplayDetector) and detector.exhausted:
            break

        result = assess_distraction(features, thresholds, ear.threshold)
        if features.ear is not None:
            ear.update(features.ear)

        unit_ready = state.update(result, ear.total_blinks)
        if unit_ready:
            state.close_unit(ear.reset_unit())

        if state.total_frames % LOG_INTERVAL == 0:
            _log_row(state, ear, result, debug_log)

        match SESSION:
            case ReplayMode():
                print(f"  재현 중... {state.total_frames}프레임", end="\r")
            case _:
                frame = draw_overlay(
                    frame, result,
                    total_blinks=ear.total_blinks,
                    ear_threshold=ear.threshold,
                    drut_history=state.drut_history,
                    scenario=intent.scenario,
                    calibrated=True,
                )
                cv2.imshow("FocusLens", frame)
                if (cv2.waitKey(1) & 0xFF == ord("q")
                        or cv2.getWindowProperty("FocusLens", cv2.WND_PROP_VISIBLE) < 1):
                    break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

    # ── Step 4: analytics ─────────────────────────────────────────────────────
    print("\n세션 종료. 분석 중...")
    summary = compute_summary(state, intent, cal)

    match SESSION:
        case ReplayMode(path=p) | RecordMode(path=p):
            out_dir = Path(p)
        case LiveMode():
            out_dir = Path(".")

    # ── Step 5: report ────────────────────────────────────────────────────────
    reporter = _make_reporter()
    report   = reporter.report(summary)

    report_fname = _report_filename()
    _save_outputs(out_dir, debug_log, summary, report, report_fname)
    _save_exp_result(reporter, report_fname)
    _print_session_summary(debug_log, summary, report, out_dir, report_fname)


# ── helpers ───────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    print("\n" + "=" * 50)
    print("FocusLens — Personalized Concentration Tracker")
    if DEV_MODE:
        print("(DEV MODE — stubs active, 'q' or Ctrl-C to stop)")
    match SESSION:
        case RecordMode(path=p): print(f"(RECORD → {p})")
        case ReplayMode(path=p): print(f"(REPLAY ← {p})")
    print("=" * 50)


def _print_session_header(intent, thresholds) -> None:
    print(f"\n집중 추적 시작! 'q'를 누르면 종료됩니다.")
    print(f"시나리오: {intent.scenario.label}")
    print(f"Pitch 범위: {thresholds.pitch_min:.1f}° ~ {thresholds.pitch_max:.1f}°")
    print(f"Yaw threshold: ±{thresholds.yaw_threshold:.1f}°")
    print("-" * 65)
    print(f"{'time':>6} | {'EAR':>6} | {'thr':>6} | {'gaze':>10} | "
          f"{'yaw':>6} | {'pitch':>6} | reason")
    print("-" * 65)


def _report_filename() -> str:
    if LLM_MODE == "slm" and SLM_MODEL_PATH:
        stem = Path(SLM_MODEL_PATH).stem
        return f"slm_{stem}_focus_report.txt"
    if LLM_MODE == "openai":
        return "openai_focus_report.txt"
    return "stub_focus_report.txt"


def _save_outputs(out_dir: Path, debug_log: list[dict], summary, report: str, report_fname: str) -> None:
    (out_dir / "debug_log.json").write_text(
        json.dumps(debug_log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "session_summary.json").write_text(
        json.dumps(summary_to_dict(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / report_fname).write_text(report, encoding="utf-8")


def _save_exp_result(reporter: LLMReporter, report_fname: str) -> None:
    metrics  = getattr(reporter, "metrics", {})
    csv_path = Path("exp_result.csv")
    row = {
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "llm_mode":          LLM_MODE,
        "model_path":        SLM_MODEL_PATH or LLM_MODE,
        "elapsed_sec":       metrics.get("elapsed_sec", ""),
        "completion_tokens": metrics.get("completion_tokens", ""),
        "tokens_per_sec":    metrics.get("tokens_per_sec", ""),
        "memory_mb":         metrics.get("memory_mb", ""),
        "report_file":       report_fname,
    }
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _print_session_summary(debug_log, summary, report: str, out_dir: Path, report_fname: str) -> None:
    print("\n=== Session Summary ===")
    print(json.dumps(summary_to_dict(summary), indent=2, ensure_ascii=False))

    if debug_log:
        reasons = [r["reason"] for r in debug_log]
        cnt     = Counter(reasons)
        total   = len(reasons)
        print("\n=== Distraction 원인 분포 ===")
        for k, v in cnt.most_common():
            print(f"  {k:20s}: {v:4d} ({v / total * 100:.1f}%)")

    print("\n" + "=" * 50)
    print(report)
    print("=" * 50)
    print(f"\n리포트가 {out_dir / report_fname}에 저장됐습니다.")


def _log_row(
    state: SessionState,
    ear:   EarTracker,
    result: DistractionResult,
    debug_log: list[dict],
) -> None:
    f       = result.features
    elapsed = round(time.time() - state.session_start, 1)
    ear_s   = f"{f.ear:.3f}"   if f.ear   is not None else "  N/A"
    gaze_s  = f.gaze.value     if f.gaze  is not None else "  N/A"
    yaw_s   = f"{f.yaw:.1f}"  if f.yaw   is not None else "  N/A"
    pitch_s = f"{f.pitch:.1f}" if f.pitch is not None else "  N/A"
    reason  = result.reason.value

    print(f"{elapsed:>6} | {ear_s:>6} | {ear.threshold:.3f} | {gaze_s:>10} | "
          f"{yaw_s:>6} | {pitch_s:>6} | {reason}")

    log_entry = dataclasses.asdict(f)
    log_entry["gaze"]      = gaze_s
    log_entry["time_s"]    = elapsed
    log_entry["ear_thr"]   = ear.threshold
    log_entry["reason"]    = reason
    log_entry["distracted"] = result.distracted
    debug_log.append(log_entry)


if __name__ == "__main__":
    main()
