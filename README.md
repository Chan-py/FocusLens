# FocusLens — Proto 0

웹캠으로 실시간 집중도를 측정하고, 세션 종료 후 LLM이 개인화된 집중 리포트를 생성하는 시스템.

프레임마다 얼굴 부재·고개 이탈·눈 감김·시선 이탈을 판정하고, 이를 분당 점수(DRUT)로 누적. 세션이 끝나면 구조화된 요약 데이터를 언어 모델에 전달해 리포트를 생성.

---

## 파이프라인

```
┌─────────────┐   UserIntent           ┌──────────────────┐
│ Calibration │ ── CalibrationData ──▶│ derive_thresholds│
└─────────────┘                        └────────┬─────────┘
                                                │ DetectionThresholds
                                                ▼
┌─────────────┐   FrameFeatures       ┌──────────────────┐   DistractionResult
│ FaceDetector│ ────────────────────▶│assess_distraction│ ──────────────────┐
└─────────────┘                       └──────────────────┘                   │
  (카메라 / 파일)                                                             ▼
                                                                   ┌─────────────────┐
                                                                   │  SessionState   │
                                                                   │  + EarTracker   │
                                                                   └────────┬────────┘
                                                                            │ drut_history …
                                                                            ▼
                                                                   ┌─────────────────┐
                                                                   │ compute_summary │
                                                                   └────────┬────────┘
                                                                            │ SessionSummary
                                                                            ▼
                                                                   ┌─────────────────┐
                                                                   │   LLMReporter   │
                                                                   └─────────────────┘
```

각 스테이지는 `models.py`에 정의된 frozen dataclass를 통해 데이터를 주고받는다.  
`assess_distraction`, `derive_thresholds`, `compute_summary`, `compute_calibration`은 순수 함수로, 카메라 없이 단독 테스트 가능.

---

## 빠른 시작

```bash
uv sync
uv run python main.py
```

캘리브레이션 창이 열리고, **Q**를 누르면 세션이 종료된다. 종료 후 세 파일이 생성된다:

| 파일 | 내용 |
|------|------|
| `debug_log.json` | 프레임별 feature 로그 |
| `session_summary.json` | 구조화된 세션 분석 결과 |
| `focus_report.txt` | LLM이 생성한 집중 리포트 |

---

## 설정

설정값은 모두 `main.py` 상단에 있다.

### 실행 모드

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `DEV_MODE` | `False` | `True` → 카메라·MediaPipe 없이 스텁으로 실행 |
| `CAMERA_INDEX` | `0` | OpenCV 카메라 인덱스 |

### LLM 리포트

백엔드별로 선택 의존성을 별도 설치해야 한다:

```bash
uv sync --extra cloud   # OpenAI 백엔드 (openai 패키지)
uv sync --extra local   # on-device SLM (llama-cpp-python)
```

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `LLM_MODE` | `"openai"` | `"openai"` / `"slm"` / `"stub"` |
| `OPENAI_KEY` | 환경변수 `OPENAI_API_KEY` | `LLM_MODE="openai"` 일 때 필요 |
| `SLM_MODEL_PATH` | `None` | `LLM_MODE="slm"` 일 때 GGUF 파일 경로 |

`"stub"`은 의존성 추가 없이 단순 텍스트 리포트를 출력한다. 파이프라인 전체를 검증할 때 유용하다.

### 녹화 / 재현

셋 중 하나를 `SESSION`에 대입한다:

```python
SESSION = LiveMode()                  # 기본값 — 카메라로 실시간 측정
SESSION = RecordMode("runs/run_01")   # 측정하면서 FrameFeatures + 캘리브레이션 저장
SESSION = ReplayMode("runs/run_01")   # 저장된 세션을 카메라 없이 재실행
```

`RecordMode`와 `ReplayMode`가 별개 타입이므로 두 모드를 동시에 설정하는 실수가 타입 수준에서 차단된다.

**녹화 모드**: 실제 detector를 감싸 프레임마다 `runs/run_01/features.jsonl`에 기록하고, 캘리브레이션 정보를 `runs/run_01/meta.json`에 저장한다.

**재현 모드**: 위 파일을 불러와 카메라·MediaPipe 없이 파이프라인을 다시 실행한다. `config.py` 상수를 바꾸거나 다른 리포터로 교체한 뒤, 동일한 입력으로 결과를 비교하는 실험에 사용한다.

---

## 테스트 실행

pytest는 `uv sync`로 함께 설치된다 (`[dependency-groups] dev`).

```bash
uv run pytest .
```

| 파일 | 검증 내용 |
|------|----------|
| `test_detection.py` | 순차 체크 순서, UNKNOWN gaze 비판정, `derive_thresholds` 시나리오 매핑 |
| `test_ear_tracker.py` | 깜빡임 카운팅, 동적 threshold, `reset_unit` 인터페이스 |
| `test_session_state.py` | DRUT 계산, 단위 경계 신호, `update` / `close_unit` 분리 |
| `test_analytics.py` | 골든타임 선택, drop event, focus_pattern, blink_trend |
| `test_calibration.py` | 기준값 평균, study posture 오버라이드, EAR fallback |
| `test_pipeline.py` | `ReplayDetector` / `JsonlFeatureStore` 왕복, 전체 파이프라인 통합 |

`tests/conftest.py`에 공유 헬퍼(`make_thresholds`, `focused_frame`, `run_pipeline` 등)가 있어, 카메라 없이 알려진 입력으로 각 단계를 검증할 수 있다.

---

## 모듈 구조

```
proto_0/
├── main.py                  진입점 — 파이프라인 조립 및 실행
├── models.py                스테이지 간 데이터 계약 (frozen dataclass + enum)
├── config.py                알고리즘 상수 — SSOT
├── detection.py             derive_thresholds(), assess_distraction()  [순수 함수]
├── analytics.py             compute_summary()  [순수 함수]
├── feature_stream.py        JsonlFeatureStore, RecordingDetector, ReplayDetector
│
├── features/
│   ├── detector.py          FaceDetector ABC, MediaPipeFaceDetector, StubFaceDetector
│   ├── ear.py               EAR 계산 (Wang et al.)
│   ├── gaze.py              동공 위치 → GazePosition
│   └── head_pose.py         PnP → yaw / pitch / roll
│
├── session/
│   ├── ear_tracker.py       슬라이딩 윈도우 EAR threshold + 깜빡임 카운팅
│   └── session_state.py     DRUT 누적 — update() / close_unit() 인터페이스
│
├── calibration/
│   ├── calibrator.py        캘리브레이션 UI; compute_calibration() [순수 함수]
│   └── __init__.py
│
├── report/
│   ├── base.py              LLMReporter ABC, build_prompt(), summary_to_dict()
│   ├── openai_reporter.py   OpenAI 백엔드
│   ├── local_reporter.py    llama-cpp-python (on-device GGUF)
│   └── stub_reporter.py     API 없이 동작하는 개발용 스텁
│
├── ui/
│   ├── overlay.py           실시간 HUD (draw_overlay)
│   └── capture.py           StubCapture — 개발 / 재현용 합성 프레임
│
└── tests/                   pytest 테스트 스위트
```

---

## 캘리브레이션 흐름

실행 시 (dev / replay 모드가 아닐 때) 캘리브레이션 창에서 세 단계를 수행한다:

1. **정면 자세** (5초) — pitch / yaw 기준값 측정
2. **자연 눈 깜빡임** (5초) — EAR 기준값 측정
3. **공부 자세** (5초, PAPER / BOOK 시나리오만) — pitch 기준값 재측정

수집된 `FrameFeatures`를 `compute_calibration()`에 넘기면 불변 `CalibrationData`가 반환된다.  
이를 `derive_thresholds()`에 전달하면 세션 전체에서 사용할 `DetectionThresholds`가 만들어진다.

---

## 판정 로직 (Wang et al. 순차 체크)

프레임마다 아래 경로 중 하나를 통과한다:

```
얼굴 없음?        → NO_FACE
고개 이탈?        → HEAD_DEVIATION     (|yaw| > 임계값 또는 pitch 범위 초과)
EAR < 임계값?    → EYES_CLOSED        (동적 임계값: mean(윈도우) − offset)
시선 이탈?        → GAZE_DEVIATION     (UNKNOWN은 판정하지 않음)
이 외            → NONE  (집중 중)
```

800프레임마다 DRUT 단위가 마감된다 (30fps 기준 약 26초):

```
DRUT = 집중_저하_프레임 수 / 전체_프레임 수

DRUT < 0.2  →  집중 단위
DRUT ≥ 0.2  →  집중 저하 단위
```

---

## 파이프라인 확장

### 임계값 조정 (가장 간단)

`config.py`의 상수 수정. `derive_thresholds()`와 `EarTracker`가 이 값을 참조하므로 다른 코드를 건드릴 필요 없음.

```python
# config.py 주요 상수
EAR_ALPHA            = 0.02   # 동적 임계값 offset
EAR_WINDOW_SIZE      = 60     # 슬라이딩 윈도우 크기
DRUT_UNIT_FRAMES     = 800    # 단위 경계 (프레임 수)
DRUT_FOCUS_THRESHOLD = 0.2    # DRUT 집중 판정 기준
YAW_TOLERANCE        = 20.0   # 고개 좌우 허용 범위 (도)
GAZE_BINARIZE_THRESHOLD = 50  # 동공 이진화 임계값
```

재현 모드(`REPLAY_FROM`)와 함께 쓰면 동일 입력으로 파라미터 효과를 비교할 수 있다.

---

### 판정 순서 / 조건 변경

`detection.py`의 `assess_distraction()`을 수정한다. 순수 함수라 `test_detection.py`로 즉시 검증할 수 있다.

```python
# detection.py — 현재 순서
if not features.face_detected:  → NO_FACE
if 고개 이탈:                    → HEAD_DEVIATION
if EAR 낮음:                    → EYES_CLOSED
if 시선 이탈:                    → GAZE_DEVIATION
```

체크 순서를 바꾸거나 새 조건을 추가한 뒤, `test_detection.py`의 순차 체크 테스트를 함께 수정한다.

---

### 새 distraction 원인 추가

1. `models.py`의 `DistractionReason`에 값 추가
2. `detection.py`의 `assess_distraction()`에 판정 로직 추가

`SessionState._reason_counts`는 `{r: 0 for r in DistractionReason}`으로 초기화되므로 자동으로 확장된다.

---

### feature 알고리즘 교체 (EAR / Gaze / HeadPose)

각 계산 함수는 `features/` 아래 독립 모듈에 있다. `FaceDetector.detect()`가 반환하는 `FrameFeatures` 계약만 지키면 하위 파이프라인은 변경 없이 동작한다.

| 파일 | 교체 범위 |
|------|----------|
| `features/ear.py` — `compute_ear()` | EAR 랜드마크 인덱스 및 수식 |
| `features/gaze.py` — `compute_gaze()` | 이진화 방식, 동공 위치 추정 알고리즘 |
| `features/head_pose.py` — `compute_head_pose()` | PnP 입력 랜드마크, 카메라 모델 |

교체 후 `DEV_MODE = True` + `StubFaceDetector` 대신 실제 알고리즘의 단위 테스트를 `test_pipeline.py`에 추가하는 것을 권장한다.

---

### 새 feature 추가 (예: 오디오)

1. `models.py`의 `FrameFeatures`에 필드 추가 (예: `audio_level: float | None`)
2. `features/` 아래 계산 모듈 추가
3. `features/detector.py`의 `detect()`에서 호출해 `FrameFeatures`에 포함
4. `detection.py`의 `assess_distraction()`에서 활용

`FrameFeatures`가 frozen dataclass이므로 필드 추가 시 타입 체커가 모든 생성 지점을 즉시 알려준다.

---

### Detector 교체

`features/detector.py`의 `FaceDetector` ABC를 구현해 `Calibrator`와 메인 루프에 주입한다. MediaPipe 외 다른 얼굴 인식 라이브러리로 교체하거나, `RecordingDetector` / `ReplayDetector`처럼 감싸는 용도로 쓸 수 있다.

### Reporter 교체

`report/base.py`의 `LLMReporter` ABC를 구현해 `reporter.report(summary)`에 전달한다. 프롬프트 구성은 `build_prompt()`에서, 직렬화는 `summary_to_dict()`에서 처리하므로 Reporter 구현체는 API 호출 부분만 작성하면 된다.

### 파이프라인 일부만 실행

`tests/conftest.py`의 `run_pipeline()`이 패턴을 보여준다. `FrameFeatures` 리스트를 직접 넘기면 카메라와 캘리브레이션 없이 detection → session → analytics를 실행할 수 있다.

---

## 미구현 / 알려진 한계

아래는 코드 동작에 직접 영향을 주는 주요 항목이다.

**`focus_gaze` 캘리브레이션 미구현** (프로젝트 novelty 핵심)  
PAPER / BOOK 시나리오에서도 `allowed_gazes = {GazePosition.CENTER}`만 허용된다. 고개를 숙여 종이를 볼 때 시선이 CENTER가 아닐 수 있어 집중 중에도 `GAZE_DEVIATION`으로 오판정된다. `calibration/calibrator.py`의 `focus_gaze: GazePosition | None = None` 줄이 아직 stub이며, 구현 시 캘리브레이션에서 시선 방향을 측정해 이 필드를 채우면 `derive_thresholds()`가 자동으로 `allowed_gazes`를 확장한다.

**시선 이진화 임계값 고정**  
`config.GAZE_BINARIZE_THRESHOLD = 50`이 조명·카메라 종류와 무관하게 모든 사용자에게 동일하게 적용된다. 어두운 환경이나 밝은 눈동자를 가진 사용자에서 정확도가 떨어질 수 있다.

**모바일 카메라 왜곡 미보정**  
`features/head_pose.py`에서 카메라 행렬을 `focal_length = frame_width`, 왜곡 계수를 `zeros`로 가정한다. 모바일 카메라는 화각과 왜곡이 웹캠과 크게 달라 head pose 각도 오차가 발생한다. 디바이스별 캘리브레이션 또는 기본값 조정이 필요하다.

**재현 소진 후 루프 자동 종료 없음**  
`ReplayDetector`가 모든 프레임을 소진한 뒤 `exhausted = True`가 되어도 `StubCapture`는 계속 빈 프레임을 생성하므로 메인 루프가 자동으로 종료되지 않는다. 재현 실험 시 직접 **Q**를 눌러야 한다.
