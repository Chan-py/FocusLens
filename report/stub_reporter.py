"""Stub LLM reporter for development — no API key or model file needed."""

from models import SessionSummary
from .base import LLMReporter


class StubLLMReporter(LLMReporter):
    """Formats session data as a plain text report without calling any LLM.
    Useful for verifying the full pipeline end-to-end in dev mode."""

    def report(self, summary: SessionSummary) -> str:
        if summary.golden_hour:
            gs, ge = summary.golden_hour
            golden = f"{gs}~{ge}분"
        else:
            golden = "측정 데이터 부족"

        drops = (
            ", ".join(f"{e.time_min}분({e.trigger.value})" for e in summary.drop_events)
            or "없음"
        )
        fatigue = (
            f"{summary.fatigue_onset_min}분" if summary.fatigue_onset_min else "미감지"
        )

        return f"""[StubLLMReporter — dev mode, no LLM call made]

[오늘 세션 요약]
시나리오: {summary.scenario.label}
세션 시간: {summary.session_duration_min:.1f}분
집중 비율: {summary.effective_focus_ratio:.0%}
골든타임: {golden}
총 눈 깜빡임: {summary.total_blinks}회

[집중 저하 원인 분석]
주요 원인: {summary.top_distraction.value}
집중 저하 이벤트: {drops}
눈 깜빡임 추세: {summary.blink_trend.value}
피로 시작 시점: {fatigue}

[내일을 위한 제안]
집중 패턴: {summary.focus_pattern.value}
DRUT 기록: {list(summary.drut_history)}

(실제 리포트를 생성하려면 OPENAI_API_KEY를 설정하거나 SLM_MODEL_PATH를 지정하세요)
"""
