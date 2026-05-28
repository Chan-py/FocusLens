"""LLMReporter ABC + shared prompt engineering."""

from __future__ import annotations

import dataclasses
import json
from abc import ABC, abstractmethod
from enum import Enum

from models import SessionSummary


SYSTEM_PROMPT = """You are FocusLens, a personal concentration coach.
You receive structured JSON data from a focus tracking session and generate a warm, practical report in Korean.

Your report must have exactly three sections:
1. [오늘 세션 요약] - What happened (duration, focus ratio, golden hour)
2. [집중 저하 원인 분석] - Why focus dropped (use drop_events, blink_trend, top_distraction)
3. [내일을 위한 제안] - Specific actionable advice (based on focus_pattern, fatigue_onset)

Rules:
- Be warm and encouraging, not critical
- Keep it concise (3-5 sentences per section)
- Give SPECIFIC suggestions based on the data, not generic advice
- If scenario is "paper" or "book", acknowledge that looking down is normal
- Never mention raw numbers like DRUT; translate them to plain language
"""


def summary_to_dict(summary: SessionSummary) -> dict:
    """Convert SessionSummary to a JSON-serialisable dict."""
    d = dataclasses.asdict(summary)
    return _enum_to_value(d)


def _enum_to_value(obj: object) -> object:
    """Recursively convert Enum instances to their .value."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _enum_to_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_enum_to_value(v) for v in obj)
    return obj


def build_prompt(summary: SessionSummary) -> str:
    d = summary_to_dict(summary)
    return (
        "아래는 오늘 집중도 세션 분석 데이터입니다. "
        "한국어로 따뜻하고 실용적인 리포트를 작성해주세요.\n\n"
        f"[세션 데이터]\n{json.dumps(d, ensure_ascii=False, indent=2)}"
    )


class LLMReporter(ABC):
    @abstractmethod
    def report(self, summary: SessionSummary) -> str: ...
