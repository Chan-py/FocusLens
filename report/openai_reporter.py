"""OpenAI-backed LLM reporter (cloud)."""

import time

from models import SessionSummary
from .base import SYSTEM_PROMPT, LLMReporter, build_prompt


class OpenAIReporter(LLMReporter):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._api_key = api_key
        self._model   = model
        self.metrics: dict = {}

    def report(self, summary: SessionSummary) -> str:
        try:
            import openai
            client = openai.OpenAI(api_key=self._api_key)

            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_prompt(summary)},
                ],
                max_tokens=800,
                temperature=0.7,
            )
            elapsed = time.perf_counter() - t0

            completion_tokens = response.usage.completion_tokens if response.usage else 0
            self.metrics = {
                "elapsed_sec":       round(elapsed, 2),
                "completion_tokens": completion_tokens,
                "tokens_per_sec":    round(completion_tokens / elapsed, 2) if elapsed > 0 else 0,
                "memory_mb":         0.0,
            }
            return response.choices[0].message.content
        except Exception as e:
            return f"[OpenAI 오류] {e}"
