"""On-device SLM reporter via llama-cpp-python."""

import os
import time

from models import SessionSummary
from .base import SYSTEM_PROMPT, LLMReporter, build_prompt


class LocalSLMReporter(LLMReporter):
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 4) -> None:
        self._model_path = model_path
        self._n_ctx      = n_ctx
        self._n_threads  = n_threads
        self.metrics: dict = {}

    def report(self, summary: SessionSummary) -> str:
        try:
            import psutil
            from llama_cpp import Llama

            llm = Llama(
                model_path=self._model_path,
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
                verbose=False,
            )
            memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

            t0 = time.perf_counter()
            output = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_prompt(summary)},
                ],
                max_tokens=800,
                temperature=0.7,
            )
            elapsed = time.perf_counter() - t0

            completion_tokens = (output.get("usage") or {}).get("completion_tokens", 0)
            self.metrics = {
                "elapsed_sec":       round(elapsed, 2),
                "completion_tokens": completion_tokens,
                "tokens_per_sec":    round(completion_tokens / elapsed, 2) if elapsed > 0 else 0,
                "memory_mb":         round(memory_mb, 1),
            }
            return output["choices"][0]["message"]["content"]
        except ImportError as e:
            return f"[SLM 오류] 필요한 패키지가 없습니다: {e}"
        except Exception as e:
            return f"[SLM 오류] {e}"
