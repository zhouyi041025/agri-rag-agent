"""进程内运行统计：请求、工具调用、token 与估算成本。"""

from __future__ import annotations

import threading


class ServiceStats:
    """线程安全的计数器，由 HTTP 层在每次问答后写入。

    金额只有在配置了单价（`AGRI_PRICE_INPUT_PER_1K` / `AGRI_PRICE_OUTPUT_PER_1K`）
    时才有意义 —— 不同模型价格差异大，默认只统计 token，不替用户猜价格。
    """

    def __init__(self, price_input_per_1k: float = 0.0, price_output_per_1k: float = 0.0) -> None:
        self._lock = threading.Lock()
        self.price_input_per_1k = float(price_input_per_1k)
        self.price_output_per_1k = float(price_output_per_1k)
        self.requests = 0
        self.cached_requests = 0
        self.errors = 0
        self.llm_calls = 0
        self.tool_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latency_ms_total = 0.0

    def record(self, answer) -> None:
        with self._lock:
            self.requests += 1
            if getattr(answer, "cached", False):
                self.cached_requests += 1
            self.llm_calls += int(getattr(answer, "llm_calls", 0) or 0)
            self.tool_calls += sum(1 for step in getattr(answer, "steps", []) if step.get("type") == "tool")
            usage = getattr(answer, "usage", {}) or {}
            self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            self.latency_ms_total += float(getattr(answer, "elapsed_ms", 0.0) or 0.0)

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def snapshot(self, cache=None) -> dict:
        with self._lock:
            payload = {
                "requests": self.requests,
                "cached_requests": self.cached_requests,
                "errors": self.errors,
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "avg_latency_ms": round(self.latency_ms_total / self.requests, 1) if self.requests else 0.0,
                "cost_usd": round(
                    self.prompt_tokens / 1000 * self.price_input_per_1k
                    + self.completion_tokens / 1000 * self.price_output_per_1k,
                    6,
                ),
            }
        if cache is not None:
            payload["cache"] = {
                "size": cache.size,
                "hits": cache.hits,
                "misses": cache.misses,
                "hit_rate": cache.hit_rate,
            }
        return payload
