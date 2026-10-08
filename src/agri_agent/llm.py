"""大模型接入层：OpenAI 兼容 Chat Completions（含 Function Calling）。

只实现一个协议、一个客户端，是为了让「换模型」变成改环境变量，
而不是改代码——DeepSeek / 通义千问 / 智谱 / Moonshot / vLLM 都兼容这套接口。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .config import settings as default_settings


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class BaseLLM:
    name = "base"
    model = ""

    def chat(self, messages: list[dict], tools: list[dict] | None = None, temperature: float | None = None) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError


class OpenAICompatLLM(BaseLLM):
    """OpenAI 兼容客户端，支持流式与非流式；工具调用走标准 tool_calls 字段。"""

    name = "openai-compatible"
    # 重试间隔上限（秒）：再长就不如让上层降级为规则 Agent，用户至少能拿到回答
    DEFAULT_BACKOFF_CAP = 8.0

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float = 0.2,
        timeout: float = 60.0,
        max_retries: int = 2,
        backoff_cap: float = DEFAULT_BACKOFF_CAP,
    ):
        import httpx

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max(0, int(max_retries))
        self.backoff_cap = float(backoff_cap)
        self._client = httpx.Client(timeout=timeout)
        # 供 /stats 观测：真实调用次数与重试次数（限流时能立刻看出来）
        self.calls = 0
        self.retries = 0

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def chat(self, messages: list[dict], tools: list[dict] | None = None, temperature: float | None = None) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        body: dict[str, Any] | None = None
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                self.calls += 1
                response = self._client.post(
                    f"{self.base_url}/chat/completions", headers=self._headers(), json=payload
                )
                response.raise_for_status()
                body = response.json()
                break
            except Exception as exc:  # 网络抖动、限流（429）、超时都走重试
                last_error = exc
                if attempt < self.max_retries:
                    self.retries += 1
                    # 指数退避：被限流后立刻重试只会继续被拒，等待通常能让第二次放行
                    time.sleep(min(0.5 * 2**attempt, self.backoff_cap))
        if body is None:
            raise RuntimeError(f"调用 LLM 失败（已重试 {self.retries} 次）：{last_error}")

        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            raw_arguments = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
            except json.JSONDecodeError:
                arguments = {}
            calls.append(
                ToolCall(
                    id=item.get("id", f"call_{len(calls)}"),
                    name=function.get("name", ""),
                    arguments=arguments,
                    raw_arguments=raw_arguments,
                    raw=item,
                )
            )
        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=calls,
            usage=body.get("usage") or {},
            model=body.get("model", self.model),
            raw=body,
        )


def build_llm(cfg: Settings | None = None) -> BaseLLM | None:
    """按配置返回大模型客户端；未配置 Key 时返回 None，由上层降级为规则 Agent。"""
    cfg = cfg or default_settings
    if not cfg.llm_enabled:
        return None
    return OpenAICompatLLM(
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
        api_key=cfg.llm_api_key,
        temperature=cfg.llm_temperature,
        timeout=cfg.llm_timeout,
        max_retries=cfg.llm_max_retries,
        backoff_cap=cfg.llm_backoff_cap,
    )
