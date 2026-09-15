"""大模型接入层：OpenAI 兼容 Chat Completions（含 Function Calling）。

只实现一个协议、一个客户端，是为了让「换模型」变成改环境变量，
而不是改代码——DeepSeek / 通义千问 / 智谱 / Moonshot / vLLM 都兼容这套接口。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import Settings, settings as default_settings


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

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float = 0.2,
        timeout: float = 60.0,
    ):
        import httpx

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self._client = httpx.Client(timeout=timeout)

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

        response = self._client.post(
            f"{self.base_url}/chat/completions", headers=self._headers(), json=payload
        )
        response.raise_for_status()
        body = response.json()

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
    )
