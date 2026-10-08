"""LLM 客户端的重试与退避。

回归背景：此前 chat() 是单次 POST，网络抖动或 429 直接把异常抛给上层，
SSE 前端只会收到一个 error 事件。现在按指数退避重试，并在客户端上
留下 calls / retries 计数供 /stats 观测。
"""

import pytest

from agri_agent.llm import OpenAICompatLLM

PAYLOAD = {
    "choices": [{"message": {"content": "答案", "tool_calls": []}}],
    "usage": {"total_tokens": 10},
}


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FlakyClient:
    """前 failures 次调用抛错（模拟 429），之后返回正常响应。"""

    def __init__(self, failures=0, payload=None):
        self.failures = failures
        self.calls = 0
        self.payload = payload or PAYLOAD

    def post(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("429 Too Many Requests")
        return FakeResponse(self.payload)


def make_llm(failures=0, max_retries=2):
    llm = OpenAICompatLLM(
        base_url="http://example.invalid",
        model="deepseek-chat",
        api_key="test-key",
        max_retries=max_retries,
        backoff_cap=0.0,  # 测试里不真的等待
    )
    llm._client = FlakyClient(failures=failures)
    return llm


def test_retries_then_succeeds():
    llm = make_llm(failures=2)

    response = llm.chat([{"role": "user", "content": "你好"}])

    assert response.content == "答案"
    assert llm.calls == 3
    assert llm.retries == 2


def test_raises_after_retry_budget_exhausted():
    llm = make_llm(failures=5, max_retries=2)

    with pytest.raises(RuntimeError, match="调用 LLM 失败"):
        llm.chat([{"role": "user", "content": "你好"}])
    assert llm.calls == 3


def test_no_retry_when_disabled():
    llm = make_llm(failures=1, max_retries=0)

    with pytest.raises(RuntimeError):
        llm.chat([{"role": "user", "content": "你好"}])
    assert llm.calls == 1
