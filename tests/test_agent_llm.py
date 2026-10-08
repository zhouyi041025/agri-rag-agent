"""LLMAgent 的工具循环：此前没有测试覆盖，全靠真实模型调用才能验证。

用脚本化的假模型（ScriptedLLM）把"模型决定调工具"这一步固定下来，
让循环逻辑、步数上限、工具失败回填和引用裁剪都能在离线单测里断言。
"""

from agri_agent.agent.agent import LLMAgent
from agri_agent.agent.tools import build_default_tools
from agri_agent.config import Settings
from agri_agent.llm import BaseLLM, LLMResponse, ToolCall


def tool_call_response(name: str, arguments: dict) -> LLMResponse:
    raw = {
        "id": "call_1",
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id="call_1", name=name, arguments=arguments, raw=raw)],
        usage={"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
    )


def final_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


class ScriptedLLM(BaseLLM):
    """按脚本逐次返回响应的假模型；记录每次调用是否允许了工具。"""

    name = "scripted"
    model = "scripted"

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.calls = 0
        self.tools_seen: list[list[dict] | None] = []

    def chat(self, messages, tools=None, temperature=None):
        self.calls += 1
        self.tools_seen.append(tools)
        assert self.responses, "脚本用完了：说明模型被调用的次数超出预期"
        return self.responses.pop(0)


def test_llm_agent_runs_tool_then_answers(kb):
    llm = ScriptedLLM(
        [
            tool_call_response("search_knowledge", {"query": "炭疽病 防治", "top_k": 3}),
            final_response("炭疽病可用代森锰锌防治，注意安全间隔期。[1]"),
        ]
    )
    agent = LLMAgent(kb, build_default_tools(kb), llm, Settings())

    answer = agent.answer("芦荟炭疽病怎么防治？")

    assert llm.calls == 2
    assert answer.steps and answer.steps[0]["name"] == "search_knowledge"
    assert answer.steps[0]["ok"] is True
    assert answer.citations, "答案里的 [1] 应该映射回引用台账"
    assert answer.llm_calls == 2
    assert answer.usage["total_tokens"] == 25
    assert answer.cached is False


def test_llm_agent_forces_final_answer_after_step_cap(kb):
    """步数用尽后必须再问一次模型（不允许再调工具），而不是抛错或空答案。"""
    cfg = Settings()
    cfg.max_agent_steps = 2
    llm = ScriptedLLM(
        [
            tool_call_response("search_knowledge", {"query": "炭疽病"}),
            tool_call_response("search_knowledge", {"query": "炭疽病 防治"}),
            final_response("综合已有结果，建议使用代森锰锌并注意间隔期。[1]"),
        ]
    )
    agent = LLMAgent(kb, build_default_tools(kb), llm, cfg)

    answer = agent.answer("炭疽病怎么防治？")

    assert llm.calls == 3
    assert llm.tools_seen[-1] is None, "最后一次调用不应再暴露工具"
    assert answer.answer
    assert len(answer.steps) == 2


def test_llm_agent_records_failed_tool_and_continues(kb):
    """模型调了不存在的工具时，错误要回填给模型继续推理，而不是中断整条链路。"""
    llm = ScriptedLLM(
        [
            tool_call_response("no_such_tool", {"foo": "bar"}),
            final_response("已根据知识库内容作答。[1]"),
        ]
    )
    agent = LLMAgent(kb, build_default_tools(kb), llm, Settings())

    answer = agent.answer("芦荟炭疽病怎么防治？")

    assert answer.steps and answer.steps[0]["ok"] is False
    assert "未注册的工具" in answer.steps[0]["preview"]
    assert answer.answer


def test_llm_agent_trims_citations_to_those_actually_used(kb):
    """检索登记了多条候选，但正文只引用 [2]：返回的引用列表必须只有这一条。"""
    llm = ScriptedLLM(
        [
            tool_call_response("search_knowledge", {"query": "炭疽病 防治", "top_k": 3}),
            final_response("按照第二条资料执行即可。[2]"),
        ]
    )
    agent = LLMAgent(kb, build_default_tools(kb), llm, Settings())

    answer = agent.answer("芦荟炭疽病怎么防治？")

    assert len(answer.citations) == 1
    assert answer.citations[0]["index"] == 1, "台账应被裁剪并重排为 [1]"
    assert "[1]" in answer.answer
    assert "[2]" not in answer.answer
