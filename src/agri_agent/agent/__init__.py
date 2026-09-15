"""Agent 层：工具定义与两种执行策略（LLM 驱动 / 规则降级）。"""

from .agent import AgentAnswer, BaseAgent, LLMAgent, OfflineAgent, build_agent
from .tools import Tool, ToolRegistry, ToolResult, build_default_tools

__all__ = [
    "AgentAnswer",
    "BaseAgent",
    "LLMAgent",
    "OfflineAgent",
    "build_agent",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "build_default_tools",
]
