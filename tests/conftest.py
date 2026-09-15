from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agri_agent.config import settings  # noqa: E402
from agri_agent.rag.pipeline import KnowledgeBase  # noqa: E402


@pytest.fixture(scope="session")
def kb() -> KnowledgeBase:
    """整个测试会话共用一个内存索引，避免每个用例重复构建。"""
    return KnowledgeBase.build(strategy="heading", embed_provider="tfidf", cfg=settings)


@pytest.fixture()
def agent(kb):
    from agri_agent.agent.agent import OfflineAgent
    from agri_agent.agent.tools import build_default_tools

    return OfflineAgent(kb, build_default_tools(kb, settings), settings)
