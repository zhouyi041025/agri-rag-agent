"""查询缓存：键包含历史签名，命中后答案带 cached 标记。"""

from agri_agent.agent.agent import OfflineAgent
from agri_agent.agent.tools import build_default_tools
from agri_agent.cache import QueryCache
from agri_agent.config import settings


def test_cache_key_includes_history_signature():
    key_a = QueryCache.make_key("炭疽病怎么防治", [{"role": "user", "content": "上一轮问题A"}])
    key_b = QueryCache.make_key("炭疽病怎么防治", [{"role": "user", "content": "上一轮问题B"}])
    assert key_a != key_b


def test_cache_evicts_least_recently_used():
    cache = QueryCache(max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")
    cache.set("c", 3)

    assert cache.size == 2
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_agent_reuses_cached_answer(kb):
    cache = QueryCache(max_size=8)
    agent = OfflineAgent(kb, build_default_tools(kb, settings), settings, cache=cache)

    first = agent.answer("芦荟炭疽病怎么防治？")
    assert first.cached is False

    second = agent.answer("芦荟炭疽病怎么防治？")

    assert second.cached is True
    assert second is not first  # 命中缓存返回的是副本
    assert first.cached is False  # 旧对象不应被一起改掉
    assert second.answer == first.answer
    assert cache.hits == 1


def test_agent_cache_can_be_bypassed(kb):
    cache = QueryCache(max_size=8)
    agent = OfflineAgent(kb, build_default_tools(kb, settings), settings, cache=cache)

    agent.answer("芦荟炭疽病怎么防治？")
    again = agent.answer("芦荟炭疽病怎么防治？", use_cache=False)

    assert again.cached is False
    assert cache.hits == 0
