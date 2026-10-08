"""进程内查询缓存：重复问题直接复用上一次的答案、引用与工具轨迹。"""

from __future__ import annotations

import hashlib
from collections import OrderedDict


class QueryCache:
    """LRU 查询缓存。

    键包含问题与最近 6 轮对话的签名：多轮场景下同一个问题配不同上文，
    不能复用同一份答案（否则追问会被上一轮的答案串掉）。命中时返回的
    AgentAnswer 会带上 `cached=True`，前端与评测都能看出来。
    """

    def __init__(self, max_size: int = 128) -> None:
        self.max_size = max(1, int(max_size))
        self._store: OrderedDict[str, object] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(question: str, history: list[dict] | None = None) -> str:
        history_signature = "|".join(
            f"{turn.get('role')}:{str(turn.get('content', '')).strip()}" for turn in (history or [])[-6:]
        )
        raw = f"{question.strip()}|{history_signature}"
        return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()

    def get(self, key: str):
        if key in self._store:
            self.hits += 1
            self._store.move_to_end(key)
            return self._store[key]
        self.misses += 1
        return None

    def set(self, key: str, value) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total else 0.0
