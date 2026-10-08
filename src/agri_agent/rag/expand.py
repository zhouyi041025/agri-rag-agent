"""查询改写：把农户口语映射到知识库里的规范术语。

真实用户问「一天中什么时候打药最合适」，知识库里写的是「施药时机」；
「拍照片」对应用户问法，知识库里是「图像采集」。这类词汇缺口靠增加
向量模型解决成本高，用一份领域同义词表做查询扩展性价比最高。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import PROJECT_ROOT

SYNONYMS_PATH = PROJECT_ROOT / "data" / "synonyms.json"


def load_synonyms(path: Path | None = None) -> dict[str, tuple[str, ...]]:
    """同义词表外置为数据文件：改词表不需要动代码，也便于测试断言完整性。"""
    source = Path(path) if path else SYNONYMS_PATH
    if not source.exists():
        raise FileNotFoundError(f"同义词表不存在：{source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    return {term: tuple(synonyms) for term, synonyms in payload.items()}


# 进程内加载一次；expand_query 只读
SYNONYMS: dict[str, tuple[str, ...]] = load_synonyms()


_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")


def expand_query(text: str, max_extra: int = 6) -> str:
    """在原始查询后追加同义术语，保持原句不变以免破坏 BM25 的短语信息。"""
    if not text:
        return text
    extras: list[str] = []
    for term, synonyms in SYNONYMS.items():
        if term in text:
            for synonym in synonyms:
                if synonym not in text and synonym not in extras:
                    extras.append(synonym)
    if not extras:
        return text
    return text + " " + " ".join(extras[:max_extra])
