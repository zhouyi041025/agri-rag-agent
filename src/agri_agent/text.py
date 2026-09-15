"""中文文本处理：归一化、分词、字符 n-gram。

分词器按可用性降级：jieba 可用时用词粒度；否则退化为「单字 + 二元组」，
保证在没装任何分词库的环境里检索链路依然可用。
"""

from __future__ import annotations

import re
import unicodedata

try:  # pragma: no cover - 取决于运行环境
    import jieba

    jieba.setLogLevel(60)
    _HAS_JIEBA = True
except Exception:  # pragma: no cover
    _HAS_JIEBA = False

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+\-]*")
_WS_RE = re.compile(r"[ \t\u3000]+")

# 中文停用词（精简版，只保留高频虚词，避免误伤专业术语）
STOPWORDS = {
    "的", "了", "和", "是", "在", "有", "与", "及", "或", "对", "为", "被", "把",
    "我", "你", "他", "它", "这", "那", "什么", "怎么", "如何", "请问", "一下",
    "吗", "呢", "啊", "呀", "吧", "着", "过", "得", "地", "中", "上", "下", "个",
    "多少", "应该", "可以", "需要", "请", "帮", "帮我", "告诉", "现在", "目前",
}


def normalize(text: str) -> str:
    """全角转半角、统一空白、去掉零宽字符。"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _WS_RE.sub(" ", text).strip()


def has_jieba() -> bool:
    return _HAS_JIEBA


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """中英混合分词，供 BM25 等稀疏检索使用。"""
    text = normalize(text)
    if not text:
        return []

    tokens: list[str] = [m.group(0).lower() for m in _WORD_RE.finditer(text)]

    for match in _CJK_RE.finditer(text):
        segment = match.group(0)
        if _HAS_JIEBA:
            pieces = [w.strip() for w in jieba.lcut(segment)]
            tokens.extend(w for w in pieces if w)
        else:
            tokens.extend(segment)
            tokens.extend(segment[i : i + 2] for i in range(len(segment) - 1))

    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def char_ngrams(text: str, n: int = 2) -> list[str]:
    """字符 n-gram 特征，用于无外部模型时的向量化。"""
    text = normalize(text)
    if not text:
        return []
    feats: list[str] = [m.group(0).lower() for m in _WORD_RE.finditer(text)]
    for match in _CJK_RE.finditer(text):
        segment = match.group(0)
        if len(segment) < n:
            feats.append(segment)
            continue
        feats.extend(segment[i : i + n] for i in range(len(segment) - n + 1))
    return feats


def split_sentences(text: str) -> list[str]:
    """按中英文标点切句，保留标点，供长段落的滑窗切分使用。"""
    text = normalize(text)
    if not text:
        return []
    parts = re.split(r"(?<=[。！？；;!?\.])\s*", text)
    return [p.strip() for p in parts if p.strip()]
