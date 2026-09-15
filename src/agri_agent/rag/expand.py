"""查询改写：把农户口语映射到知识库里的规范术语。

真实用户问「一天中什么时候打药最合适」，知识库里写的是「施药时机」；
「拍照片」对应用户问法，知识库里是「图像采集」。这类词汇缺口靠增加
向量模型解决成本高，用一份领域同义词表做查询扩展性价比最高。
"""

from __future__ import annotations

import re

SYNONYMS: dict[str, tuple[str, ...]] = {
    # 农事操作
    "打药": ("施药", "喷药"),
    "喷药": ("施药",),
    "用药": ("施药",),
    "打多少": ("用药量", "剂量"),
    "兑水": ("稀释", "兑水量"),
    "配药": ("配制", "混配"),
    # 图像与识别
    "照片": ("图像", "拍摄"),
    "拍照": ("拍摄", "图像"),
    "拍一张": ("拍摄", "图像"),
    "叶子": ("叶片",),
    "长什么样": ("症状", "特征"),
    "怎么看": ("识别", "症状"),
    "识别": ("检测",),
    # 病害口语
    "烂": ("腐烂", "软腐"),
    "发黑": ("黑斑", "煤污"),
    "发黄": ("褪绿", "黄化"),
    "长斑": ("病斑",),
    "斑点": ("病斑",),
    # 环境与管理
    "含水率": ("墒情", "水分"),
    "多少合适": ("适宜", "区间"),
    "什么时候": ("时机", "时期"),
    "多久": ("频次", "间隔"),
    "晒太阳": ("光照",),
    "遮阴": ("遮阳", "遮光"),
    "怕涝": ("积水", "排水"),
    # 药剂
    "残留": ("安全间隔期",),
    "抗性": ("抗药性",),
}

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
