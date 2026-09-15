"""文本切分：标题感知切分（默认）与固定窗口切分（对照基线）。

标题感知切分会把 Markdown 的小节标题带进每个 chunk 的上下文，减少
「一段话脱离标题后语义不完整」造成的召回损失。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..text import split_sentences
from .loader import Document

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    position: int
    source: str

    @property
    def display_text(self) -> str:
        """给模型看的文本：带上来源标题，提升可引用性。"""
        header = f"【{self.title} · {self.section}】" if self.section else f"【{self.title}】"
        return f"{header}\n{self.text}"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "section": self.section,
            "text": self.text,
            "position": self.position,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


def _window_split(text: str, max_chars: int, overlap: int) -> list[str]:
    """对超长段落先按句子聚合，再退化为定长滑窗，避免切断句子。"""
    sentences = split_sentences(text) or [text]
    pieces: list[str] = []
    buffer = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            step = max(1, max_chars - overlap)
            pieces.extend(sentence[i : i + max_chars] for i in range(0, len(sentence), step))
            continue
        if len(buffer) + len(sentence) <= max_chars:
            buffer += sentence
        else:
            pieces.append(buffer)
            buffer = sentence
    if buffer:
        pieces.append(buffer)
    return [p.strip() for p in pieces if p.strip()]


def _heading_blocks(text: str) -> list[tuple[str, str]]:
    """把文档拆成 (小节标题, 小节正文) 序列。

    同一小节内的段落先合并，再交给长度滑窗切分——按空行切块会把「症状 /
    发病条件 / 防治要点」拆成三个互不相干的短块，检索时既丢上下文，
    又让病名的文档频率虚高、IDF 被稀释。
    """
    blocks: list[tuple[str, str]] = []
    section = ""
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if buffer:
            joined = "\n".join(buffer).strip()
            if joined:
                blocks.append((section, joined))
            buffer = []

    for line in text.split("\n"):
        stripped = line.strip()
        match = _HEADING_RE.match(stripped)
        if match:
            flush()
            level, heading = len(match.group(1)), match.group(2).strip()
            section = heading if level > 1 else section
            continue
        if stripped:
            buffer.append(stripped)
    flush()
    return blocks


def chunk_document(
    doc: Document,
    max_chars: int = 420,
    overlap: int = 80,
    strategy: str = "heading",
) -> list[Chunk]:
    chunks: list[Chunk] = []
    position = 0

    if strategy == "fixed":
        segments = _window_split(doc.text, max_chars, overlap)
        for segment in segments:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{position:03d}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section="",
                    text=segment,
                    position=position,
                    source=doc.source,
                )
            )
            position += 1
        return chunks

    for section, block in _heading_blocks(doc.text):
        parts = _window_split(block, max_chars, overlap) if len(block) > max_chars else [block]
        for part in parts:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{position:03d}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section=section,
                    text=part,
                    position=position,
                    source=doc.source,
                )
            )
            position += 1
    return chunks


def chunk_documents(
    docs: list[Document],
    max_chars: int = 420,
    overlap: int = 80,
    strategy: str = "heading",
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, max_chars=max_chars, overlap=overlap, strategy=strategy))
    return chunks
