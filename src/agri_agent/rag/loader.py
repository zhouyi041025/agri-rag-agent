"""知识库文档加载：支持 Markdown / 纯文本 / PDF，统一成 Document。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..text import normalize

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}
_TITLE_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("解析 PDF 需要安装 pypdf：pip install pypdf") from exc

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(p for p in pages if p)


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def load_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的文档类型：{path.name}")

    raw = _read_pdf(path) if suffix == ".pdf" else _read_text(path)
    text = normalize(raw)
    if not text:
        raise ValueError(f"文档为空：{path.name}")

    match = _TITLE_RE.search(text)
    title = match.group(1).strip() if match else path.stem
    return Document(
        doc_id=path.stem,
        title=title,
        text=text,
        source=str(path),
        metadata={"suffix": suffix, "bytes": path.stat().st_size},
    )


def load_documents(kb_dir: str | Path) -> list[Document]:
    """加载目录下所有受支持的文档，按文件名排序保证索引可复现。"""
    directory = Path(kb_dir)
    if not directory.exists():
        raise FileNotFoundError(f"知识库目录不存在：{directory}")

    documents: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            documents.append(load_document(path))

    if not documents:
        raise ValueError(f"知识库目录中没有可用文档：{directory}")
    return documents
