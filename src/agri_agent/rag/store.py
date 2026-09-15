"""向量索引：numpy 暴力检索 + 落盘持久化。

chunk 数量在万级以内时，numpy 矩阵点积（已归一化，等价于余弦相似度）
比引入 FAISS 更快也更省事；超过十万级再换 HNSW/IVF 索引，接口不变。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .chunker import Chunk


class VectorStore:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder_name: str = "unknown"):
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunk 数量与向量数量不一致")
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)
        self.embedder_name = embedder_name

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1]) if self.embeddings.size else 0

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        """返回 [(chunk 下标, 相似度分数)]，按分数降序。"""
        if not len(self.chunks) or query_vector.size == 0:
            return []
        query = query_vector.reshape(-1).astype(np.float32)
        if query.shape[0] != self.dim:
            raise ValueError(f"查询向量维度 {query.shape[0]} 与索引维度 {self.dim} 不一致")
        scores = self.embeddings @ query
        top_k = max(1, min(top_k, scores.shape[0]))
        order = np.argpartition(-scores, top_k - 1)[:top_k]
        order = order[np.argsort(-scores[order])]
        return [(int(idx), float(scores[idx])) for idx in order]

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        (directory / "store_meta.json").write_text(
            json.dumps(
                {
                    "embedder": self.embedder_name,
                    "chunks": len(self.chunks),
                    "dim": self.dim,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        directory = Path(directory)
        embeddings = np.load(directory / "embeddings.npy")
        chunks: list[Chunk] = []
        with (directory / "chunks.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    chunks.append(Chunk.from_dict(json.loads(line)))
        meta_path = directory / "store_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(chunks, embeddings, embedder_name=meta.get("embedder", "unknown"))
