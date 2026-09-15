"""检索器：BM25（稀疏）+ 向量（稠密）+ RRF 融合。

单一检索通道在中文农业问答上各有一半短板：BM25 抓不到同义表述，
向量检索对品种名、药剂名、数值这类低频精确词不敏感。混合检索
用 RRF（Reciprocal Rank Fusion）把两路排名融合，不依赖分数可比性。
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from ..text import tokenize
from .chunker import Chunk
from .embedder import BaseEmbedder
from .store import VectorStore


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    retriever: str
    rank: int = 0

    @property
    def citation(self) -> str:
        label = f"{self.chunk.title} · {self.chunk.section}" if self.chunk.section else self.chunk.title
        return label

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "section": self.chunk.section,
            "score": round(self.score, 4),
            "retriever": self.retriever,
            "rank": self.rank,
            "text": self.chunk.text,
        }


class BM25Retriever:
    """BM25（k1=1.5, b=0.75）+ 标题字段加权（BM25F 简化实现）。

    正文里往往只写「病原为胶孢炭疽菌」，病名只出现在小节标题上；如果标题
    不单独加权，只按正文词频打分，「常见药剂」这类反复提到病名的小节反而
    会盖过病名对应的小节。因此把标题/小节名按 title_boost 倍计入词频。
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75, title_boost: int = 4):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.title_boost = title_boost
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []

        for index, chunk in enumerate(chunks):
            counts = Counter(tokenize(chunk.text))
            for term, freq in Counter(tokenize(f"{chunk.title} {chunk.section}")).items():
                counts[term] += freq * title_boost
            self.doc_len.append(sum(counts.values()))
            for term, freq in counts.items():
                self.postings[term].append((index, freq))

        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0

    def _idf(self, term: str) -> float:
        total = len(self.chunks)
        freq = len(self.postings.get(term, ()))
        if freq == 0:
            return 0.0
        return math.log(1 + (total - freq + 0.5) / (freq + 0.5))

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        terms = tokenize(query)
        if not terms or not self.chunks:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in set(terms):
            idf = self._idf(term)
            if idf <= 0:
                continue
            for doc_index, freq in self.postings[term]:
                length = self.doc_len[doc_index] or 1
                denominator = freq + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                scores[doc_index] += idf * freq * (self.k1 + 1) / denominator

        ranked = sorted(scores.items(), key=lambda item: -item[1])[:top_k]
        return [
            RetrievalResult(chunk=self.chunks[idx], score=score, retriever="bm25", rank=rank)
            for rank, (idx, score) in enumerate(ranked, start=1)
        ]


class DenseRetriever:
    def __init__(self, store: VectorStore, embedder: BaseEmbedder):
        self.store = store
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        vector = self.embedder.encode([query])
        if vector.size == 0:
            return []
        hits = self.store.search(vector, top_k=top_k)
        return [
            RetrievalResult(
                chunk=self.store.chunks[idx],
                score=score,
                retriever="dense",
                rank=rank,
            )
            for rank, (idx, score) in enumerate(hits, start=1)
        ]


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]], k: int = 60, top_k: int = 5
) -> list[RetrievalResult]:
    """RRF：score(d) = Σ 1 / (k + rank_i(d))。"""
    fused: dict[str, float] = defaultdict(float)
    registry: dict[str, RetrievalResult] = {}
    for results in result_lists:
        for position, result in enumerate(results, start=1):
            key = result.chunk.chunk_id
            fused[key] += 1.0 / (k + position)
            registry.setdefault(key, result)

    ordered = sorted(fused.items(), key=lambda item: -item[1])[:top_k]
    return [
        RetrievalResult(
            chunk=registry[key].chunk,
            score=score,
            retriever="hybrid",
            rank=rank,
        )
        for rank, (key, score) in enumerate(ordered, start=1)
    ]


class HybridRetriever:
    def __init__(self, sparse: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60, fetch_k: int = 20):
        self.sparse = sparse
        self.dense = dense
        self.rrf_k = rrf_k
        self.fetch_k = fetch_k

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        fetch_k = max(top_k, self.fetch_k)
        sparse_hits = self.sparse.search(query, top_k=fetch_k)
        dense_hits = self.dense.search(query, top_k=fetch_k)
        return reciprocal_rank_fusion([sparse_hits, dense_hits], k=self.rrf_k, top_k=top_k)
