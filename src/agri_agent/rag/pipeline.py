"""知识库门面：构建索引、持久化、按模式检索、拼装带引用的上下文。"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import Settings, settings as default_settings
from .chunker import Chunk, chunk_documents
from .embedder import BaseEmbedder, HashingTfidfEmbedder, build_embedder
from .expand import expand_query
from .loader import load_documents
from .retriever import BM25Retriever, DenseRetriever, HybridRetriever, RetrievalResult
from .store import VectorStore


class KnowledgeBase:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: BaseEmbedder,
        store: VectorStore,
        meta: dict | None = None,
    ):
        self.chunks = chunks
        self.embedder = embedder
        self.store = store
        self.meta = meta or {}
        self.sparse = BM25Retriever(chunks)
        self.dense = DenseRetriever(store, embedder)

    # ---------- 构建与持久化 ----------
    @classmethod
    def build(
        cls,
        *,
        kb_dir: str | Path | None = None,
        strategy: str | None = None,
        embed_provider: str | None = None,
        cfg: Settings | None = None,
    ) -> "KnowledgeBase":
        cfg = cfg or default_settings
        kb_dir = Path(kb_dir or cfg.kb_dir)
        strategy = strategy or cfg.chunk_strategy
        provider = embed_provider or cfg.embed_provider

        documents = load_documents(kb_dir)
        chunks = chunk_documents(
            documents,
            max_chars=cfg.chunk_max_chars,
            overlap=cfg.chunk_overlap,
            strategy=strategy,
        )
        texts = [chunk.display_text for chunk in chunks]
        embedder = build_embedder(
            provider,
            texts,
            embed_base_url=cfg.embed_base_url,
            embed_model=cfg.embed_model,
            embed_api_key=cfg.embed_api_key,
            max_features=cfg.embed_dim,
        )
        embeddings = embedder.encode(texts)
        store = VectorStore(chunks, embeddings, embedder_name=embedder.name)
        meta = {
            "strategy": strategy,
            "chunk_max_chars": cfg.chunk_max_chars,
            "chunk_overlap": cfg.chunk_overlap,
            "documents": len(documents),
            "chunks": len(chunks),
            "embedder": embedder.name,
            "embed_dim": store.dim,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return cls(chunks, embedder, store, meta)

    def save(self, directory: str | Path | None = None, cfg: Settings | None = None) -> Path:
        cfg = cfg or default_settings
        directory = Path(directory or cfg.artifacts_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.store.save(directory)
        self.embedder.save(directory / "embedder.json")
        (directory / "kb_meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path, cfg: Settings | None = None) -> "KnowledgeBase":
        cfg = cfg or default_settings
        directory = Path(directory)
        store = VectorStore.load(directory)
        embedder_path = directory / "embedder.json"
        payload = json.loads(embedder_path.read_text(encoding="utf-8"))
        if payload["name"] == "tfidf":
            embedder: BaseEmbedder = HashingTfidfEmbedder.load(embedder_path)
        else:  # 云端向量需要重新构造客户端
            embedder = build_embedder(
                "api" if payload["name"] == "api" else "local",
                [],
                embed_base_url=cfg.embed_base_url,
                embed_model=payload.get("model", cfg.embed_model),
                embed_api_key=cfg.embed_api_key,
            )
        meta_path = directory / "kb_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(store.chunks, embedder, store, meta)

    @classmethod
    def load_or_build(cls, cfg: Settings | None = None, rebuild: bool = False) -> "KnowledgeBase":
        cfg = cfg or default_settings
        index_file = Path(cfg.artifacts_dir) / "embeddings.npy"
        if index_file.exists() and not rebuild:
            try:
                return cls.load(cfg.artifacts_dir, cfg)
            except Exception:
                pass
        kb = cls.build(cfg=cfg)
        try:
            kb.save(cfg=cfg)
        except Exception:
            pass
        return kb

    # ---------- 检索 ----------
    def search(self, query: str, top_k: int = 5, mode: str = "hybrid", expand: bool = True) -> list[RetrievalResult]:
        if expand:
            query = expand_query(query)
        if mode == "bm25":
            return self.sparse.search(query, top_k=top_k)
        if mode == "dense":
            return self.dense.search(query, top_k=top_k)
        hybrid = HybridRetriever(self.sparse, self.dense, rrf_k=default_settings.rrf_k)
        return hybrid.search(query, top_k=top_k)

    def build_context(self, query: str, top_k: int = 5, mode: str = "hybrid", expand: bool = True) -> tuple[str, list[dict]]:
        """返回 (拼装好的上下文文本, 引用列表)，引用编号与正文 [1][2] 一一对应。"""
        results = self.search(query, top_k=top_k, mode=mode, expand=expand)
        blocks, citations = [], []
        for number, result in enumerate(results, start=1):
            blocks.append(f"[{number}] {result.chunk.display_text}")
            citations.append(
                {
                    "index": number,
                    "chunk_id": result.chunk.chunk_id,
                    "doc": result.chunk.title,
                    "section": result.chunk.section,
                    "score": round(result.score, 4),
                    "retriever": result.retriever,
                }
            )
        return "\n\n".join(blocks), citations

    @property
    def stats(self) -> dict:
        return {
            "chunks": len(self.chunks),
            "documents": len({chunk.doc_id for chunk in self.chunks}),
            "embedder": self.embedder.name,
            "dim": self.store.dim,
            **{k: v for k, v in self.meta.items() if k in {"strategy", "built_at"}},
        }
