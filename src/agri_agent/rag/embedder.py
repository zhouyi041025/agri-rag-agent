"""向量化：三层可插拔实现，按环境自动降级。

1. local  —— sentence-transformers 加载中文 BGE（效果最好，需额外安装）
2. api    —— 调用 OpenAI 兼容的 /embeddings 接口
3. tfidf  —— 内置纯 numpy 实现，零外部依赖，保证任何环境都能跑通

把「换向量模型」做成一层抽象，是为了让检索效果对比实验可以真正跑起来，
而不是只能写「我们使用了向量检索」一句空话。
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..text import char_ngrams, tokenize


class BaseEmbedder:
    name = "base"
    dim = 0

    def fit(self, texts: list[str]) -> "BaseEmbedder":
        return self

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - 抽象方法
        raise NotImplementedError

    def save(self, path: Path) -> None:  # pragma: no cover - 抽象方法
        raise NotImplementedError


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _features(text: str) -> list[str]:
    return char_ngrams(text, 2) + tokenize(text, drop_stopwords=False)


@dataclass
class HashingTfidfEmbedder(BaseEmbedder):
    """词表级 TF-IDF：中文用字符二元组 + 分词结果作为特征。"""

    max_features: int = 4096
    vocab: dict[str, int] = None  # type: ignore[assignment]
    idf: np.ndarray = None  # type: ignore[assignment]
    name: str = "tfidf"

    def __post_init__(self) -> None:
        if self.vocab is None:
            self.vocab = {}
        if self.idf is None:
            self.idf = np.zeros(0, dtype=np.float32)
        self.dim = len(self.vocab)

    def fit(self, texts: list[str]) -> "HashingTfidfEmbedder":
        doc_freq: Counter[str] = Counter()
        for text in texts:
            doc_freq.update(set(_features(text)))

        kept = [feat for feat, _ in doc_freq.most_common(self.max_features)]
        self.vocab = {feat: idx for idx, feat in enumerate(kept)}
        total = max(1, len(texts))
        self.idf = np.array(
            [math.log((1 + total) / (1 + doc_freq[feat])) + 1.0 for feat in kept],
            dtype=np.float32,
        )
        self.dim = len(self.vocab)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), max(1, self.dim)), dtype=np.float32)
        if self.dim == 0:
            return matrix
        for row, text in enumerate(texts):
            counts = Counter(feat for feat in _features(text) if feat in self.vocab)
            if not counts:
                continue
            length = sum(counts.values())
            for feat, count in counts.items():
                idx = self.vocab[feat]
                matrix[row, idx] = (1.0 + math.log(count)) / length * self.idf[idx]
        return _l2_normalize(matrix)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {"name": self.name, "vocab": list(self.vocab), "idf": self.idf.tolist()},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "HashingTfidfEmbedder":
        payload = json.loads(path.read_text(encoding="utf-8"))
        vocab = {feat: idx for idx, feat in enumerate(payload["vocab"])}
        embedder = cls(vocab=vocab, idf=np.array(payload["idf"], dtype=np.float32))
        embedder.dim = len(vocab)
        return embedder


class ApiEmbedder(BaseEmbedder):
    """OpenAI 兼容的云端向量化接口。"""

    name = "api"

    def __init__(self, base_url: str, model: str, api_key: str, timeout: float = 60.0, batch_size: int = 32):
        import httpx

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.batch_size = batch_size
        self._client = httpx.Client(timeout=timeout)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self._client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": batch},
            )
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda item: item.get("index", 0))
            vectors.extend(item["embedding"] for item in data)
        matrix = np.asarray(vectors, dtype=np.float32)
        self.dim = matrix.shape[1] if matrix.size else 0
        return _l2_normalize(matrix)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"name": self.name, "model": self.model, "dim": self.dim}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, base_url: str, api_key: str) -> "ApiEmbedder":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(base_url=base_url, model=payload["model"], api_key=api_key)


class SentenceTransformerEmbedder(BaseEmbedder):
    """本地 BGE 向量模型（可选依赖）。"""

    name = "local"

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer  # 可选依赖

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(matrix, dtype=np.float32)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"name": self.name, "model": self.model_name, "dim": self.dim}, ensure_ascii=False),
            encoding="utf-8",
        )


def build_embedder(provider: str, texts: list[str], *, embed_base_url: str = "", embed_model: str = "", embed_api_key: str = "", max_features: int = 4096) -> BaseEmbedder:
    """按 provider 构造向量化器，任何失败都回退到内置 TF-IDF。"""
    provider = (provider or "auto").lower()
    prefer_local, prefer_api = provider == "local", provider == "api"
    if provider == "auto":
        prefer_local, prefer_api = True, bool(embed_api_key)

    if prefer_local:
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            pass
    if prefer_api and embed_api_key:
        try:
            embedder = ApiEmbedder(embed_base_url, embed_model, embed_api_key)
            embedder.encode(["连通性测试"])
            return embedder
        except Exception:
            pass
    return HashingTfidfEmbedder(max_features=max_features).fit(texts)
