"""构建并持久化检索索引。

用法：
    python scripts/build_index.py
    python scripts/build_index.py --strategy fixed --embed-provider tfidf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agri_agent.config import settings  # noqa: E402
from agri_agent.rag.pipeline import KnowledgeBase  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="构建芦荟病虫害知识库索引")
    parser.add_argument("--strategy", default=settings.chunk_strategy, choices=["heading", "fixed"])
    parser.add_argument("--embed-provider", default=settings.embed_provider, choices=["auto", "local", "api", "tfidf"])
    parser.add_argument("--max-chars", type=int, default=settings.chunk_max_chars)
    parser.add_argument("--overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--out", default=str(settings.artifacts_dir))
    args = parser.parse_args()

    settings.chunk_max_chars = args.max_chars
    settings.chunk_overlap = args.overlap

    kb = KnowledgeBase.build(strategy=args.strategy, embed_provider=args.embed_provider, cfg=settings)
    kb.save(args.out, cfg=settings)

    print("索引构建完成")
    for key, value in kb.stats.items():
        print(f"  {key}: {value}")
    print(f"  输出目录: {args.out}")

    benchmark = ["炭疽病 症状 识别", "炭疽病 发病条件", "安全间隔期"]
    for query in benchmark:
        top = kb.search(query, top_k=2)
        print(f"  自检 [{query}] -> " + " / ".join(f"{r.chunk.section or r.chunk.title}({r.score:.4f})" for r in top))


if __name__ == "__main__":
    main()
