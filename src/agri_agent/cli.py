"""命令行入口：构建索引、直接提问、检索调试、交互式对话。

用法：
    python -m agri_agent.cli build
    python -m agri_agent.cli ask "芦荟炭疽病怎么防治？"
    python -m agri_agent.cli retrieve "炭疽病 发病条件" --mode hybrid
    python -m agri_agent.cli chat
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import settings
from .rag.pipeline import KnowledgeBase


def cmd_build(args: argparse.Namespace) -> int:
    kb = KnowledgeBase.build(strategy=args.strategy, embed_provider=args.embed_provider, cfg=settings)
    kb.save(cfg=settings)
    print(json.dumps(kb.stats, ensure_ascii=False, indent=2))
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    kb = KnowledgeBase.load_or_build(settings)
    for result in kb.search(args.query, top_k=args.top_k, mode=args.mode):
        label = f"{result.chunk.title} · {result.chunk.section}" if result.chunk.section else result.chunk.title
        print(f"[{result.rank}] score={result.score:.4f} ({result.retriever}) {label}")
        print(f"    {result.chunk.text[:160]}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .agent.agent import build_agent

    agent = build_agent(settings)
    answer = agent.answer(args.query)
    print(answer.answer)
    if answer.citations:
        print("\n--- 引用来源 ---")
        for item in answer.citations:
            label = f"{item['doc']} · {item['section']}" if item.get("section") else item["doc"]
            print(f"[{item['index']}] {label} (score={item['score']}, {item['retriever']})")
    if args.trace and answer.steps:
        print("\n--- 工具调用轨迹 ---")
        for step in answer.steps:
            print(f"  {step['name']}({step['arguments']}) ok={step['ok']} {step['latency_ms']}ms")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from .agent.agent import build_agent

    agent = build_agent(settings)
    history: list[dict[str, str]] = []
    print(f"已进入对话模式（agent={agent.name}），输入 exit 退出。")
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", "退出"}:
            break
        if not question:
            continue
        answer = agent.answer(question, history=history)
        print(f"\n助手：{answer.answer}")
        history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer.answer}])
        history = history[-6:]
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agri-agent", description="芦荟病虫害智能问答与诊断 Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="构建索引")
    build.add_argument("--strategy", default=settings.chunk_strategy, choices=["heading", "fixed"])
    build.add_argument("--embed-provider", default=settings.embed_provider, choices=["auto", "local", "api", "tfidf"])
    build.set_defaults(func=cmd_build)

    ask = sub.add_parser("ask", help="单次提问")
    ask.add_argument("query")
    ask.add_argument("--trace", action="store_true", help="打印工具调用轨迹")
    ask.set_defaults(func=cmd_ask)

    retrieve = sub.add_parser("retrieve", help="仅做检索，便于调参")
    retrieve.add_argument("query")
    retrieve.add_argument("--mode", default="hybrid", choices=["hybrid", "bm25", "dense"])
    retrieve.add_argument("--top-k", type=int, default=5)
    retrieve.set_defaults(func=cmd_retrieve)

    chat = sub.add_parser("chat", help="交互式对话")
    chat.set_defaults(func=cmd_chat)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
