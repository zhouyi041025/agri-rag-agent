"""检索效果评测：对比分块策略与检索通道，输出可复现的指标报告。

指标口径：
- Recall@k / Hit@k：前 k 个片段中，是否存在同时命中该题全部关键词的片段
  （比「命中同一文档」严格，能真实反映"这段话能不能回答问题"）；
- MRR：第一个相关片段的排名倒数均值；
- Union@k：前 k 个片段合起来是否覆盖全部关键词（衡量上下文完整性）。

用法：
    python eval/run_eval.py
    python eval/run_eval.py --strategy fixed --retriever bm25
    python eval/run_eval.py --grid            # 跑完整对比矩阵
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agri_agent.config import settings  # noqa: E402
from agri_agent.rag.pipeline import KnowledgeBase  # noqa: E402

TOPK = [1, 3, 5]


@dataclass
class CaseResult:
    case_id: str
    question: str
    hits: dict[int, bool] = field(default_factory=dict)
    union: dict[int, bool] = field(default_factory=dict)
    first_rank: int | None = None
    top_sections: list[str] = field(default_factory=list)


def load_testset(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def evaluate(kb: KnowledgeBase, cases: list[dict], mode: str = "hybrid", top_k: int = 5) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in cases:
        keywords = case["keywords"]
        ranked = kb.search(case["question"], top_k=top_k, mode=mode)
        case_result = CaseResult(case_id=case["id"], question=case["question"])
        case_result.top_sections = [r.chunk.section or r.chunk.title for r in ranked]

        for k in TOPK:
            subset = ranked[:k]
            case_result.hits[k] = any(all(kw in r.chunk.text for kw in keywords) for r in subset)
            merged = "\n".join(r.chunk.text for r in subset)
            case_result.union[k] = all(kw in merged for kw in keywords)

        for rank, result in enumerate(ranked, start=1):
            if all(kw in result.chunk.text for kw in keywords):
                case_result.first_rank = rank
                break
        results.append(case_result)
    return results


def summarize(results: list[CaseResult]) -> dict:
    total = len(results)
    summary = {f"recall@{k}": sum(r.hits[k] for r in results) / total for k in TOPK}
    summary.update({f"union@{k}": sum(r.union[k] for r in results) / total for k in TOPK})
    reciprocal = [1.0 / r.first_rank if r.first_rank else 0.0 for r in results]
    summary["mrr"] = sum(reciprocal) / total
    summary["miss"] = [r.case_id for r in results if not r.hits[max(TOPK)]]
    summary["total"] = total
    return summary


def format_report(label: str, summary: dict, results: list[CaseResult], cases_meta: dict) -> str:
    lines = [f"## {label}", ""]
    lines.append(f"- 用例数：{summary['total']}")
    lines.append(f"- Recall@1 / @3 / @5：{summary['recall@1']:.3f} / {summary['recall@3']:.3f} / {summary['recall@5']:.3f}")
    lines.append(f"- Union@1 / @3 / @5：{summary['union@1']:.3f} / {summary['union@3']:.3f} / {summary['union@5']:.3f}")
    lines.append(f"- MRR：{summary['mrr']:.3f}")
    if summary["miss"]:
        lines.append(f"- 未命中用例：{', '.join(summary['miss'])}")
    lines.append("")
    lines.append("| 用例 | 问题 | Recall@3 | 首个相关排名 | Top3 命中章节 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for result in results:
        rank = str(result.first_rank) if result.first_rank else "未命中"
        top3 = " / ".join(result.top_sections[:3])
        lines.append(f"| {result.case_id} | {cases_meta[result.case_id]['question']} | {'✅' if result.hits[3] else '❌'} | {rank} | {top3} |")
    lines.append("")
    return "\n".join(lines)


def run_once(strategy: str, retriever: str, cases: list[dict], top_k: int) -> tuple[dict, list[CaseResult]]:
    cfg = settings
    cfg.chunk_strategy = strategy
    kb = KnowledgeBase.build(strategy=strategy, embed_provider="tfidf", cfg=cfg)
    results = evaluate(kb, cases, mode=retriever, top_k=top_k)
    return summarize(results), results


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 检索评测")
    parser.add_argument("--testset", default=str(ROOT / "eval" / "testset.jsonl"))
    parser.add_argument("--strategy", default="heading", choices=["heading", "fixed"])
    parser.add_argument("--retriever", default="hybrid", choices=["hybrid", "bm25", "dense"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--grid", action="store_true", help="运行 分块策略 × 检索通道 的对比矩阵")
    parser.add_argument("--report", default=str(ROOT / "eval" / "reports"))
    args = parser.parse_args()

    cases = load_testset(Path(args.testset))
    cases_meta = {case["id"]: case for case in cases}
    report_dir = Path(args.report)
    report_dir.mkdir(parents=True, exist_ok=True)

    combos = (
        [(s, r) for s in ("heading", "fixed") for r in ("bm25", "dense", "hybrid")]
        if args.grid
        else [(args.strategy, args.retriever)]
    )

    markdown = ["# 检索效果评测报告", "", f"用例数：{len(cases)}，评测脚本：`eval/run_eval.py`", ""]
    metrics_dump = {}
    for strategy, retriever in combos:
        summary, results = run_once(strategy, retriever, cases, args.top_k)
        label = f"chunk={strategy} / retriever={retriever}"
        markdown.append(format_report(label, summary, results, cases_meta))
        metrics_dump[label] = {k: v for k, v in summary.items() if k != "miss"}
        print(
            f"{label:34s} Recall@1={summary['recall@1']:.3f} Recall@3={summary['recall@3']:.3f} "
            f"Recall@5={summary['recall@5']:.3f} MRR={summary['mrr']:.3f}"
        )

    (report_dir / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    (report_dir / "metrics.json").write_text(json.dumps(metrics_dump, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 {report_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
