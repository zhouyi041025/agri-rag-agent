"""生成端评测：衡量答案的引用正确性、忠实度与拒答行为。

与 run_eval.py（只评检索）互补。指标口径：

- 引用正确率：答案里出现的 [n] 全部能在 citations 里找到，且 citations
  里的编号都在正文出现。这是「幽灵引用 / 幻觉编号」的直接检测。
- 引用覆盖率：至少标注了一个合法引用的答案占比。
- 忠实度（规则近似）：答案里带引用标记的句子，其关键词能在被引片段的
  原文中找到。纯规则实现，不依赖额外模型；适合做回归而非绝对质量评分。
- 拒答正确率：对明确超出知识库范围的问题，答案应给出「暂未收录」类提示
  而不是编造内容。

用法：
    python eval/run_gen_eval.py
    python eval/run_gen_eval.py --json eval/reports/gen_eval.json

注意：默认用 OfflineAgent（无需 API Key、无需网络），结果可复现。
若要评 LLMAgent，需自行配置 Key，且指标波动会明显更大。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agri_agent.agent.agent import build_agent  # noqa: E402
from agri_agent.agent.tools import build_default_tools  # noqa: E402
from agri_agent.config import settings  # noqa: E402
from agri_agent.rag.pipeline import KnowledgeBase  # noqa: E402
from agri_agent.text import tokenize  # noqa: E402

CITATION_RE = re.compile(r"\[(\d+)\]")
REFUSAL_MARKERS = ("暂未收录", "不做推测", "未检索到")

# 知识库未覆盖的话题，用于测拒答行为
OUT_OF_SCOPE = [
    "火星上怎么种芦荟",
    "芦荟 stocks 今天涨跌如何",
    "帮我写一段 Python 快速排序代码",
]


def citation_correctness(answer_text: str, citations: list[dict]) -> bool:
    in_text = {int(number) for number in CITATION_RE.findall(answer_text)}
    in_ledger = {int(item["index"]) for item in citations}
    return in_text == in_ledger


def faithfulness(answer_text: str, citations: list[dict]) -> tuple[int, int]:
    """带引用句子的关键词在被引原文中的命中情况，返回 (命中数, 检查数)。"""
    by_index = {int(item["index"]): item for item in citations}
    checked = supported = 0
    for line in answer_text.splitlines():
        match = CITATION_RE.search(line)
        if not match:
            continue
        item = by_index.get(int(match.group(1)))
        if item is None:
            continue
        source_tokens = set(tokenize(item.get("text", "")))
        answer_tokens = {token for token in tokenize(line) if len(token) > 1}
        if not answer_tokens:
            continue
        checked += 1
        overlap = len(answer_tokens & source_tokens) / len(answer_tokens)
        if overlap >= 0.3:
            supported += 1
    return supported, checked


def refusal_correct(answer_text: str) -> bool:
    return any(marker in answer_text for marker in REFUSAL_MARKERS)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行生成端评测")
    parser.add_argument("--json", type=Path, help="将明细报告写入指定 JSON 文件")
    args = parser.parse_args()

    kb = KnowledgeBase.load_or_build(settings)
    tools = build_default_tools(kb, settings)
    # 显式禁用 LLM，使用可复现的离线降级策略
    agent = build_agent(settings, kb=kb, tools=tools, llm=None)

    cases = [
        json.loads(line)
        for line in (ROOT / "eval" / "testset.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    details: list[dict] = []
    correct = covered = supported = checked = 0
    for case in cases:
        answer = agent.answer(case["question"], use_cache=False)
        ok = citation_correctness(answer.answer, answer.citations)
        hit, total = faithfulness(answer.answer, answer.citations)
        correct += int(ok)
        covered += int(bool(answer.citations))
        supported += hit
        checked += total
        details.append(
            {
                "id": case["id"],
                "question": case["question"],
                "citation_correct": ok,
                "citation_count": len(answer.citations),
                "faithfulness": round(hit / total, 3) if total else None,
            }
        )

    refused = 0
    refusal_details: list[dict] = []
    for question in OUT_OF_SCOPE:
        answer = agent.answer(question, use_cache=False)
        ok = refusal_correct(answer.answer)
        refused += int(ok)
        refusal_details.append({"question": question, "refused": ok})

    total = len(cases)
    report = {
        "agent": "offline (rule-based)",
        "sample_count": total,
        "citation_correctness": round(correct / total, 3),
        "citation_coverage": round(covered / total, 3),
        "faithfulness_rule_based": round(supported / checked, 3) if checked else None,
        "refusal_correctness": round(refused / len(OUT_OF_SCOPE), 3),
        "note": "忠实度为规则近似指标，用于版本间回归对比，不代表绝对事实正确率。",
        "details": details,
        "refusal_details": refusal_details,
    }

    print(f"样本数: {total}")
    print(f"引用正确率: {report['citation_correctness']:.1%}")
    print(f"引用覆盖率: {report['citation_coverage']:.1%}")
    print(f"忠实度(规则): {report['faithfulness_rule_based']}")
    print(f"拒答正确率: {report['refusal_correctness']:.1%}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"明细已写入 {args.json}")


if __name__ == "__main__":
    main()
