"""两种 Agent 执行策略，共用同一套工具与引用台账。

- LLMAgent：Function Calling 循环，模型自主决定调哪个工具、调几次；
- OfflineAgent：无 API Key 时的降级实现，规则路由 + 抽取式回答。

两者输出结构完全一致（答案 + 引用 + 执行轨迹），因此前端、评测、
测试都不需要关心底层用的是哪种策略——这也是能对「无 Key 也能演示」的原因。
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..config import Settings, settings as default_settings
from ..llm import BaseLLM, build_llm
from ..rag.pipeline import KnowledgeBase
from ..text import split_sentences, tokenize
from .tools import CitationLedger, PRODUCTS, ToolRegistry, build_default_tools

SYSTEM_PROMPT = """你是「荟诊」，服务芦荟种植户的 AI 农技助手。你必须严守以下规则：

1. 事实性内容（病虫害症状、发病条件、防治方法、栽培管理）必须先调用 search_knowledge 检索本地知识库，回答只能基于检索结果，并用 [1][2] 标注来源编号。
2. 涉及用药量、兑水量、面积换算时必须调用 calc_spray_dosage，禁止自行心算。
3. 涉及当前田间状况必须先调用 get_field_env 取监测数据，禁止自行假设温湿度。
4. 用户提供叶片照片路径或要求看图时调用 diagnose_leaf_image，并结合知识库给出处置建议。
5. 判断施药窗口时调用 get_forecast 查看降雨情况。
6. 知识库没有覆盖的内容，直接说明「知识库暂未收录」，不要编造农药名称、剂量或安全间隔期。
7. 回答结构：先给结论，再给依据（带引用编号），最后给 2-4 条可执行的农事动作。
8. 涉及高毒农药、超范围用药或大面积病害暴发时，提醒用户咨询当地植保站。回答仅供参考，不替代属地农技人员现场诊断。"""

DISEASES = ["炭疽病", "褐斑病", "黑斑病", "叶枯病", "根腐病", "白绢病", "软腐病", "煤烟病"]


@dataclass
class AgentAnswer:
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    agent: str = ""
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "steps": self.steps,
            "agent": self.agent,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


class BaseAgent:
    name = "base"

    def __init__(self, kb: KnowledgeBase, tools: ToolRegistry, cfg: Settings | None = None):
        self.kb = kb
        self.tools = tools
        self.cfg = cfg or default_settings
        self._on_step = None

    def _execute(self, name: str, arguments: dict[str, Any], ledger: CitationLedger, steps: list[dict[str, Any]]) -> Any:
        result = self.tools.call(name, arguments, context={"ledger": ledger})
        record = {
            "type": "tool",
            "name": name,
            "arguments": arguments,
            "ok": result.ok,
            "latency_ms": round(result.latency_ms, 1),
            "preview": result.content[:240],
        }
        steps.append(record)
        if self._on_step:
            try:
                self._on_step(record)
            except Exception:
                pass
        return result

    @staticmethod
    def _finalize(answer_text: str, ledger: CitationLedger) -> tuple[str, list[dict[str, Any]]]:
        """只保留答案里真正引用到的片段，并重排编号。

        检索阶段会一次性把 top_k 个片段登记进台账（LLM 需要看到带编号的上下文），
        但最终回答通常只引用其中几条。若不裁剪，溯源面板会出现正文里根本没提过的
        「幽灵引用」，也会出现 [2] 这种断号。
        """
        items = ledger.snapshot()
        used = sorted({int(match) for match in re.findall(r"\[(\d+)\]", answer_text)})
        by_index = {item["index"]: item for item in items}
        keep = [number for number in used if number in by_index]
        if not keep or len(keep) == len(items):
            return answer_text, items

        mapping: dict[int, int] = {}
        renumbered: list[dict[str, Any]] = []
        for new_number, old_number in enumerate(keep, start=1):
            mapping[old_number] = new_number
            item = dict(by_index[old_number])
            item["index"] = new_number
            renumbered.append(item)

        rewritten = re.sub(r"\[(\d+)\]", lambda m: f"[{mapping.get(int(m.group(1)), int(m.group(1)))}]", answer_text)
        return rewritten, renumbered

    def answer(self, question: str, history: list[dict[str, str]] | None = None, on_step=None) -> AgentAnswer:
        """对外统一入口；on_step 用于把工具调用过程实时推给前端。"""
        self._on_step = on_step
        try:
            return self._answer(question, history or [])
        finally:
            self._on_step = None

    def _answer(self, question: str, history: list[dict[str, str]]) -> AgentAnswer:  # pragma: no cover
        raise NotImplementedError


class LLMAgent(BaseAgent):
    name = "llm"

    def __init__(self, kb: KnowledgeBase, tools: ToolRegistry, llm: BaseLLM, cfg: Settings | None = None):
        super().__init__(kb, tools, cfg)
        self.llm = llm

    def _answer(self, question: str, history: list[dict[str, str]]) -> AgentAnswer:
        started = time.perf_counter()
        ledger = CitationLedger()
        steps: list[dict[str, Any]] = []

        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history[-6:]:
            if turn.get("role") in {"user", "assistant"} and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": question})

        answer_text = ""
        for _ in range(max(1, self.cfg.max_agent_steps)):
            response = self.llm.chat(messages, tools=self.tools.schemas(), temperature=self.cfg.llm_temperature)
            if not response.tool_calls:
                answer_text = response.content.strip()
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [call.raw for call in response.tool_calls if call.raw],
                }
            )
            for call in response.tool_calls:
                result = self._execute(call.name, call.arguments, ledger, steps)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result.content})
        else:
            messages.append({"role": "user", "content": "请基于已有工具结果直接给出最终回答，并标注引用编号。"})
            response = self.llm.chat(messages, tools=None, temperature=self.cfg.llm_temperature)
            answer_text = response.content.strip()

        if not answer_text:
            answer_text = "未能生成回答，请重试或换一种问法。"

        answer_text, citations = self._finalize(answer_text, ledger)
        return AgentAnswer(
            answer=answer_text,
            citations=citations,
            steps=steps,
            agent=self.name,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )


class OfflineAgent(BaseAgent):
    """规则路由 + 抽取式问答，用于无大模型环境下的可运行演示与自动化测试。"""

    name = "rule-based"

    _AREA_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:亩|mu)")
    _SITE_RE = re.compile(r"\b(S[1-5])\b", re.IGNORECASE)
    _IMAGE_RE = re.compile(r"[\w\\/:.\\-]+\.(?:jpg|jpeg|png|bmp|webp)", re.IGNORECASE)

    _FORECAST_KEYWORDS = ("天气", "降雨", "下雨", "预报", "未来几天", "能不能打药")
    _DOSAGE_KEYWORDS = ("剂量", "用量", "兑水", "施药量", "配比", "稀释", "打多少", "多少量", "多少克", "多少毫升", "用多少", "几克", "几毫升")
    _IMAGE_KEYWORDS = ("图片", "照片", "看图", "识别一下", "病斑", "叶片图", "拍了一张", "叶子有斑")
    _ENV_KEYWORDS = ("环境", "温湿度", "墒情", "田间情况", "田间数据", "大棚", "监测点", "实时数据", "传感器", "现在多少")

    # 省略式追问：短句且以指代词/连词开头时，语义依赖上一轮
    _FOLLOWUP_HINTS = ("那", "这", "它", "该", "此", "上述", "刚才", "那么", "还有", "再", "另外", "顺便")
    _FOLLOWUP_MAX_LEN = 20
    # 追问自带领域词的 IDF 占比达到该值，就认为它能独立检索，不必借用上文
    _DOMAIN_TERM_RATIO = 0.5

    # 关键词覆盖率阈值：低于 MIN 直接弃答，低于 CONFIDENT 给出匹配度提示
    MIN_COVERAGE = 0.42
    CONFIDENT_COVERAGE = 0.62
    _idf_ceiling: float | None = None

    def _answer(self, question: str, history: list[dict[str, str]]) -> AgentAnswer:
        started = time.perf_counter()
        ledger = CitationLedger()
        steps: list[dict[str, Any]] = []
        text = question.strip()
        query, borrowed = self._rewrite_query(text, history)

        if self._match(query, self._FORECAST_KEYWORDS):
            site = self._site(text)
            result = self._execute("get_forecast", {"site_id": site}, ledger, steps)
            answer = (
                f"【天气与施药窗口】\n{result.content}\n\n"
                "施药建议：降雨前后 24 小时内不宜施药；雨后及时排水降湿，并抓住降雨间隙补施保护性杀菌剂。"
            )

        elif self._is_dosage_question(query):
            answer = self._handle_dosage(query, ledger, steps)

        elif self._match(text, self._IMAGE_KEYWORDS) or self._IMAGE_RE.search(text):
            match = self._IMAGE_RE.search(text)
            result = self._execute(
                "diagnose_leaf_image",
                {"image_path": match.group(0) if match else "", "site_id": self._site(text)},
                ledger,
                steps,
            )
            before = len(ledger)
            self._execute("search_knowledge", {"query": "炭疽病 症状 识别", "top_k": 3}, ledger, steps)
            answer = (
                f"【图像诊断】\n{result.content}\n\n"
                f"{self._format_citations(ledger, since=before, question=text)}\n\n"
                "建议动作：1) 摘除病叶并带出田外销毁；2) 降低田间湿度、改善通风；"
                "3) 发病初期选用保护性杀菌剂，并与内吸性药剂轮换；4) 3-5 天后复查新叶。"
            )

        elif self._match(query, self._ENV_KEYWORDS):
            site = self._site(text)
            result = self._execute("get_field_env", {"site_id": site, "hours": 24}, ledger, steps)
            before = len(ledger)
            self._execute("search_knowledge", {"query": "炭疽病 发病条件", "top_k": 2}, ledger, steps)
            answer = (
                f"【田间环境】\n{result.content}\n\n"
                f"{self._format_citations(ledger, since=before, question=text)}"
            )

        else:
            answer = self._handle_knowledge(query, ledger, steps)

        if borrowed:
            answer = (
                f"【多轮理解】本轮追问「{text}」不含可检索的实词，"
                f"已结合上一轮问题「{borrowed[:24]}」补全检索意图。\n\n{answer}"
            )

        answer, citations = self._finalize(answer, ledger)
        return AgentAnswer(
            answer=answer,
            citations=citations,
            steps=steps,
            agent=self.name,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    # ---------------- 内部实现 ----------------
    @staticmethod
    def _match(text: str, keywords: Iterable[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    def _is_follow_up(self, text: str) -> bool:
        """是否是依赖上一轮的省略式追问，如「那这种药要打几次」。"""
        stripped = text.strip().strip("？?。！!，,、 ")
        if not stripped or len(stripped) > self._FOLLOWUP_MAX_LEN:
            return False
        return stripped.startswith(self._FOLLOWUP_HINTS)

    def _has_domain_terms(self, text: str) -> bool:
        """追问自身是否带够分量的领域词。

        只要有一个知识库认识的实词就算数，但像「这个多久用一次」虽然撞上了
        「一次」这种泛词，IDF 占比很低，仍判为依赖上文的追问。
        """
        terms = self._query_terms(text)
        if not terms:
            return False
        total = sum(self._term_idf(term) for term in terms)
        known = sum(self.kb.sparse._idf(term) for term in terms if self.kb.sparse._idf(term) > 0)
        return total > 0 and known / total >= self._DOMAIN_TERM_RATIO

    def _rewrite_query(self, text: str, history: list[dict[str, str]]) -> tuple[str, str]:
        """把省略式追问补全，返回（用于检索的问题, 被借用的上一轮问题）。

        「那这种药要打几次」这类追问本身没有可检索的实词，直接拿去检索会因
        覆盖率过低而弃答。此时沿用上一轮的检索意图，多轮对话才真正接得上。
        """
        previous = ""
        for message in reversed(history or []):
            if str(message.get("role")) == "user":
                candidate = str(message.get("content") or "").strip()
                if candidate:
                    previous = candidate
                    break
        if not previous or not self._is_follow_up(text) or self._has_domain_terms(text):
            return text, ""
        return previous, previous

    def _is_dosage_question(self, text: str) -> bool:
        if self._match(text, self._DOSAGE_KEYWORDS):
            return True
        return bool(self._AREA_RE.search(text)) and any(name in text for name in PRODUCTS)

    def _site(self, text: str) -> str:
        match = self._SITE_RE.search(text)
        return match.group(1).upper() if match else "S1"

    def _chunk_of(self, chunk_id: str):
        for chunk in self.kb.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        raise KeyError(chunk_id)

    @staticmethod
    def _query_terms(text: str) -> set[str]:
        """保留有区分度的查询词：过滤单字虚词与停用词。"""
        return {term for term in tokenize(text) if len(term) > 1}

    def _max_idf(self) -> float:
        if self._idf_ceiling is None:
            total = max(1, len(self.kb.chunks))
            type(self)._idf_ceiling = math.log(1 + (total - 0.5) / 1.5)
        return type(self)._idf_ceiling

    def _term_idf(self, term: str) -> float:
        """语料中未出现的词按最高 IDF 处理，让「问了个知识库完全没有的词」能被识别出来。"""
        value = self.kb.sparse._idf(term)
        return value if value > 0 else self._max_idf()

    def _coverage(self, query_terms: set[str], text: str) -> float:
        text_terms = set(tokenize(text))
        weights = {term: self._term_idf(term) for term in query_terms}
        total = sum(weights.values())
        if total <= 0:
            return 0.0
        return sum(weight for term, weight in weights.items() if term in text_terms) / total

    def _format_citations(self, ledger: CitationLedger, since: int = 0, question: str = "") -> str:
        """把本轮新产生的引用整理成「依据」段落，每条引用挑一句最相关的原文。"""
        items = ledger.snapshot()[since:]
        if not items:
            return "【依据】\n未检索到可引用的知识库片段。"

        query_terms = self._query_terms(question)
        lines = ["【依据】"]
        for item in items[:3]:
            try:
                chunk = self._chunk_of(item["chunk_id"])
            except KeyError:
                continue
            sentences = [s for s in split_sentences(chunk.text) if len(s) >= 8]
            if query_terms and sentences:
                sentences.sort(key=lambda s: -self._coverage(query_terms, f"{chunk.title} {chunk.section} {s}"))
            picked = sentences[0] if sentences else chunk.text[:120]
            label = f"{chunk.title} · {chunk.section}" if chunk.section else chunk.title
            lines.append(f"[{item['index']}] {label}：{picked}")
        return "\n".join(lines)

    @staticmethod
    def _no_answer() -> str:
        return (
            "知识库中暂未收录与该问题直接相关的内容，为避免给出错误建议，这里不做推测。\n\n"
            "可以尝试：① 换用具体病害名称提问（如「炭疽病怎么防治」）；② 提供叶片照片做图像诊断；"
            "③ 查询田间环境数据判断病害风险。"
        )

    def _handle_dosage(self, text: str, ledger: CitationLedger, steps: list[dict[str, Any]]) -> str:
        area_match = self._AREA_RE.search(text)
        if not area_match:
            return "请补充施药面积（亩），例如：我有 5 亩芦荟，防治炭疽病，代森锰锌要用多少？"

        product = next((name for name in PRODUCTS if name in text), "")
        if not product:
            return f"请指定药剂名称，当前登记清单包含：{'、'.join(PRODUCTS)}。"

        severity = next((level for level in ("轻度", "中度", "重度", "轻", "中", "重") if level in text), "中")
        disease = next((name for name in DISEASES if name in text), "")
        result = self._execute(
            "calc_spray_dosage",
            {
                "area_mu": float(area_match.group(1)),
                "product": product,
                "disease": disease,
                "severity": severity,
            },
            ledger,
            steps,
        )
        before = len(ledger)
        self._execute(
            "search_knowledge",
            {"query": f"{disease or '病害'} 施药 注意事项 安全间隔期 抗性", "top_k": 2},
            ledger,
            steps,
        )
        return f"【用药方案】\n{result.content}\n\n{self._format_citations(ledger, since=before, question=text)}"

    def _handle_knowledge(self, question: str, ledger: CitationLedger, steps: list[dict[str, Any]]) -> str:
        result = self._execute("search_knowledge", {"query": question, "top_k": self.cfg.top_k}, ledger, steps)
        hits = (result.data or {}).get("results") or []
        if not result.ok or not hits:
            return self._no_answer()

        query_terms = self._query_terms(question)
        if not query_terms:
            return "请补充更具体的信息，例如病害名称、种植场景或施药面积。"

        scored: list[tuple[float, int, str]] = []
        for item in hits:
            try:
                chunk = self._chunk_of(item["chunk_id"])
            except KeyError:
                continue
            label = f"{chunk.title} {chunk.section}"
            for sentence in split_sentences(chunk.text):
                if len(sentence) < 8:
                    continue
                scored.append((self._coverage(query_terms, f"{label} {sentence}"), item["index"], sentence))

        if not scored:
            return self._no_answer()
        scored.sort(key=lambda triple: (-triple[0], triple[1]))
        best = [item for item in scored if item[0] > 0][:4]

        if not best or best[0][0] < self.MIN_COVERAGE:
            return self._no_answer()

        lines = [f"[{index}] {sentence}" for _, index, sentence in best]
        verified = len({index for _, index, _ in best})
        note = ""
        if best[0][0] < self.CONFIDENT_COVERAGE:
            note = "\n\n提示：该问题与知识库的直接匹配度一般，建议补充具体的病害名称或种植场景，以便给出更准确的建议。"
        return (
            "【知识库回答】\n" + "\n".join(lines)
            + f"\n\n以上内容来自 {verified} 条知识库片段，已标注来源编号，可在溯源面板查看原文。" + note
        )


def build_agent(
    cfg: Settings | None = None,
    kb: KnowledgeBase | None = None,
    tools: ToolRegistry | None = None,
    llm: BaseLLM | None = None,
) -> BaseAgent:
    cfg = cfg or default_settings
    kb = kb or KnowledgeBase.load_or_build(cfg)
    tools = tools or build_default_tools(kb, cfg)
    llm = llm if llm is not None else build_llm(cfg)
    if llm is not None:
        return LLMAgent(kb, tools, llm, cfg)
    return OfflineAgent(kb, tools, cfg)
