"""Agent 工具集：检索、环境查询、用药计算、图像诊断、天气预报。

设计原则：
- 事实性知识 -> 走知识库检索，不让模型自由发挥；
- 数值计算 -> 走确定性代码，不让模型心算；
- 实时数据 -> 走数据源读取，不让模型假设。
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config import Settings, settings as default_settings
from ..rag.pipeline import KnowledgeBase


@dataclass
class ToolResult:
    ok: bool
    content: str
    data: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "content": self.content,
            "data": self.data,
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., ToolResult]
    needs_context: bool = False

    def run(self, arguments: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> ToolResult:
        kwargs = dict(arguments or {})
        if self.needs_context:
            kwargs["context"] = context or {}
        started = time.perf_counter()
        try:
            result = self.func(**kwargs)
        except Exception as exc:  # 工具异常不应打断整个 Agent 循环
            result = ToolResult(ok=False, content=f"工具执行失败：{type(exc).__name__}: {exc}")
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def call(self, name: str, arguments: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(ok=False, content=f"未注册的工具：{name}")
        return tool.run(arguments, context)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools


class CitationLedger:
    """全流程统一的引用编号台账，保证正文 [1][2] 与溯源面板一一对应。"""

    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []
        self._index: dict[str, int] = {}

    def add(self, chunk, score: float, retriever: str) -> int:
        if chunk.chunk_id in self._index:
            return self._index[chunk.chunk_id]
        number = len(self._items) + 1
        self._index[chunk.chunk_id] = number
        self._items.append(
            {
                "index": number,
                "chunk_id": chunk.chunk_id,
                "doc": chunk.title,
                "section": chunk.section,
                "score": round(float(score), 4),
                "retriever": retriever,
                "text": chunk.text,
            }
        )
        return number

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


# --------------------------------------------------------------------------- #
# 确定性业务数据
# --------------------------------------------------------------------------- #

ENV_THRESHOLDS = {
    "air_temp_c": (15.0, 32.0, "空气温度"),
    "air_humidity_pct": (50.0, 85.0, "空气湿度"),
    "soil_moisture_pct": (55.0, 65.0, "土壤含水率"),
    "soil_ph": (6.0, 7.2, "土壤 pH"),
    "light_lux": (3000.0, 45000.0, "光照强度"),
}

PRODUCTS: dict[str, dict[str, Any]] = {
    "代森锰锌": {"kind": "保护性杀菌剂", "dose_g_per_mu": 160, "water_l_per_mu": 45, "dilution": "600-800 倍", "safety_days": 15, "max_times": 3, "targets": ["炭疽病", "褐斑病", "黑斑病"]},
    "多菌灵": {"kind": "内吸性杀菌剂", "dose_g_per_mu": 100, "water_l_per_mu": 45, "dilution": "800-1000 倍", "safety_days": 20, "max_times": 2, "targets": ["炭疽病", "叶枯病", "根腐病"]},
    "苯醚甲环唑": {"kind": "三唑类内吸杀菌剂", "dose_g_per_mu": 30, "water_l_per_mu": 45, "dilution": "1500-2000 倍", "safety_days": 21, "max_times": 3, "targets": ["炭疽病", "黑斑病"]},
    "咪鲜胺": {"kind": "咪唑类杀菌剂", "dose_g_per_mu": 50, "water_l_per_mu": 45, "dilution": "1000-1500 倍", "safety_days": 14, "max_times": 3, "targets": ["炭疽病"]},
    "吡唑醚菌酯": {"kind": "甲氧基丙烯酸酯类杀菌剂", "dose_g_per_mu": 25, "water_l_per_mu": 45, "dilution": "2000-2500 倍", "safety_days": 14, "max_times": 3, "targets": ["炭疽病", "褐斑病"]},
    "春雷霉素": {"kind": "农用抗生素", "dose_g_per_mu": 40, "water_l_per_mu": 45, "dilution": "1000-1200 倍", "safety_days": 10, "max_times": 4, "targets": ["软腐病", "叶枯病"]},
}

SEVERITY_FACTOR = {"轻": 0.8, "轻度": 0.8, "中": 1.0, "中度": 1.0, "重": 1.2, "重度": 1.2}


def _read_env_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {"site_id": row.get("site_id", ""), "site_name": row.get("site_name", ""), "timestamp": row.get("timestamp", "")}
            for key in ENV_THRESHOLDS:
                try:
                    parsed[key] = float(row.get(key, ""))
                except (TypeError, ValueError):
                    parsed[key] = None
            rows.append(parsed)
    return rows


# --------------------------------------------------------------------------- #
# 工具实现
# --------------------------------------------------------------------------- #


def make_search_tool(kb: KnowledgeBase, default_top_k: int = 5) -> Tool:
    def search_knowledge(query: str, top_k: int = 0, context: dict[str, Any] | None = None) -> ToolResult:
        ledger: CitationLedger | None = (context or {}).get("ledger")
        results = kb.search(query, top_k=top_k or default_top_k)
        if not results:
            return ToolResult(ok=False, content="知识库中未检索到相关内容。", data={"hits": 0})

        blocks, hits = [], []
        for result in results:
            number = ledger.add(result.chunk, result.score, result.retriever) if ledger is not None else result.rank
            label = result.chunk.title + (f" · {result.chunk.section}" if result.chunk.section else "")
            blocks.append(f"[{number}] 来源：{label}\n{result.chunk.text}")
            hits.append(
                {
                    "index": number,
                    "chunk_id": result.chunk.chunk_id,
                    "doc": result.chunk.title,
                    "section": result.chunk.section,
                    "score": round(result.score, 4),
                }
            )

        return ToolResult(
            ok=True,
            content="检索到的知识库片段（回答时请用 [编号] 标注来源）：\n\n" + "\n\n".join(blocks),
            data={"hits": len(hits), "results": hits},
        )

    return Tool(
        name="search_knowledge",
        description="检索本地农业知识库，获取芦荟病虫害症状、发病条件、防治方法、栽培管理等事实性依据。回答专业问题前必须调用。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索查询，建议使用问题中的关键术语，如“炭疽病 症状 防治”"},
                "top_k": {"type": "integer", "description": "返回片段数量，默认 5"},
            },
            "required": ["query"],
        },
        func=search_knowledge,
        needs_context=True,
    )


def make_env_tool(cfg: Settings) -> Tool:
    path = Path(cfg.env_data_path)

    def get_field_env(site_id: str = "", hours: int = 24) -> ToolResult:
        rows = _read_env_rows(path)
        if not rows:
            return ToolResult(ok=False, content=f"环境数据文件缺失或为空：{path}")

        sites = {row["site_id"]: row["site_name"] for row in rows}
        site_id = site_id or next(iter(sites))
        site_rows = [row for row in rows if row["site_id"] == site_id]
        if not site_rows:
            return ToolResult(ok=False, content=f"未找到监测点 {site_id}，可用监测点：{', '.join(sites)}", data={"sites": sites})

        site_rows = site_rows[-max(1, hours // 6) :]
        latest = site_rows[-1]
        alerts, summary = [], {}
        for key, (low, high, label) in ENV_THRESHOLDS.items():
            values = [row[key] for row in site_rows if row.get(key) is not None]
            if not values:
                continue
            summary[key] = {"avg": round(sum(values) / len(values), 1), "min": round(min(values), 1), "max": round(max(values), 1)}
            value = latest.get(key)
            if value is None:
                continue
            if value < low:
                alerts.append(f"{label}偏低：当前 {value}（适宜 {low}-{high}）")
            elif value > high:
                alerts.append(f"{label}偏高：当前 {value}（适宜 {low}-{high}）")

        risk = "偏高" if (latest.get("air_temp_c") or 0) >= 26 and (latest.get("air_humidity_pct") or 0) >= 75 else "一般"
        lines = [
            f"监测点：{site_id} {site_rows[-1]['site_name']}（共 {len(site_rows)} 条最新记录）",
            f"最新采集时间：{latest['timestamp']}",
            "指标：" + "；".join(
                f"{ENV_THRESHOLDS[key][2]} {latest[key]}" for key in ENV_THRESHOLDS if latest.get(key) is not None
            ),
            f"近 {hours} 小时均值：" + "；".join(
                f"{ENV_THRESHOLDS[key][2]} {value['avg']}" for key, value in summary.items()
            ),
            "炭疽病流行风险：" + risk + ("（高温高湿，需重点巡田）" if risk == "偏高" else ""),
            "阈值告警：" + ("；".join(alerts) if alerts else "无"),
        ]
        return ToolResult(ok=True, content="\n".join(lines), data={"site_id": site_id, "latest": latest, "summary": summary, "alerts": alerts, "risk": risk})

    return Tool(
        name="get_field_env",
        description="读取田间物联网监测终端的最新环境数据，包括空气温湿度、土壤含水率、土壤 pH 与光照，并给出阈值告警与病害流行风险判断。",
        parameters={
            "type": "object",
            "properties": {
                "site_id": {"type": "string", "description": "监测点编号，如 S1/S2/S3/S4/S5，留空则取第一个"},
                "hours": {"type": "integer", "description": "统计窗口（小时），默认 24"},
            },
        },
        func=get_field_env,
    )


def make_dosage_tool() -> Tool:
    def calc_spray_dosage(area_mu: float, product: str, disease: str = "", severity: str = "中", sprayer_l_per_mu: float = 0.0) -> ToolResult:
        info = PRODUCTS.get(product)
        if info is None:
            return ToolResult(ok=False, content=f"药剂“{product}”不在登记用药清单中，可选：{'、'.join(PRODUCTS)}", data={"products": list(PRODUCTS)})

        factor = SEVERITY_FACTOR.get(str(severity).strip(), 1.0)
        dose_g = info["dose_g_per_mu"] * factor * area_mu
        water_l = (sprayer_l_per_mu or info["water_l_per_mu"]) * area_mu
        bucket_g = info["dose_g_per_mu"] * factor
        target_note = ""
        if disease and disease not in info["targets"]:
            target_note = f"注意：{product} 的登记靶标为 {', '.join(info['targets'])}，对「{disease}」并非首选，建议咨询当地植保站确认。"

        lines = [
            f"作物面积：{area_mu:g} 亩；防治对象：{disease or '未指定'}；病情程度：{severity}",
            f"药剂：{product}（{info['kind']}）  推荐稀释倍数：{info['dilution']}",
            f"亩用药量：{bucket_g:.1f} g（基准 {info['dose_g_per_mu']} g × 程度系数 {factor:g}）",
            f"总用药量：{dose_g:.1f} g；建议兑水量：{water_l:.0f} L（约 {water_l / 15:.0f} 喷雾器，按 15 L/桶计）",
            f"安全间隔期：{info['safety_days']} 天；每季最多使用 {info['max_times']} 次，注意与不同作用机理药剂轮换以延缓抗性",
        ]
        if target_note:
            lines.append(target_note)
        lines.append("提示：以上为按登记用量推算的参考值，实际施药请以产品标签与当地植保部门指导为准。")

        return ToolResult(
            ok=True,
            content="\n".join(lines),
            data={
                "product": product,
                "dose_g": round(dose_g, 1),
                "water_l": round(water_l, 1),
                "safety_days": info["safety_days"],
                "dilution": info["dilution"],
                "warning": target_note,
            },
        )

    return Tool(
        name="calc_spray_dosage",
        description="按面积、药剂与病情程度计算用药量和兑水量，并给出安全间隔期与轮换建议。涉及数值计算时必须调用，不要自行估算。",
        parameters={
            "type": "object",
            "properties": {
                "area_mu": {"type": "number", "description": "施药面积（亩）"},
                "product": {"type": "string", "description": "药剂名称，如 代森锰锌、多菌灵、苯醚甲环唑"},
                "disease": {"type": "string", "description": "防治对象，如 炭疽病"},
                "severity": {"type": "string", "description": "病情程度：轻 / 中 / 重"},
                "sprayer_l_per_mu": {"type": "number", "description": "每亩实际用水量（升），留空取默认值"},
            },
            "required": ["area_mu", "product"],
        },
        func=calc_spray_dosage,
    )


def make_diagnose_tool(cfg: Settings) -> Tool:
    model_path = Path(cfg.project_root) / "models" / "leaf_disease.pt"

    def diagnose_leaf_image(image_path: str = "", site_id: str = "") -> ToolResult:
        path = Path(image_path) if image_path else None
        if path and not path.exists():
            return ToolResult(ok=False, content=f"图像不存在：{image_path}")

        if path and model_path.exists():
            try:
                from ultralytics import YOLO  # 可选依赖

                model = YOLO(str(model_path))
                result = model.predict(str(path), verbose=False)[0]
                names = result.names
                detections = [
                    {"class": names[int(box.cls)], "confidence": round(float(box.conf), 3)}
                    for box in result.boxes
                ]
                return ToolResult(ok=True, content=f"检测到 {len(detections)} 个目标：{json.dumps(detections, ensure_ascii=False)}", data={"source": "model", "detections": detections})
            except Exception:
                pass

        placeholder = {
            "source": "placeholder",
            "note": "未部署本地检测模型（models/leaf_disease.pt），以下为演示用结构化返回；接入 YOLOv11 检测/分割模型后自动切换为真实推理结果。",
            "site_id": site_id or "S1",
            "top_k": [
                {"disease": "炭疽病", "confidence": 0.86, "lesion_ratio": 0.17, "grade": "中度"},
                {"disease": "褐斑病", "confidence": 0.09, "lesion_ratio": 0.05, "grade": "轻度"},
                {"disease": "健康", "confidence": 0.05, "lesion_ratio": 0.0, "grade": "-"},
            ],
        }
        lines = [
            "图像诊断结果（演示数据，未接入本地模型）：",
            *[f"  {item['disease']}：置信度 {item['confidence']:.2f}，病斑面积占比 {item['lesion_ratio']:.0%}，分级 {item['grade']}" for item in placeholder["top_k"]],
            "建议：结合下方知识库中的防治方案与田间环境数据制定处置措施。",
        ]
        return ToolResult(ok=True, content="\n".join(lines), data=placeholder)

    return Tool(
        name="diagnose_leaf_image",
        description="对叶片照片做病害识别与分级，返回病害类别、置信度和病斑面积占比。用户提供图片路径或描述图片时调用。",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "本地图片路径"},
                "site_id": {"type": "string", "description": "对应监测点编号，可选"},
            },
        },
        func=diagnose_leaf_image,
    )


def make_forecast_tool() -> Tool:
    def get_forecast(site_id: str = "S1", days: int = 3) -> ToolResult:
        days = max(1, min(days, 7))
        payload = {
            "source": "offline-stub",
            "site_id": site_id,
            "days": [
                {"date": f"D+{offset + 1}", "weather": ["多云", "阵雨", "晴"][offset % 3], "temp_c": [27, 24, 30][offset % 3], "rain_mm": [0, 12, 0][offset % 3]}
                for offset in range(days)
            ],
        }
        lines = [f"监测点 {site_id} 未来 {days} 天天气（离线示例数据）："]
        for day in payload["days"]:
            lines.append(f"  {day['date']}：{day['weather']}，{day['temp_c']}℃，降水 {day['rain_mm']} mm")
        lines.append("说明：离线模式下不访问外部天气服务，接入天气 API 后此处为真实预报。")
        return ToolResult(ok=True, content="\n".join(lines), data=payload)

    return Tool(
        name="get_forecast",
        description="查询监测点未来天气预报，用于判断施药窗口（降雨前后不宜施药）。",
        parameters={
            "type": "object",
            "properties": {
                "site_id": {"type": "string", "description": "监测点编号，如 S1"},
                "days": {"type": "integer", "description": "预报天数，默认 3，最多 7"},
            },
        },
        func=get_forecast,
    )


def build_default_tools(kb: KnowledgeBase, cfg: Settings | None = None) -> ToolRegistry:
    cfg = cfg or default_settings
    registry = ToolRegistry()
    registry.register(make_search_tool(kb, default_top_k=cfg.top_k))
    registry.register(make_env_tool(cfg))
    registry.register(make_dosage_tool())
    registry.register(make_diagnose_tool(cfg))
    registry.register(make_forecast_tool())
    return registry
