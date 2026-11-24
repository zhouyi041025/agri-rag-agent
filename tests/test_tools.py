
from agri_agent.agent.tools import CitationLedger, build_default_tools


def _tools(kb):
    return build_default_tools(kb)


def test_registry_exposes_openai_tool_schemas(kb):
    tools = _tools(kb)
    schemas = tools.schemas()
    names = {schema["function"]["name"] for schema in schemas}
    assert {"search_knowledge", "get_field_env", "calc_spray_dosage", "diagnose_leaf_image", "get_forecast"} == names
    assert all(schema["type"] == "function" for schema in schemas)


def test_search_tool_registers_citations(kb):
    tools = _tools(kb)
    ledger = CitationLedger()
    result = tools.call("search_knowledge", {"query": "炭疽病 防治"}, context={"ledger": ledger})
    assert result.ok
    assert len(ledger) > 0
    assert "[1]" in result.content


def test_dosage_calculation_is_deterministic(kb):
    tools = _tools(kb)
    result = tools.call("calc_spray_dosage", {"area_mu": 5, "product": "代森锰锌", "disease": "炭疽病", "severity": "中"})
    assert result.ok
    assert result.data["dose_g"] == 800.0
    assert result.data["water_l"] == 225.0
    assert result.data["safety_days"] == 15


def test_dosage_severity_scaling_and_unknown_product(kb):
    tools = _tools(kb)
    mild = tools.call("calc_spray_dosage", {"area_mu": 1, "product": "多菌灵", "severity": "轻"})
    severe = tools.call("calc_spray_dosage", {"area_mu": 1, "product": "多菌灵", "severity": "重"})
    assert mild.data["dose_g"] < severe.data["dose_g"]

    unknown = tools.call("calc_spray_dosage", {"area_mu": 1, "product": "百草枯"})
    assert not unknown.ok
    assert "清单" in unknown.content


def test_env_tool_reads_csv_and_reports_alerts(kb):
    tools = _tools(kb)
    result = tools.call("get_field_env", {"site_id": "S3", "hours": 24})
    assert result.ok
    assert result.data["site_id"] == "S3"
    assert "炭疽病流行风险" in result.content
    assert isinstance(result.data["alerts"], list)


def test_env_tool_handles_unknown_site(kb):
    tools = _tools(kb)
    result = tools.call("get_field_env", {"site_id": "S99"})
    assert not result.ok
    assert "可用监测点" in result.content


def test_tool_exception_is_captured(kb):
    tools = _tools(kb)
    result = tools.call("calc_spray_dosage", {"area_mu": "五亩", "product": "多菌灵"})
    assert not result.ok
    assert "工具执行失败" in result.content


def test_forecast_tool_offline_payload(kb):
    tools = _tools(kb)
    result = tools.call("get_forecast", {"site_id": "S1", "days": 3})
    assert result.ok
    assert result.data["source"] == "offline-stub"
    assert len(result.data["days"]) == 3


def test_diagnose_falls_back_to_placeholder_without_model(kb):
    """没有模型文件时必须回退到演示数据，并在 note 里说清接入方式。"""
    tools = _tools(kb)
    result = tools.call("diagnose_leaf_image", {"image_path": ""})
    assert result.ok
    assert result.data["source"] == "placeholder"
    assert "ONNX" in result.data["note"]


def test_diagnose_reports_missing_image(kb):
    tools = _tools(kb)
    result = tools.call("diagnose_leaf_image", {"image_path": "不存在的图片.png"})
    assert not result.ok
    assert "图像不存在" in result.content


def test_diagnose_with_unloadable_onnx_falls_back(tmp_path):
    """放了一个坏 .onnx 也不能抛异常：推理失败要回退到占位实现。"""
    from agri_agent.agent import tools as tools_module
    from agri_agent.config import Settings

    model = tmp_path / "leaf_disease.onnx"
    model.write_bytes(b"not-a-real-onnx")
    image = tmp_path / "leaf.png"
    image.write_bytes(b"not-a-real-image")

    cfg = Settings()
    cfg.leaf_model_path = str(model)
    tool = tools_module.make_diagnose_tool(cfg)

    result = tool.run({"image_path": str(image)})
    assert result.ok
    assert result.data["source"] == "placeholder"


def test_products_are_externalized_with_source_fields():
    """用药数据应来自外置配置且每条都带来源字段，便于审计。"""
    from agri_agent.agent.tools import PRODUCTS, PRODUCTS_META, load_products

    assert PRODUCTS, "用药清单不应为空"
    assert PRODUCTS_META.get("version"), "配置应带数据版本"
    assert any("待核实" in str(item.get("source", "")) for item in PRODUCTS.values())
    for name, item in PRODUCTS.items():
        assert "source" in item, f"{name} 缺少 source 字段"
        assert "dose_g_per_mu" in item


def test_load_products_rejects_empty_file(tmp_path):
    """配置损坏时要明确报错，不能静默回退到硬编码数据。"""
    import json
    import pytest
    from agri_agent.agent.tools import load_products

    empty = tmp_path / "products.json"
    empty.write_text(json.dumps({"products": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="为空"):
        load_products(empty)


def test_gen_eval_metrics_are_computable(tmp_path):
    """生成端评测的核心指标函数应能直接调用，避免脚本腐烂。"""
    from eval.run_gen_eval import citation_correctness, refusal_correct

    assert citation_correctness("回答内容[1]和[2]", [{"index": 1}, {"index": 2}]) is True
    assert citation_correctness("回答内容[9]", [{"index": 1}]) is False
    assert citation_correctness("无引用内容", [{"index": 1}]) is False
    assert refusal_correct("知识库中暂未收录相关内容") is True
    assert refusal_correct("建议您使用代森锰锌") is False
