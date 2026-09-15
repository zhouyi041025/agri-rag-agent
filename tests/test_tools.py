import json

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
