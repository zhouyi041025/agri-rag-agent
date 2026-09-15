import asyncio

import httpx
from fastapi.testclient import TestClient

from agri_agent.serving.api import create_app


def test_health_and_tools_endpoints():
    with TestClient(create_app()) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["kb"]["chunks"] > 0
        assert "search_knowledge" in health["tools"]

        tools = client.get("/tools").json()["tools"]
        assert any(item["function"]["name"] == "calc_spray_dosage" for item in tools)


def test_retrieve_endpoint_exposes_scores():
    with TestClient(create_app()) as client:
        payload = client.get("/retrieve", params={"q": "炭疽病 症状", "mode": "hybrid", "top_k": 3}).json()
        assert payload["mode"] == "hybrid"
        assert payload["results"]
        assert {"chunk_id", "score", "retriever", "section"} <= set(payload["results"][0])


def test_chat_endpoint_returns_answer_and_citations():
    with TestClient(create_app()) as client:
        response = client.post("/chat", json={"message": "芦荟炭疽病怎么防治？"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert body["citations"]
        assert body["agent"] in {"rule-based", "llm"}


def test_chat_stream_emits_step_and_final_events():
    with TestClient(create_app()) as client:
        with client.stream("POST", "/chat/stream", json={"message": "S3 现在田间环境怎么样？"}) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())
    assert '"type": "step"' in body or '"type":"step"' in body
    assert '"type": "final"' in body or '"type":"final"' in body
    assert "get_field_env" in body


def test_index_page_is_served():
    with TestClient(create_app()) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "荟诊" in response.text


def test_endpoints_work_without_lifespan():
    """Serverless / 冷启动场景下 lifespan 可能不触发，接口仍必须可用。

    用 httpx.ASGITransport 直接调用 ASGI app（不会执行 startup 事件），
    验证惰性初始化逻辑生效。
    """
    application = create_app()
    transport = httpx.ASGITransport(app=application)

    async def run():
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json()["kb"]["chunks"] > 0

            chat = await client.post("/chat", json={"message": "芦荟炭疽病怎么防治？"})
            assert chat.status_code == 200
            assert chat.json()["answer"]

    asyncio.run(run())
