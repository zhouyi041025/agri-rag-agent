"""FastAPI 服务：对话（含 SSE 流式）、检索调试、健康检查与内置 Web 界面。"""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..agent.agent import build_agent
from ..agent.tools import build_default_tools
from ..config import Settings, settings as default_settings
from ..llm import build_llm
from ..rag.pipeline import KnowledgeBase

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户问题")
    history: list[dict] = Field(default_factory=list, description="多轮历史，形如 [{'role':'user','content':'...'}]")


class RetrieveResponse(BaseModel):
    query: str
    mode: str
    results: list[dict]


def create_app(cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or default_settings

    _init_lock = threading.Lock()

    def _ensure_state(application: FastAPI) -> None:
        """惰性初始化知识库与工具，重复调用安全。

        Serverless / 冷启动环境下 lifespan 不一定被触发，因此每个入口都先调用它。
        初始化完成后写入 app.state._ready，后续调用直接返回。
        """
        if getattr(application.state, "_ready", False):
            return
        with _init_lock:
            if getattr(application.state, "_ready", False):
                return
            kb = KnowledgeBase.load_or_build(cfg)
            application.state.kb = kb
            application.state.tools = build_default_tools(kb, cfg)
            application.state.llm = build_llm(cfg)
            application.state.cfg = cfg
            application.state._ready = True

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _ensure_state(app)
        yield

    app = FastAPI(title="芦荟病虫害智能问答与诊断 Agent", version="0.3.0", lifespan=lifespan)

    def _new_agent():
        _ensure_state(app)
        return build_agent(cfg, kb=app.state.kb, tools=app.state.tools, llm=app.state.llm)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        page = STATIC_DIR / "index.html"
        if not page.exists():
            raise HTTPException(status_code=404, detail="未找到前端页面")
        return FileResponse(page)

    @app.get("/health")
    async def health() -> dict:
        _ensure_state(app)
        kb = app.state.kb
        return {
            "status": "ok",
            "agent": "llm" if app.state.llm is not None else "rule-based",
            "llm_model": cfg.llm_model if app.state.llm is not None else None,
            "kb": kb.stats,
            "tools": app.state.tools.names(),
        }

    @app.get("/tools")
    async def tools() -> dict:
        _ensure_state(app)
        return {"tools": app.state.tools.schemas()}

    @app.post("/chat")
    async def chat(payload: ChatRequest) -> dict:
        agent = _new_agent()
        answer = await asyncio.to_thread(agent.answer, payload.message, payload.history)
        return answer.to_dict()

    @app.post("/chat/stream")
    async def chat_stream(payload: ChatRequest) -> StreamingResponse:
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit(item: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def on_step(step: dict) -> None:
            emit({"type": "step", "step": step})

        def work() -> None:
            try:
                agent = _new_agent()
                answer = agent.answer(payload.message, payload.history, on_step=on_step)
                emit({"type": "final", "payload": answer.to_dict()})
            except Exception as exc:  # 把异常也推给前端，避免连接悬挂
                emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                emit({"type": "__end__"})

        threading.Thread(target=work, daemon=True).start()

        async def event_stream():
            while True:
                item = await queue.get()
                if item.get("type") == "__end__":
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/retrieve", response_model=RetrieveResponse)
    async def retrieve(
        q: str = Query(..., description="检索语句"),
        mode: str = Query("hybrid", pattern="^(hybrid|bm25|dense)$"),
        top_k: int = Query(5, ge=1, le=20),
    ) -> RetrieveResponse:
        _ensure_state(app)
        results = app.state.kb.search(q, top_k=top_k, mode=mode)
        return RetrieveResponse(query=q, mode=mode, results=[r.to_dict() for r in results])

    return app


app = create_app()
