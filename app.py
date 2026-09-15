"""线上部署入口：以 7860 端口（或 $PORT）启动服务。

容器平台与魔搭创空间默认探测 7860 端口；本地运行等价于
`uvicorn agri_agent.serving.api:app --app-dir src --host 0.0.0.0 --port 7860`。

用法：python app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import uvicorn  # noqa: E402


def main() -> None:
    port = int(os.environ.get("PORT") or os.environ.get("AGRI_PORT") or "7860")
    uvicorn.run(
        "agri_agent.serving.api:app",
        host="0.0.0.0",
        port=port,
        app_dir=str(ROOT / "src"),
        log_level="info",
    )


if __name__ == "__main__":
    main()
