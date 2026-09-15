"""集中配置：环境变量 / .env 读取，全局单例 Settings。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path | None = None) -> None:
    """极简 .env 解析（避免为一个小项目引入额外依赖）。"""
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _get(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def _get_int(key: str, default: int) -> int:
    try:
        return int(_get(key, str(default)))
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    try:
        return float(_get(key, str(default)))
    except ValueError:
        return default


def _get_bool(key: str, default: bool) -> bool:
    return _get(key, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    project_root: Path = PROJECT_ROOT
    kb_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "kb")
    env_data_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "env" / "field_env.csv")
    artifacts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "index")

    llm_provider: str = "offline"
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_api_key: str = ""
    llm_temperature: float = 0.2
    llm_timeout: float = 60.0

    embed_provider: str = "auto"
    embed_base_url: str = "https://api.siliconflow.cn/v1"
    embed_model: str = "BAAI/bge-m3"
    embed_api_key: str = ""
    embed_dim: int = 4096

    chunk_strategy: str = "heading"
    chunk_max_chars: int = 420
    chunk_overlap: int = 80
    top_k: int = 5
    rrf_k: int = 60
    use_rerank: bool = True
    max_agent_steps: int = 6

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            llm_provider=_get("AGRI_LLM_PROVIDER", "offline").lower(),
            llm_base_url=_get("AGRI_LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/"),
            llm_model=_get("AGRI_LLM_MODEL", "deepseek-chat"),
            llm_api_key=_get("AGRI_LLM_API_KEY", ""),
            llm_temperature=_get_float("AGRI_LLM_TEMPERATURE", 0.2),
            embed_provider=_get("AGRI_EMBED_PROVIDER", "auto").lower(),
            embed_base_url=_get("AGRI_EMBED_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/"),
            embed_model=_get("AGRI_EMBED_MODEL", "BAAI/bge-m3"),
            embed_api_key=_get("AGRI_EMBED_API_KEY", ""),
            embed_dim=_get_int("AGRI_EMBED_DIM", 4096),
            chunk_strategy=_get("AGRI_CHUNK_STRATEGY", "heading").lower(),
            chunk_max_chars=_get_int("AGRI_CHUNK_MAX_CHARS", 420),
            chunk_overlap=_get_int("AGRI_CHUNK_OVERLAP", 80),
            top_k=_get_int("AGRI_TOP_K", 5),
            rrf_k=_get_int("AGRI_RRF_K", 60),
            use_rerank=_get_bool("AGRI_USE_RERANK", True),
            max_agent_steps=_get_int("AGRI_MAX_AGENT_STEPS", 6),
        )

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider not in {"offline", "none"} and bool(self.llm_api_key)


settings = Settings.from_env()
