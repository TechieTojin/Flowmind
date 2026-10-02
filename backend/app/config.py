"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent

# Load backend/.env if present. Real environment variables take precedence.
load_dotenv(BACKEND_DIR / ".env")


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    app_name: str
    cors_origins: tuple[str, ...]
    ollama_base_url: str
    ollama_model: str
    # Timeouts in seconds.
    ollama_connect_timeout: float
    ollama_status_timeout: float
    ollama_generation_timeout: float


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name="FlowMind API",
        cors_origins=("http://localhost:5173",),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        ollama_connect_timeout=_float_env("OLLAMA_CONNECT_TIMEOUT", 5.0),
        ollama_status_timeout=_float_env("OLLAMA_STATUS_TIMEOUT", 10.0),
        # CPU inference of qwen3:8b can easily exceed 60s, so be generous.
        ollama_generation_timeout=_float_env("OLLAMA_GENERATION_TIMEOUT", 300.0),
    )
