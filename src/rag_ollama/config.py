"""Environment-driven configuration for rag-ollama."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - useful before dependencies are installed
    pass


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    database_connect_timeout: int
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    embedding_device: str
    embedding_use_fp16: bool
    embedding_batch_size: int
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    llm_base_url: str | None
    llm_api_key: str | None
    llm_model: str | None
    llm_timeout_seconds: float
    max_upload_bytes: int
    api_key: str | None
    auto_init_db: bool

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=os.getenv(
                "DATABASE_URL",
                "postgresql://raglab:raglab-dev-only@localhost:5432/raglab",
            ),
            database_connect_timeout=_int_env("DATABASE_CONNECT_TIMEOUT", 3),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "flag").strip().lower(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            embedding_dim=_int_env("EMBEDDING_DIM", 1024),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            embedding_use_fp16=_bool_env("EMBEDDING_USE_FP16", False),
            embedding_batch_size=_int_env("EMBEDDING_BATCH_SIZE", 8),
            chunk_size=_int_env("CHUNK_SIZE", 1600),
            chunk_overlap=_int_env("CHUNK_OVERLAP", 240),
            retrieval_top_k=_int_env("RETRIEVAL_TOP_K", 5),
            llm_base_url=os.getenv("LLM_BASE_URL") or None,
            llm_api_key=os.getenv("LLM_API_KEY") or None,
            llm_model=os.getenv("LLM_MODEL") or None,
            llm_timeout_seconds=_float_env("LLM_TIMEOUT_SECONDS", 60.0),
            max_upload_bytes=_int_env("MAX_UPLOAD_BYTES", 20 * 1024 * 1024),
            api_key=os.getenv("RAG_LAB_API_KEY") or None,
            auto_init_db=_bool_env("AUTO_INIT_DB", True),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
