# =============================================================================
# 파일명: config.py
# 경로: src/rag_ollama/config.py
# 목적: 환경변수를 애플리케이션 설정으로 변환하고 입력 제약 검증함
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""환경변수를 rag-ollama 설정으로 변환하고 입력 제약 검증함"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # NOTE: 의존성 설치 전에도 설정 모듈을 불러올 수 있도록 선택 처리함
    pass


def _bool_env(name: str, default: bool) -> bool:
    """환경변수의 대표적인 참 값 표현을 bool로 변환함"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    """환경변수의 정수값을 변환하고 잘못된 입력을 즉시 거부함"""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _float_env(name: str, default: float) -> float:
    """환경변수의 실수값을 변환하고 잘못된 입력을 즉시 거부함"""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


@dataclass(frozen=True, slots=True)
class Settings:
    """RAG 저장·임베딩·검색·LLM 실행에 필요한 불변 설정 모음임"""

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

    def __post_init__(self) -> None:
        """인제스트나 검색을 사용할 수 없게 만드는 설정을 조기 검증함"""

        if self.database_connect_timeout <= 0:
            raise ValueError("DATABASE_CONNECT_TIMEOUT must be positive")
        if self.embedding_provider not in {"flag", "hash"}:
            raise ValueError("EMBEDDING_PROVIDER must be 'flag' or 'hash'")
        if self.embedding_dim <= 0:
            raise ValueError("EMBEDDING_DIM must be positive")
        if self.embedding_batch_size <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE must be positive")
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be between 0 and CHUNK_SIZE - 1")
        if self.retrieval_top_k <= 0:
            raise ValueError("RETRIEVAL_TOP_K must be positive")
        if self.llm_timeout_seconds <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS must be positive")
        if self.max_upload_bytes <= 0:
            raise ValueError("MAX_UPLOAD_BYTES must be positive")

    @classmethod
    def from_env(cls) -> "Settings":
        """현재 프로세스 환경변수와 기본값으로 설정 객체를 생성함

        Returns:
            Settings: 유효성 검증을 마친 애플리케이션 설정임

        Raises:
            ValueError: 숫자 형식이나 서비스 제약을 만족하지 못할 때 발생함
        """
        return cls(
            database_url=os.getenv(
                "DATABASE_URL",
                "postgresql://raglab:raglab-dev-only@localhost:55432/raglab",
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
    """프로세스 전체에서 재사용할 설정 객체를 반환함

    Caveats:
        환경변수 변경은 캐시를 지우거나 프로세스를 재시작해야 반영됨
    """
    return Settings.from_env()
