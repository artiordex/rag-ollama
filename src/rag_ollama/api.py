# =============================================================================
# 파일명: api.py
# 경로: src/rag_ollama/api.py
# 목적: 문서 등록·검색·품질진단 FastAPI 엔드포인트와 수명주기 관리함
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""문서 등록·검색·품질진단 FastAPI 엔드포인트와 수명주기 관리함"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

from .config import get_settings
from .db import DatabaseError, check_db, init_db
from .embeddings import EmbeddingError
from .llm import LLMError
from .models import (
    DocumentResponse,
    IngestResponse,
    QualityDiagnoseRequest,
    QualityReport,
    QueryRequest,
    QueryResponse,
    TextIngestRequest,
)
from .parsers import MAX_DOCUMENT_INPUT_BYTES, ParseError, parse_document
from .quality import diagnose_text
from .service import diagnose_document_quality, document_summary, ingest_text, query_rag

logger = logging.getLogger(__name__)
settings = get_settings()


def _require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """설정된 API 키가 있을 때 요청 헤더를 검증함

    Raises:
        HTTPException: 키가 없거나 설정값과 다를 때 발생함
    """
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="X-API-Key가 필요합니다.")


def _service_error(exc: Exception) -> HTTPException:
    """내부 예외를 외부에 노출 가능한 HTTP 오류로 변환함

    Returns:
        HTTPException: 오류 종류에 대응하는 상태 코드와 안전한 메시지임

    Caveats:
        매핑되지 않은 예외는 상세 내용을 숨기고 서버 로그에만 기록함
    """
    if isinstance(exc, ParseError):
        return HTTPException(status_code=415, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, (DatabaseError, EmbeddingError, LLMError)):
        return HTTPException(status_code=503, detail=str(exc))
    logger.exception("rag-ollama request failed")
    return HTTPException(status_code=500, detail="요청을 처리하지 못했습니다.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """애플리케이션 시작 시 설정된 경우 PostgreSQL 스키마를 초기화함

    Caveats:
        DB 자동 초기화 실패는 API 프로세스를 중단하지 않고 상태 확인에서
        degraded로 표시하도록 위임함
    """
    if settings.auto_init_db:
        try:
            init_db(settings)
        except DatabaseError as exc:
            logger.warning("DB 자동 초기화를 건너뜁니다: %s", exc)
    yield


app = FastAPI(
    title="rag-ollama API",
    version="0.1.0",
    description="Hugging Face Transformers model experiments, document diagnostics, and pgvector RAG retrieval.",
    lifespan=lifespan,
)
router = APIRouter(dependencies=[Depends(_require_api_key)])


@app.get("/health")
def health() -> dict[str, Any]:
    """DB와 임베딩·LLM 설정의 현재 상태를 반환함

    Returns:
        dict[str, Any]: 서비스 상태, DB 연결 상태, 모델 설정 요약임
    """
    try:
        database_ok = check_db(settings)
    except DatabaseError as exc:
        return {"status": "degraded", "database": "down", "detail": str(exc)}
    return {
        "status": "ok" if database_ok else "degraded",
        "database": "up" if database_ok else "down",
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "llm_backend": settings.llm_backend,
        "llm_configured": settings.llm_configured,
    }


@router.post("/documents/text", response_model=IngestResponse)
def ingest_text_endpoint(request: TextIngestRequest) -> IngestResponse:
    """텍스트 본문을 정규화·진단·청킹·임베딩 후 저장함

    Args:
        request: 문서명, 본문, 원천 유형, MIME과 추가 메타데이터임

    Returns:
        IngestResponse: 문서 ID, 중복 여부, 청크 수와 품질 리포트임

    Raises:
        HTTPException: 입력·DB·임베딩 오류를 API 오류로 변환해 발생함
    """
    try:
        return ingest_text(
            settings,
            name=request.name,
            text=request.text,
            source_type=request.source_type,
            mime_type=request.mime_type,
            metadata=request.metadata,
        )
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/documents/file", response_model=IngestResponse)
async def ingest_file_endpoint(
    file: UploadFile = File(...),
    metadata: str = Form(default="{}"),
) -> IngestResponse:
    """업로드 파일을 제한된 메모리로 읽고 텍스트 문서로 저장함

    Args:
        file: 지원 확장자의 업로드 파일임
        metadata: 호출자가 전달한 JSON object 문자열임

    Returns:
        IngestResponse: 파싱·품질진단·인제스트 결과임

    Raises:
        HTTPException: 크기·형식·메타데이터·저장 오류에 대응해 발생함

    Caveats:
        파서가 추출한 메타데이터가 호출자 값보다 우선하며, 업로드 스트림은
        요청 종료 시 닫힘
    """
    try:
        # SECURITY: 허용량보다 한 바이트만 더 읽어 초과 multipart 본문이 무제한 메모리에 복사되지 않게 함
        upload_limit = min(settings.max_upload_bytes, MAX_DOCUMENT_INPUT_BYTES)
        payload = await file.read(upload_limit + 1)
        if len(payload) > upload_limit:
            raise HTTPException(status_code=413, detail=f"파일은 {upload_limit}바이트 이하만 허용됩니다.")
        parsed_metadata = json.loads(metadata)
        if not isinstance(parsed_metadata, dict):
            raise ValueError("metadata는 JSON object여야 합니다.")
        parsed = parse_document(file.filename or "uploaded-file", payload, file.content_type)
        # NOTE: 같은 키가 충돌하면 호출자 입력보다 파서가 확인한 파일 사실값을 우선함
        merged_metadata = {**parsed_metadata, **parsed.metadata}
        return ingest_text(
            settings,
            name=file.filename or "uploaded-file",
            text=parsed.text,
            source_type="file",
            mime_type=parsed.mime_type,
            metadata=merged_metadata,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _service_error(exc) from exc
    finally:
        await file.close()


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document_endpoint(document_id: UUID) -> DocumentResponse:
    """문서 요약과 품질 리포트를 ID로 조회함

    Args:
        document_id: 조회할 문서 UUID임

    Returns:
        DocumentResponse: 문서 메타데이터와 청크·품질 요약임

    Raises:
        HTTPException: 문서가 없거나 저장소 조회에 실패할 때 발생함
    """
    try:
        result = document_summary(settings, document_id)
    except Exception as exc:
        raise _service_error(exc) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="문서를 찾지 못했습니다.")
    return result


@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest) -> QueryResponse:
    """질문을 임베딩해 pgvector에서 근거를 검색하고 선택적으로 답변 생성함

    Args:
        request: 질문, 검색 범위, 품질 하한과 LLM 사용 여부임

    Returns:
        QueryResponse: 근거 청크, 유사도와 선택적 LLM 답변임

    Raises:
        HTTPException: 질문 검증·임베딩·DB·LLM 오류에 대응해 발생함
    """
    try:
        return query_rag(
            settings,
            question=request.question,
            top_k=request.top_k,
            document_id=request.document_id,
            min_quality_score=request.min_quality_score,
            use_llm=request.use_llm,
        )
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/quality/diagnose", response_model=QualityReport)
def quality_diagnose_endpoint(request: QualityDiagnoseRequest) -> QualityReport:
    """인라인 텍스트 또는 저장 문서의 현재 품질 baseline을 계산함

    Args:
        request: 문서 ID 또는 텍스트 중 하나와 진단 메타데이터임

    Returns:
        QualityReport: 점수, 상태, 지표와 품질 이슈 목록임

    Raises:
        HTTPException: 두 입력을 함께 주거나 대상 문서를 찾지 못할 때 발생함
    """
    if request.document_id is not None and request.text is not None:
        raise HTTPException(status_code=422, detail="document_id와 text 중 하나만 지정해야 합니다.")
    if request.document_id is not None:
        try:
            result = diagnose_document_quality(settings, request.document_id)
        except Exception as exc:
            raise _service_error(exc) from exc
        if result is None:
            raise HTTPException(status_code=404, detail="문서를 찾지 못했습니다.")
        return result
    if request.text is None:
        raise HTTPException(status_code=422, detail="document_id 또는 text 중 하나는 필요합니다.")
    return diagnose_text(request.text, request.name, request.metadata)


app.include_router(router)


def run() -> None:
    """환경 설정의 주소와 포트로 uvicorn 서버를 실행함"""
    import uvicorn

    uvicorn.run(
        "rag_ollama.api:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )
