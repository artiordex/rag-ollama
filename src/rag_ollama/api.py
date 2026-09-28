"""FastAPI application for ingestion, retrieval, and quality diagnostics."""

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
from .parsers import ParseError, parse_document
from .quality import diagnose_text
from .service import document_summary, ingest_text, query_rag

logger = logging.getLogger(__name__)
settings = get_settings()


def _require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="X-API-Key가 필요합니다.")


def _service_error(exc: Exception) -> HTTPException:
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
    if settings.auto_init_db:
        try:
            init_db(settings)
        except DatabaseError as exc:
            logger.warning("DB 자동 초기화를 건너뜁니다: %s", exc)
    yield


app = FastAPI(
    title="rag-ollama API",
    version="0.1.0",
    description="Unstructured document ingestion, quality diagnostics, and pgvector RAG retrieval.",
    lifespan=lifespan,
)
router = APIRouter(dependencies=[Depends(_require_api_key)])


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        database_ok = check_db(settings)
    except DatabaseError as exc:
        return {"status": "degraded", "database": "down", "detail": str(exc)}
    return {
        "status": "ok" if database_ok else "degraded",
        "database": "up" if database_ok else "down",
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "llm_configured": bool(settings.llm_base_url and settings.llm_model),
    }


@router.post("/documents/text", response_model=IngestResponse)
def ingest_text_endpoint(request: TextIngestRequest) -> IngestResponse:
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
    payload = await file.read()
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"파일은 {settings.max_upload_bytes}바이트 이하만 허용됩니다.")
    try:
        parsed_metadata = json.loads(metadata)
        if not isinstance(parsed_metadata, dict):
            raise ValueError("metadata는 JSON object여야 합니다.")
        parsed = parse_document(file.filename or "uploaded-file", payload, file.content_type)
        merged_metadata = {**parsed.metadata, **parsed_metadata}
        return ingest_text(
            settings,
            name=file.filename or "uploaded-file",
            text=parsed.text,
            source_type="file",
            mime_type=parsed.mime_type,
            metadata=merged_metadata,
        )
    except Exception as exc:
        raise _service_error(exc) from exc


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document_endpoint(document_id: UUID) -> DocumentResponse:
    try:
        result = document_summary(settings, document_id)
    except Exception as exc:
        raise _service_error(exc) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="문서를 찾지 못했습니다.")
    return result


@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest) -> QueryResponse:
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
    if request.document_id is not None:
        try:
            result = document_summary(settings, request.document_id)
        except Exception as exc:
            raise _service_error(exc) from exc
        if result is None:
            raise HTTPException(status_code=404, detail="문서를 찾지 못했습니다.")
        return result["quality"]
    if request.text is None:
        raise HTTPException(status_code=422, detail="document_id 또는 text 중 하나는 필요합니다.")
    return diagnose_text(request.text, request.name, request.metadata)


app.include_router(router)


def run() -> None:
    import uvicorn

    uvicorn.run(
        "rag_ollama.api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )
