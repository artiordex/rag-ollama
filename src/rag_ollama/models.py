# =============================================================================
# 파일명: models.py
# 경로: src/rag_ollama/models.py
# 목적: 문서 등록·검색·품질진단 API의 입력·출력 스키마 정의함
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""문서 등록·검색·품질진단 API의 입력·출력 스키마 정의함"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from .chunking import MAX_TEXT_CHARACTERS


class TextIngestRequest(BaseModel):
    """텍스트 문서 등록 요청 스키마임"""

    name: str = Field(min_length=1, max_length=512)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARACTERS)
    source_type: str = Field(default="text", max_length=64)
    mime_type: str | None = "text/plain"
    metadata: dict[str, Any] = Field(default_factory=dict)


class QualityIssue(BaseModel):
    """품질 진단에서 발견한 단일 이슈 스키마임"""

    severity: str
    code: str
    message: str
    value: float | int | str | None = None


class QualityReport(BaseModel):
    """문서 품질 점수·상태·지표·이슈를 담는 스키마임"""

    score: int
    status: str
    assessment_scope: str | None = None
    rule_set_version: str | None = None
    metrics: dict[str, float | int | str]
    issues: list[QualityIssue]


class IngestResponse(BaseModel):
    """문서 등록 결과와 중복·품질 요약을 담는 스키마임"""

    document_id: UUID
    name: str
    chunk_count: int
    duplicate: bool
    quality: QualityReport


class SourceHit(BaseModel):
    """검색된 단일 근거 청크의 순위와 품질 정보를 담는 스키마임"""

    rank: int
    document_id: UUID
    source_name: str
    chunk_index: int
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    quality_score: int | None = None


class QueryRequest(BaseModel):
    """RAG 질문과 검색·생성 조건을 담는 요청 스키마임"""

    question: str = Field(min_length=1, max_length=5_000)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: UUID | None = None
    min_quality_score: int | None = Field(default=None, ge=0, le=100)
    use_llm: bool = True


class QueryResponse(BaseModel):
    """RAG 답변과 검색 문맥·근거 목록을 담는 응답 스키마임"""

    answer: str | None
    context: str
    llm_configured: bool
    sources: list[SourceHit]


class QualityDiagnoseRequest(BaseModel):
    """인라인 텍스트 또는 저장 문서 품질진단 요청 스키마임"""

    document_id: UUID | None = None
    name: str = Field(default="inline", max_length=512)
    text: str | None = Field(default=None, max_length=MAX_TEXT_CHARACTERS)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    """문서 상세 메타데이터와 품질·청크 요약을 담는 스키마임"""

    document_id: UUID
    name: str
    source_type: str
    mime_type: str | None
    metadata: dict[str, Any]
    quality: QualityReport
    chunk_count: int
