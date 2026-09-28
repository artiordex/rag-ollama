"""API request and response models."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class TextIngestRequest(BaseModel):
    name: str = Field(min_length=1, max_length=512)
    text: str = Field(min_length=1)
    source_type: str = Field(default="text", max_length=64)
    mime_type: str | None = "text/plain"
    metadata: dict[str, Any] = Field(default_factory=dict)


class QualityIssue(BaseModel):
    severity: str
    code: str
    message: str
    value: float | int | str | None = None


class QualityReport(BaseModel):
    score: int
    status: str
    metrics: dict[str, float | int | str]
    issues: list[QualityIssue]


class IngestResponse(BaseModel):
    document_id: UUID
    name: str
    chunk_count: int
    duplicate: bool
    quality: QualityReport


class SourceHit(BaseModel):
    rank: int
    document_id: UUID
    source_name: str
    chunk_index: int
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    quality_score: int | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: UUID | None = None
    min_quality_score: int | None = Field(default=None, ge=0, le=100)
    use_llm: bool = True


class QueryResponse(BaseModel):
    answer: str | None
    context: str
    llm_configured: bool
    sources: list[SourceHit]


class QualityDiagnoseRequest(BaseModel):
    document_id: UUID | None = None
    name: str = Field(default="inline", max_length=512)
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    document_id: UUID
    name: str
    source_type: str
    mime_type: str | None
    metadata: dict[str, Any]
    quality: QualityReport
    chunk_count: int
