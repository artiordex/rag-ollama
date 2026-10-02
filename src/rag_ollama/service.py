# =============================================================================
# 파일명: service.py
# 경로: src/rag_ollama/service.py
# 목적: 문서 인제스트·검색·LLM 답변·품질진단의 애플리케이션 서비스 제공함
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""문서 인제스트·검색·LLM 답변·품질진단의 애플리케이션 서비스 제공함"""

from __future__ import annotations

import hashlib
from typing import Any
from uuid import UUID

from .chunking import MAX_TEXT_CHARACTERS, chunk_text, normalize_text
from .config import Settings
from .db import find_document_by_hash, get_document, get_document_content, save_document, search_chunks
from .embeddings import get_embedder
from .llm import LLMClient
from .quality import diagnose_text


def _content_hash(text: str) -> str:
    """정규화된 본문을 중복 판별용 SHA-256 해시로 변환함"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ingest_text(
    settings: Settings,
    *,
    name: str,
    text: str,
    source_type: str,
    mime_type: str | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """텍스트를 정규화·진단·청킹·임베딩해 중복 없이 저장함

    Args:
        settings: 저장·임베딩 실행 설정임
        name: 원천 문서 이름임
        text: 저장할 원문 텍스트임
        source_type: 텍스트 또는 파일 등 입력 유형임
        mime_type: 원천 MIME 타입임
        metadata: 문서에 함께 저장할 메타데이터임

    Returns:
        dict[str, Any]: 문서 ID, 청크 수, 중복 여부와 품질 리포트임

    Raises:
        ValueError: 본문이 비어 있거나 입력·청킹 한도를 초과할 때 발생함
        DatabaseError: 저장소에서 문서·청크를 저장하지 못할 때 발생함

    Caveats:
        본문 해시가 같은 문서는 기존 결과를 반환해 재시도 시 중복 저장하지 않음
    """
    if len(text) > MAX_TEXT_CHARACTERS:
        raise ValueError(f"입력 텍스트가 허용 한도({MAX_TEXT_CHARACTERS}자)를 초과했습니다.")
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("추출된 텍스트가 비어 있습니다.")

    content_hash = _content_hash(normalized)
    existing = find_document_by_hash(settings, name, content_hash)
    quality = diagnose_text(normalized, name, metadata)
    if existing:
        return {
            "document_id": existing["id"],
            "name": existing["source_name"],
            "chunk_count": existing["chunk_count"],
            "duplicate": True,
            "quality": existing["quality_report"],
        }

    text_chunks = chunk_text(normalized, settings.chunk_size, settings.chunk_overlap)
    if not text_chunks:
        raise ValueError("문서를 청크로 나누지 못했습니다.")

    chunks = [
        {
            "index": chunk.index,
            "text": chunk.text,
            "metadata": {"char_start": chunk.start, "char_end": chunk.end},
        }
        for chunk in text_chunks
    ]
    embeddings = get_embedder(settings).embed_documents([chunk["text"] for chunk in chunks])
    document_id, duplicate, chunk_count = save_document(
        settings,
        source_name=name,
        source_type=source_type,
        mime_type=mime_type,
        content_hash=content_hash,
        content=normalized,
        metadata=metadata,
        quality_report=quality,
        chunks=chunks,
        embeddings=embeddings,
    )
    return {
        "document_id": document_id,
        "name": name,
        "chunk_count": chunk_count,
        "duplicate": duplicate,
        "quality": quality,
    }


def build_context(hits: list[dict[str, Any]]) -> str:
    """검색 결과를 답변 생성 프롬프트용 번호 매긴 문맥으로 조합함"""
    return "\n\n".join(
        f"[{index}] source={hit['source_name']} chunk={hit['chunk_index']}\n{hit['text']}"
        for index, hit in enumerate(hits, start=1)
    )


def query_rag(
    settings: Settings,
    *,
    question: str,
    top_k: int,
    document_id: UUID | None,
    min_quality_score: int | None,
    use_llm: bool,
) -> dict[str, Any]:
    """질문을 임베딩하고 pgvector 근거를 검색한 뒤 선택적으로 답변 생성함

    Args:
        settings: 검색·임베딩·LLM 실행 설정임
        question: 사용자의 검색 질문임
        top_k: 반환할 최대 근거 수임
        document_id: 지정하면 한 문서로 검색을 제한함
        min_quality_score: 지정하면 품질 점수 미달 문서를 제외함
        use_llm: 근거가 있을 때 생성 답변을 요청할지 여부임

    Returns:
        dict[str, Any]: 답변, 문맥, LLM 상태와 근거 목록임

    Raises:
        ValueError: 질문이 비어 있을 때 발생함
        DatabaseError: 검색 또는 저장소 조회에 실패할 때 발생함
        EmbeddingError: 질문 임베딩 계산에 실패할 때 발생함
        LLMError: 선택적 답변 생성에 실패할 때 발생함
    """
    question = question.strip()
    if not question:
        raise ValueError("질문이 비어 있습니다.")

    embedder = get_embedder(settings)
    query_embedding = embedder.embed_query(question)
    hits = search_chunks(
        settings,
        query_embedding,
        top_k=top_k,
        document_id=document_id,
        min_quality_score=min_quality_score,
    )
    context = build_context(hits)
    llm = LLMClient(settings)
    completion = llm.complete(question, context) if use_llm and hits else None
    return {
        "answer": completion.answer if completion else None,
        "context": context,
        "llm_configured": llm.configured,
        "generation": completion.to_dict() if completion else None,
        "sources": [
            {
                "rank": index,
                "document_id": hit["document_id"],
                "source_name": hit["source_name"],
                "chunk_index": hit["chunk_index"],
                "score": hit["score"],
                "text": hit["text"],
                "metadata": hit["metadata"],
                "quality_score": hit["quality_score"],
            }
            for index, hit in enumerate(hits, start=1)
        ],
    }


def document_summary(settings: Settings, document_id: UUID) -> dict[str, Any] | None:
    """문서 상세 조회 결과를 API 응답 형태로 축약함

    Returns:
        dict[str, Any] | None: 문서 요약 또는 대상이 없을 때 None임
    """
    row = get_document(settings, document_id)
    if row is None:
        return None
    return {
        "document_id": row["id"],
        "name": row["source_name"],
        "source_type": row["source_type"],
        "mime_type": row["mime_type"],
        "metadata": row["metadata"] or {},
        "quality": row["quality_report"],
        "chunk_count": row["chunk_count"],
    }


def diagnose_document_quality(settings: Settings, document_id: UUID) -> dict[str, Any] | None:
    """저장된 문서 본문에 현재 학습용 품질 baseline을 다시 적용함

    Returns:
        dict[str, Any] | None: 최신 품질 리포트 또는 대상이 없을 때 None임

    Caveats:
        기존 저장 리포트를 갱신하지 않고 현재 규칙 결과만 반환함
    """
    row = get_document_content(settings, document_id)
    if row is None:
        return None
    return diagnose_text(row["content"], row["source_name"], row.get("metadata") or {})
