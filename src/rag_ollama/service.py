"""Application services shared by the HTTP API and future workers."""

from __future__ import annotations

import hashlib
from typing import Any
from uuid import UUID

from .chunking import chunk_text, normalize_text
from .config import Settings
from .db import find_document_by_hash, get_document, save_document, search_chunks
from .embeddings import get_embedder
from .llm import LLMClient
from .quality import diagnose_text


def _content_hash(text: str) -> str:
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
    answer = llm.complete(question, context) if use_llm and hits else None
    return {
        "answer": answer,
        "context": context,
        "llm_configured": llm.configured,
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
