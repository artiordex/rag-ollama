"""PostgreSQL/pgvector persistence."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID, uuid4

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .config import Settings

logger = logging.getLogger(__name__)


class DatabaseError(RuntimeError):
    """Raised when the storage layer cannot complete an operation."""


def _connect(settings: Settings, register_embedding: bool = False) -> psycopg.Connection[Any]:
    try:
        connection = psycopg.connect(
            settings.database_url,
            row_factory=dict_row,
            connect_timeout=settings.database_connect_timeout,
        )
        if register_embedding:
            register_vector(connection)
        return connection
    except Exception as exc:
        raise DatabaseError("PostgreSQL에 연결하지 못했습니다. DATABASE_URL을 확인하세요.") from exc


def init_db(settings: Settings) -> None:
    """Create the extension and the initial schema if they do not exist."""

    connection = _connect(settings)
    try:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.commit()
        register_vector(connection)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_documents (
                id UUID PRIMARY KEY,
                source_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                mime_type TEXT,
                content_hash TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                quality_report JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        embedding_dimension = int(settings.embedding_dim)
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id BIGSERIAL PRIMARY KEY,
                document_id UUID NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector({embedding_dimension}) NOT NULL,
                UNIQUE(document_id, chunk_index)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS rag_documents_source_hash_idx
            ON rag_documents (source_name, content_hash)
            """
        )
        connection.commit()

        # HNSW is available in modern pgvector releases. Keep schema setup
        # usable with older extension versions by treating the index as optional.
        try:
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS rag_chunks_embedding_hnsw_idx
                ON rag_chunks USING hnsw (embedding vector_cosine_ops)
                """
            )
            connection.commit()
        except Exception as exc:  # pragma: no cover - depends on server extension version
            connection.rollback()
            logger.warning("HNSW 인덱스를 만들지 못했습니다. 소규모 데이터는 선형 검색으로 동작합니다: %s", exc)
    except Exception as exc:
        connection.rollback()
        raise DatabaseError("RAG 스키마를 초기화하지 못했습니다.") from exc
    finally:
        connection.close()


def check_db(settings: Settings) -> bool:
    connection = _connect(settings)
    try:
        connection.execute("SELECT 1")
        return True
    except Exception as exc:
        raise DatabaseError("PostgreSQL 상태 확인에 실패했습니다.") from exc
    finally:
        connection.close()


def find_document_by_hash(settings: Settings, source_name: str, content_hash: str) -> dict[str, Any] | None:
    connection = _connect(settings)
    try:
        return connection.execute(
            """
            SELECT d.id, d.source_name, d.source_type, d.mime_type, d.metadata,
                   d.quality_report, COUNT(c.id)::integer AS chunk_count
            FROM rag_documents d
            LEFT JOIN rag_chunks c ON c.document_id = d.id
            WHERE d.source_name = %s AND d.content_hash = %s
            GROUP BY d.id
            """,
            (source_name, content_hash),
        ).fetchone()
    except Exception as exc:
        raise DatabaseError("문서 중복 여부를 확인하지 못했습니다.") from exc
    finally:
        connection.close()


def save_document(
    settings: Settings,
    *,
    source_name: str,
    source_type: str,
    mime_type: str | None,
    content_hash: str,
    content: str,
    metadata: dict[str, Any],
    quality_report: dict[str, Any],
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> tuple[UUID, bool, int]:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks와 embeddings의 개수가 다릅니다.")

    connection = _connect(settings, register_embedding=True)
    try:
        with connection.transaction():
            existing = connection.execute(
                "SELECT id FROM rag_documents WHERE source_name = %s AND content_hash = %s",
                (source_name, content_hash),
            ).fetchone()
            if existing:
                count = connection.execute(
                    "SELECT COUNT(*)::integer AS count FROM rag_chunks WHERE document_id = %s",
                    (existing["id"],),
                ).fetchone()["count"]
                return existing["id"], True, count

            document_id = uuid4()
            connection.execute(
                """
                INSERT INTO rag_documents
                    (id, source_name, source_type, mime_type, content_hash, content, metadata, quality_report)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    document_id,
                    source_name,
                    source_type,
                    mime_type,
                    content_hash,
                    content,
                    Jsonb(metadata),
                    Jsonb(quality_report),
                ),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                connection.execute(
                    """
                    INSERT INTO rag_chunks (document_id, chunk_index, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        document_id,
                        chunk["index"],
                        chunk["text"],
                        Jsonb(chunk["metadata"]),
                        Vector(embedding),
                    ),
                )
            return document_id, False, len(chunks)
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError("문서와 임베딩을 저장하지 못했습니다.") from exc
    finally:
        connection.close()


def get_document(settings: Settings, document_id: UUID) -> dict[str, Any] | None:
    connection = _connect()
    try:
        return connection.execute(
            """
            SELECT d.id, d.source_name, d.source_type, d.mime_type, d.metadata,
                   d.quality_report, COUNT(c.id)::integer AS chunk_count
            FROM rag_documents d
            LEFT JOIN rag_chunks c ON c.document_id = d.id
            WHERE d.id = %s
            GROUP BY d.id
            """,
            (document_id,),
        ).fetchone()
    except Exception as exc:
        raise DatabaseError("문서를 조회하지 못했습니다.") from exc
    finally:
        connection.close()


def search_chunks(
    settings: Settings,
    embedding: list[float],
    *,
    top_k: int,
    document_id: UUID | None = None,
    min_quality_score: int | None = None,
) -> list[dict[str, Any]]:
    conditions: list[str] = []
    filter_params: list[Any] = []
    if document_id is not None:
        conditions.append("c.document_id = %s")
        filter_params.append(document_id)
    if min_quality_score is not None:
        conditions.append("COALESCE(NULLIF(d.quality_report->>'score', ''), '0')::integer >= %s")
        filter_params.append(min_quality_score)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"""
        SELECT c.id, c.document_id, d.source_name, c.chunk_index, c.content,
               c.metadata, d.quality_report,
               1 - (c.embedding <=> %s) AS score
        FROM rag_chunks c
        JOIN rag_documents d ON d.id = c.document_id
        {where_clause}
        ORDER BY c.embedding <=> %s
        LIMIT %s
    """
    vector = Vector(embedding)
    params = [vector, *filter_params, vector, top_k]

    connection = _connect(settings, register_embedding=True)
    try:
        rows = connection.execute(query, params).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            quality = row.get("quality_report") or {}
            results.append(
                {
                    "id": row["id"],
                    "document_id": row["document_id"],
                    "source_name": row["source_name"],
                    "chunk_index": row["chunk_index"],
                    "text": row["content"],
                    "metadata": row["metadata"] or {},
                    "score": float(row["score"]),
                    "quality_score": quality.get("score"),
                }
            )
        return results
    except Exception as exc:
        raise DatabaseError("벡터 검색에 실패했습니다.") from exc
    finally:
        connection.close()
