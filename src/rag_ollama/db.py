# =============================================================================
# 파일명: db.py
# 경로: src/rag_ollama/db.py
# 목적: PostgreSQL·pgvector 연결, 스키마, 문서·청크 저장과 검색 제공함
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""PostgreSQL·pgvector 연결, 스키마, 문서·청크 저장과 검색 제공함"""

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
    """저장 계층이 요청을 완료하지 못했음을 나타내는 예외임"""


def _connect(settings: Settings, register_embedding: bool = False) -> psycopg.Connection[Any]:
    """설정된 PostgreSQL에 dict row와 선택적 vector 타입을 연결함

    Args:
        settings: DB 주소와 연결 제한 시간을 포함한 애플리케이션 설정임
        register_embedding: pgvector 타입 등록 여부임

    Returns:
        psycopg.Connection[Any]: 연결된 PostgreSQL 세션임

    Raises:
        DatabaseError: 연결 또는 vector 타입 등록에 실패할 때 발생함
    """
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
    """pgvector 확장과 문서·청크·검색 인덱스 스키마를 멱등적으로 생성함

    Caveats:
        구형 pgvector에서 HNSW 생성을 실패해도 소규모 데이터는 선형 검색으로
        계속 사용할 수 있도록 선택 기능으로 취급함
    """

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

        # NOTE: 최신 pgvector의 HNSW를 사용하되 구버전 확장에서도 스키마 초기화가 계속되도록 선택 처리함
        try:
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS rag_chunks_embedding_hnsw_idx
                ON rag_chunks USING hnsw (embedding vector_cosine_ops)
                """
            )
            connection.commit()
        except Exception as exc:  # NOTE: 서버 pgvector 확장 버전에 따라 인덱스 생성 실패가 가능함
            connection.rollback()
            logger.warning("HNSW 인덱스를 만들지 못했습니다. 소규모 데이터는 선형 검색으로 동작합니다: %s", exc)
    except Exception as exc:
        connection.rollback()
        raise DatabaseError("RAG 스키마를 초기화하지 못했습니다.") from exc
    finally:
        connection.close()


def check_db(settings: Settings) -> bool:
    """PostgreSQL 연결 가능 여부를 짧은 쿼리로 확인함

    Returns:
        bool: `SELECT 1` 실행이 완료되면 True임

    Raises:
        DatabaseError: 연결 또는 상태 확인에 실패할 때 발생함
    """
    connection = _connect(settings)
    try:
        connection.execute("SELECT 1")
        return True
    except Exception as exc:
        raise DatabaseError("PostgreSQL 상태 확인에 실패했습니다.") from exc
    finally:
        connection.close()


def find_document_by_hash(settings: Settings, source_name: str, content_hash: str) -> dict[str, Any] | None:
    """원천 이름과 정규화 본문 해시가 같은 문서의 요약을 조회함

    Returns:
        dict[str, Any] | None: 중복 문서 요약 또는 대상이 없을 때 None임

    Raises:
        DatabaseError: 중복 조회에 실패할 때 발생함
    """
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
    """문서와 모든 청크·임베딩을 하나의 트랜잭션으로 저장함

    Args:
        settings: DB 연결과 임베딩 차원 설정임
        source_name: 원천 파일 또는 문서 이름임
        source_type: 입력 경로를 나타내는 유형임
        mime_type: 원천 MIME 타입임
        content_hash: 정규화 본문의 중복 판별 해시임
        content: 저장할 정규화 원문임
        metadata: 문서 메타데이터임
        quality_report: 저장 시점의 품질 리포트임
        chunks: 청크 본문과 원문 위치 메타데이터 목록임
        embeddings: 청크별 정규화 임베딩 목록임

    Returns:
        tuple[UUID, bool, int]: 문서 ID, 중복 여부, 저장·기존 청크 수임

    Raises:
        ValueError: 청크와 임베딩 개수가 다를 때 발생함
        DatabaseError: 저장 트랜잭션이 실패할 때 발생함

    Caveats:
        동일 이름과 해시의 문서는 새로 쓰지 않아 인제스트 재시도가 멱등적임
    """
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
    """문서 메타데이터와 품질 요약 및 청크 수를 조회함

    Returns:
        dict[str, Any] | None: 문서 요약 또는 대상이 없을 때 None임

    Raises:
        DatabaseError: 문서 조회에 실패할 때 발생함
    """
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


def get_document_content(settings: Settings, document_id: UUID) -> dict[str, Any] | None:
    """요약 조회와 분리해 저장된 추출 본문을 조회함

    Caveats:
        본문은 크기가 클 수 있어 일반 문서 목록 조회에서 불필요하게 읽지 않도록
        별도 함수로 유지함
    """
    connection = _connect()
    try:
        return connection.execute(
            "SELECT source_name, content, metadata FROM rag_documents WHERE id = %s",
            (document_id,),
        ).fetchone()
    except Exception as exc:
        raise DatabaseError("문서 본문을 조회하지 못했습니다.") from exc
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
    """pgvector cosine distance로 품질·문서 조건을 적용해 청크를 검색함

    Args:
        settings: DB 연결과 검색 설정임
        embedding: 질문 임베딩 벡터임
        top_k: 반환할 최대 결과 수임
        document_id: 지정하면 해당 문서로 검색을 제한함
        min_quality_score: 지정하면 품질 점수 이상인 문서만 사용함

    Returns:
        list[dict[str, Any]]: 점수 내림차순 검색 결과 목록임

    Raises:
        DatabaseError: 벡터 검색에 실패할 때 발생함
    """
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
