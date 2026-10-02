# =============================================================================
# 파일명: evaluation.py
# 경로: src/rag_ollama/evaluation.py
# 목적: 고정 RAG 질문 묶음을 실행하고 재현 가능한 JSONL 실험 기록을 저장함
# 작성일: 2026-10-02
# =============================================================================

"""고정 평가 질문을 반복 실행하고 모델·검색 결과를 JSONL로 기록함"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

import httpx

from .config import Settings, get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = PROJECT_ROOT / "evals" / "questions.jsonl"
DEFAULT_FIXTURE = PROJECT_ROOT / "evals" / "transformers-basics.md"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results"
DEFAULT_API_URL = f"http://127.0.0.1:{os.getenv('API_PORT', '8000')}"


def _sha256(payload: bytes) -> str:
    """바이트 데이터의 재현성 확인용 SHA-256 해시를 반환함"""
    return hashlib.sha256(payload).hexdigest()


def _package_version(package: str) -> str | None:
    """설치된 패키지 버전을 읽고 없으면 None을 반환함"""
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _load_questions(path: Path) -> list[dict[str, Any]]:
    """JSONL 질문 파일을 읽고 필수 항목과 ID 중복을 검증함"""
    questions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: JSON 형식이 잘못되었습니다.") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number}: 각 줄은 JSON object여야 합니다.")

        case_id = str(item.get("id", "")).strip()
        question = str(item.get("question", "")).strip()
        expected_evidence = str(item.get("expected_evidence", "")).strip()
        if not case_id or not question or not expected_evidence:
            raise ValueError(
                f"{path}:{line_number}: id, question, expected_evidence가 필요합니다."
            )
        if case_id in seen_ids:
            raise ValueError(f"{path}:{line_number}: 중복 case id입니다: {case_id}")
        seen_ids.add(case_id)

        raw_top_k = item.get("top_k")
        top_k = int(raw_top_k) if raw_top_k is not None else None
        if top_k is not None and not 1 <= top_k <= 50:
            raise ValueError(f"{path}:{line_number}: top_k는 1부터 50까지여야 합니다.")
        questions.append(
            {
                "id": case_id,
                "question": question,
                "expected_answer": str(item.get("expected_answer", "")).strip(),
                "expected_evidence": expected_evidence,
                "top_k": top_k,
            }
        )

    if not questions:
        raise ValueError(f"질문이 없습니다: {path}")
    return questions


def _normalize_phrase(value: str) -> str:
    """대소문자와 공백 차이를 무시할 비교 문자열을 반환함"""
    return " ".join(value.casefold().split())


def _settings_snapshot(settings: Settings) -> dict[str, Any]:
    """실험 재현에 필요한 설정만 추려 비밀값 없이 반환함"""
    if settings.llm_backend == "transformers":
        model = settings.hf_model_id
        revision = settings.hf_revision
    elif settings.llm_backend == "openai-compatible":
        model = settings.llm_model
        revision = None
    else:
        model = None
        revision = None

    return {
        "llm": {
            "backend": settings.llm_backend,
            "model": model,
            "requested_revision": revision,
            "device_map": settings.hf_device_map if settings.llm_backend == "transformers" else None,
            "dtype": settings.hf_dtype if settings.llm_backend == "transformers" else None,
            "quantization": (
                settings.hf_quantization if settings.llm_backend == "transformers" else None
            ),
            "max_new_tokens": settings.llm_max_new_tokens,
            "temperature": settings.llm_temperature,
        },
        "embedding": {
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
            "dimension": settings.embedding_dim,
            "device": settings.embedding_device,
            "use_fp16": settings.embedding_use_fp16,
        },
        "retrieval": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "default_top_k": settings.retrieval_top_k,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
    }


def _write_record(file: TextIO, record: dict[str, Any]) -> None:
    """JSONL 한 줄을 기록하고 중단 시에도 결과를 보존함"""
    file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    file.flush()


def run_evaluation(
    *,
    questions_path: Path = DEFAULT_QUESTIONS,
    fixture_path: Path = DEFAULT_FIXTURE,
    output_path: Path | None = None,
    api_url: str = DEFAULT_API_URL,
    top_k_override: int | None = None,
    use_llm: bool = True,
) -> Path:
    """고정 fixture에 질문을 실행하고 한 줄당 하나의 JSON 기록을 저장함"""
    questions_path = questions_path.resolve()
    fixture_path = fixture_path.resolve()
    questions = _load_questions(questions_path)
    fixture_bytes = fixture_path.read_bytes()
    fixture_text = fixture_bytes.decode("utf-8")
    settings = get_settings()

    if top_k_override is not None and not 1 <= top_k_override <= 50:
        raise ValueError("--top-k는 1부터 50까지여야 합니다.")
    headers = {"X-API-Key": settings.api_key} if settings.api_key else {}
    client = httpx.Client(
        base_url=api_url.rstrip("/"),
        headers=headers,
        timeout=1800.0,
        trust_env=False,
    )
    try:
        health_response = client.get("/health")
        health_response.raise_for_status()
        health = health_response.json()
        if health.get("database") != "up":
            raise RuntimeError("API 데이터베이스가 준비되지 않았습니다.")

        fixture_response = client.post(
            "/documents/text",
            json={
                "name": fixture_path.name,
                "text": fixture_text,
                "source_type": "text",
                "mime_type": "text/markdown",
                "metadata": {
                    "purpose": "fixed-evaluation-fixture",
                    "fixture_sha256": _sha256(fixture_bytes),
                },
            },
        )
        fixture_response.raise_for_status()
        fixture = fixture_response.json()
    except httpx.HTTPStatusError as exc:
        client.close()
        response_text = exc.response.text[:500]
        raise RuntimeError(
            f"평가 API 요청 실패 ({exc.response.status_code}): {response_text}"
        ) from exc
    except httpx.HTTPError as exc:
        client.close()
        raise RuntimeError(
            f"평가 API에 연결하지 못했습니다: {api_url} ({type(exc).__name__})"
        ) from exc
    except Exception:
        client.close()
        raise

    run_id = str(uuid4())
    started_at = datetime.now(timezone.utc)
    if output_path is None:
        timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
        output_path = DEFAULT_OUTPUT_DIR / f"{timestamp}-{run_id[:8]}.jsonl"
    else:
        output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evidence_hits = 0
    evidence_total = 0
    error_count = 0
    elapsed_values: list[float] = []
    try:
        with output_path.open("a", encoding="utf-8") as output_file:
            _write_record(
                output_file,
                {
                    "record_type": "run",
                    "run_id": run_id,
                    "started_at_utc": started_at.isoformat(),
                    "api_url": api_url,
                    "question_set": questions_path.name,
                    "question_set_sha256": _sha256(questions_path.read_bytes()),
                    "fixture": {
                        "name": fixture_path.name,
                        "sha256": _sha256(fixture_bytes),
                        "document_id": fixture["document_id"],
                        "chunk_count": fixture["chunk_count"],
                    },
                    "case_count": len(questions),
                    "use_llm": use_llm,
                    "settings": _settings_snapshot(settings),
                },
            )

            for case in questions:
                actual_top_k = top_k_override or case["top_k"] or settings.retrieval_top_k
                case_started = time.perf_counter()
                evidence_total += 1
                try:
                    response = client.post(
                        "/query",
                        json={
                            "question": case["question"],
                            "top_k": actual_top_k,
                            "document_id": str(fixture["document_id"]),
                            "min_quality_score": None,
                            "use_llm": use_llm,
                        },
                    )
                    response.raise_for_status()
                    result = response.json()
                    elapsed_ms = round((time.perf_counter() - case_started) * 1000, 2)
                    elapsed_values.append(elapsed_ms)
                    expected_phrase = _normalize_phrase(case["expected_evidence"])
                    evidence_found = any(
                        expected_phrase in _normalize_phrase(source["text"])
                        for source in result["sources"]
                    )
                    evidence_hits += int(evidence_found)
                    record = {
                        "record_type": "case",
                        "run_id": run_id,
                        "case_id": case["id"],
                        "question": case["question"],
                        "expected_answer": case["expected_answer"],
                        "expected_evidence": case["expected_evidence"],
                        "expected_evidence_found": evidence_found,
                        "top_k": actual_top_k,
                        "elapsed_ms": elapsed_ms,
                        "answer": result["answer"],
                        "generation": result["generation"],
                        "sources": result["sources"],
                    }
                except Exception as exc:
                    error_count += 1
                    elapsed_ms = round((time.perf_counter() - case_started) * 1000, 2)
                    elapsed_values.append(elapsed_ms)
                    record = {
                        "record_type": "case",
                        "run_id": run_id,
                        "case_id": case["id"],
                        "question": case["question"],
                        "expected_answer": case["expected_answer"],
                        "expected_evidence": case["expected_evidence"],
                        "top_k": actual_top_k,
                        "elapsed_ms": elapsed_ms,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                _write_record(output_file, record)

            average_ms = round(sum(elapsed_values) / len(elapsed_values), 2)
            _write_record(
                output_file,
                {
                    "record_type": "summary",
                    "run_id": run_id,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "case_count": len(elapsed_values),
                    "error_count": error_count,
                    "average_elapsed_ms": average_ms,
                    "expected_evidence_hits": evidence_hits,
                    "expected_evidence_total": evidence_total,
                    "expected_evidence_hit_rate": (
                        round(evidence_hits / evidence_total, 4) if evidence_total else None
                    ),
                },
            )
    finally:
        client.close()

    return output_path


def main() -> None:
    """명령행 인자를 받아 기본 평가 질문 묶음을 실행함"""
    parser = argparse.ArgumentParser(
        description="고정 RAG 질문을 실행하고 모델·근거·지연시간을 JSONL로 기록합니다."
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="모델 생성을 생략하고 검색 근거만 비교합니다.",
    )
    args = parser.parse_args()
    try:
        output_path = run_evaluation(
            questions_path=args.questions,
            fixture_path=args.fixture,
            output_path=args.output,
            api_url=args.api_url,
            top_k_override=args.top_k,
            use_llm=not args.retrieval_only,
        )
    except (OSError, ValueError, RuntimeError, httpx.HTTPError) as exc:
        parser.error(str(exc))
    print(f"평가 기록: {output_path}")


if __name__ == "__main__":
    main()
