# =============================================================================
# 파일명: quality.py
# 경로: src/rag_ollama/quality.py
# 목적: 비정형 텍스트의 설명 가능한 기초 품질 지표와 이슈 산출함
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""비정형 텍스트의 설명 가능한 기초 품질 지표와 이슈 산출함"""

from __future__ import annotations

import re
from collections import Counter

from .chunking import MAX_TEXT_CHARACTERS, normalize_text


def _issue(severity: str, code: str, message: str, value: float | int | str | None = None) -> dict[str, object]:
    """품질 진단 결과에서 일관된 이슈 구조를 생성함"""
    return {"severity": severity, "code": code, "message": message, "value": value}


def diagnose_text(text: str, source_name: str = "", metadata: dict[str, object] | None = None) -> dict[str, object]:
    """비정형 텍스트에서 재현 가능한 기초 품질 지표와 이슈를 산출함

    Args:
        text: 진단할 원문 텍스트임
        source_name: 품질 리포트에 기록할 원천 이름임
        metadata: 페이지·표 개수처럼 파서가 제공한 보조 지표임

    Returns:
        dict[str, object]: 점수, 상태, 규칙 버전, 지표와 이슈 목록임

    Raises:
        ValueError: 입력 텍스트가 최대 문자 수를 초과할 때 발생함

    Caveats:
        도메인 승인 기준이 아닌 학습용 baseline이며, 문서와 함께 저장해
        규칙 버전별 재진단 비교에 사용함
    """

    if len(text) > MAX_TEXT_CHARACTERS:
        raise ValueError(f"진단 텍스트가 허용 한도({MAX_TEXT_CHARACTERS}자)를 초과했습니다.")
    normalized = normalize_text(text)
    lines = normalized.splitlines() if normalized else []
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    counts = Counter(non_empty_lines)
    duplicate_line_count = sum(count - 1 for count in counts.values() if count > 1)
    line_count = len(lines)
    non_empty_count = len(non_empty_lines)
    char_count = len(normalized)
    word_count = len(re.findall(r"\S+", normalized))
    replacement_count = normalized.count("�")
    control_count = sum(1 for char in normalized if ord(char) < 32 and char not in {"\n", "\t"})
    blank_line_ratio = ((line_count - non_empty_count) / line_count) if line_count else 0.0
    duplicate_line_ratio = (duplicate_line_count / non_empty_count) if non_empty_count else 0.0

    metrics: dict[str, float | int | str] = {
        "char_count": char_count,
        "line_count": line_count,
        "non_empty_line_count": non_empty_count,
        "word_count": word_count,
        "replacement_character_count": replacement_count,
        "control_character_count": control_count,
        "blank_line_ratio": round(blank_line_ratio, 4),
        "duplicate_line_ratio": round(duplicate_line_ratio, 4),
        "source_name": source_name,
    }
    if metadata:
        for key in ("page_count", "table_count"):
            if key in metadata:
                metrics[key] = metadata[key]  # type: ignore[assignment]

    issues: list[dict[str, object]] = []
    score = 100

    if not normalized:
        issues.append(_issue("error", "EMPTY_DOCUMENT", "추출된 텍스트가 없습니다."))
        score = 0
    else:
        if char_count < 100:
            issues.append(_issue("warning", "SHORT_DOCUMENT", "문서 본문이 너무 짧습니다.", char_count))
            score -= 15
        if replacement_count:
            issues.append(
                _issue(
                    "warning",
                    "ENCODING_REPLACEMENT",
                    "인코딩 손상으로 보이는 대체 문자가 포함되어 있습니다.",
                    replacement_count,
                )
            )
            score -= min(30, replacement_count * 2)
        if control_count:
            issues.append(_issue("warning", "CONTROL_CHARACTERS", "제어 문자가 포함되어 있습니다.", control_count))
            score -= min(20, control_count * 2)
        if duplicate_line_ratio >= 0.25:
            issues.append(
                _issue(
                    "warning",
                    "DUPLICATED_LINES",
                    "동일한 줄이 반복되어 추출 오류 또는 중복 콘텐츠가 의심됩니다.",
                    duplicate_line_ratio,
                )
            )
            score -= 20
        elif duplicate_line_ratio >= 0.10:
            issues.append(_issue("info", "REPEATED_LINES", "반복되는 줄이 일부 존재합니다.", duplicate_line_ratio))
            score -= 8
        if not source_name.strip():
            issues.append(_issue("warning", "MISSING_SOURCE_NAME", "원천 문서명이 없습니다."))
            score -= 5

    score = max(0, min(100, score))
    status = "pass" if score >= 90 else "warn" if score >= 70 else "fail"
    return {
        "score": score,
        "status": status,
        "assessment_scope": "rag-ollama-learning-baseline",
        "rule_set_version": "baseline-text-v1",
        "metrics": metrics,
        "issues": issues,
    }
