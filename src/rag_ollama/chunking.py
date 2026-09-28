"""Small, dependency-free text normalizer and chunker."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextChunk:
    index: int
    text: str
    start: int
    end: int


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_BOUNDARY = re.compile(r"\n\n|(?<=[.!?。！？])\s+|\n")


def _best_boundary(text: str, start: int, end: int) -> int:
    if end >= len(text):
        return len(text)
    window_start = start + int((end - start) * 0.55)
    candidates = [match.end() for match in _BOUNDARY.finditer(text, window_start, end)]
    if candidates:
        return max(candidates)
    return end


def chunk_text(text: str, max_chars: int = 1600, overlap: int = 240) -> list[TextChunk]:
    """Split text while preferring paragraph/sentence boundaries.

    Character sizes are intentionally used instead of model token sizes so this
    component works before an embedding model is loaded. The defaults are a
    reasonable starting point for Korean and English mixed documents.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be between 0 and max_chars - 1")

    normalized = normalize_text(text)
    if not normalized:
        return []

    chunks: list[TextChunk] = []
    start = 0
    index = 0
    while start < len(normalized):
        raw_end = min(start + max_chars, len(normalized))
        end = _best_boundary(normalized, start, raw_end)
        piece = normalized[start:end].strip()

        if not piece:
            start = max(end, start + 1)
            continue

        left_trim = len(normalized[start:end]) - len(normalized[start:end].lstrip())
        right_trimmed_end = end - (len(normalized[start:end]) - len(normalized[start:end].rstrip()))
        actual_start = start + left_trim
        actual_end = max(actual_start, right_trimmed_end)
        chunks.append(TextChunk(index=index, text=piece, start=actual_start, end=actual_end))
        index += 1

        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)

    return chunks
