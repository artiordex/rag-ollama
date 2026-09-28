"""Text extraction for common unstructured document formats."""

from __future__ import annotations

import json
import mimetypes
from dataclasses import dataclass, field
from html.parser import HTMLParser
from io import BytesIO
from pathlib import PurePath

from .chunking import normalize_text


class ParseError(ValueError):
    """Raised when a file cannot be converted into text."""


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    text: str
    mime_type: str | None
    metadata: dict[str, object] = field(default_factory=dict)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        elif tag.lower() in {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag.lower() in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)


def _decode_bytes(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "utf-16", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def _parse_pdf(payload: bytes) -> ParsedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ParseError("PDF 파싱을 위해 pypdf를 설치해야 합니다.") from exc

    reader = PdfReader(BytesIO(payload))
    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = normalize_text(page.extract_text() or "")
        if page_text:
            pages.append(f"[Page {page_number}]\n{page_text}")
    return ParsedDocument(
        text=normalize_text("\n\n".join(pages)),
        mime_type="application/pdf",
        metadata={"page_count": len(reader.pages)},
    )


def _parse_docx(payload: bytes) -> ParsedDocument:
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ParseError("DOCX 파싱을 위해 python-docx를 설치해야 합니다.") from exc

    document = Document(BytesIO(payload))
    parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    table_count = 0
    for table in document.tables:
        table_count += 1
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return ParsedDocument(
        text=normalize_text("\n".join(parts)),
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        metadata={"table_count": table_count},
    )


def parse_document(filename: str, payload: bytes, content_type: str | None = None) -> ParsedDocument:
    """Extract text from supported files.

    Scanned PDFs and HWP files need an OCR/HWP-specific adapter and are
    deliberately reported as unsupported/empty rather than silently indexed as
    garbage.
    """

    suffix = PurePath(filename or "").suffix.lower()
    mime_type = (content_type or mimetypes.guess_type(filename or "")[0] or "").split(";", 1)[0] or None

    if suffix == ".pdf" or mime_type == "application/pdf":
        return _parse_pdf(payload)
    if suffix == ".docx":
        return _parse_docx(payload)
    if suffix in {".html", ".htm"} or mime_type == "text/html":
        parser = _HTMLTextExtractor()
        parser.feed(_decode_bytes(payload))
        return ParsedDocument(normalize_text("".join(parser.parts)), "text/html")
    if suffix == ".json" or mime_type == "application/json":
        raw = _decode_bytes(payload)
        try:
            value = json.loads(raw)
            text = json.dumps(value, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            text = raw
        return ParsedDocument(normalize_text(text), "application/json")
    if suffix in {".hwp", ".hwpx"}:
        raise ParseError("HWP/HWPX는 현재 기본 파서에 포함되지 않았습니다. HWPX 또는 OCR 어댑터를 추가하세요.")

    return ParsedDocument(normalize_text(_decode_bytes(payload)), mime_type or "text/plain")
