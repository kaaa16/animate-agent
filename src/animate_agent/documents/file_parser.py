"""Deterministic file (pptx/docx/pdf) to DocumentIR parsers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from animate_agent.documents.models import DocumentBlock, DocumentIR, DocumentSource, Section

SUPPORTED_EXTENSIONS = frozenset({".pptx", ".docx", ".pdf"})


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _document_id(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _is_heading_style(style_name: str) -> bool:
    lowered = style_name.lower()
    return lowered.startswith("heading") or "标题" in style_name or lowered == "title"


def _paragraph_block(section_id: str, index: int, text: str) -> DocumentBlock | None:
    cleaned = _clean_text(text)
    if not cleaned:
        return None
    return DocumentBlock(id=f"{section_id}-block-{index}", type="paragraph", text=cleaned)


def _resolve_title(sections: list[Section]) -> str:
    for section in sections:
        if section.title:
            return section.title
    for section in sections:
        for block in section.blocks:
            if block.text:
                return block.text
    return "Untitled document"


def _finalize(path: Path, sections: list[Section]) -> DocumentIR:
    return DocumentIR(
        document_id=_document_id(path),
        title=_resolve_title(sections),
        source=DocumentSource(type="file"),
        sections=sections,
    )


def parse_pptx(path: str | Path) -> DocumentIR:
    """Convert a PPTX presentation into DocumentIR (one Section per slide)."""
    from pptx import Presentation

    p = Path(path)
    presentation = Presentation(str(p))
    sections: list[Section] = []
    for slide_index, slide in enumerate(presentation.slides, start=1):
        lines: list[str] = []
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for paragraph in shape.text_frame.paragraphs:
                text = "".join(run.text for run in paragraph.runs)
                cleaned = _clean_text(text)
                if cleaned:
                    lines.append(cleaned)
        if not lines:
            continue
        section = Section(id=f"section-{slide_index}", title=lines[0], level=1)
        for index, text in enumerate(lines[1:], start=1):
            block = _paragraph_block(section.id, index, text)
            if block is not None:
                section.blocks.append(block)
        sections.append(section)
    return _finalize(p, sections)


def parse_docx(path: str | Path) -> DocumentIR:
    """Convert a DOCX document into DocumentIR, grouping paragraphs by heading styles."""
    from docx import Document

    p = Path(path)
    document = Document(str(p))
    sections: list[Section] = []
    current: Section | None = None
    section_counter = 0

    for paragraph in document.paragraphs:
        style_name = paragraph.style.name if paragraph.style is not None else ""
        text = _clean_text(paragraph.text)
        if not text:
            continue
        if _is_heading_style(style_name):
            section_counter += 1
            current = Section(id=f"section-{section_counter}", title=text, level=1)
            sections.append(current)
            continue
        if current is None:
            section_counter += 1
            current = Section(id=f"section-{section_counter}", title="", level=1)
            sections.append(current)
        block = _paragraph_block(current.id, len(current.blocks) + 1, text)
        if block is not None:
            current.blocks.append(block)

    return _finalize(p, sections)


def parse_pdf(path: str | Path) -> DocumentIR:
    """Convert a PDF into DocumentIR (one Section per page)."""
    from pypdf import PdfReader

    p = Path(path)
    reader = PdfReader(str(p))
    sections: list[Section] = []
    for page_index, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        lines = [cleaned for line in raw.splitlines() if (cleaned := _clean_text(line))]
        if not lines:
            continue
        section = Section(id=f"section-{page_index}", title=lines[0], level=1)
        for index, text in enumerate(lines[1:], start=1):
            block = _paragraph_block(section.id, index, text)
            if block is not None:
                section.blocks.append(block)
        sections.append(section)
    return _finalize(p, sections)


def parse_file(path: str | Path) -> DocumentIR:
    """Dispatch on file extension and return a DocumentIR."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pptx":
        return parse_pptx(p)
    if ext == ".docx":
        return parse_docx(p)
    if ext == ".pdf":
        return parse_pdf(p)
    raise ValueError(f"不支持的文件格式: {ext}。支持: .pptx, .docx, .pdf")
