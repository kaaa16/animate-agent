"""Tests for file (pptx/docx/pdf) ingestion into DocumentIR."""

from pathlib import Path

import pytest

from animate_agent.documents.file_parser import parse_docx, parse_file, parse_pdf, parse_pptx
from animate_agent.documents.models import DocumentIR


def _write_pptx(path: Path) -> None:
    from pptx import Presentation

    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[0])
    slide.shapes.title.text = "力学基础"
    presentation.save(str(path))


def _write_docx(path: Path) -> None:
    from docx import Document

    document = Document()
    document.add_heading("电路基础", level=1)
    document.add_paragraph("电压与电流的关系")
    document.add_heading("欧姆定律", level=1)
    document.add_paragraph("U = I * R")
    document.save(str(path))


def _write_pdf(path: Path) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)


def test_parse_pptx_extracts_text(tmp_path: Path) -> None:
    path = tmp_path / "sample.pptx"
    _write_pptx(path)

    document = parse_pptx(path)

    assert document.source.type == "file"
    assert document.document_id
    assert document.sections[0].title == "力学基础"


def test_parse_docx_groups_headings(tmp_path: Path) -> None:
    path = tmp_path / "sample.docx"
    _write_docx(path)

    document = parse_docx(path)

    assert document.source.type == "file"
    assert [section.title for section in document.sections if section.title] == [
        "电路基础",
        "欧姆定律",
    ]
    assert any(
        block.type == "paragraph" and "电压" in block.text
        for section in document.sections
        for block in section.blocks
    )


def test_parse_pdf_produces_document(tmp_path: Path) -> None:
    path = tmp_path / "sample.pdf"
    _write_pdf(path)

    document = parse_pdf(path)

    assert document.source.type == "file"
    assert document.document_id


def test_parse_file_dispatches(tmp_path: Path) -> None:
    path = tmp_path / "sample.docx"
    _write_docx(path)

    assert isinstance(parse_file(path), DocumentIR)


def test_parse_file_rejects_unknown_extension(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError):
        parse_file(path)
