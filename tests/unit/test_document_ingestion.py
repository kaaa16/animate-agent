import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from animate_agent import api
from animate_agent.documents.fetcher import fetch_html
from animate_agent.documents.models import DocumentIR
from animate_agent.documents.parser import parse_html
from animate_agent.documents.service import ingest_url

FIXTURE = Path(__file__).parents[1] / "fixtures" / "manim_quickstart.html"
SOURCE_URL = "https://docs.manim.community/en/stable/tutorials/quickstart.html"


def fixture_html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_fetch_html_downloads_a_page() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=fixture_html(),
            request=request,
        )
    )

    async def run() -> str:
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_html(SOURCE_URL, client=client)

    assert "Manim Quickstart" in asyncio.run(run())


def test_parser_extracts_title() -> None:
    document = parse_html(fixture_html(), SOURCE_URL)

    assert document.title == "Manim Quickstart"


def test_parser_extracts_headings() -> None:
    document = parse_html(fixture_html(), SOURCE_URL)

    assert [section.title for section in document.sections] == [
        "Manim Quickstart",
        "Introduction",
        "Creating a Scene",
        "Rendering",
    ]


def test_parser_extracts_paragraph() -> None:
    document = parse_html(fixture_html(), SOURCE_URL)

    assert any(
        block.type == "paragraph"
        for section in document.sections
        for block in section.blocks
    )


def test_parser_extracts_code_block() -> None:
    document = parse_html(fixture_html(), SOURCE_URL)

    code_blocks = [
        block for section in document.sections for block in section.blocks if block.type == "code"
    ]
    assert code_blocks[0].language == "python"
    assert "class CreateCircle" in code_blocks[0].text


def test_parser_extracts_lists_and_images() -> None:
    document = parse_html(fixture_html(), SOURCE_URL)
    blocks = [block for section in document.sections for block in section.blocks]

    assert any(block.type == "list" and "Save the file." in block.text for block in blocks)
    assert any(
        block.type == "image"
        and block.source_ref == "https://docs.manim.community/en/stable/tutorials/_images/create-circle.png"
        for block in blocks
    )


def test_parser_gives_each_list_item_its_own_block() -> None:
    document = parse_html(fixture_html(), SOURCE_URL)
    lists = [
        block for section in document.sections for block in section.blocks if block.type == "list"
    ]

    assert any("Save the file." in block.text for block in lists)
    # Each item is its own block, so no list block spans multiple lines...
    assert all("\n" not in block.text for block in lists)
    # ...and the ids handed out are distinct.
    assert len({block.id for block in lists}) == len(lists)


def test_parser_excludes_page_chrome_and_executable_content() -> None:
    document = parse_html(fixture_html(), SOURCE_URL)
    serialized = document.model_dump_json()

    for excluded in (
        "Documentation navigation",
        "Sidebar content",
        "Accept cookies",
        "Footer content",
        "window.tracking",
        "display: none",
    ):
        assert excluded not in serialized


def test_document_ir_passes_pydantic_validation() -> None:
    document = parse_html(fixture_html(), SOURCE_URL)

    validated = DocumentIR.model_validate(document.model_dump())

    assert validated.document_id == document.document_id


def test_from_url_api_returns_document_ir(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = parse_html(fixture_html(), SOURCE_URL)

    async def fake_ingest_url(url: str) -> DocumentIR:
        assert url == SOURCE_URL
        return expected

    monkeypatch.setattr(api, "ingest_url", fake_ingest_url)

    response = TestClient(api.app).post(
        "/api/documents/from-url",
        json={"url": SOURCE_URL},
    )

    assert response.status_code == 200
    assert DocumentIR.model_validate(response.json()) == expected


def test_ingest_url_persists_valid_json(tmp_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text=fixture_html(),
            request=request,
        )
    )

    async def run() -> DocumentIR:
        async with httpx.AsyncClient(transport=transport) as client:
            return await ingest_url(SOURCE_URL, output_dir=tmp_path, client=client)

    document = asyncio.run(run())
    saved = json.loads((tmp_path / f"{document.document_id}.json").read_text(encoding="utf-8"))

    assert DocumentIR.model_validate(saved) == document
