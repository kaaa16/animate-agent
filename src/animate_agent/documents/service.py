"""Document ingestion orchestration and persistence."""

import json
from pathlib import Path

import httpx

from animate_agent.documents.fetcher import fetch_html
from animate_agent.documents.models import DocumentIR
from animate_agent.documents.parser import parse_html

DEFAULT_DOCUMENTS_DIR = Path("data/documents")


async def ingest_url(
    url: str,
    *,
    output_dir: Path = DEFAULT_DOCUMENTS_DIR,
    client: httpx.AsyncClient | None = None,
) -> DocumentIR:
    """Fetch, parse, validate, and persist one URL document."""

    html = await fetch_html(url, client=client)
    document = parse_html(html, url)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{document.document_id}.json"
    destination.write_text(
        json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return document
