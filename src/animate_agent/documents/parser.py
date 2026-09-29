"""Deterministic HTML-to-DocumentIR parser."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from animate_agent.documents.models import DocumentBlock, DocumentIR, DocumentSource, Section

ReadabilityDocument: Any
try:
    from readability import Document as _ReadabilityDocument  # type: ignore[import-untyped]

    ReadabilityDocument = _ReadabilityDocument
except ImportError:  # pragma: no cover - exercised only in minimal installations
    ReadabilityDocument = None

REMOVED_TAGS = ("script", "style", "nav", "footer", "aside", "noscript", "template")
NOISE_SELECTORS = (
    "[role='navigation']",
    "[role='complementary']",
    "[aria-label*='cookie' i]",
    "[class*='cookie' i]",
    "[id*='cookie' i]",
    "[class*='sidebar' i]",
    "[id*='sidebar' i]",
    "[class*='breadcrumb' i]",
    ".headerlink",
    "[class*='toc' i]",
    "[id*='toc' i]",
)
CONTENT_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "ul", "ol", "img")


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if slug:
        return slug
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]  # noqa: S324


def _page_title(soup: BeautifulSoup) -> str:
    if soup.h1 and _clean_text(soup.h1.get_text(" ", strip=True)):
        heading = BeautifulSoup(str(soup.h1), "html.parser")
        for permalink in heading.select(".headerlink"):
            permalink.decompose()
        return _clean_text(heading.get_text(" ", strip=True))
    if soup.title and _clean_text(soup.title.get_text(" ", strip=True)):
        return _clean_text(soup.title.get_text(" ", strip=True))
    return "Untitled document"


def _main_content(original_html: str, soup: BeautifulSoup) -> BeautifulSoup | Tag:
    semantic_root = (
        soup.find("main")
        or soup.find("article")
        or soup.select_one("[role='main']")
    )
    if semantic_root is not None:
        return semantic_root
    if ReadabilityDocument is not None:
        readable = BeautifulSoup(ReadabilityDocument(original_html).summary(), "html.parser")
        if readable.find(("h1", "h2", "h3", "p", "pre")):
            return readable
    return soup.body or soup


def _remove_noise(root: BeautifulSoup | Tag) -> None:
    for element in root.find_all(REMOVED_TAGS):
        element.decompose()
    for selector in NOISE_SELECTORS:
        for element in root.select(selector):
            element.decompose()


def _top_level_content(root: BeautifulSoup | Tag) -> Iterable[Tag]:
    for element in root.find_all(CONTENT_TAGS):
        if any(parent.name in CONTENT_TAGS for parent in element.parents if parent is not root):
            continue
        yield element


def _blocks_from_element(element: Tag, block_id: str, base_url: str) -> list[DocumentBlock]:
    """Return the blocks for one element.

    Usually one, but a list yields one block per item so that source_refs can
    name a single step instead of the whole list. `block_id` is the id of the
    first block; later blocks take consecutive numbers so the caller's
    len(blocks)+1 numbering stays correct.
    """
    source_ref = f"#{element['id']}" if element.get("id") else None
    if element.name == "p":
        text = _clean_text(element.get_text(" ", strip=True))
        if not text:
            return []
        return [DocumentBlock(id=block_id, type="paragraph", text=text, source_ref=source_ref)]
    if element.name == "pre":
        code = element.find("code")
        text = (code or element).get_text("\n", strip=True)
        if not text:
            return []
        raw_classes = (code or element).get("class")
        classes = raw_classes if isinstance(raw_classes, list) else []
        language = next(
            (item.split("language-", 1)[1] for item in classes if item.startswith("language-")),
            None,
        )
        return [
            DocumentBlock(
                id=block_id,
                type="code",
                text=text,
                language=language,
                source_ref=source_ref,
            )
        ]
    if element.name in {"ul", "ol"}:
        items = [
            _clean_text(item.get_text(" ", strip=True))
            for item in element.find_all("li", recursive=False)
        ]
        items = [item for item in items if item]
        if not items:
            return []
        prefix, _, number = block_id.rpartition("-block-")
        first = int(number)
        return [
            DocumentBlock(
                id=f"{prefix}-block-{first + offset}",
                type="list",
                text=item,
                source_ref=source_ref,
            )
            for offset, item in enumerate(items)
        ]
    if element.name == "img":
        src = element.get("src")
        if not isinstance(src, str) or not src:
            return []
        alt = element.get("alt")
        return [
            DocumentBlock(
                id=block_id,
                type="image",
                text=_clean_text(alt if isinstance(alt, str) else ""),
                source_ref=urljoin(base_url, src),
            )
        ]
    return []


def parse_html(html: str, source_url: str) -> DocumentIR:
    """Clean HTML and convert its ordered semantic content into DocumentIR."""

    source_soup = BeautifulSoup(html, "html.parser")
    title = _page_title(source_soup)
    root = _main_content(html, source_soup)
    _remove_noise(root)

    document_id = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:16]
    sections: list[Section] = []
    current: Section | None = None
    used_ids: dict[str, int] = {}

    for element in _top_level_content(root):
        if element.name and re.fullmatch(r"h[1-6]", element.name):
            heading = _clean_text(element.get_text(" ", strip=True))
            if not heading:
                continue
            base_id = _slug(heading)
            used_ids[base_id] = used_ids.get(base_id, 0) + 1
            suffix = f"-{used_ids[base_id]}" if used_ids[base_id] > 1 else ""
            current = Section(
                id=f"section-{base_id}{suffix}",
                title=heading,
                level=int(element.name[1]),
            )
            sections.append(current)
            continue

        if current is None:
            current = Section(id="section-overview", title=title, level=1)
            sections.append(current)
        block_id = f"{current.id}-block-{len(current.blocks) + 1}"
        current.blocks.extend(_blocks_from_element(element, block_id, source_url))

    return DocumentIR(
        document_id=document_id,
        title=title,
        source=DocumentSource(type="url", url=source_url),
        sections=sections,
    )
