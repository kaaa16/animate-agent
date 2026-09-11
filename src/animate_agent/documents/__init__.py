"""Document ingestion and semantic extraction package."""

from animate_agent.documents.file_parser import parse_file
from animate_agent.documents.models import DocumentBlock, DocumentIR, DocumentSource, Section
from animate_agent.documents.parser import parse_html

__all__ = [
    "DocumentBlock",
    "DocumentIR",
    "DocumentSource",
    "Section",
    "parse_file",
    "parse_html",
]
