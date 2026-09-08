"""Validated intermediate representation shared by all document adapters."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentSource(BaseModel):
    """Where the source document came from."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["url", "file"]
    url: str | None = None


class DocumentBlock(BaseModel):
    """A single ordered content block within a section."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["paragraph", "code", "list", "image"]
    text: str
    language: str | None = None
    source_ref: str | None = None


class Section(BaseModel):
    """A heading and the content that follows it."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    level: int = Field(ge=1, le=6)
    blocks: list[DocumentBlock] = Field(default_factory=list)


class DocumentIR(BaseModel):
    """Format-independent, validated document structure."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str
    source: DocumentSource
    sections: list[Section] = Field(default_factory=list)
