"""Validated intermediate representation produced by the Knowledge Agent."""

from pydantic import BaseModel, ConfigDict, Field


class LessonScene(BaseModel):
    """One teachable scene in a generated lesson."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    objective: str
    narration: str
    key_points: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)


class LessonIR(BaseModel):
    """Knowledge-level lesson plan derived from a DocumentIR."""

    model_config = ConfigDict(extra="forbid")

    lesson_id: str
    document_id: str
    title: str
    subject: str
    summary: str
    learning_objectives: list[str] = Field(default_factory=list)
    scenes: list[LessonScene] = Field(default_factory=list)
