"""Validated intermediate representation produced by the Knowledge Agent."""

from pydantic import BaseModel, ConfigDict, Field


class LessonScene(BaseModel):
    """One teachable scene in a generated lesson.

    `narration` carries a floor as well as a ceiling: a scene too thin to fill
    ~80 characters is a scene too thin to carry its own animation, and should
    have been merged with a neighbour (see the Knowledge Agent prompt).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    objective: str
    narration: str = Field(min_length=80, max_length=200)
    key_points: list[str] = Field(default_factory=list, min_length=2, max_length=5)
    source_refs: list[str] = Field(default_factory=list)


class LessonIR(BaseModel):
    """Knowledge-level lesson plan derived from a DocumentIR."""

    model_config = ConfigDict(extra="forbid")

    lesson_id: str
    document_id: str
    title: str
    subject: str
    summary: str
    learning_objectives: list[str] = Field(default_factory=list, min_length=2, max_length=4)
    scenes: list[LessonScene] = Field(default_factory=list, min_length=1)
