"""Validated intermediate representation produced by the Knowledge Agent."""

from pydantic import BaseModel, ConfigDict, Field


class LessonScene(BaseModel):
    """One teachable scene in a generated lesson.

    `narration` used to run 80~200 characters, on the argument that a scene too
    thin to fill 80 is a scene too thin to carry its own animation. That was
    written when nothing in the pipeline had a length at all: 80 minimum across
    six to eight scenes put the shortest possible video at 80+ seconds, so a
    one-minute cut was not merely unachieved — it was unreachable, and no layer
    was in a position to notice.

    What the number is now derived from is the finished video. Narration is what
    gets said aloud, the beat rate is six characters a second
    (`rendering/layout.SPEECH_CHARS_PER_SECOND`), and the target is sixty
    seconds, so the whole script has about 350 characters in it and a scene gets
    its share. The work that used to be done by writing more per scene is done by
    having more scenes: the cap on scenes went from 8 to 12 at the same time.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    objective: str
    narration: str = Field(min_length=20, max_length=60)
    # The cap must stay above the longest single list the source can hand a
    # scene: a source block listing N items forces one scene to cover all N, and
    # a cap below N makes the model silently drop items (and then contradict its
    # own objective, which still says "N").
    key_points: list[str] = Field(default_factory=list, min_length=2, max_length=8)
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
