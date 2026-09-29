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

    What the number is derived from is still the finished video. Narration is what
    gets said aloud, the beat rate is six characters a second
    (`rendering/layout.SPEECH_CHARS_PER_SECOND`), and the target is ninety
    seconds, so the whole script has about 540 characters in it and a scene gets
    its share — about 70, which is four beats of one screen each.

    **The bounds here are the union of two modes, not a rule.** A scene is one
    thing when the input is a document found in the world and another when it is
    a script somebody wrote for this film: compressing a slide deck to ninety
    seconds is a different job from carrying the author's own paragraphs, and the
    two want different lengths. Pydantic sees one model, so it gets the envelope;
    the band that actually applies is chosen per run in `KnowledgeAgent._validate`
    (`_NARRATION_BOUNDS`). That is not belt-and-braces — widening these bounds
    without that check would silently drop the ceiling document mode has today,
    and nothing downstream would notice a lesson that had stopped compressing.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    objective: str
    narration: str = Field(min_length=20, max_length=120)
    # The cap must stay above the longest single list the source can hand a
    # scene: a source block listing N items forces one scene to cover all N, and
    # a cap below N makes the model silently drop items (and then contradict its
    # own objective, which still says "N").
    #
    # 12, not 8, since the scene count became derived rather than fixed. At eight
    # scenes the longest list in the corpus was one scene's problem; at five or
    # six, a scene covers more of the source and its points add up. The current
    # lesson already puts 8 on one scene, so the next merge makes 10 or 12 — and
    # the failure mode of a cap that is too low is not a refusal, it is the model
    # quietly keeping the first eight and dropping the rest, after which the
    # lesson contradicts its own objective and no check in the pipeline can tell.
    key_points: list[str] = Field(default_factory=list, min_length=2, max_length=12)
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
