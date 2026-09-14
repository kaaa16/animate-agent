"""Validated intermediate representation produced by the Storyboard Agent.

Node in the pipeline: `LessonIR -> [Storyboard Agent] -> StoryboardIR -> RenderSpec`.

The contract here is deliberately **semantic**: objects carry roles and labels,
never coordinates. Geometry is assigned afterwards by the deterministic layout
pass, so the same StoryboardIR always renders to the same picture. See
`docs/storyboard-milestone.md` (decision D3) for why, and for the observable
symptom of breaking it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Object, step and control ids are authored by the model, so the pattern is a
# guard against punctuation soup rather than a naming preference.
ID_PATTERN = r"^[a-z][a-z0-9_-]*$"

# Scalar values a semantic prop may hold. The *keys* are checked against the
# rendering registry (`storyboard/validation.py`); the values are intentionally
# open, because a prop's unit and range are a rendering concern.
PropValue = str | int | float | bool


class StoryboardObject(BaseModel):
    """A semantic object on screen.

    No geometry: no x/y, no pixel size, no colour. `role` is resolved against
    the rendering registry, which is what makes an unregistered role a hard
    failure instead of a silently dropped object.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=48, pattern=ID_PATTERN)
    role: str = Field(min_length=1, max_length=32)
    label: str = Field(min_length=1, max_length=24)
    props: dict[str, PropValue] = Field(default_factory=dict)
    source_refs: list[str] = Field(default_factory=list)


class StoryboardStep(BaseModel):
    """One beat within a scene.

    A step is a *visual* beat, not a slide: it must change something on screen.
    Without this rule a model can emit seven steps of pure narration, which
    validates fine and animates to nothing.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=48, pattern=ID_PATTERN)
    title: str = Field(min_length=1, max_length=24)
    # 12, not 20. The quality baseline's own beat captions run 14~22 characters
    # ("最近回波被标记为危险候选点。"), so a 20-character floor rejects the very
    # writing this project is trying to match — 2 of its 4 steps would fail.
    # See docs/storyboard-milestone.md (D5).
    description: str = Field(min_length=12, max_length=200)
    highlights: list[str] = Field(default_factory=list, max_length=12)
    object_states: dict[str, dict[str, PropValue]] = Field(default_factory=dict)
    key_points: list[str] = Field(default_factory=list, min_length=1, max_length=8)

    @model_validator(mode="after")
    def _must_change_something(self) -> StoryboardStep:
        if not self.highlights and not self.object_states:
            raise ValueError("步骤必须至少高亮一个对象或设置一个对象状态（不能只有文字）")
        return self


class StoryboardControl(BaseModel):
    """An interactive control bound to a property the animation actually reads.

    `target_property` is the project rule "interactive parameters must change the
    result, not just the UI number" made addressable: validation resolves it
    against the registry's behaviours, so a control nothing consumes is rejected.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=48, pattern=ID_PATTERN)
    type: Literal["slider", "toggle", "button"]
    label: str = Field(min_length=1, max_length=16)
    # "<object_id>.<prop>" or "scene.<prop>".
    target_property: str = Field(min_length=1, max_length=64)
    min: float | None = None
    max: float | None = None
    default: float | None = None
    step: float | None = None
    unit: str = Field(default="", max_length=8)
    action: str | None = Field(default=None, max_length=32)

    @model_validator(mode="after")
    def _type_specific_fields(self) -> StoryboardControl:
        if self.type == "slider":
            missing = [
                name
                for name, value in (
                    ("min", self.min),
                    ("max", self.max),
                    ("default", self.default),
                    ("step", self.step),
                )
                if value is None
            ]
            if missing:
                raise ValueError(f"滑杆必须给出 {'/'.join(missing)}")
        if self.type == "button" and self.action is None:
            raise ValueError("按钮必须给出 action")
        return self


class StoryboardScene(BaseModel):
    """One animated scene: a set of objects, the beats that move them, and controls.

    `steps` bounds are loose here on purpose. The 3..7 shot rule lives in
    `StoryboardSettings` and is enforced by the validation pass, so the repo
    states that rule exactly once instead of restating it in the schema.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=48, pattern=ID_PATTERN)
    scene_type: str = Field(min_length=1, max_length=32)
    teaching_goal: str = Field(min_length=4, max_length=80)
    # Which LessonIR scenes this storyboard scene covers. 1~2, and the union
    # across all scenes must cover every LessonScene.
    lesson_scene_ids: list[str] = Field(min_length=1, max_length=2)
    objects: list[StoryboardObject] = Field(min_length=1, max_length=12)
    steps: list[StoryboardStep] = Field(min_length=1, max_length=12)
    controls: list[StoryboardControl] = Field(default_factory=list, max_length=6)
    params: dict[str, PropValue] = Field(default_factory=dict)
    renderer_hint: str | None = Field(default=None, max_length=32)


class StoryboardIR(BaseModel):
    """Storyboard-level animation plan derived from a LessonIR."""

    model_config = ConfigDict(extra="forbid")

    storyboard_id: str
    lesson_id: str
    document_id: str
    title: str
    subject: str
    eyebrow: str = ""
    scenes: list[StoryboardScene] = Field(min_length=1, max_length=10)
