"""The render contract: a fully geometric spec the player can draw without guessing.

Node in the pipeline: `StoryboardIR -> [layout] -> RenderSpec -> [player] -> pixels`.

Why this is a new strict model instead of `animation/elements.py`
----------------------------------------------------------------

`elements.py` is an unvalidated dataclass whose `Scene.elements` is typed
`list[SpecSerializable]` — a `Protocol`, not a union. Pydantic cannot
discriminate on that, and the file also carries two untyped escape hatches
(`TimelineStep.actions: list[dict[str, Any]]`, `Scene.metadata: dict[str, Any]`).
The boundary is exactly where validation matters most, so this module defines its
own `kind`-keyed union and never touches the frozen baseline.

Two rules this module keeps
---------------------------

**Geometry here, semantics in `props`.** Element classes carry identity and
geometry (x/y/size) and nothing else that a renderer must interpret. Every
semantic value — `speed`, `danger`, `radius`, `visible` — stays in `props`,
copied verbatim from the StoryboardIR. This is decision D3 made concrete: the
layout pass must not need to understand the domain, and playback mutates `props`
through `object_states` without touching geometry. Hoisting `danger` into a typed
field would give one concept two homes and two ways to fall out of sync.

**Colour is absent.** No hex, no palette, no theme. `tone` names a slot the
player resolves against its own `:root` variables, because colour belongs to the
theme layer (decision D1) and because `body.danger` flips at playback time — a
frozen colour cannot follow it.

Unknown kinds are a hard failure, never a silent drop: `RenderElement` is a
discriminated union, so a `kind` nobody registered raises with the name in the
message.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from animate_agent.rendering.registry import Glyph
from animate_agent.storyboard.models import PropValue

#: Bumped when a change would make an older player mis-draw a newer spec. The
#: player refuses a version it does not know rather than rendering something
#: plausible-but-wrong; see `docs/storyboard-milestone.md` on visible errors.
SPEC_VERSION = 1

#: Theme slot an element draws through. The player maps these to its
#: `:root` variables, so changing the palette never touches a spec.
Tone = Literal["normal", "accent", "danger", "muted", "success"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RenderPoint(_Model):
    """A point in stage coordinates: origin top-left, +y downwards."""

    x: float
    y: float


class _Element(_Model):
    """Identity, geometry and tone. Everything else lives in `props`."""

    id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    label: str = ""
    x: float
    y: float
    tone: Tone = "normal"
    #: Semantic values carried over verbatim from the StoryboardIR. The player
    #: reads them through the behaviours in `rendering/registry.py`.
    props: dict[str, PropValue] = Field(default_factory=dict)
    #: Geometry prop -> the control target that drives it, e.g.
    #: `{"radius": "scene.safe_distance"}`.
    #:
    #: Layout bakes a concrete number into every geometric field, so playback
    #: needs to be told which knob moves which number. It cannot be inferred:
    #: the safe-distance circle's radius and the danger threshold are the same
    #: quantity, but a lidar's `radius` is a different one, and a rule that
    #: matched on the property name would move both. Naming the binding here
    #: keeps that distinction in the spec instead of in a heuristic.
    binds: dict[str, str] = Field(default_factory=dict)


class BodyElement(_Element):
    """A solid object. `glyph` names a T2 glyph; without one, `shape` is drawn."""

    kind: Literal["body"] = "body"
    shape: Literal["rect", "circle", "polygon", "capsule"] = "rect"
    width: float = Field(default=0.0, ge=0)
    height: float = Field(default=0.0, ge=0)
    #: Vertex count for `shape: "polygon"`, ignored otherwise. `inner_ratio` below
    #: 1 turns it into a star, which is how the baseline's obstacles are drawn
    #: (`app.js:696`: 8 vertices alternating between r and 0.78r). Parametric, not
    #: a glyph — a glyph would freeze a `d` string for a shape an atom computes
    #: exactly, and D1 forbids hand-written `d` for that reason.
    sides: int = Field(default=6, ge=3, le=64)
    inner_ratio: float = Field(default=1.0, gt=0, le=1)
    #: Degrees, clockwise from +x. The car and its lidar fan share it.
    heading: float = 0.0
    glyph: str | None = None
    #: Where this body goes during playback, when the preset gave it a path.
    #:
    #: `field` bakes a parabola here — the same `_arc_points` curve the `trace`
    #: gets, so the ball flies along the trajectory the lesson draws rather than
    #: along a second one computed independently. Nothing in the player evaluates
    #: physics: it interpolates along points layout already sampled (decision D3,
    #: "语义给模型，布局给代码" — the arithmetic is the code's job, and the
    #: player's job is only to move through it).
    path: list[RenderPoint] = Field(default_factory=list)
    #: Seconds for one pass along `path`. Baked rather than derived from `speed`
    #: because a throw's duration is a property of the arc, not of a prop the
    #: model chose a unit for.
    duration: float = Field(default=0.0, ge=0)


class EmitterElement(_Element):
    """A fan of ranging rays. `radius` is the sensing distance in stage units."""

    kind: Literal["emitter"] = "emitter"
    radius: float = Field(gt=0)
    fov: float = Field(default=180.0, gt=0, le=360)
    rays: int = Field(default=13, ge=1, le=181)
    heading: float = 0.0
    #: Id of the body this emitter is mounted on, already resolved by layout.
    anchor: str | None = None


class ZoneElement(_Element):
    """A dashed circle or ring marking a threshold distance."""

    kind: Literal["zone"] = "zone"
    radius: float = Field(gt=0)
    anchor: str | None = None


class LinkElement(_Element):
    """A straight connection between two bodies; dashed when inactive."""

    kind: Literal["link"] = "link"
    points: list[RenderPoint] = Field(min_length=2)
    active: bool = True


class TravelerElement(_Element):
    """A token that rides along a link or a trace. Position comes from `path_id`."""

    kind: Literal["traveler"] = "traveler"
    path_id: str = Field(min_length=1)
    progress: float = Field(default=0.0, ge=0, le=1)
    size: float = Field(default=10.0, gt=0)


class TraceElement(_Element):
    """A polyline: a body's history, or a curve computed from a formula.

    The points are baked at layout time, which is what makes the curve a piece
    of geometry rather than a formula the renderer would have to evaluate
    (decision D4). `anchor` is what is left of the relation, so the player can
    translate the whole curve when the body it belongs to moves.
    """

    kind: Literal["trace"] = "trace"
    points: list[RenderPoint] = Field(min_length=2)
    dashed: bool = False
    anchor: str | None = None
    #: The `length` value that draws the whole curve, so the player can turn a
    #: live `length` into "how much of it to draw" without restating layout's
    #: `TRACE_LENGTH_REFERENCE`.
    length_reference: float = Field(default=0.0, gt=0)


class VectorElement(_Element):
    """An arrow. `dx`/`dy` are already scaled to stage units by layout."""

    kind: Literal["vector"] = "vector"
    dx: float
    dy: float
    head: float = Field(default=10.0, gt=0)
    #: Stage pixels per unit of `magnitude`, for this scene.
    #:
    #: Layout normalises arrow lengths *within a scene* — the largest `magnitude`
    #: gets `VECTOR_MAX_LENGTH`, the rest are proportional — because a velocity
    #: and an acceleration drawn to one scale would be a category error. A live
    #: `magnitude` needs that ratio to resize the arrow, and the alternative was
    #: to restate the normalisation in JavaScript, which is two implementations
    #: of one rule and therefore a drift. Baked, the player only multiplies.
    length_scale: float = Field(default=0.0, ge=0)
    #: Id of the body the arrow is drawn from, already resolved by layout.
    #:
    #: Kept for the same reason `EmitterElement.anchor` is: a vector is not *at*
    #: a coordinate, it acts on something. Layout bakes a number in either way,
    #: but a body that moves during playback would otherwise leave its arrows
    #: behind — the failure the mount pass was written to fix for emitters.
    anchor: str | None = None


class AxisElement(_Element):
    """A graduated axis. `length` runs along `heading` from (x, y)."""

    kind: Literal["axis"] = "axis"
    length: float = Field(gt=0)
    heading: float = 0.0
    ticks: int = Field(default=5, ge=0, le=50)
    tick_labels: list[str] = Field(default_factory=list)


class DimensionElement(_Element):
    """A double-headed annotation between two points, for a range or a height.

    `from_id`/`to_id` are the two objects being measured, when the annotation
    measures between objects rather than between fixed points. They are what
    lets the callout follow a car down a lane; without them the number would sit
    still while the thing it names moved out from under it — the same
    label-and-thing-disagree failure the `safe_distance` binding exists to fix.
    """

    kind: Literal["dimension"] = "dimension"
    start: RenderPoint
    end: RenderPoint
    from_id: str | None = None
    to_id: str | None = None


class AngleElement(_Element):
    """An arc marking the angle at a vertex between two directions."""

    kind: Literal["angle"] = "angle"
    center: RenderPoint
    from_degrees: float
    to_degrees: float
    radius: float = Field(default=40.0, gt=0)


class RegionElement(_Element):
    """A translucent filled area, optionally outlined."""

    kind: Literal["region"] = "region"
    points: list[RenderPoint] = Field(min_length=3)
    opacity: float = Field(default=0.15, ge=0, le=1)


class WaveElement(_Element):
    """A sampled curve — sine or envelope. Layout bakes the samples."""

    kind: Literal["wave"] = "wave"
    points: list[RenderPoint] = Field(min_length=2)


class ReadoutElement(_Element):
    """A text panel.

    `text` is drawn with `fillText`/`textContent` and never parsed. A formula
    reaches this field already turned into geometry at build time, not as an
    executable string (decision D4).
    """

    kind: Literal["readout"] = "readout"
    text: str = ""
    width: float = Field(default=0.0, ge=0)
    height: float = Field(default=0.0, ge=0)
    align: Literal["left", "center", "right"] = "left"


#: Discriminated on `kind`, so an unregistered one raises by name instead of
#: being dropped — the Fabric.js `ClassRegistry` behaviour, minus the registry.
RenderElement = Annotated[
    BodyElement
    | EmitterElement
    | ZoneElement
    | LinkElement
    | TravelerElement
    | TraceElement
    | VectorElement
    | AxisElement
    | DimensionElement
    | AngleElement
    | RegionElement
    | WaveElement
    | ReadoutElement,
    Field(discriminator="kind"),
]


class RenderStep(_Model):
    """One beat. `states` overrides element props for this step only."""

    id: str = Field(min_length=1)
    title: str = ""
    narration: str = ""
    highlights: list[str] = Field(default_factory=list)
    states: dict[str, dict[str, PropValue]] = Field(default_factory=dict)


class RenderControl(_Model):
    """A control, resolved against the elements that exist in its scene.

    Mirrors `StoryboardControl` and adds nothing: `unit` and `action` are here
    because the player needs them to label a slider and to run a button.
    """

    id: str = Field(min_length=1)
    type: Literal["slider", "toggle", "button"]
    label: str = ""
    target_property: str = Field(min_length=1)
    min: float | None = None
    max: float | None = None
    default: float | None = None
    step: float | None = None
    unit: str = ""
    action: str | None = None


class RenderScene(_Model):
    """One drawable scene: geometry resolved, semantics intact."""

    id: str = Field(min_length=1)
    title: str = ""
    teaching_goal: str = ""
    #: The preset the layout used. Kept so a comparison can say *why* two specs
    #: differ, and so `--render-sample` can assert it matches the sample.
    preset: str = ""
    #: Element id -> id of the element it is drawn relative to. Layout resolves
    #: relations into coordinates; this preserves the relation for the player,
    #: which needs it to move an emitter with the car it rides on.
    attachment: dict[str, str] = Field(default_factory=dict)
    #: Body id -> the id of the `safe_distance`-role zone mounted on it.
    #:
    #: Derived from the scene graph, not from a prop name, and that is the whole
    #: point. `proximity_gate` needs a threshold; `behaviors.js` used to look for
    #: one on the *body* (`lookup(bodyId, "safe_distance")`), which works for the
    #: hand-written baseline because it puts `safe_distance` in `scene.params` and
    #: fails silently for every document that roled a zone instead. The avoidance
    #: document wrote `safe_zone.radius: 0.8`, the lookup found nothing, and the
    #: car drove straight past its obstacles with the danger flag never set —
    #: while the lesson on screen was about deciding to avoid them.
    #:
    #: A zone whose role is `safe_distance` and whose `of` is a body *is* that
    #: body's threshold. Reading that off the relation is the same move
    #: `_chain_boxes` makes when it asks "does a link touch it?" instead of
    #: checking a role name.
    thresholds: dict[str, str] = Field(default_factory=dict)
    elements: list[RenderElement] = Field(default_factory=list)
    steps: list[RenderStep] = Field(default_factory=list)
    controls: list[RenderControl] = Field(default_factory=list)
    #: Scene-level parameters, addressed as `scene.<name>` by controls.
    params: dict[str, PropValue] = Field(default_factory=dict)


class RenderStage(_Model):
    """The drawing surface in CSS pixels. The player handles device pixel ratio."""

    width: float = Field(default=960, gt=0)
    height: float = Field(default=600, gt=0)


class RenderSpec(_Model):
    """Everything needed to draw, and nothing that needs interpreting."""

    spec_version: int = SPEC_VERSION
    storyboard_id: str = ""
    lesson_id: str = ""
    document_id: str = ""
    title: str = ""
    subject: str = ""
    eyebrow: str = ""
    stage: RenderStage = Field(default_factory=RenderStage)
    #: The glyph geometry the scenes below refer to, by name.
    #:
    #: Carried in the spec rather than fetched by the player, for the same reason
    #: every other number is: the spec is the player's *only* input, and a spec
    #: that draws differently depending on what else is on disk is not a contract.
    #: It also keeps the player's fetch count at one, which is asserted.
    #:
    #: Only the glyphs this spec actually names are here — the other four in
    #: `registry.T2_GLYPHS` are not copied into every file that does not use them.
    glyphs: dict[str, Glyph] = Field(default_factory=dict)
    scenes: list[RenderScene] = Field(default_factory=list)

    @model_validator(mode="after")
    def _a_named_glyph_must_travel_with_the_spec(self) -> RenderSpec:
        """A `body.glyph` with no geometry behind it is not a drawable spec.

        Without this the player is handed a name it cannot resolve and has to
        decide between throwing and drawing the plain shape — and both are worse
        than a layout that refused to produce the file. This is the same shape as
        `required_relations`: the failure is knowable here, so it is a failure
        here, rather than something the next layer has to notice.
        """
        named = {
            element.glyph
            for scene in self.scenes
            for element in scene.elements
            if isinstance(element, BodyElement) and element.glyph is not None
        }
        missing = sorted(named - set(self.glyphs))
        if missing:
            raise ValueError(
                f"这些字形被 body 引用了，spec 却没有带上它们的几何：{'、'.join(missing)}"
            )
        return self
