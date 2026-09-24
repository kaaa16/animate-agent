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
    #: The point `heading` turns this body about, in the body's own local frame:
    #: `(0, 0)` is the centre of the shape and `+y` points down.
    #:
    #: `(0, 0)` is not a sentinel — it is a real and common answer, and it is what
    #: every body did before this field existed. A car turns about its middle and
    #: looks right doing it. An `arm` turns about its shoulder instead, and that
    #: difference is the whole of what this carries.
    #:
    #: Measured against **what the body draws**, not against `width`/`height`.
    #: Those two differ whenever a glyph is involved — the player fits the glyph's
    #: ink into the box and centres it, so a box taller than the ink leaves space
    #: above and below. Layout answers this with `_drawn_size`, and the field
    #: carries the result rather than the recipe.
    #:
    #: Baked by layout from the object's role, never read from `props`: see
    #: `_BODY_PIVOTS` in `layout.py` for why a hinge is the code's decision rather
    #: than a coordinate the model gets to write.
    pivot: tuple[float, float] = (0.0, 0.0)
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


class VerdictElement(_Element):
    """A ✓ / ✗ / ! pinned through another object's corner.

    `mark` is a **word**, not a shape. `registry.py` prints the three legal names
    into the vocabulary and `validation.py` refuses any other, so by the time a
    spec exists the value is one of three and the drawer's job is to turn it into
    a colour and a glyph. That is the arrangement `tone` already has, and it is
    here for the same reason: a name the drawer cannot place would otherwise
    degrade quietly into a mark that reads as deliberate.

    `glyphs` is a table for the same reason `CardElement.tags` is one, and the
    mistake it prevents was caught by `player_smoke.mjs` on the first fixture
    that flipped a mark mid-scene. `mark` is live, so a beat may turn a 存疑 into
    a 通过 — and a single `glyph` field baked from the *declared* mark then draws
    a warning triangle inside a badge the drawer has just coloured green. The
    spec would not contain `check` at all, so the player could not have drawn it
    even if it had asked. Only the marks some beat can reach are in here.

    A mark with no glyph data (a half-installed checkout) is simply absent from
    the table, which leaves the badge drawn as a bare disc in that mark's colour.

    `label_side` and `label_width` are baked for the reason `readout.align` is:
    they are positions, and positions are the layout's business (decision D3).
    """

    kind: Literal["verdict"] = "verdict"
    mark: str = "warn"
    text: str = ""
    size: float = Field(default=0.0, ge=0)
    glyphs: dict[str, str] = Field(default_factory=dict)
    label_width: float = Field(default=0.0, ge=0)
    label_side: Literal["right", "left", "below"] = "right"
    #: The object this mark judges. Kept so the badge follows a body that moves
    #: during playback, exactly as an emitter's or a vector's does.
    anchor: str | None = None


class CardTag(_Model):
    """One type as a card draws it: the words, and the pill that holds them.

    Baked per type rather than derived at draw time because both halves are
    decided above the player. The words are the glossary in `registry.py` — a
    fact about the vocabulary, not about this card — and the width is a text
    measurement, which in this project is the layout's business. `drawReadout`
    reading `element.width` instead of calling `measureText` is the precedent.
    """

    text: str = ""
    width: float = Field(default=0.0, ge=0)


class CardElement(_Element):
    """One value set large, with the type it is an example of beneath it.

    `text` and `type` are both live, so a beat can rewrite either — which is the
    teaching move this primitive is for ("同一个位置，换一个值"). That is why
    `tags` is a table rather than a single pair of fields: a beat that sets
    `type: "boolean"` needs the words and the pill for *that* type, and there is
    nowhere in the player to derive them from. Only the types some beat can
    actually write are in here, so the table is the size of the lesson rather
    than six entries times every card.

    `value_type` is the name the model wrote, kept because the tag's *colour* is
    keyed on the name while its words come out of `tags` — the glossary lives in
    `registry.py` and the palette in the theme, which is the same split `tone`
    has.
    """

    kind: Literal["card"] = "card"
    text: str = ""
    value_type: str = ""
    tags: dict[str, CardTag] = Field(default_factory=dict)
    width: float = Field(default=0.0, ge=0)
    height: float = Field(default=0.0, ge=0)


#: What a tree node's value *looks like*, for the one purpose of choosing a
#: colour. Named rather than inlined so `layout._value_kind` can be annotated
#: with it — a function that sniffs a type and returns `str` is a function the
#: type checker cannot tell from one that returns anything at all.
TreeValueKind = Literal["string", "number", "literal", "text"]


class TreeLine(_Model):
    """One row of a `tree`: how deep it sits, and what it says.

    `prefix` holds the key **with its separator**, so the drawer places the value
    without measuring anything: it draws `prefix`, then `text` at the width
    `prefix` came out. That is the arrangement `CardTag.width` and
    `ReadoutElement.width` already have — a text measurement is the layout's
    business (decision D3) — and it is why the colon is baked into the string
    rather than reconstructed on the canvas, where `": "` would have to come out
    the same width in two different languages.

    `value_kind` is the *sniffed* type of `text` — `"20"` is a number, `"\\"x\\""` a
    string — and it is baked for the reason `CardElement.tags` is: the drawer is
    handed a name and looks up a colour, and the words a person would read the
    kind from are on the canvas, not in a parser. It is a hint and never a claim:
    a JSON tree is text a lesson wrote, not a document anyone validated.

    `depth` is what every form reads. Which node is under which is carried by
    indentation, so a row that lost its depth would still draw every character it
    has and stop being a picture of a hierarchy.

    `icon` is a **name**, resolved by the player against `RenderSpec.glyphs` the
    way `body.glyph` is, and it is written by the model *inside* `text` — see
    `registry.TREE_ICONS` for the syntax and for why a separate per-row list
    would have been worse.

    `x` and `y` are where the row draws, as an offset from the panel's text
    origin, and they are baked for **every** form. Two of the five need it and
    the other three would be fine without it, but one rule for all five is one
    rule to keep: a form that computed its own positions in the drawer would be
    the second place the geometry lives, and the two would part company the first
    time a constant moved. `None` means the spec was written before the fields
    existed, and the drawer falls back to the outline's own rule — `depth *
    TREE_INDENT` across, `index * TREE_LINE_HEIGHT` down — which is exactly what
    those specs were drawn with.
    """

    depth: int = Field(default=0, ge=0)
    prefix: str = ""
    text: str = ""
    value_kind: TreeValueKind = "text"
    icon: str = ""
    x: float | None = None
    y: float | None = None


#: How a `tree` arranges its rows. The closed set is `registry.TREE_FORMS`,
#: written out here rather than unpacked so the annotation is readable — and held
#: equal to it by `test_render_block.py`, the way `CodeKind` is held to
#: `registry.CODE_KINDS`.
TreeForm = Literal["outline", "branch", "mind", "brace", "boxes"]


class TreeElement(_Element):
    """A nested structure, drawn in whichever of the five forms fits it.

    **Both axes are bounded, and that is the design constraint rather than a
    preference.** The stage is 960 x 600 and the frame a block may occupy is
    narrower than that, so a form whose width grows with the number of *leaves*
    is a form that eventually cannot be rendered at all — and the failure would
    arrive after the last model call, which is the worst possible time to learn
    it. Every form here therefore puts **siblings on `y` and depth on `x`**:
    width is a function of how deep the content goes, never of how much of it
    there is.

    An earlier version of this docstring said a tidy tree's width grows with its
    leaves and that six of them is the whole stage. That measurement is right for
    one way of drawing a tidy tree and wrong for the other, and the difference is
    worth keeping: spreading the leaves across the available width is a choice,
    not a property of tree diagrams. Put each *depth* at a fixed x instead and a
    chain of eight nodes costs one row rather than eight, which is the case the
    outline is worst at.

    `form` is settled here rather than by the drawer for the reason `code`'s
    `language` is: it changes the geometry, and geometry is baked. It is static —
    a beat cannot change it — because every form decides the panel's size
    differently, and a box cut for one form cannot hold another.

    The five, and what each is for, are argued in `registry.TREE_FORMS`.
    """

    kind: Literal["tree"] = "tree"
    form: TreeForm = "outline"
    lines: list[TreeLine] = Field(default_factory=list)
    #: The rows to emphasise, verbatim from the prop (`"3"`, `"2-4"`). Live, so a
    #: beat can walk a lesson down a structure one node at a time; parsed by the
    #: drawer, because a range is data and parsing four characters is not
    #: executing code (decision D4 is about the latter).
    focus: str = ""
    width: float = Field(default=0.0, ge=0)
    height: float = Field(default=0.0, ge=0)


#: What one span of code *is*. The closed set is `registry.CODE_KINDS`, written
#: out here rather than unpacked so the annotation is readable — and held equal to
#: the registry's list by a test, the same arrangement `Tone` and `TONE_GLOSSES`
#: have. `text`, `name` and `operator` are deliberately three names for what the
#: drawer colours the same way: they are different *facts* (whitespace between
#: tokens, an identifier, punctuation) and one *appearance*.
CodeKind = Literal[
    "text", "name", "keyword", "builtin", "string", "number", "comment", "operator", "literal"
]


class CodeSpan(_Model):
    """A run of characters in one code line that shares a colour."""

    text: str = ""
    kind: CodeKind = "text"


class CodeLine(_Model):
    """One row of a `code` block. A line with no tokens is a blank one."""

    spans: list[CodeSpan] = Field(default_factory=list)


class CodeElement(_Element):
    """A block of source, tokenised at build time and coloured at draw time.

    The split is Shiki's — what the code *is* is decided once, above the player,
    and the player turns names into colours. The alternative, shipping the source
    and a tokeniser, would put a second implementation of the highlighting in
    JavaScript and give the spec a string the player has to parse.
    """

    kind: Literal["code"] = "code"
    lines: list[CodeLine] = Field(default_factory=list)
    #: The name the model wrote, kept because the *colour* is keyed on it while
    #: the choice of tokeniser was made above — the same split `CardElement.value_type`
    #: has. Nothing in the drawer reads it today; it is here so a spec says which
    #: language it was told it was, and so `player_smoke` can print it.
    language: str = ""
    focus: str = ""
    width: float = Field(default=0.0, ge=0)
    height: float = Field(default=0.0, ge=0)


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
    | ReadoutElement
    | VerdictElement
    | CardElement
    | TreeElement
    | CodeElement,
    Field(discriminator="kind"),
]


class RenderStep(_Model):
    """One beat. `states` overrides element props for this step only."""

    id: str = Field(min_length=1)
    title: str = ""
    narration: str = ""
    #: How long this beat holds before the player moves on, in seconds.
    #:
    #: `0.0` means **"no timing was declared"**, and it is the default because
    #: every spec written before this field existed has to keep loading. The
    #: player reads it as "sit on this beat and never move by yourself", which
    #: is exactly what it did before — so an old file behaves like an old file
    #: rather than silently acquiring a rhythm nobody chose for it.
    #:
    #: The layout layer fills it in (`layout.beat_duration`) from the
    #: narration. Not from the model: `storyboard/prompts.py` forbids the
    #: storyboard layer from writing 时长 in as many words, alongside 坐标 and
    #: 颜色. A throw's flight time is derived down here for the same reason, and
    #: a beat's length has the same shape of answer — the narration is what the
    #: beat is *for*, so how long it takes to read is how long it needs.
    duration: float = Field(default=0.0, ge=0)
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
        # A tree row's icon arrives the same way and fails the same way: the
        # drawer holds a name and looks its geometry up, so a name with nothing
        # behind it is a row marker that silently is not there.
        named.update(
            line.icon
            for scene in self.scenes
            for element in scene.elements
            if isinstance(element, TreeElement)
            for line in element.lines
            if line.icon
        )
        missing = sorted(named - set(self.glyphs))
        if missing:
            raise ValueError(f"这些字形被 spec 引用了，却没有带上它们的几何：{'、'.join(missing)}")
        return self
