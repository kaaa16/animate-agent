"""Deterministic layout: a semantic storyboard scene becomes coordinates.

Node in the pipeline: `StoryboardIR -> [this] -> RenderSpec -> [player] -> pixels`.

Why this is the layer that decides whether the picture is worth looking at
-----------------------------------------------------------------------

`StoryboardIR` carries no coordinates at all (decision D3): the model says a car
is a `vehicle` and an octagon is an `obstacle`, and *this* module decides where
they go. So the quality of every generated frame is settled here, not by the
model. That is the whole reason the model is not asked for numbers — a model's
coordinates are neither reliable nor reproducible, and "same input, same frames"
is the property every comparison in this project rests on.

What it is not
--------------

**Never force-directed.** Four fixed slot grids and a fallback, and a scene that
does not fit raises `LayoutError` rather than being nudged into place. Readable
placement of a semantic graph is a research problem; guessing at it produces a
picture that is wrong in a way nobody can point at. A hard failure names the
scene, and a hard failure is something a retry or a preset change can fix.

**Never invents semantics.** Every `props` value is copied verbatim from the
storyboard. Layout reads props to *choose geometry* (a `radius` becomes a size)
and never rewrites one, so playback mutating `props` through `object_states`
cannot fight with a value layout baked in.

Proportions, not pixels
-----------------------

Every constant below is a ratio of `RenderStage`, or a default for a value the
storyboard did not give. The `lane` ratios reproduce the quality baseline's own
placement — `animation/templates.py` puts its car at 120/960 across and its
first obstacle at 410/960 — so a laid-out scene and the translated baseline can
be put side by side and be talking about the same picture. That comparison is
the only reason M1 (`legacy.py`) was built before this module.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from animate_agent.rendering.models import (
    AngleElement,
    AxisElement,
    BodyElement,
    DimensionElement,
    EmitterElement,
    LinkElement,
    ReadoutElement,
    RenderControl,
    RenderElement,
    RenderPoint,
    RenderScene,
    RenderSpec,
    RenderStage,
    RenderStep,
    TraceElement,
    TravelerElement,
    VectorElement,
    ZoneElement,
)
from animate_agent.rendering.registry import (
    PENDING_ALTERNATIVES,
    PENDING_PRIMITIVES,
    ROLE_TO_PRIMITIVE,
    Glyph,
    stage_range,
)
from animate_agent.storyboard.models import (
    PropValue,
    StoryboardControl,
    StoryboardIR,
    StoryboardObject,
    StoryboardScene,
    StoryboardStep,
)


class LayoutError(ValueError):
    """A scene cannot be arranged: no placer for the preset, or no geometry to compute.

    Raised rather than worked around. A scene that silently loses an object, or
    gets one placed at the origin, is a picture with a hole in it that reports
    success — the failure this project spends the most effort avoiding.
    """


@dataclass(frozen=True, slots=True)
class _Box:
    """Where an object goes and how big it is. Centre-anchored, stage coordinates."""

    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class _Frame:
    """What a preset may arrange in, once the panels have taken their corner.

    **Panels are a corner, not a column**, and that distinction is the whole
    reason this exists. `FIELD_RIGHT_RATIO = 0.60` and `HUB_RADIUS_X_RATIO = 0.22`
    were both chosen to keep clear of the readouts — sized, in each case, for a
    stack of panels that the scenes using that preset have never produced. A
    `field` scene with one caption was still laying itself out around a
    seven-panel wall.

    So a preset asks *how far right may I go at this height*, and gets an answer
    about the panels that are actually there. Above, below or between them the
    answer is the stage margin; level with one, it is that panel's left edge.

    A preset that would reach into a panel it cannot see — because it places
    several things at different heights — is still caught by `_check_fit`. This
    is a shape a preset can use, not a gate.
    """

    #: Stage minus `STAGE_MARGIN_RATIO` on all four sides.
    left: float
    top: float
    right: float
    bottom: float
    #: The panels the layout decided, as boxes. Empty when the scene has none.
    panels: tuple[_Box, ...] = ()

    def _beside(self, y: float, half_height: float) -> list[_Box]:
        """The panels at this height — the ones that are in the way sideways.

        `_check_fit` calls two boxes overlapping only when *both* axes overlap, so
        one axis of clearance is enough for them to read as separate. This is that
        rule, applied to a band instead of to one box.
        """
        return [
            box
            for box in self.panels
            if box.y - box.height / 2 < y + half_height and box.y + box.height / 2 > y - half_height
        ]

    def right_of(self, y: float, half_height: float = 0.0) -> float:
        """The rightmost **edge** allowed for something occupying `y ± half_height`.

        An edge rather than a centre, so a caller subtracts half of whatever it
        is placing. That keeps the one number a preset cannot derive — the size
        of the thing it is about to put down — at the call site, where the size
        came from in the first place.
        """
        edges = [box.x - box.width / 2 for box in self._beside(y, half_height)]
        return min([self.right, *[edge - PANEL_CLEARANCE for edge in edges]])

    def left_of(self, y: float, half_height: float = 0.0) -> float:
        """The mirror of `right_of`, for the presets that grow both ways."""
        edges = [box.x + box.width / 2 for box in self._beside(y, half_height)]
        return max([self.left, *[edge + PANEL_CLEARANCE for edge in edges]])


def _frame(stage: RenderStage, panels: dict[str, _Box]) -> _Frame:
    margin_x = stage.width * STAGE_MARGIN_RATIO
    margin_y = stage.height * STAGE_MARGIN_RATIO
    return _Frame(
        left=margin_x,
        top=margin_y,
        right=stage.width - margin_x,
        bottom=stage.height - margin_y,
        panels=tuple(panels.values()),
    )


# --------------------------------------------------------------------------
# Proportions. Ratios of the stage, so a larger canvas scales rather than crops.
# --------------------------------------------------------------------------

#: The baseline's lane sits at 310/600 (`animation/templates.py`: the car is at
#: y=310). Below centre, which leaves room above for the fan and the swerve.
LANE_Y_RATIO = 310 / 600

#: 120/960, the baseline's car position.
LANE_VEHICLE_X_RATIO = 120 / 960

#: Where the baseline puts its obstacles: x=410 and x=570 of 960 across.
#:
#: Written as the baseline's own fractions rather than as round decimals, so a
#: laid-out scene lands on the *same* coordinates as the translated baseline and
#: not merely near them. `0.42` would put the first obstacle 6px to the left of
#: where `templates.py` puts it, and the side-by-side comparison this module
#: exists to enable would be off by a nudge nobody could account for.
#: `test_the_lane_ratios_are_the_baselines_own` holds these to those numbers.
LANE_OBSTACLE_FIRST_X_RATIO = 410 / 960
LANE_OBSTACLE_SPACING_RATIO = (570 - 410) / 960

#: How far off the lane an obstacle sits, as a fraction of `safe_distance`.
#:
#: This number is load-bearing and couples to the player, so it is derived from
#: the constraint rather than picked: for the car to ever leave the danger
#: envelope it must be able to move clear of an obstacle, i.e.
#: `offset + max_dodge > safe_distance`. With the player's `MAX_DODGE = 46` and
#: the baseline's `safe_distance = 76`, anything under 30 pins the car in a
#: permanent danger state — the picture still moves, so nothing looks broken.
#: `test_the_car_can_escape_the_danger_envelope` reads `MAX_DODGE` out of
#: `behaviors.js` and checks the inequality, so the two layers cannot drift.
LANE_OBSTACLE_OFFSET_RATIO = 0.5

# --------------------------------------------------------------------------
# Defaults for values the storyboard did not provide.
# --------------------------------------------------------------------------

#: `Obstacle.radius`'s default in the frozen baseline.
DEFAULT_OBSTACLE_RADIUS = 32.0
DEFAULT_SENSOR_RADIUS = 150.0
DEFAULT_SENSOR_RAYS = 13
DEFAULT_FOV = 180.0
DEFAULT_SAFE_DISTANCE = 76.0

#: The range `EmitterElement` accepts. Named here so a prop the model wrote
#: outside it can be pulled back in rather than reaching the model and throwing;
#: see `_bounded`.
MIN_FOV = 1.0
MAX_FOV = 360.0
MAX_SENSOR_RAYS = 181

#: Slack on the bounds and overlap checks, in stage units. Not zero: a box whose
#: edge lands exactly on the canvas edge is fine, and floating-point division on
#: the ratios above routinely produces 403.19999999999993 rather than 403.2.
FIT_TOLERANCE = 2.0

#: The baseline draws an obstacle as 8 vertices alternating between `r` and
#: `0.78r` (`frontend/demo/app.js:696`). Two numbers an atom computes exactly —
#: which is why this is parametric and not a T2 glyph (decision D1).
OBSTACLE_SIDES = 8
OBSTACLE_INNER_RATIO = 0.78

#: A `chain` is a left-to-right pipeline, so its nodes sit on a horizontal spine
#: at 252/600 — above centre, which leaves the lower third for the actor row.
#: How far the spine runs, left and right, is the frame's to say: see
#: `_chain_boxes`.
CHAIN_Y_RATIO = 0.42
#: Bodies that are not part of the pipeline itself (the thing being driven) sit
#: below it. Same relationship the ROS baseline has between its two nodes on
#: y=230 and the `RobotCar` "runtime" at y=430: the chain is not the subject.
CHAIN_ACTOR_Y_RATIO = 0.76

#: `generic` has no geometry to express, only an order, so it is an honest column:
#: objects top to bottom in the order the model wrote them, which is the order the
#: narration introduces them.
GENERIC_COLUMN_X_RATIO = 0.3

#: `hub` is a star: one centre and its leaves on a ring around it.
#:
#: An **ellipse** rather than a circle, because the stage is not square: a circle
#: sized to fit the height leaves the leaves bunched in a vertical line, and one
#: sized to fit the width leaves nothing above or below.
#:
#: The x radius is **measured, not kept here** (`_hub_radius_x`). It was 0.22,
#: with a note saying that 0.30 would collide with the panel column — and
#: sweeping the constant says otherwise: 0.29 fails on the *left* leaf leaving
#: the *left edge of the canvas*, and the panels never appear in the failure.
#: Four leaves sit at 0°/90°/180°/270°, so nothing shares a corner with the
#: readouts to collide with. The number was set by one constraint and explained
#: by another.
HUB_CENTER_X_RATIO = 0.32
HUB_CENTER_Y_RATIO = 0.5
HUB_RADIUS_Y_RATIO = 0.30

#: `field` is a 2D plot: axes crossing at an origin, one launch slot per
#: projectile, trajectories arcing up and to the right.
#:
#: The origin sits low and left so "+y up" is the direction with room in it — a
#: parabola drawn from a centred origin has nowhere to arc. Where the plot *ends*
#: on the right is the frame's to say: it used to be 0.60 of the width, which was
#: chosen to stay clear of the panel column and then paid for that room in every
#: scene, whether or not the panels were there.
FIELD_ORIGIN_X_RATIO = 0.15
FIELD_ORIGIN_Y_RATIO = 0.82
FIELD_TOP_RATIO = 0.12

#: Trajectories are drawn from a **fixed launch speed**, with the 45° throw made
#: to span its whole slot; `_arc_points` derives every other angle from
#: `R = v²sin2θ/g` and `H = v²sin²θ/(2g)`. That is not an aesthetic choice — it
#: is what makes the document's own 注意事项 visible. At one speed, 30° and 60°
#: land in the same place, 45° goes furthest, and 90° goes nowhere: three facts
#: the lesson states, all of which a "fit the biggest arc into the box" rule
#: would quietly contradict.

#: How many points a trajectory is sampled at. Enough that the curve reads as a
#: curve at 960px wide, few enough that the spec stays a readable size.
TRACE_SAMPLES = 48

#: `trace.length` has **no declared unit** yet (see M4 跟进 1 in
#: `docs/storyboard-milestone.md`), so this is a reference point rather than a
#: conversion: a trace is drawn across this fraction of its slot, clamped so a
#: legal-but-absurd value still produces a visible arc. The real chain writes
#: 100~140 for a trajectory it has fully drawn, which is what this is calibrated
#: against; when the vocabulary declares the unit, this constant retires.
TRACE_LENGTH_REFERENCE = 200.0
TRACE_SPAN_MIN = 0.25
TRACE_SPAN_MAX = 1.0

#: Vector arrows. `magnitude` carries no unit and is not comparable across roles
#: — a velocity of 50 m/s and a gravitational acceleration of 9.8 m/s² drawn to
#: the same scale would say the acceleration is 5× smaller, which is a category
#: error, not a fact. So arrows are scaled **within** a scene, where all the
#: magnitudes come from one lesson about one quantity: the largest gets the full
#: length, the rest are proportional.
#:
#: `VECTOR_MIN_FRACTION` is the part that is not proportional. A lone vector —
#: scene-1 gives the projectile its `v0` *and* its gravity — would otherwise be
#: drawn at 20% of the others and read as decoration. The floor keeps it legible
#: without touching the ratios that carry meaning (30:40 stays 0.6:0.8).
VECTOR_MAX_LENGTH = 170.0
VECTOR_MIN_FRACTION = 0.35

#: An angle arc's radius, clamped rather than taken literally for the same
#: missing-unit reason. A value that is obviously already in stage pixels is used
#: as written.
ANGLE_RADIUS_DEFAULT = 46.0
ANGLE_RADIUS_MIN = 20.0
ANGLE_RADIUS_MAX = 120.0

#: Stage pixels a thrown body covers per second, and the floor on how quick a
#: short throw is allowed to be. The duration is baked onto the body so the
#: player only interpolates; it is derived from the arc's *length* rather than
#: from a prop, because `speed` on a projectile would need a unit the model has
#: never been given — and a 竖直上抛 and a 平抛 are the same launch at the same
#: speed, so the shorter arc is genuinely over sooner.
THROW_PIXELS_PER_SECOND = 260.0
THROW_MIN_DURATION = 1.2

#: How long one beat lasts, derived from the narration it carries.
#:
#: Same reasoning as `THROW_MIN_DURATION` one paragraph up, applied to the beat
#: rather than to the body: the storyboard layer is told in as many words not to
#: write 时长 (`storyboard/prompts.py`, alongside 坐标 and 颜色), so the number
#: comes from here out of the one thing the model did write.
#:
#: Until this existed nothing computed a beat's length at all, and the player
#: made up the difference by never moving: it loops on whatever beat is loaded,
#: so **beats 2..N were only ever visible if a person clicked through them**.
#: An entire scene could be written, validated, rendered and reviewed without
#: anybody noticing that no viewer would ever see past its first frame.
#:
#: The rate is **not** a reading speed, and the first version calling it one was
#: the mistake. Nothing reads the narration at the beat's pace — it is printed
#: in the list beside the picture, and what the beat has to give the eye is time
#: to watch whatever changed. A rate of 6 characters per second was a reading
#: clock doing a job that has no reading in it, and it made the length of the
#: *sentence* decide the length of the *shot*: the same document cut into the
#: same beats ran 99 seconds where the hand-written baseline ran 22, purely
#: because its descriptions were written one clause longer.
#:
#: So the rate is now a nudge and nothing more — one extra second per 18
#: characters — and `BEAT_BASE_SECONDS` sets the pace. Across the whole legal
#: range of `description` that is a band of roughly 4.5 to 7 seconds, which is
#: where the baseline sits and, on the evidence of the one scene anybody could
#: stand to watch, where a beat wants to be.
#:
#: What is still true is the trade underneath: two beats of equal length hold
#: for equal time, one of them the point of the scene and the other a
#: transition. Buying editorial rhythm back means letting the model write a
#: duration, which means reversing the rule in `prompts.py` — a change of
#: position, not a parameter.
BEAT_BASE_SECONDS = 3.8
BEAT_CHARS_PER_SECOND = 18.0
BEAT_MIN_SECONDS = 3.8
BEAT_MAX_SECONDS = 7.0

#: Graduations on an axis when the storyboard did not say how many.
DEFAULT_AXIS_TICKS = 5

#: How far past an axis's tip its own label sits, in stage units. Mirrored from
#: `drawAxis` (`primitives.js`), and there is a test holding the two equal.
#:
#: It is a number this layer needs even though this layer does not draw the
#: label: the field has to stop short of the margin by the label's half-width
#: plus this, or the label is written off the edge of the canvas. Which is what
#: happened — widening the field to the margin pushed 水平方向 off the right edge,
#: and nothing caught it, because `_check_fit` compares boxes and an annotation
#: has none.
AXIS_LABEL_GAP = 18.0

#: Readouts stack in a right-hand column for **every** preset. One rule, because a
#: text panel has no relation to the geometry to hang off — it needs a slot that
#: does not move when the objects do. Colliding with a body is a `LayoutError`
#: like anything else; a panel is not more important than the picture.
#:
#: The column is **as wide as the widest text in it and no wider**, right-aligned
#: against the stage margin.
#:
#: It used to be a fixed 24% of the stage whatever the panel held, which cost two
#: things that did not look like each other. A four-character caption was handed
#: a 230×88 box and drew an empty panel with a word in the corner of it. And
#: `publisher-code` was handed the same 230px box and drew *past its own right
#: edge* — that line is 448px wide, the box was collision-checked at 230, and the
#: text went to 448 because nothing anywhere had looked at it. Nothing failed.
#:
#: So there is no fraction here any more. A panel takes what its text needs, up
#: to the width the stage can actually show, and a line wider than that is a
#: `LayoutError` rather than a line written off the canvas (`_panel_boxes`).
PANEL_RIGHT_RATIO = 0.97
#: Where a comfortable stack starts and ends — `y` centres, not edges. Not a
#: limit: `_panel_span` widens them when the panels have more to say than this.
PANEL_TOP_RATIO = 0.14
PANEL_BOTTOM_RATIO = 0.86
#: Between two stacked panels.
MIN_PANEL_GAP = 8.0
#: Between a panel and anything placed beside it. Not zero, and not small:
#: `_check_fit` accepts boxes that merely touch, and a body flush against a
#: panel's edge reads as part of the panel rather than as something the panel is
#: talking about.
PANEL_CLEARANCE = 24.0

#: The canvas keeps its own margin, so nothing is drawn flush against the edge.
#: It is also what the panel column is measured back from.
STAGE_MARGIN_RATIO = 0.03

#: Panel text metrics, in stage units.
#:
#: These used to live in the drawer: `drawReadout` fits a panel to its text when
#: `element.width` is falsy. But the layout always set `element.width`, so the
#: rule never once ran on a laid-out spec — and the layout's fixed column was the
#: thing it was meant to correct. It lives here now because a box is geometry and
#: geometry is this layer's job (decision D3), and because `_check_fit` can only
#: check a box this layer decided.
PANEL_FONT_SIZE = 14.0
#: Advance width per character, in ems. Two numbers rather than one average,
#: because these strings mix scripts — `雷达有效半径2到8米` is half full-width and
#: half narrow — and any single number is wrong at both ends. `W`/`F` are the
#: Unicode East Asian Width classes for a full-width glyph; the rest is `Inter`,
#: where an average Latin advance runs a little over half an em.
PANEL_WIDE_EM = 1.0
PANEL_NARROW_EM = 0.55
PANEL_PADDING = 12.0
PANEL_LINE_HEIGHT = 20.0
#: The narrowest panel still worth drawing: a two-character caption needs a box,
#: not a sliver.
MIN_PANEL_WIDTH = 120.0

#: role -> (width, height) in stage units, for roles a preset places as a box.
_BODY_SIZES: dict[str, tuple[float, float]] = {
    "vehicle": (62.0, 40.0),  # `RobotCar`'s frozen defaults
    "obstacle": (DEFAULT_OBSTACLE_RADIUS * 2, DEFAULT_OBSTACLE_RADIUS * 2),
    "node": (96.0, 48.0),
    "endpoint": (84.0, 44.0),
    "device": (72.0, 56.0),
    "agent": (56.0, 56.0),
    "projectile": (28.0, 28.0),
    # Tall and narrow, because an arm is read from its far end: the shape has to
    # be long enough that a swing moves something, and thin enough that the near
    # end does not swallow the pivot it turns about.
    "arm": (34.0, 96.0),
    "object": (64.0, 48.0),
}

#: role -> the local point a body turns about, as a fraction of its own box.
#:
#: `(0.0, 0.0)` is the centre of the shape, which is what every body did before
#: this table existed and what a car or a wheel still wants: a body that turns
#: about its own middle pivots in place. An `arm` wants the other thing — a hinge
#: at the *shoulder*, which for a box is the middle of its bottom edge — because
#: an arm pinned through its middle swings both ends at once and reads as a
#: spinning stick rather than as something reaching for a point.
#:
#: Keyed by role and not by a prop, which is the same call `_BODY_SIZES` makes
#: and the same one `_flight` makes for a throw's duration: where a thing hinges
#: is a fact about the kind of thing it is, not a number a lesson should have to
#: state. Decision D3 — 语义给模型，呈现给代码 — and it is also what keeps the
#: pivot out of reach of a model that would otherwise have to be trusted with a
#: coordinate.
#:
#: Fractions of **what the body draws**, not of its box — see `_drawn_size`. For
#: every role but `arm` there is nothing to say, because `(0, 0)` is the centre
#: of the shape in both readings.
_BODY_PIVOTS: dict[str, tuple[float, float]] = {
    "arm": (0.0, 0.5),
}

#: Glyph data lives in `assets/glyphs/` as build-time products of
#: `tools/build_glyphs.py`. Layout reads the directory rather than the
#: declaration list, so a glyph whose data is missing degrades to the parametric
#: shape instead of handing the player a name it cannot resolve.
#:
#: The gap that gate used to cover is closed from the other side now: the test
#: suite requires the declared set and the directory to be the same set, so
#: "declared but undrawable" is a failing test rather than a picture that is
#: quietly a rounded rectangle. What is left here is the honest fallback for a
#: half-installed checkout.
GLYPH_DATA_DIR = Path(__file__).resolve().parents[3] / "assets" / "glyphs"


def _available_glyphs() -> frozenset[str]:
    if not GLYPH_DATA_DIR.is_dir():
        return frozenset()
    return frozenset(path.stem for path in GLYPH_DATA_DIR.glob("*.json"))


def _load_glyph(name: str) -> Glyph:
    """One glyph's geometry, validated against the schema it ships in.

    A malformed asset is a `LayoutError` and not a pydantic error escaping from
    six frames down: the CLI and the pipeline both know how to report the first
    one, and the file has a name that belongs in the message.
    """
    path = GLYPH_DATA_DIR / f"{name}.json"
    try:
        return Glyph.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LayoutError(f"字形数据 `{path}` 读不出来或不合 schema：{exc}") from exc


# --------------------------------------------------------------------------
# Reading props — defensively, because a prop's type is not checked for shape
# --------------------------------------------------------------------------


def _number(obj: StoryboardObject, prop: str, default: float) -> float:
    """A numeric prop, or `default` when it is absent or not a number.

    `bool` is excluded explicitly: it is an `int` subclass in Python, so
    `float(True)` is 1.0 and a `visible: true` would silently become a size.
    """
    value = obj.props.get(prop)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _bounded(obj: StoryboardObject, prop: str, default: float, low: float, high: float) -> float:
    """A numeric prop pulled inside the range the render contract can express.

    The vocabulary declares prop *names* and not their ranges, so the model can
    write a legal-but-undrawable value: the real chain wrote `fov: 0` for a
    lidar that is not scanning yet, which the strict element model rejects
    outright. Clamping rather than raising is the deliberate half of that: a
    zero-width fan is a fan that is switched off, and failing the whole document
    over it would trade a missing sliver for a missing picture.

    The other half is the guard in `layout_scene`, which turns any remaining
    out-of-range value into a `LayoutError` naming the object. Without it the
    failure surfaces as a Pydantic traceback, which tells the reader nothing
    about which object, in which scene, the model got wrong.
    """
    return min(max(_number(obj, prop, default), low), high)


def _stage_distance(
    obj: StoryboardObject,
    primitive: str,
    prop: str,
    default: float,
) -> float:
    """A distance prop, held inside the range the stage can show.

    The backstop for the units problem, and only the backstop — the fix is the
    vocabulary saying `radius` is in 舞台像素 and `prop_out_of_range` rejecting a
    value that is not. This exists for the two cases validation cannot reach: a
    hand-written sample, and a `scene.safe_distance` that arrives from `params`
    rather than from a prop.

    Why clamp rather than raise: raising here is a `LayoutError` after the last
    LLM call, which is the exact failure the `drawable` gate was written to stop
    doing. A clamped circle is a visible circle. `_bounded` makes the same
    argument for `fov: 0`, and `_to_angle` has made it for its `radius` since the
    day the angle arcs landed.

    What it fixes, measured: `lidar_emitter.radius: 8` — eight metres in the
    document — drew an eight-pixel fan on a 960-pixel stage, and
    `safe_zone.radius: 0.8` a circle under one pixel across. Both were legible
    numbers in every layer's terms, and both were invisible on screen.
    """
    return _clamp_stage(primitive, prop, _number(obj, prop, default))


def _clamp_stage(primitive: str, prop: str, value: float) -> float:
    """`value` held inside the stage range `prop` is declared with, if it has one."""
    bounds = stage_range(primitive, prop)
    return value if bounds is None else bounds.clamp(value)


def _clamp_value(primitive: str | None, prop: str, value: PropValue) -> PropValue:
    """The same hold, for a value that might not be a number at all.

    Beat states go through here. `speed` is not consumed into geometry — nothing
    in this module reads it, `linear_motion` does, at playback — so a body's
    `speed: 60` sits in `props` until the player multiplies it by 90 and the car
    vanishes into a blur. Held here for the same reason the emitter's radius is:
    a number outside the legible range has no reader that can notice.
    """
    if primitive is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return _clamp_stage(primitive, prop, float(value))


def _props_for(obj: StoryboardObject, primitive: str) -> dict[str, PropValue]:
    """`obj.props`, carried verbatim — except a distance, which is held to the stage.

    The verbatim rule is what lets the player read `direction` or `magnitude` at
    playback: layout consumes a prop into geometry *and* keeps the value, rather
    than transferring it. This is the one exception, and it is not a rewrite so
    much as a consistency fix — `_stage_distance` already clamped the number that
    got drawn, and leaving the raw one in `props` would mean the circle was
    radius 24 while the danger gate that reads the same prop fired at 0.8.

    For any storyboard that passes validation the two are the same number, since
    `prop_out_of_range` rejects a value outside this range. The clamp only bites
    on a hand-written sample or a scene param, where there is no validator to ask.
    """
    props = dict(obj.props)
    for prop in props:
        props[prop] = _clamp_value(primitive, prop, props[prop])
    return props


def _safe_distance(scene: StoryboardScene) -> float:
    """The one number that is this scene's safety threshold.

    Three places need it and they must not disagree:

    - `_lane_boxes` offsets the obstacles by half of it;
    - `_to_zone` draws the circle that is labelled 安全距离;
    - `behaviors.js` fires `proximity_gate` when the nearest obstacle is closer.

    They used to read two different sources. The placer took `scene.params`, the
    gate took the zone's own `radius` — so a document that roled a zone instead
    of setting a scene param got a circle drawn at one distance and a car that
    reacted at another, and the reaction never happened at all. Reading one
    number in one place is the fix; which number it is matters less than that.

    Resolution order is "most specific wins": the zone the document declared,
    then the scene parameter the baseline uses, then the default. Both sources
    are clamped to the stage range, because a threshold in metres is not a
    threshold in pixels.
    """
    zone = next(
        (
            obj
            for obj in scene.objects
            if ROLE_TO_PRIMITIVE.get(obj.role) == "zone"
            and obj.role == "safe_distance"
            and isinstance(obj.props.get("radius"), (int, float))
            and not isinstance(obj.props.get("radius"), bool)
        ),
        None,
    )
    if zone is not None:
        return _stage_distance(zone, "zone", "radius", DEFAULT_SAFE_DISTANCE)
    declared = _scene_number(scene, "safe_distance", DEFAULT_SAFE_DISTANCE)
    return _clamp_stage("zone", "radius", declared)


def _relation(obj: StoryboardObject, name: str) -> str | None:
    """A relation prop's target id, or None. Validation has already run."""
    value = obj.props.get(name)
    return value if isinstance(value, str) and value else None


def _scene_number(scene: StoryboardScene, name: str, default: float) -> float:
    value = scene.params.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _primitive_of(obj: StoryboardObject) -> str | None:
    return ROLE_TO_PRIMITIVE.get(obj.role)


# --------------------------------------------------------------------------
# Placement — the preset decides where things go
# --------------------------------------------------------------------------


def _lane_boxes(scene: StoryboardScene, stage: RenderStage, frame: _Frame) -> dict[str, _Box]:
    """A corridor: bodies on the lane line, obstacles alternating either side.

    Obstacles alternate because the fourth beat teaches *comparing* left and
    right clearance. A preset that put every obstacle on the same side would
    make that beat a foregone conclusion — which is what the baseline does, and
    why its car always swerves the same way.

    **The only preset that ignores `frame`, and the only one that should.** Every
    position here is a fraction of the stage that `templates.py` also uses —
    `LANE_VEHICLE_X_RATIO` is its car, `LANE_OBSTACLE_FIRST_X_RATIO` its first
    obstacle — so that a laid-out scene and the translated baseline can be put
    side by side and be talking about the same picture (M2's acceptance). Moving
    the obstacles to fill the frame would make the comparison meaningless to buy
    room this preset is not short of: a corridor with a car, an obstacle and a
    dashed line through it does not look emptier for being narrower.
    """
    lane_y = stage.height * LANE_Y_RATIO
    obstacles = [obj for obj in scene.objects if obj.role == "obstacle"]
    movers = _bodies_but(scene, "obstacle")
    if not obstacles:
        raise LayoutError(f"场景 `{scene.id}` 选了 `lane`，却没有 `obstacle` 角色的对象")
    if not movers:
        raise LayoutError(
            f"场景 `{scene.id}` 选了 `lane`，却没有车道上的主体（非 obstacle 的 body）"
        )

    boxes: dict[str, _Box] = {}

    # The same number `_to_zone` draws and `proximity_gate` fires at. It has to
    # be: this ratio decides how far off the lane the obstacles sit, and the gate
    # decides whether the car reacts to them. Two sources meant a document could
    # get a car that reacted at one distance and obstacles placed for another —
    # which is exactly how the avoidance run came out with a car that drove
    # through everything.
    safe = _safe_distance(scene)
    offset = safe * LANE_OBSTACLE_OFFSET_RATIO
    first_x = stage.width * LANE_OBSTACLE_FIRST_X_RATIO
    spacing = stage.width * LANE_OBSTACLE_SPACING_RATIO
    for index, obj in enumerate(obstacles):
        radius = _number(obj, "radius", DEFAULT_OBSTACLE_RADIUS)
        side = -1.0 if index % 2 == 0 else 1.0
        boxes[obj.id] = _Box(
            x=first_x + spacing * index,
            y=lane_y + side * offset,
            width=radius * 2,
            height=radius * 2,
        )

    # Movers queue up from the left. More than one is rare, but stacking them
    # along x beats overlapping them at a single point.
    cursor = stage.width * LANE_VEHICLE_X_RATIO
    for obj in movers:
        width, height = _body_size(obj)
        boxes[obj.id] = _Box(x=cursor, y=lane_y, width=width, height=height)
        cursor += width + 24.0

    boxes.update(_panel_boxes(scene, stage))
    return boxes


# `node` and `readout` read alike but are different levels of the vocabulary — the
# first is a role under the `body` primitive, the second is a primitive name. Two
# explicitly named helpers, because getting it backwards fails as "there are no
# nodes in this scene" rather than as anything that points at the mistake.


def _with_role(scene: StoryboardScene, *roles: str) -> list[StoryboardObject]:
    return [obj for obj in scene.objects if obj.role in roles]


def _with_primitive(scene: StoryboardScene, primitive: str) -> list[StoryboardObject]:
    return [obj for obj in scene.objects if _primitive_of(obj) == primitive]


def _bodies_but(scene: StoryboardScene, *roles: str) -> list[StoryboardObject]:
    return [obj for obj in _with_primitive(scene, "body") if obj.role not in roles]


def _column(
    objects: list[StoryboardObject],
    x: float,
    top: float,
    bottom: float,
    size_of: Callable[[StoryboardObject], tuple[float, float]],
) -> dict[str, _Box]:
    """Stack objects evenly down a column, first at `top` and last at `bottom`.

    Even spacing rather than fixed pitch: it means the column stays inside the
    canvas for any count, and `_check_fit` is then only left to catch the case
    where the objects are simply too tall to be stacked at all.
    """
    if not objects:
        return {}
    step = 0.0 if len(objects) == 1 else (bottom - top) / (len(objects) - 1)
    boxes: dict[str, _Box] = {}
    for index, obj in enumerate(objects):
        width, height = size_of(obj)
        boxes[obj.id] = _Box(x=x, y=top + step * index, width=width, height=height)
    return boxes


def _is_wide(character: str) -> bool:
    """Is this glyph drawn at full width? The Unicode East Asian Width property."""
    return unicodedata.east_asian_width(character) in ("W", "F")


def _text_extent(text: str, *, minimum: float = MIN_PANEL_WIDTH) -> tuple[float, float]:
    """The box `text` needs, as `(width, height)` in stage units.

    An estimate, and only ever asked about one font at one size — `drawReadout`
    sets `14px Inter, 'Microsoft YaHei'`. Widths come from a Unicode property
    rather than from a measurement because there is no font here to measure
    with, and the point is not precision: it is that the number is derived from
    the text instead of being a constant that never looked at it.

    `minimum` is the panel floor, and an annotation passes 0: `MIN_PANEL_WIDTH`
    is about the smallest *panel* worth drawing, and a one-character axis label
    is a one-character label. The padding is the panel's either way, which
    over-reserves an annotation by a few pixels — the harmless direction, and
    the estimate is the coarse part regardless.
    """
    lines = text.split("\n")
    widest = max((_line_width(line) for line in lines), default=0.0)
    return (
        max(minimum, widest + PANEL_PADDING * 2),
        len(lines) * PANEL_LINE_HEIGHT + PANEL_PADDING * 2,
    )


def _line_width(line: str) -> float:
    return sum(
        PANEL_FONT_SIZE * (PANEL_WIDE_EM if _is_wide(ch) else PANEL_NARROW_EM) for ch in line
    )


def _panel_texts(obj: StoryboardObject, scene: StoryboardScene) -> list[str]:
    """Every string this panel can be made to show.

    `text` is a live prop (`LIVE_PROPS.readout` in `registry.py`), so a beat can
    rewrite it — and beats do. Across the three sample documents **11 of the 16
    readouts are given a longer string by some beat than the object declares**:
    `param_panel` declares 四个关键参数 and beat 7 writes
    雷达有效半径2到8米；控制循环80ms, nearly three times as wide.

    So the box has to hold every value the prop can take, not the one it starts
    at. Sizing from the declared string drew the beat's longer text out over the
    panel's own edge, and nothing failed.
    """
    candidates: list[PropValue | None] = [obj.props.get("text"), obj.label]
    for step in scene.steps:
        state = step.object_states.get(obj.id)
        if isinstance(state, dict):
            candidates.append(state.get("text"))
    return [value for value in candidates if isinstance(value, str) and value]


def _panel_extent(texts: Sequence[str]) -> tuple[float, float]:
    """The box that holds every one of `texts`.

    Component-wise, not per-string. The widest string and the tallest string need
    not be the same one — a beat can rewrite a caption into two short lines — and
    a box sized from whichever the loop happened to look at first would fit one
    of them and not the other.
    """
    extents = [_text_extent(text) for text in texts]
    return (
        max((width for width, _ in extents), default=0.0),
        max((height for _, height in extents), default=0.0),
    )


def _panel_span(heights: Sequence[float], stage: RenderStage) -> tuple[float, float]:
    """Where the first and last panel centres go, given how tall they came out.

    `PANEL_TOP_RATIO`/`PANEL_BOTTOM_RATIO` are where a comfortable stack sits,
    not a limit, and the span widens before anything fails.

    The first version went the other way round: it shrank every panel to fit that
    span. That produces boxes the text does not fit in, so the drawer grew them
    back on its way to the canvas and two panels drew over each other with no
    gate in between — a `_check_fit` that passes because it was checking a
    rectangle nobody drew. Sizing from the text and widening the span keeps the
    failure where it belongs: the panels genuinely have more to say than the
    stage can show, and `_check_fit` says so by name.
    """
    margin = stage.height * STAGE_MARGIN_RATIO
    low = margin + heights[0] / 2
    high = stage.height - margin - heights[-1] / 2
    top = stage.height * PANEL_TOP_RATIO
    bottom = stage.height * PANEL_BOTTOM_RATIO
    if len(heights) == 1:
        return min(top, high), min(top, high)
    step = max((above + below) / 2 + MIN_PANEL_GAP for above, below in pairwise(heights))
    needed = step * (len(heights) - 1)
    if needed <= bottom - top or needed > high - low:
        return top, bottom
    middle = (low + high) / 2
    return middle - needed / 2, middle + needed / 2


def _panel_boxes(scene: StoryboardScene, stage: RenderStage) -> dict[str, _Box]:
    """The right-hand text column, shared by every preset.

    One width for the whole stack rather than one per panel: ragged left edges
    read as misalignment, and the widest panel is what the column has to clear
    anyway. Heights are per panel, because a one-line caption and a two-line one
    are not the same box — rounding both up to a constant is what produced the
    empty panels.
    """
    readouts = _with_primitive(scene, "readout")
    if not readouts:
        return {}

    text = {obj.id: _panel_texts(obj, scene) for obj in readouts}
    extent = {obj_id: _panel_extent(held) for obj_id, held in text.items()}
    needed = max(width for width, _ in extent.values())
    room = stage.width * (1 - 2 * STAGE_MARGIN_RATIO)
    if needed > room:
        widest = max(extent, key=lambda obj_id: extent[obj_id][0])
        line = max(text[widest], key=lambda held: _text_extent(held)[0])
        raise LayoutError(
            f"场景 `{scene.id}` 里 `{widest}` 的面板文字有 {needed:.0f}px 宽，"
            f"而画布只留得出 {room:.0f}px；写出去的话会跑到画布外面，"
            f"拆成两行、或者换一句短一点的：{line[:40]!r}"
        )

    top, bottom = _panel_span([extent[obj.id][1] for obj in readouts], stage)
    x = stage.width * PANEL_RIGHT_RATIO - needed / 2
    return _column(readouts, x, top, bottom, lambda obj: (needed, extent[obj.id][1]))


def _link_endpoint_ids(scene: StoryboardScene) -> set[str]:
    """Ids that some link connects to. These are the chain's participants."""
    return {
        target
        for obj in scene.objects
        if _primitive_of(obj) == "link"
        for target in (_relation(obj, "from"), _relation(obj, "to"))
        if target is not None
    }


def _chain_boxes(scene: StoryboardScene, stage: RenderStage, frame: _Frame) -> dict[str, _Box]:
    """A left-to-right pipeline: participants on a spine, everything else below it.

    **Who counts as a participant is derived, not read off the role.** The first
    version put every `node` on the spine and everything else on the lower row,
    which is fine until a real generated storyboard roles a lidar *node* as a
    `device` — it then got drawn on the actor row, and the link from it to the
    next node cut diagonally across the whole picture. Nothing errored; the
    diagram just stopped being readable.

    The robust question is the structural one: **does a link touch it?** A body
    the chain connects is part of the chain whatever it is called, and a body
    nothing connects is not — which is also exactly what the lower row means
    (the runtime the chain drives, not a hop in it).

    The link itself is not placed here — it is an *edge*, and its geometry lives
    on the two bodies it joins (`_link_points`). A box would be a position that
    means nothing.

    The spine runs between the frame's two edges, not `CHAIN_LEFT_RATIO` to
    `CHAIN_RIGHT_RATIO`. The right-hand one was 0.84 to stay clear of the panel
    column, and the spine sits at 0.42 of the height — below the panels a chain
    scene actually has, so the reservation was paying for a wall that was not
    there. Both ends move together rather than only the right: a pipeline with
    one margin nearly twice the other reads as a mistake, and the frame's left
    edge is the honest answer for where a chain starts.
    """
    endpoints = _link_endpoint_ids(scene)
    bodies = _with_primitive(scene, "body")
    on_spine = [obj for obj in bodies if obj.role == "node" or obj.id in endpoints]
    if len(on_spine) < 2:
        raise LayoutError(
            f"场景 `{scene.id}` 选了 `chain`，但脊线上只有 {len(on_spine)} 个对象；"
            "链路至少要两个节点才连得起来"
        )

    spine_y = stage.height * CHAIN_Y_RATIO
    # Two different halves, because they answer two different questions: the
    # height is the band whose panels are in the way, the width is how far a node
    # reaches sideways once it is placed. Using one for both put the end nodes
    # *outside* the margin — on canvas, so nothing failed, and off the edge the
    # frame had just said was the limit.
    half_w = max(width for width, _ in map(_body_size, on_spine)) / 2
    half_h = max(height for _, height in map(_body_size, on_spine)) / 2
    left = frame.left_of(spine_y, half_h) + half_w
    right = frame.right_of(spine_y, half_h) - half_w
    if right < left:
        raise LayoutError(
            f"场景 `{scene.id}` 选了 `chain`，但脊线所在的高度上左右只剩 "
            f"{right - left:.0f}px 可用；面板把这一行占满了"
        )

    boxes: dict[str, _Box] = {}
    step = (right - left) / (len(on_spine) - 1)
    for index, node in enumerate(on_spine):
        width, height = _body_size(node)
        boxes[node.id] = _Box(x=left + step * index, y=spine_y, width=width, height=height)

    # The actor row starts at the spine's own left edge rather than at its first
    # node's centre, so a row of different-sized bodies does not drift left of the
    # pipeline it belongs to.
    cursor = left - half_w
    for actor in [obj for obj in bodies if obj not in on_spine]:
        width, height = _body_size(actor)
        boxes[actor.id] = _Box(
            x=cursor + width / 2, y=stage.height * CHAIN_ACTOR_Y_RATIO, width=width, height=height
        )
        cursor += width + 24.0

    return boxes


def _generic_boxes(scene: StoryboardScene, stage: RenderStage, frame: _Frame) -> dict[str, _Box]:
    """A vertical column plus the text column. The last resort, and it looks like it.

    Deliberately no slots: `generic` is what a scene falls back to when no preset
    describes it, so pretending to arrange it would mean inventing a structure
    nobody asked for. An ordered list is the honest answer.

    Takes `frame` and does not read it, which is the same statement: a column has
    no horizontal structure to stretch. Its span down the page is the panel
    column's own span for the same non-reason — it is where an evenly spaced list
    looked right, not a measurement of anything.
    """
    return _column(
        _bodies_but(scene),
        stage.width * GENERIC_COLUMN_X_RATIO,
        stage.height * PANEL_TOP_RATIO,
        stage.height * PANEL_BOTTOM_RATIO,
        _body_size,
    )


def _hub_spokes(
    leaves: list[StoryboardObject],
) -> list[tuple[StoryboardObject, float]]:
    """Each leaf with the angle it sits at, from straight up and clockwise.

    Clockwise from the top is the order the narration introduces the parts, so
    the first leaf lands where the eye starts. Shared by the placer and the
    radius computation, because a radius worked out for one set of angles and
    applied to another is a collision nobody would look for.
    """
    return [
        (leaf, -math.pi / 2 + (2 * math.pi * index) / len(leaves))
        for index, leaf in enumerate(leaves)
    ]


def _hub_radius_x(
    scene_id: str,
    centre: RenderPoint,
    radius_y: float,
    spokes: list[tuple[StoryboardObject, float]],
    frame: _Frame,
) -> float:
    """The widest the ring may be before a leaf leaves the frame.

    `HUB_RADIUS_X_RATIO` used to do this job, and its comment says it is 0.22
    because "at 0.30 the rightmost leaf collides with the panel column, which is
    a `LayoutError` and so would have failed the scene rather than merely
    crowding it." Sweeping the constant says something else. 0.29 fails — on
    `target`, the **left** leaf, off the *left edge of the canvas*. The panels do
    not appear in the failure at all, because a four-leaf hub has no leaf in the
    top-right corner where the panels are. The constant was sized by the left
    margin and explained by the panels; the two happen to be close, and nothing
    checked which one was binding.

    So the room is measured instead of remembered. Each leaf asks about the
    height it will actually occupy: beside a panel it is the panel's edge, above
    or below one it is the stage margin. A constant cannot do that, and a hub
    that shares its corner with a wide code readout needs it to.
    """
    room = math.inf
    for leaf, angle in spokes:
        width, height = _body_size(leaf)
        y = centre.y + math.sin(angle) * radius_y
        across = math.cos(angle)
        if across > 0:
            edge = frame.right_of(y, height / 2) - width / 2
            room = min(room, (edge - centre.x) / across)
        elif across < 0:
            edge = frame.left_of(y, height / 2) + width / 2
            room = min(room, (centre.x - edge) / -across)
    if room < 0:
        raise LayoutError(f"场景 `{scene_id}` 选了 `hub`，但中心的左右都被面板占满，环展不开")
    return room


def _hub_boxes(scene: StoryboardScene, stage: RenderStage, frame: _Frame) -> dict[str, _Box]:
    """A star: the first `node` at the centre, every other body on a ring.

    The centre is chosen by role because that is what `hub`'s `required_roles`
    already promises the model — the validator rejects a `hub` with no `node`, so
    the first one is the hub by construction rather than by guess.

    Spokes are **not** synthesised. A star's edges are the model's to write: if
    it wants the centre joined to its leaves it should say so with a `link`, and
    if it does not, what it drew is a rosette, which is what this returns.
    Inventing edges here would put lines on the picture that no beat refers to.
    """
    centres = [obj for obj in _with_primitive(scene, "body") if obj.role == "node"]
    leaves = [obj for obj in _with_primitive(scene, "body") if obj.role != "node"]
    if not centres:
        raise LayoutError(f"场景 `{scene.id}` 选了 `hub`，但没有 `node` 角色的中心对象")
    if not leaves:
        raise LayoutError(
            f"场景 `{scene.id}` 选了 `hub`，但除了中心以外没有别的对象；"
            "星形至少要有一个叶子才画得出来"
        )

    centre = RenderPoint(
        x=stage.width * HUB_CENTER_X_RATIO,
        y=stage.height * HUB_CENTER_Y_RATIO,
    )
    width, height = _body_size(centres[0])
    boxes = {centres[0].id: _Box(x=centre.x, y=centre.y, width=width, height=height)}

    radius_y = stage.height * HUB_RADIUS_Y_RATIO
    spokes = _hub_spokes(leaves)
    radius_x = _hub_radius_x(scene.id, centre, radius_y, spokes, frame)
    for leaf, angle in spokes:
        width, height = _body_size(leaf)
        boxes[leaf.id] = _Box(
            x=centre.x + math.cos(angle) * radius_x,
            y=centre.y + math.sin(angle) * radius_y,
            width=width,
            height=height,
        )
    return boxes


def _field_boxes(scene: StoryboardScene, stage: RenderStage, frame: _Frame) -> dict[str, _Box]:
    """A 2D plot: every body on the ground line, one launch slot each.

    Bodies share the field width evenly rather than stacking at the origin. A
    comparison scene — the projectile document draws 平抛/斜抛/竖直上抛 side by
    side — has three projectiles and no trajectory between them, and three boxes
    at one point is a `LayoutError` about an overlap that the scene never meant.
    Giving each body its own slot also gives each trajectory a width to arc into.

    The right edge is what the frame leaves; see `_field_right` for why it is
    measured at the ground line.
    """
    bodies = _with_primitive(scene, "body")
    if not bodies:
        raise LayoutError(f"场景 `{scene.id}` 选了 `field`，却没有任何 body 可以发射")

    origin_x = stage.width * FIELD_ORIGIN_X_RATIO
    origin_y = stage.height * FIELD_ORIGIN_Y_RATIO
    field_width = _field_right(scene, stage, frame) - origin_x
    if field_width <= 0:
        raise LayoutError(
            f"场景 `{scene.id}` 选了 `field`，但从原点 ({origin_x:.0f}) 往右已经没有地方了；"
            "面板把这一行占满了"
        )
    slot = field_width / len(bodies)

    boxes: dict[str, _Box] = {}
    for index, obj in enumerate(bodies):
        width, height = _body_size(obj)
        boxes[obj.id] = _Box(x=origin_x + slot * index, y=origin_y, width=width, height=height)

    return boxes


def _field_origin(stage: RenderStage) -> RenderPoint:
    """Where the axes cross. A pure function of the stage, so the placer and the
    axis converter cannot disagree about it."""
    return RenderPoint(
        x=stage.width * FIELD_ORIGIN_X_RATIO,
        y=stage.height * FIELD_ORIGIN_Y_RATIO,
    )


#: preset name -> placer. A preset with no entry raises rather than falling back
#: to `generic`: a scene that asked for axes and silently got a vertical list is
#: a picture that lies about what it is.
#:
#: Every placer takes the frame whether or not it reads it. `lane` and `generic`
#: both ignore it and both say why; a signature that varied per preset would put
#: that decision somewhere other than the preset it belongs to.
_PLACERS: dict[str, Callable[[StoryboardScene, RenderStage, _Frame], dict[str, _Box]]] = {
    "lane": _lane_boxes,
    "chain": _chain_boxes,
    "hub": _hub_boxes,
    "field": _field_boxes,
    "generic": _generic_boxes,
}


def _place(scene: StoryboardScene, stage: RenderStage) -> tuple[dict[str, _Box], _Frame]:
    """Panels first, then the picture around them.

    The order used to be the other way round in effect: each placer arranged
    against a constant that *assumed* a panel column, then the panels were laid
    on top. Deciding the panels first is what makes `_Frame` possible, and it is
    also the honest order — the panels are the only thing here whose size is not
    a free choice.
    """
    placer = _PLACERS.get(scene.scene_type)
    if placer is None:
        raise LayoutError(
            f"场景 `{scene.id}` 的预设 `{scene.scene_type}` 还没有布局实现；"
            f"已实现：{'、'.join(sorted(_PLACERS))}。其余预设属于 M3，"
            "在实现之前宁可失败，也不要摆出一张意思不对的画面"
        )
    panels = _panel_boxes(scene, stage)
    frame = _frame(stage, panels)
    boxes = placer(scene, stage, frame)
    boxes.update(panels)
    _check_fit(boxes, stage, scene.id)
    return boxes, frame


def _check_fit(boxes: dict[str, _Box], stage: RenderStage, scene_id: str) -> None:
    """Off the canvas, or on top of each other, is a failure — not a compromise.

    Illegible output has to fail generation rather than reach a person to notice.
    Nobody can file a bug against "the car is halfway off the right edge"; they
    just conclude the tool is bad. A `LayoutError` names the scene and the
    object, which is something a retry or a different preset can act on.

    Only independently placed boxes are compared. An emitter and a zone are
    *supposed* to sit exactly on the body they are mounted on — they have no box
    of their own — so counting that as an overlap would reject every correct
    scene.
    """
    for element_id, box in sorted(boxes.items()):
        if (
            box.x - box.width / 2 < -FIT_TOLERANCE
            or box.x + box.width / 2 > stage.width + FIT_TOLERANCE
            or box.y - box.height / 2 < -FIT_TOLERANCE
            or box.y + box.height / 2 > stage.height + FIT_TOLERANCE
        ):
            raise LayoutError(
                f"场景 `{scene_id}` 里 `{element_id}` 画到画布外面去了："
                f"中心 ({box.x:.0f}, {box.y:.0f})、尺寸 {box.width:.0f}×{box.height:.0f}，"
                f"画布只有 {stage.width:.0f}×{stage.height:.0f}"
            )

    items = sorted(boxes.items())
    for index, (left_id, left) in enumerate(items):
        for right_id, right in items[index + 1 :]:
            gap_x = abs(left.x - right.x) - (left.width + right.width) / 2
            gap_y = abs(left.y - right.y) - (left.height + right.height) / 2
            # Both axes must be violated for the boxes to really intersect; one
            # axis of clearance is enough for the two to read as separate.
            if gap_x < -FIT_TOLERANCE and gap_y < -FIT_TOLERANCE:
                raise LayoutError(
                    f"场景 `{scene_id}` 里 `{left_id}` 和 `{right_id}` 叠在一起了"
                    f"（横向重叠 {-gap_x:.0f}px，纵向重叠 {-gap_y:.0f}px）；"
                    "叠着的两个对象在画面上分不出谁是谁"
                )


def _body_size(obj: StoryboardObject) -> tuple[float, float]:
    return _BODY_SIZES.get(obj.role, (64.0, 48.0))


def _body_pivot(role: str, drawn_width: float, drawn_height: float) -> tuple[float, float]:
    """Where this body turns about, in its own local frame (`+y` is down)."""
    fx, fy = _BODY_PIVOTS.get(role, (0.0, 0.0))
    return (fx * drawn_width, fy * drawn_height)


# --------------------------------------------------------------------------
# Conversion — one dispatcher on primitive, shared by every preset
# --------------------------------------------------------------------------


def _anchored_box(
    obj: StoryboardObject,
    anchor: str | None,
    boxes: dict[str, _Box],
    what: str,
) -> _Box:
    """The box of the object this one hangs off.

    An emitter is not *at* a coordinate, it is mounted on a body — which is why
    `attachment` survives into the spec at all. A relation pointing at something
    that got no box (another emitter, a zone, or nothing) has no position to
    hang from, and saying so beats drawing it at the origin.
    """
    if anchor is None:
        raise LayoutError(f"`{obj.id}`（{what}）没有声明 `of`，不知道挂在谁身上")
    box = boxes.get(anchor)
    if box is None:
        raise LayoutError(
            f"`{obj.id}`（{what}）的 `of` 指向 `{anchor}`，但那个对象没有位置；"
            "`of` 必须指向本场景里的一个 body"
        )
    return box


def _glyph_to_draw(obj: StoryboardObject) -> str | None:
    requested = obj.props.get("glyph")
    if isinstance(requested, str) and requested in _available_glyphs():
        return requested
    return None


def _drawn_size(glyph_name: str | None, width: float, height: float) -> tuple[float, float]:
    """How big a body actually draws, which is not always how big its box is.

    A body with no glyph fills its box exactly. A body with one does not: the
    player measures the *ink* against the box by the smaller of the two ratios
    and centres it, so a glyph whose ink is squarer than the box it was given is
    drawn smaller than that box, with empty space above and below it.

    Anything that has to line up with the *edge* of a body has to ask this
    rather than `_body_size`, and the robot arm is the case that makes it
    concrete. Its ink is 18.5x20 units; given the `arm` box of 34x96 it is drawn
    34x37, centred. A hinge measured against the box would sit 30 units below
    the arm it is meant to be bolted to, and the picture would be a six-axis arm
    swinging about a point in empty space — which reads as "the animation is
    wrong" and points at nothing in this file.
    """
    if glyph_name is None:
        return (width, height)
    _, _, ink_width, ink_height = _load_glyph(glyph_name).ink_box
    if not (ink_width > 0 and ink_height > 0):
        return (width, height)
    scale = min(width / ink_width, height / ink_height)
    return (ink_width * scale, ink_height * scale)


def _to_body(
    obj: StoryboardObject,
    box: _Box,
    scene: StoryboardScene,
    stage: RenderStage,
    frame: _Frame,
) -> BodyElement:
    width, height = _body_size(obj)
    is_obstacle = obj.role == "obstacle"
    glyph_name = _glyph_to_draw(obj)
    path, duration = _flight(obj, box, scene, stage, frame)
    return BodyElement(
        id=obj.id,
        role=obj.role,
        label=obj.label,
        x=box.x,
        y=box.y,
        shape="polygon" if is_obstacle else "rect",
        width=width,
        height=height,
        sides=OBSTACLE_SIDES if is_obstacle else 6,
        inner_ratio=OBSTACLE_INNER_RATIO if is_obstacle else 1.0,
        heading=_number(obj, "heading", 0.0),
        # Against what the glyph draws, not against the box: see `_drawn_size`.
        pivot=_body_pivot(obj.role, *_drawn_size(glyph_name, width, height)),
        glyph=glyph_name,
        path=path,
        duration=duration,
        # `_props_for`, not the verbatim `dict(obj.props)` this used to carry.
        # `body.speed` is now a ranged prop, and the range is worth nothing if
        # the one layer that writes the spec does not apply it — the emitter and
        # the zone have gone through `_props_for` since the stage ranges landed,
        # and the body was left out because it had no ranged prop to hold.
        props=_props_for(obj, "body"),
    )


def _flight(
    obj: StoryboardObject,
    box: _Box,
    scene: StoryboardScene,
    stage: RenderStage,
    frame: _Frame,
) -> tuple[list[RenderPoint], float]:
    """The parabola this body travels, and how long one pass takes.

    Empty on every preset but `field`, and that is not a special case so much as
    the definition: `lane` carries a car along a corridor, `chain` and `hub` hold
    their bodies still, and `field` is the preset whose bodies are thrown.

    The curve is `_arc_points` — **the same one the `trace` gets**. That is the
    point of putting it here rather than in the player: the trajectory the lesson
    draws and the path the ball flies cannot disagree, because there is one of
    them. The player interpolates along these samples and evaluates no physics at
    all (decision D3).

    The speed is `PIXELS_PER_SPEED`-shaped in spirit but lives here, so a throw
    takes about the same time whatever its range: a 45° throw and a 15° one are
    the same launch at the same speed, and the longer arc is the longer flight.
    """
    if scene.scene_type != "field":
        return [], 0.0
    if obj.role == "obstacle":
        return [], 0.0

    heading = _number(obj, "heading", 45.0)
    span = _field_slot(scene, stage, frame)
    points = _arc_points(RenderPoint(x=box.x, y=box.y), heading, span, stage)
    if len(points) < 2:
        return [], 0.0

    arc = sum(
        math.hypot(points[index].x - points[index - 1].x, points[index].y - points[index - 1].y)
        for index in range(1, len(points))
    )
    return points, max(arc / THROW_PIXELS_PER_SECOND, THROW_MIN_DURATION)


def beat_duration(narration: str) -> float:
    """How long one beat holds, in seconds, given what it has to say.

    The one input is the narration, and that is deliberate: it is the only
    thing a beat carries that has a length at all. A beat's `highlights` count
    says how much changes at once, its `key_points` how much is being asked of
    the memory, and neither converts to seconds without a second pile of
    made-up constants. Characters are countable; the rest would be numbers I
    chose and then dressed up as derivation.

    The floor is currently inert: the shortest legal `description` is 12
    characters, which already yields 4.5s. It stays as a guard, so that
    loosening that bound some day cannot hand the player a beat it flips past
    inside a frame. The ceiling binds at ~58 characters, which is past the
    length of an ordinary beat and well short of the 200 the field allows — the
    long tail gets 7s and no more, because a beat that outlasts the viewer's
    patience is worse than one that moves on early.

    Public because `legacy.py` needs the same answer for the frozen baseline,
    which reaches the player by a different road and would otherwise keep the
    old behaviour of sitting on its first beat forever.
    """
    readable = len(narration) / BEAT_CHARS_PER_SECOND
    return min(max(BEAT_BASE_SECONDS + readable, BEAT_MIN_SECONDS), BEAT_MAX_SECONDS)


def _to_emitter(obj: StoryboardObject, boxes: dict[str, _Box]) -> EmitterElement:
    anchor = _relation(obj, "of")
    box = _anchored_box(obj, anchor, boxes, "emitter")
    radius = _stage_distance(obj, "emitter", "radius", DEFAULT_SENSOR_RADIUS)
    if radius <= 0:
        raise LayoutError(f"`{obj.id}` 的 `radius` 是 {radius}，必须大于 0")
    return EmitterElement(
        id=obj.id,
        role=obj.role,
        label=obj.label,
        x=box.x,
        y=box.y,
        radius=radius,
        fov=_bounded(obj, "fov", DEFAULT_FOV, MIN_FOV, MAX_FOV),
        rays=int(_bounded(obj, "rays", float(DEFAULT_SENSOR_RAYS), 1, float(MAX_SENSOR_RAYS))),
        anchor=anchor,
        props=_props_for(obj, "emitter"),
    )


def _to_zone(obj: StoryboardObject, scene: StoryboardScene, boxes: dict[str, _Box]) -> ZoneElement:
    """A threshold circle, hung off its anchor.

    When the storyboard gives no `radius`, the value comes from the **scene**
    tier — this is how the baseline's `scene.safe_distance` slider reaches a
    circle the scene never declared (see `legacy.py`, which synthesises the same
    circle). Laying it out here means the number is baked into the spec, so the
    element also records the binding: without it the slider would move the
    danger threshold while the circle labelled 安全距离 sat still.
    """
    anchor = _relation(obj, "of")
    box = _anchored_box(obj, anchor, boxes, "zone")
    declared = obj.props.get("radius")
    uses_scene_param = not isinstance(declared, (int, float)) or isinstance(declared, bool)
    radius = _stage_distance(obj, "zone", "radius", _safe_distance(scene))
    if radius <= 0:
        raise LayoutError(f"`{obj.id}` 的半径是 {radius}，必须大于 0")
    return ZoneElement(
        id=obj.id,
        role=obj.role,
        label=obj.label,
        x=box.x,
        y=box.y,
        radius=radius,
        anchor=anchor,
        binds={"radius": "scene.safe_distance"} if uses_scene_param else {},
        props=_props_for(obj, "zone"),
    )


def _link_points(
    scene: StoryboardScene, boxes: dict[str, _Box]
) -> dict[str, tuple[RenderPoint, RenderPoint]]:
    """Resolve every link's two ends. A link is an edge, so this is its geometry.

    Done as a pass before the elements are built because a `traveler` rides a
    link and must not depend on the link appearing earlier in the object list —
    the model's ordering is a teaching order, not a dependency order.
    """
    points: dict[str, tuple[RenderPoint, RenderPoint]] = {}
    for obj in scene.objects:
        if _primitive_of(obj) != "link":
            continue
        start, end = _edge_ends(scene, obj, boxes, "link")
        points[obj.id] = (start, end)
    return points


def _edge_ends(
    scene: StoryboardScene,
    obj: StoryboardObject,
    boxes: dict[str, _Box],
    what: str,
) -> tuple[RenderPoint, RenderPoint]:
    """Resolve a link's two ends, refusing anything that is not a placed body.

    Not merely "has a box": a readout gets a box too, and a link that runs to a
    text panel reads as a connection that means something. The registry says a
    link joins two bodies, so that is what is enforced.
    """
    by_id = {item.id: item for item in scene.objects}
    ends: list[RenderPoint] = []
    for name in ("from", "to"):
        target = _relation(obj, name)
        if target is None:
            raise LayoutError(f"`{obj.id}`（{what}）缺少 `{name}`，两端不知道连哪里")
        target_obj = by_id.get(target)
        if target_obj is None or _primitive_of(target_obj) != "body":
            raise LayoutError(
                f"`{obj.id}`（{what}）的 `{name}` 指向 `{target}`，但那不是一个 body；"
                "两个端点都必须是本场景里能落位的物体"
            )
        box = boxes.get(target)
        if box is None:
            raise LayoutError(
                f"`{obj.id}`（{what}）的 `{name}` 指向 `{target}`，但那个对象没有位置"
            )
        ends.append(RenderPoint(x=box.x, y=box.y))
    return ends[0], ends[1]


def _to_link(
    obj: StoryboardObject,
    ends: dict[str, tuple[RenderPoint, RenderPoint]],
) -> LinkElement:
    if obj.id not in ends:
        raise LayoutError(f"`{obj.id}`（link）的两端没有解析出来")
    start, end = ends[obj.id]
    active = obj.props.get("active")
    return LinkElement(
        id=obj.id,
        role=obj.role,
        label=obj.label,
        x=(start.x + end.x) / 2,
        y=(start.y + end.y) / 2,
        points=[start, end],
        active=True if active is None else bool(active),
        props=dict(obj.props),
    )


def _to_traveler(
    obj: StoryboardObject,
    ends: dict[str, tuple[RenderPoint, RenderPoint]],
) -> TravelerElement:
    """A token that rides a link. Its position is *derived*, never authored.

    `progress` is the share of the path it has covered, so the two coordinates
    follow from the link's own ends — which is why a traveler needs the link to
    be a real, resolved edge rather than a lookalike.
    """
    path_id = _relation(obj, "along")
    if path_id is None:
        raise LayoutError(f"`{obj.id}`（traveler）没有声明 `along`，不知道沿什么移动")
    if path_id not in ends:
        raise LayoutError(
            f"`{obj.id}`（traveler）的 `along` 指向 `{path_id}`，但那个对象不是本场景里的 link"
        )
    start, end = ends[path_id]
    progress = _number(obj, "progress", 0.0)
    progress = min(max(progress, 0.0), 1.0)
    return TravelerElement(
        id=obj.id,
        role=obj.role,
        label=obj.label,
        x=start.x + (end.x - start.x) * progress,
        y=start.y + (end.y - start.y) * progress,
        path_id=path_id,
        progress=progress,
        props=dict(obj.props),
    )


def _align_of(obj: StoryboardObject) -> Literal["left", "center", "right"]:
    """Coerce the alignment prop. An unrecognised value falls back to left.

    Not raised on: alignment changes nothing about *what* the panel says, so a
    value the vocabulary never offered is a wasted flourish, not a broken scene.
    """
    value = obj.props.get("align")
    if value == "center":
        return "center"
    if value == "right":
        return "right"
    return "left"


def _to_readout(obj: StoryboardObject, box: _Box) -> ReadoutElement:
    """A text panel. `text` is drawn, never parsed (decision D4)."""
    declared = obj.props.get("text")
    return ReadoutElement(
        id=obj.id,
        role=obj.role,
        label=obj.label,
        x=box.x,
        y=box.y,
        text=declared if isinstance(declared, str) and declared else obj.label,
        width=box.width,
        height=box.height,
        align=_align_of(obj),
        props=dict(obj.props),
    )


# --------------------------------------------------------------------------
# The `field` vocabulary — directions, curves and callouts
# --------------------------------------------------------------------------


def _object_by_id(scene: StoryboardScene, object_id: str | None) -> StoryboardObject | None:
    if object_id is None:
        return None
    return next((obj for obj in scene.objects if obj.id == object_id), None)


def _to_canvas_angle(degrees: float) -> float:
    """A physics angle (y-up, anticlockwise) as a canvas angle (y-down, clockwise).

    The one place the two conventions meet, and the reason `heading` does not
    mean the same thing in every preset. A `lane` scene describes a car on a
    screen, so its heading is already canvas convention — the player's
    `drawHeading` reads it straight. A `field` scene describes a throw, and the
    real chain writes `heading: 90` for 竖直上抛 (straight up) and
    `direction: -90` for gravitational force (straight down). Read as canvas
    angles those point at the floor and the ceiling.

    `atoms.js` already states that reconciling the two is the layout's job and
    not the renderer's. This is that job, and it is deliberately the only place
    either convention is converted.
    """
    return -degrees


def _axis_label_room(scene: StoryboardScene) -> float:
    """Room an `x_axis`'s own label needs past the tip it is drawn at.

    `drawAxis` centres the label `AXIS_LABEL_GAP` beyond the tip, so the field
    has to stop short by half a label plus that gap. The estimate is the layout's
    usual one (`_text_extent`) and is deliberately on the generous side: a
    reservation that is a few pixels too wide costs a few pixels of axis, and one
    that is too narrow writes text off the canvas with nothing to catch it.
    """
    labels = [
        obj.label
        for obj in scene.objects
        if _primitive_of(obj) == "axis" and obj.role == "x_axis" and obj.label
    ]
    if not labels:
        return 0.0
    return AXIS_LABEL_GAP + max(_text_extent(label, minimum=0.0)[0] for label in labels) / 2


def _field_right(scene: StoryboardScene, stage: RenderStage, frame: _Frame) -> float:
    """Where the plot stops on the right.

    One function, so the placer, the curve and the axis cannot disagree about it
    — the same reason `_field_origin` exists.

    The question is asked at the **ground line**, not over the picture's whole
    height, and that is not an approximation: a parabola's highest point is the
    midpoint of its range, so the right edge of the field is only ever reached at
    ground level. Asking about the full band instead costs the projectile
    document 200px of axis on every scene — measured — to guard against an apex
    that is always at half the width.

    The axis's own label is subtracted last, because it is drawn at the end of
    whatever this returns.
    """
    bodies = _with_primitive(scene, "body")
    half_height = max((h for _, h in map(_body_size, bodies)), default=0.0) / 2
    right = frame.right_of(stage.height * FIELD_ORIGIN_Y_RATIO, half_height)
    return right - _axis_label_room(scene)


def _field_slot(scene: StoryboardScene, stage: RenderStage, frame: _Frame) -> float:
    """The width one body in a `field` scene owns, to arc its trajectory into.

    Computed from the same quantities `_field_boxes` places with, so the placer
    and the curve cannot disagree about how much room there is — which is why the
    frame has to reach this far down. A trajectory drawn into a span its body was
    not placed at runs out from under that body, and nothing about the picture
    would say so.
    """
    bodies = _with_primitive(scene, "body")
    if not bodies:
        return 0.0
    return (_field_right(scene, stage, frame) - stage.width * FIELD_ORIGIN_X_RATIO) / len(bodies)


def _trace_span(obj: StoryboardObject) -> float:
    """How much of its slot a trajectory is drawn across, from `length`."""
    value = _number(obj, "length", TRACE_LENGTH_REFERENCE)
    return min(max(value / TRACE_LENGTH_REFERENCE, TRACE_SPAN_MIN), TRACE_SPAN_MAX)


def _arc_points(
    start: RenderPoint,
    heading: float,
    span: float,
    stage: RenderStage,
) -> list[RenderPoint]:
    """Sample the parabola a body launched at `heading` degrees would trace.

    `heading` is a physics angle: 0 is horizontal, 90 is straight up. `span` is
    the range a 45° throw gets; every other angle is that same throw at the same
    speed, so `R = span·sin 2θ` and `H = span·sin²θ / 2`.

    That pair is worth stating plainly because it is where the picture's
    honesty lives. `R` is symmetric about 45° — 30° and 60° land in the same
    place — and it is zero at 90°, which is 竖直上抛 having no range at all. A
    model fitted to the box instead ("draw the biggest arc that fits") gets a
    taller arc for a steeper throw and says nothing about either fact.

    The shape is a genuine parabola: the slope at the launch point works out to
    `tan θ`, so the drawn curve and the number in the lesson agree.

    The two degenerate ends are drawn rather than rejected. At 90° the arc
    collapses to the vertical line that *is* 竖直上抛, and at 0° to the flat line
    that *is* an object with no vertical component.
    """
    theta = math.radians(min(max(heading, 0.0), 90.0))
    if theta < math.radians(1.0):
        return [start, RenderPoint(x=start.x + span, y=start.y)]

    room = max(start.y - stage.height * FIELD_TOP_RATIO, 0.0)
    # A vertical throw reaches `span / 2`, so cap the span at what the field can
    # hold above the launch point and the 90° case cannot run off the top.
    reach = min(span, 2 * room)
    width = reach * math.sin(2 * theta)
    height = reach * math.sin(theta) ** 2 / 2
    if width < 1.0:
        return [start, RenderPoint(x=start.x, y=start.y - height)]

    points: list[RenderPoint] = []
    for index in range(TRACE_SAMPLES + 1):
        x = width * index / TRACE_SAMPLES
        lift = 4 * height * x * (width - x) / (width * width)
        points.append(RenderPoint(x=start.x + x, y=start.y - lift))
    return points


def _vector_lengths(scene: StoryboardScene) -> dict[str, float]:
    """Arrow length per vector, scaled within its own scene.

    See `VECTOR_MAX_LENGTH`: magnitudes are not comparable across roles, so the
    scaling is per scene, where a lesson is about one quantity at a time.
    """
    magnitudes = {
        obj.id: abs(_number(obj, "magnitude", 1.0)) for obj in _with_primitive(scene, "vector")
    }
    largest = max(magnitudes.values(), default=0.0)
    if largest <= 0.0:
        return {obj_id: VECTOR_MAX_LENGTH for obj_id in magnitudes}
    return {
        obj_id: VECTOR_MAX_LENGTH * max(VECTOR_MIN_FRACTION, value / largest)
        for obj_id, value in magnitudes.items()
    }


def _to_vector(
    obj: StoryboardObject,
    scene: StoryboardScene,
    boxes: dict[str, _Box],
) -> VectorElement:
    anchor = _relation(obj, "of")
    box = _anchored_box(obj, anchor, boxes, "vector")
    length = _vector_lengths(scene)[obj.id]
    angle = math.radians(_to_canvas_angle(_number(obj, "direction", 0.0)))
    magnitude = abs(_number(obj, "magnitude", 1.0))
    return VectorElement(
        id=obj.id,
        role=obj.role,
        label=obj.label,
        x=box.x,
        y=box.y,
        dx=math.cos(angle) * length,
        dy=math.sin(angle) * length,
        # The ratio the player multiplies a live `magnitude` by. Handed over
        # rather than re-derived so the normalisation above stays the only
        # description of how long an arrow is.
        length_scale=length / magnitude if magnitude > 0 else 0.0,
        anchor=anchor,
        props=dict(obj.props),
    )


def _to_trace(
    obj: StoryboardObject,
    scene: StoryboardScene,
    boxes: dict[str, _Box],
    stage: RenderStage,
    frame: _Frame,
) -> TraceElement:
    anchor = _relation(obj, "of")
    box = _anchored_box(obj, anchor, boxes, "trace")
    body = _object_by_id(scene, anchor)
    # The trajectory is the *body's* throw, not the trace's own: a trace is the
    # history of something, so it has no direction of its own to draw.
    heading = _number(body, "heading", 45.0) if body is not None else 45.0
    span = _field_slot(scene, stage, frame) * _trace_span(obj)
    return TraceElement(
        id=obj.id,
        role=obj.role,
        label=obj.label,
        x=box.x,
        y=box.y,
        points=_arc_points(RenderPoint(x=box.x, y=box.y), heading, span, stage),
        anchor=anchor,
        # So the player can turn a live `length` into "how much of the curve to
        # draw" without restating `TRACE_LENGTH_REFERENCE`.
        length_reference=TRACE_LENGTH_REFERENCE,
        props=dict(obj.props),
    )


def _to_axis(
    obj: StoryboardObject,
    scene: StoryboardScene,
    stage: RenderStage,
    frame: _Frame,
) -> AxisElement:
    """A graduated axis from the field origin.

    Which way it runs follows from the role, not from a prop: `x_axis` and
    `y_axis` are already the two directions, and asking the model for an angle
    as well would give one fact two places to be written.

    The `x_axis` reaches `_field_right`, the same call the bodies and the
    trajectories use — an axis that stopped somewhere else than the plot does
    would be measuring a quantity nobody drew.
    """
    origin = _field_origin(stage)
    is_vertical = obj.role == "y_axis"
    if is_vertical:
        length = origin.y - stage.height * FIELD_TOP_RATIO
        heading = -90.0
    else:
        length = _field_right(scene, stage, frame) - origin.x
        heading = 0.0
    return AxisElement(
        id=obj.id,
        role=obj.role,
        label=obj.label,
        x=origin.x,
        y=origin.y,
        length=max(length, 1.0),
        heading=heading,
        ticks=int(_number(obj, "ticks", DEFAULT_AXIS_TICKS)),
        props=dict(obj.props),
    )


def _to_dimension(
    obj: StoryboardObject,
    scene: StoryboardScene,
    boxes: dict[str, _Box],
) -> DimensionElement:
    """A callout between two objects, keeping their ids so it can follow them."""
    start, end = _edge_ends(scene, obj, boxes, "dimension")
    return DimensionElement(
        id=obj.id,
        role=obj.role,
        label=obj.label,
        x=(start.x + end.x) / 2,
        y=(start.y + end.y) / 2,
        start=start,
        end=end,
        from_id=_relation(obj, "from"),
        to_id=_relation(obj, "to"),
        props=dict(obj.props),
    )


def _to_angle(
    obj: StoryboardObject,
    scene: StoryboardScene,
    boxes: dict[str, _Box],
) -> AngleElement:
    """An arc at a body, measuring a throw's elevation.

    The angle is measured from the horizontal, which is what 发射角 means, and
    falls back to the anchor body's own heading when `degrees` is absent — the
    one case where a relation is a better source than the object's own prop.
    """
    anchor = _relation(obj, "of")
    box = _anchored_box(obj, anchor, boxes, "angle")
    body = _object_by_id(scene, anchor)
    default = _number(body, "heading", 45.0) if body is not None else 45.0
    radius = _number(obj, "radius", ANGLE_RADIUS_DEFAULT)
    return AngleElement(
        id=obj.id,
        role=obj.role,
        label=obj.label,
        x=box.x,
        y=box.y,
        center=RenderPoint(x=box.x, y=box.y),
        from_degrees=0.0,
        to_degrees=_number(obj, "degrees", default),
        radius=min(max(radius, ANGLE_RADIUS_MIN), ANGLE_RADIUS_MAX),
        props=dict(obj.props),
    )


def _to_element(
    obj: StoryboardObject,
    scene: StoryboardScene,
    boxes: dict[str, _Box],
    links: dict[str, tuple[RenderPoint, RenderPoint]],
    stage: RenderStage,
    frame: _Frame,
) -> RenderElement:
    """Dispatch on the primitive, so every preset shares one converter.

    An unimplemented primitive raises, naming what is missing rather than being
    drawn as something plausible: an `axis` rendered as a rectangle is worse
    than no picture, because it looks deliberate.

    The check is against `registry.PENDING_PRIMITIVES`, not a list kept here, so
    the name the model was warned about in the prompt is the same name that
    fails here. By the time this raises, the validator has already had its say
    (`primitive_not_drawable`) and the model has already had its retry — reaching
    this line means a storyboard was written by hand, or the validator was
    bypassed.
    """
    primitive = _primitive_of(obj)
    if primitive is None:
        raise LayoutError(f"`{obj.id}` 的角色 `{obj.role}` 不在注册表里")
    if primitive in PENDING_PRIMITIVES:
        alternatives = PENDING_ALTERNATIVES.get(primitive, "别的图元")
        raise LayoutError(
            f"`{obj.id}`（角色 `{obj.role}`）的图元 `{primitive}` 还没有布局实现——"
            f"改用 {alternatives} 就能画出来"
        )
    if primitive in ("body", "readout"):
        box = boxes.get(obj.id)
        if box is None:
            raise LayoutError(f"`{obj.id}`（{primitive}）没有被摆放：预设没有给它位置")
        if primitive == "readout":
            return _to_readout(obj, box)
        return _to_body(obj, box, scene, stage, frame)
    if primitive == "emitter":
        return _to_emitter(obj, boxes)
    if primitive == "zone":
        return _to_zone(obj, scene, boxes)
    if primitive == "link":
        return _to_link(obj, links)
    if primitive == "traveler":
        return _to_traveler(obj, links)
    if primitive == "vector":
        return _to_vector(obj, scene, boxes)
    if primitive == "trace":
        return _to_trace(obj, scene, boxes, stage, frame)
    if primitive == "axis":
        return _to_axis(obj, scene, stage, frame)
    if primitive == "dimension":
        return _to_dimension(obj, scene, boxes)
    if primitive == "angle":
        return _to_angle(obj, scene, boxes)
    # Unreachable for every primitive the registry declares drawable, and a test
    # walks all of them to keep it that way. If it fires, the registry is lying:
    # `drawable=True` promised a picture the dispatch below cannot draw. Naming
    # the mismatch rather than the primitive is the point — whoever reads this
    # needs to edit `registry.py`, not go hunting through the storyboard.
    raise LayoutError(
        f"`{obj.id}` 的图元 `{primitive}` 在注册表里声明为可绘制，"
        f"但 `_to_element` 没有它的分支——注册表与布局层不同步"
    )


def _to_step(step: StoryboardStep, primitives: dict[str, str]) -> RenderStep:
    """One beat, with its state values held to the stage the way props are.

    `primitives` maps object id -> primitive name, built once per scene by
    `layout_scene`. Without it a beat state would be the one road into the spec
    that skips the range check — and it is the road a lesson takes when it wants
    to show a car slowing down, which is exactly the beat where the number is
    most likely to have been copied out of the document in the document's own
    unit.
    """
    return RenderStep(
        id=step.id,
        title=step.title,
        narration=step.description,
        duration=beat_duration(step.description),
        highlights=list(step.highlights),
        states={
            target: {
                prop: _clamp_value(primitives.get(target), prop, value)
                for prop, value in value_by_prop.items()
            }
            for target, value_by_prop in step.object_states.items()
        },
    )


def _to_control(control: StoryboardControl) -> RenderControl:
    """Copy a `StoryboardControl`. The two models declare the same fields on
    purpose: the spec adds `unit` and `action`, and nothing else to interpret."""
    return RenderControl.model_validate(control.model_dump())


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def layout_scene(scene: StoryboardScene, *, stage: RenderStage | None = None) -> RenderScene:
    """Place one scene. Deterministic: same input, byte-identical output.

    Elements come out in the storyboard's own object order — the same order the
    model wrote them in, which is the order the narration introduces them.
    Sorting them would be more "canonical" and would also mean the drawing order
    stops matching the teaching order, for no gain.
    """
    stage = stage or RenderStage()
    boxes, frame = _place(scene, stage)
    links = _link_points(scene, boxes)
    elements: list[RenderElement] = []
    for obj in scene.objects:
        try:
            elements.append(_to_element(obj, scene, boxes, links, stage, frame))
        except ValidationError as exc:
            # The element models are strict and the model's props are not, so a
            # value can be legal in the vocabulary and out of range in the
            # contract. Letting that surface as a Pydantic traceback names the
            # *field* and not the object, the scene, or the fact that layout is
            # where it went wrong — a real run produced exactly that for
            # `fov: 0`. `LayoutError` is what the CLI knows how to report, and
            # what a retry can act on.
            raise LayoutError(
                f"场景 `{scene.id}` 的 `{obj.id}`（{obj.role}）超出了渲染契约的取值范围：{exc}"
            ) from exc

    # object id -> primitive name, for the beat states below. An object whose
    # role is unknown is left out rather than guessed at: its states have already
    # been reported by the validator, and clamping against the wrong primitive's
    # range would be a second wrong answer to a question already answered.
    primitives = {
        obj.id: ROLE_TO_PRIMITIVE[obj.role]
        for obj in scene.objects
        if obj.role in ROLE_TO_PRIMITIVE
    }

    return RenderScene(
        id=scene.id,
        # `StoryboardScene` has no title of its own — the beats carry the words.
        teaching_goal=scene.teaching_goal,
        preset=scene.scene_type,
        # Which element rides on which. Resolved into coordinates above; kept
        # because the player needs it to move a sensor with the car it is
        # mounted on, and a relation the layout understood is worth passing on
        # rather than making the renderer infer it back out of coincident
        # coordinates.
        #
        # Vectors and traces are in here for the same reason and with one extra
        # step: their coordinates are baked, so the drawer translates them by how
        # far the body has moved since layout. Without it a car that drives off
        # leaves its arrows and its trajectory standing where it started.
        attachment={
            element.id: element.anchor
            for element in elements
            if isinstance(element, (EmitterElement, ZoneElement, TraceElement, VectorElement))
            and element.anchor is not None
        },
        # Which zone is whose threshold, read off the relations rather than off
        # prop names. `behaviors.js` needs a number for `proximity_gate`, and it
        # used to look for `safe_distance` on the *body* — a name that happens to
        # exist in the hand-written baseline (it is a `scene` param there) and
        # does not exist in any document that roles a zone instead. The avoidance
        # run wrote `safe_zone.radius`, the lookup came back null, the gate
        # `continue`d, and the car drove straight through its obstacles while the
        # lesson on screen was about deciding to avoid them.
        #
        # A `safe_distance`-role zone whose `of` is a body *is* that body's
        # threshold. Nothing needs the model to name a prop correctly.
        thresholds={
            element.anchor: element.id
            for element in elements
            if isinstance(element, ZoneElement)
            and element.role == "safe_distance"
            and element.anchor is not None
        },
        elements=elements,
        steps=[_to_step(step, primitives) for step in scene.steps],
        controls=[_to_control(control) for control in scene.controls],
        params=dict(scene.params),
    )


def layout_storyboard(storyboard: StoryboardIR, *, stage: RenderStage | None = None) -> RenderSpec:
    """Place every scene. The one call the pipeline and the CLI both use."""
    stage = stage or RenderStage()
    scenes = [layout_scene(scene, stage=stage) for scene in storyboard.scenes]
    # Collected from the laid-out scenes rather than from the storyboard, so what
    # is embedded is exactly what survived `_glyph_to_draw` — a glyph the model
    # asked for and layout dropped does not get shipped anyway.
    used = sorted(
        {
            element.glyph
            for scene in scenes
            for element in scene.elements
            if isinstance(element, BodyElement) and element.glyph is not None
        }
    )
    return RenderSpec(
        storyboard_id=storyboard.storyboard_id,
        lesson_id=storyboard.lesson_id,
        document_id=storyboard.document_id,
        title=storyboard.title,
        subject=storyboard.subject,
        eyebrow=storyboard.eyebrow,
        stage=stage,
        glyphs={name: _load_glyph(name) for name in used},
        scenes=scenes,
    )
