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
from collections.abc import Callable
from dataclasses import dataclass
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
#: at 252/600 — above centre, which leaves the lower third for the actor row and
#: keeps the spine clear of the readouts' column.
CHAIN_Y_RATIO = 0.42
CHAIN_LEFT_RATIO = 0.16
CHAIN_RIGHT_RATIO = 0.84
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
#: sized to fit the height overflows into the readout column on the right, and one
#: sized to fit the width leaves the leaves bunched in a vertical line. The x
#: radius is smaller than it looks it should be for the same reason — at 0.30 the
#: rightmost leaf collides with the panel column, which is a `LayoutError` and so
#: would have failed the scene rather than merely crowding it.
HUB_CENTER_X_RATIO = 0.32
HUB_CENTER_Y_RATIO = 0.5
HUB_RADIUS_X_RATIO = 0.22
HUB_RADIUS_Y_RATIO = 0.30

#: `field` is a 2D plot: axes crossing at an origin, one launch slot per
#: projectile, trajectories arcing up and to the right.
#:
#: The origin sits low and left so "+y up" is the direction with room in it —
#: a parabola drawn from a centred origin has nowhere to arc. The right edge
#: stops short of the panel column for the same reason `hub`'s x radius is small.
FIELD_ORIGIN_X_RATIO = 0.15
FIELD_ORIGIN_Y_RATIO = 0.82
FIELD_RIGHT_RATIO = 0.60
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

#: Graduations on an axis when the storyboard did not say how many.
DEFAULT_AXIS_TICKS = 5

#: Readouts stack in a right-hand column for **every** preset. One rule, because a
#: text panel has no relation to the geometry to hang off — it needs a slot that
#: does not move when the objects do. Colliding with a body is a `LayoutError`
#: like anything else; a panel is not more important than the picture.
PANEL_X_RATIO = 0.74
PANEL_WIDTH_RATIO = 0.24
PANEL_HEIGHT = 88.0
PANEL_TOP_RATIO = 0.14
PANEL_BOTTOM_RATIO = 0.86
#: Clearance between two stacked panels, and the smallest one still worth
#: drawing. Both only bind when a scene has many readouts; see `_panel_height`.
MIN_PANEL_GAP = 8.0
MIN_PANEL_HEIGHT = 24.0

#: role -> (width, height) in stage units, for roles a preset places as a box.
_BODY_SIZES: dict[str, tuple[float, float]] = {
    "vehicle": (62.0, 40.0),  # `RobotCar`'s frozen defaults
    "obstacle": (DEFAULT_OBSTACLE_RADIUS * 2, DEFAULT_OBSTACLE_RADIUS * 2),
    "node": (96.0, 48.0),
    "endpoint": (84.0, 44.0),
    "device": (72.0, 56.0),
    "agent": (56.0, 56.0),
    "projectile": (28.0, 28.0),
    "object": (64.0, 48.0),
}

#: Glyph data lives in `assets/glyphs/` and lands in M3. Until a glyph has data
#: the player refuses to draw it, so layout keeps the *requested* name in
#: `props.glyph` (copied verbatim like everything else) and draws the parametric
#: shape instead of emitting a `glyph` field the renderer would reject.
#:
#: This is not a silent downgrade: for an obstacle, an 8-point star *is* the
#: picture, and for the car the gap is visible and recorded. The check reads the
#: directory, so the day the data exists the glyphs start appearing with no code
#: change here.
GLYPH_DATA_DIR = Path(__file__).resolve().parents[3] / "assets" / "glyphs"


def _available_glyphs() -> frozenset[str]:
    if not GLYPH_DATA_DIR.is_dir():
        return frozenset()
    return frozenset(path.stem for path in GLYPH_DATA_DIR.glob("*.json"))


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
    if bounds is None:
        return value
    low, high = bounds
    return min(max(value, low), high)


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
        if stage_range(primitive, prop) is None:
            continue
        value = props[prop]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        low, high = stage_range(primitive, prop) or (value, value)
        props[prop] = min(max(float(value), low), high)
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


def _lane_boxes(scene: StoryboardScene, stage: RenderStage) -> dict[str, _Box]:
    """A corridor: bodies on the lane line, obstacles alternating either side.

    Obstacles alternate because the fourth beat teaches *comparing* left and
    right clearance. A preset that put every obstacle on the same side would
    make that beat a foregone conclusion — which is what the baseline does, and
    why its car always swerves the same way.
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


def _panel_height(count: int, span: float) -> float:
    """How tall each panel may be so `count` of them fit down the column.

    `PANEL_HEIGHT` is a default, not a promise. A scene of seven symbol cards —
    the projectile document's 关键参数 scene is exactly that — cannot be stacked
    at 88px inside a 432px column, and the first version of this raised
    `LayoutError` over an overlap, failing a scene whose only crime was having
    more to say than most. Shrinking the cards keeps the failure mode where it
    belongs: the column stays readable and inside the canvas at any count the
    object cap permits.
    """
    if count <= 1:
        return PANEL_HEIGHT
    step = span / (count - 1)
    return min(PANEL_HEIGHT, max(step - MIN_PANEL_GAP, MIN_PANEL_HEIGHT))


def _panel_boxes(scene: StoryboardScene, stage: RenderStage) -> dict[str, _Box]:
    """The right-hand text column, shared by every preset."""
    readouts = _with_primitive(scene, "readout")
    width = stage.width * PANEL_WIDTH_RATIO
    top = stage.height * PANEL_TOP_RATIO
    bottom = stage.height * PANEL_BOTTOM_RATIO
    height = _panel_height(len(readouts), bottom - top)
    return _column(readouts, stage.width * PANEL_X_RATIO, top, bottom, lambda _obj: (width, height))


def _link_endpoint_ids(scene: StoryboardScene) -> set[str]:
    """Ids that some link connects to. These are the chain's participants."""
    return {
        target
        for obj in scene.objects
        if _primitive_of(obj) == "link"
        for target in (_relation(obj, "from"), _relation(obj, "to"))
        if target is not None
    }


def _chain_boxes(scene: StoryboardScene, stage: RenderStage) -> dict[str, _Box]:
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
    """
    endpoints = _link_endpoint_ids(scene)
    bodies = _with_primitive(scene, "body")
    on_spine = [obj for obj in bodies if obj.role == "node" or obj.id in endpoints]
    if len(on_spine) < 2:
        raise LayoutError(
            f"场景 `{scene.id}` 选了 `chain`，但脊线上只有 {len(on_spine)} 个对象；"
            "链路至少要两个节点才连得起来"
        )

    boxes: dict[str, _Box] = {}
    left = stage.width * CHAIN_LEFT_RATIO
    right = stage.width * CHAIN_RIGHT_RATIO
    spine_y = stage.height * CHAIN_Y_RATIO
    step = (right - left) / (len(on_spine) - 1)
    for index, node in enumerate(on_spine):
        width, height = _body_size(node)
        boxes[node.id] = _Box(x=left + step * index, y=spine_y, width=width, height=height)

    cursor = left
    for actor in [obj for obj in bodies if obj not in on_spine]:
        width, height = _body_size(actor)
        boxes[actor.id] = _Box(
            x=cursor, y=stage.height * CHAIN_ACTOR_Y_RATIO, width=width, height=height
        )
        cursor += width + 24.0

    boxes.update(_panel_boxes(scene, stage))
    return boxes


def _generic_boxes(scene: StoryboardScene, stage: RenderStage) -> dict[str, _Box]:
    """A vertical column plus the text column. The last resort, and it looks like it.

    Deliberately no slots: `generic` is what a scene falls back to when no preset
    describes it, so pretending to arrange it would mean inventing a structure
    nobody asked for. An ordered list is the honest answer.
    """
    boxes = _column(
        _bodies_but(scene),
        stage.width * GENERIC_COLUMN_X_RATIO,
        stage.height * PANEL_TOP_RATIO,
        stage.height * PANEL_BOTTOM_RATIO,
        _body_size,
    )
    boxes.update(_panel_boxes(scene, stage))
    return boxes


def _hub_boxes(scene: StoryboardScene, stage: RenderStage) -> dict[str, _Box]:
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

    centre = centres[0]
    centre_x = stage.width * HUB_CENTER_X_RATIO
    centre_y = stage.height * HUB_CENTER_Y_RATIO
    width, height = _body_size(centre)
    boxes = {centre.id: _Box(x=centre_x, y=centre_y, width=width, height=height)}

    radius_x = stage.width * HUB_RADIUS_X_RATIO
    radius_y = stage.height * HUB_RADIUS_Y_RATIO
    # From straight up, clockwise — which is the order the narration introduces
    # the parts, so the first leaf lands where the eye starts.
    for index, leaf in enumerate(leaves):
        angle = -math.pi / 2 + (2 * math.pi * index) / len(leaves)
        width, height = _body_size(leaf)
        boxes[leaf.id] = _Box(
            x=centre_x + math.cos(angle) * radius_x,
            y=centre_y + math.sin(angle) * radius_y,
            width=width,
            height=height,
        )

    boxes.update(_panel_boxes(scene, stage))
    return boxes


def _field_boxes(scene: StoryboardScene, stage: RenderStage) -> dict[str, _Box]:
    """A 2D plot: every body on the ground line, one launch slot each.

    Bodies share the field width evenly rather than stacking at the origin. A
    comparison scene — the projectile document draws 平抛/斜抛/竖直上抛 side by
    side — has three projectiles and no trajectory between them, and three boxes
    at one point is a `LayoutError` about an overlap that the scene never meant.
    Giving each body its own slot also gives each trajectory a width to arc into.
    """
    bodies = _with_primitive(scene, "body")
    if not bodies:
        raise LayoutError(f"场景 `{scene.id}` 选了 `field`，却没有任何 body 可以发射")

    origin_x = stage.width * FIELD_ORIGIN_X_RATIO
    origin_y = stage.height * FIELD_ORIGIN_Y_RATIO
    field_width = stage.width * FIELD_RIGHT_RATIO - origin_x
    slot = field_width / len(bodies)

    boxes: dict[str, _Box] = {}
    for index, obj in enumerate(bodies):
        width, height = _body_size(obj)
        boxes[obj.id] = _Box(x=origin_x + slot * index, y=origin_y, width=width, height=height)

    boxes.update(_panel_boxes(scene, stage))
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
_PLACERS: dict[str, Callable[[StoryboardScene, RenderStage], dict[str, _Box]]] = {
    "lane": _lane_boxes,
    "chain": _chain_boxes,
    "hub": _hub_boxes,
    "field": _field_boxes,
    "generic": _generic_boxes,
}


def _place(scene: StoryboardScene, stage: RenderStage) -> dict[str, _Box]:
    placer = _PLACERS.get(scene.scene_type)
    if placer is None:
        raise LayoutError(
            f"场景 `{scene.id}` 的预设 `{scene.scene_type}` 还没有布局实现；"
            f"已实现：{'、'.join(sorted(_PLACERS))}。其余预设属于 M3，"
            "在实现之前宁可失败，也不要摆出一张意思不对的画面"
        )
    boxes = placer(scene, stage)
    _check_fit(boxes, stage, scene.id)
    return boxes


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


def _to_body(
    obj: StoryboardObject,
    box: _Box,
    scene: StoryboardScene,
    stage: RenderStage,
) -> BodyElement:
    width, height = _body_size(obj)
    is_obstacle = obj.role == "obstacle"
    path, duration = _flight(obj, box, scene, stage)
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
        glyph=_glyph_to_draw(obj),
        path=path,
        duration=duration,
        props=dict(obj.props),
    )


def _flight(
    obj: StoryboardObject,
    box: _Box,
    scene: StoryboardScene,
    stage: RenderStage,
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
    span = _field_slot(scene, stage)
    points = _arc_points(RenderPoint(x=box.x, y=box.y), heading, span, stage)
    if len(points) < 2:
        return [], 0.0

    arc = sum(
        math.hypot(points[index].x - points[index - 1].x, points[index].y - points[index - 1].y)
        for index in range(1, len(points))
    )
    return points, max(arc / THROW_PIXELS_PER_SECOND, THROW_MIN_DURATION)


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


def _field_slot(scene: StoryboardScene, stage: RenderStage) -> float:
    """The width one body in a `field` scene owns, to arc its trajectory into.

    Computed from the same quantities `_field_boxes` places with, so the placer
    and the curve cannot disagree about how much room there is.
    """
    bodies = _with_primitive(scene, "body")
    if not bodies:
        return 0.0
    left = stage.width * FIELD_ORIGIN_X_RATIO
    right = stage.width * FIELD_RIGHT_RATIO
    return (right - left) / len(bodies)


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
) -> TraceElement:
    anchor = _relation(obj, "of")
    box = _anchored_box(obj, anchor, boxes, "trace")
    body = _object_by_id(scene, anchor)
    # The trajectory is the *body's* throw, not the trace's own: a trace is the
    # history of something, so it has no direction of its own to draw.
    heading = _number(body, "heading", 45.0) if body is not None else 45.0
    span = _field_slot(scene, stage) * _trace_span(obj)
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


def _to_axis(obj: StoryboardObject, stage: RenderStage) -> AxisElement:
    """A graduated axis from the field origin.

    Which way it runs follows from the role, not from a prop: `x_axis` and
    `y_axis` are already the two directions, and asking the model for an angle
    as well would give one fact two places to be written.
    """
    origin = _field_origin(stage)
    is_vertical = obj.role == "y_axis"
    if is_vertical:
        length = origin.y - stage.height * FIELD_TOP_RATIO
        heading = -90.0
    else:
        length = stage.width * FIELD_RIGHT_RATIO - origin.x
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
        return _to_body(obj, box, scene, stage)
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
        return _to_trace(obj, scene, boxes, stage)
    if primitive == "axis":
        return _to_axis(obj, stage)
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


def _to_step(step: StoryboardStep) -> RenderStep:
    return RenderStep(
        id=step.id,
        title=step.title,
        narration=step.description,
        highlights=list(step.highlights),
        states={key: dict(value) for key, value in step.object_states.items()},
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
    boxes = _place(scene, stage)
    links = _link_points(scene, boxes)
    elements: list[RenderElement] = []
    for obj in scene.objects:
        try:
            elements.append(_to_element(obj, scene, boxes, links, stage))
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
        steps=[_to_step(step) for step in scene.steps],
        controls=[_to_control(control) for control in scene.controls],
        params=dict(scene.params),
    )


def layout_storyboard(storyboard: StoryboardIR, *, stage: RenderStage | None = None) -> RenderSpec:
    """Place every scene. The one call the pipeline and the CLI both use."""
    stage = stage or RenderStage()
    return RenderSpec(
        storyboard_id=storyboard.storyboard_id,
        lesson_id=storyboard.lesson_id,
        document_id=storyboard.document_id,
        title=storyboard.title,
        subject=storyboard.subject,
        eyebrow=storyboard.eyebrow,
        stage=stage,
        scenes=[layout_scene(scene, stage=stage) for scene in storyboard.scenes],
    )
