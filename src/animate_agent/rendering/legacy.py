"""Adapt a hand-written `animation.Scene` into a `RenderSpec`. No LLM, no layout.

Why this exists
---------------

This is checkpoint **P2**, and it is deliberately built before the layout pass.
The reason is the same one that put the real StoryboardIR before the validator:
a coordinate producer needs a picture that is already known to be right, or it
can only be tuned against its own imagination. `animation/templates.py` holds
three hand-written scenes whose intended appearance is settled — the car
avoidance demo is the project's quality baseline. Translating one of them gives
two things for free:

1. A **reference picture** the layout pass can be compared against.
2. A **real test of the `RenderSpec` contract**: everything the baseline draws
   must fit in it. A contract that only ever holds my own fixtures is untested.

The adapter is pure and total: same `Scene` in, byte-identical `RenderSpec` out,
and an unrecognised element type raises with the type name in the message rather
than being dropped on the floor.

Two conversions are judgement calls, both marked in place
-------------------------------------------------------

**The safe-distance circle is synthesised.** `build_robot_obstacle_avoidance_scene`
exposes `scene.safe_distance` as a slider and `frontend/demo/app.js:634` draws a
dashed circle at that radius, but the scene has no `Circle` element for it — the
circle exists only as a side effect of the slider. A `RenderSpec` cannot express
"a control that also means 'draw this'", so the adapter adds the `zone` element
the drawing code always implied. Declared here, once, rather than re-derived in
the player per preset.

**`RobotCar.heading`'s unit is not documented anywhere in the frozen source.**
Every frozen scene leaves it at the default `0`, so nothing today can tell
degrees from radians. The adapter passes it through untouched and the player
reads degrees clockwise from +x; `test_every_frozen_scene_leaves_heading_at_zero`
turns the ambiguity into a failing test the moment a nonzero heading appears,
which is the point at which somebody has to answer the question for real.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from animate_agent.animation.elements import (
    AnimatedElement,
    Circle,
    Label,
    LidarSensor,
    ManualObject,
    MessagePacket,
    Obstacle,
    RobotCar,
    RosNode,
    RosTopic,
    Scene,
    TimelineStep,
)
from animate_agent.animation.templates import (
    build_robot_obstacle_avoidance_scene,
    build_ros_pub_sub_scene,
)
from animate_agent.interaction.controls import (
    ButtonControl,
    InteractionControl,
    SliderControl,
    ToggleControl,
)
from animate_agent.rendering.models import (
    BodyElement,
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
    TravelerElement,
    ZoneElement,
)
from animate_agent.storyboard.models import PropValue

#: Every `type` literal a frozen element class can carry. The adapter's dispatch
#: is written to be exhaustive over this tuple, and a test harvests the literals
#: straight off the dataclasses and asserts the two sets are equal — so adding an
#: element class to the frozen baseline fails the test rather than silently
#: producing a spec that drops it.
#:
#: `AnimatedElement` itself is absent on purpose: its `type: str` is unconstrained.
KNOWN_ELEMENT_TYPES: tuple[str, ...] = (
    "manual_object",
    "robot_car",
    "lidar_sensor",
    "obstacle",
    "circle",
    "label",
    "ros_node",
    "ros_topic",
    "message_packet",
)

#: Frozen element type -> the registry role the adapter assigns it.
#:
#: This lives here, not in the tests, because it describes what the converters
#: below actually do. It was in the tests first, and the two immediately
#: disagreed: the test called the baseline's `Circle` a `safe_distance`, while
#: the adapter — which also *synthesises* a separate `safe_distance` zone from
#: the slider — called it a `threshold`. Two tables for one mapping is one table
#: too many. `test_render_legacy.py` now asserts this one against the output.
LEGACY_TYPE_TO_ROLE: dict[str, str] = {
    "manual_object": "caption",
    "robot_car": "vehicle",
    "lidar_sensor": "sensor",
    "obstacle": "obstacle",
    "circle": "threshold",
    "label": "caption",
    "ros_node": "node",
    "ros_topic": "topic",
    "message_packet": "message",
}

#: `Scene.metadata["template"]` -> the registry preset it corresponds to. Used
#: only to label the output; layout never reads it back.
TEMPLATE_TO_PRESET: dict[str, str] = {
    "robot_obstacle_avoidance": "lane",
    "ros_pub_sub": "chain",
}

#: The frozen scenes this adapter can translate, by template name. Keyed off the
#: same names as `TEMPLATE_TO_PRESET` so `--render-template <name>` and the preset
#: label it produces can never disagree; a test asserts the keys match.
FROZEN_TEMPLATES: dict[str, Callable[[], Scene]] = {
    "robot_obstacle_avoidance": build_robot_obstacle_avoidance_scene,
    "ros_pub_sub": build_ros_pub_sub_scene,
}

DEFAULT_STAGE = RenderStage()


class LegacyConversionError(ValueError):
    """A frozen `Scene` contains something `RenderSpec` cannot express."""


@dataclass(frozen=True, slots=True)
class _Context:
    """What a converter may need beyond the element itself.

    `positions` is collected in a first pass because `ros_topic` is an edge: its
    geometry lives on the two nodes it joins, not on the topic.
    """

    positions: dict[str, tuple[float, float]]


def _point(x: float, y: float) -> RenderPoint:
    return RenderPoint(x=x, y=y)


def _convert_manual_object(element: ManualObject, context: _Context) -> RenderElement:
    # The fallback class for "something was extracted but no domain class fits".
    # It has no geometry of its own, so it becomes the text panel it always was.
    return ReadoutElement(
        id=element.id,
        role="caption",
        label=element.label or "",
        x=element.x,
        y=element.y,
        text=element.description or element.label or element.id,
    )


def _convert_robot_car(element: RobotCar, context: _Context) -> RenderElement:
    return BodyElement(
        id=element.id,
        role="vehicle",
        label=element.label or "",
        x=element.x,
        y=element.y,
        shape="rect",
        width=element.width,
        height=element.height,
        heading=element.heading,
        props={
            "speed": element.speed,
            "show_heading": element.show_heading,
            "show_trail": element.show_trail,
        },
    )


def _convert_lidar_sensor(element: LidarSensor, context: _Context) -> RenderElement:
    # `LidarSensor` redeclares x/y with defaults of 0, so an unset one would sit
    # at the origin. When an owner is named, the owner's position wins: the fan
    # is mounted on the car, and the baseline sets both to the same value.
    owner = context.positions.get(element.owner_id)
    x = owner[0] if owner is not None else element.x
    y = owner[1] if owner is not None else element.y
    return EmitterElement(
        id=element.id,
        role="sensor",
        label=element.label or "",
        x=x,
        y=y,
        radius=element.radius,
        fov=element.fov_degrees,
        rays=element.ray_count,
        anchor=element.owner_id or None,
        props={"scan_interval_ms": element.scan_interval_ms},
    )


def _convert_obstacle(element: Obstacle, context: _Context) -> RenderElement:
    # A star, not a circle, and not a glyph either: `app.js:696` draws each
    # obstacle as 8 vertices alternating between `r` and `0.78r`, which is two
    # numbers an atom computes exactly. Shipping that as a T2 glyph would mean
    # freezing a hand-written `d` string for it, which decision D1 rules out.
    return BodyElement(
        id=element.id,
        role="obstacle",
        label=element.label or "",
        x=element.x,
        y=element.y,
        shape="polygon",
        width=element.radius * 2,
        height=element.radius * 2,
        sides=8,
        inner_ratio=0.78,
        props={"radius": element.radius, "draggable": element.draggable},
    )


def _convert_circle(element: Circle, context: _Context) -> RenderElement:
    return ZoneElement(
        id=element.id,
        role="threshold",
        label=element.label or "",
        x=element.x,
        y=element.y,
        radius=element.radius,
        props={"dashed": element.dashed, "opacity": element.fill_opacity},
    )


def _convert_label(element: Label, context: _Context) -> RenderElement:
    return ReadoutElement(
        id=element.id,
        role="caption",
        label=element.label or "",
        x=element.x,
        y=element.y,
        text=element.text,
    )


def _convert_ros_node(element: RosNode, context: _Context) -> RenderElement:
    return BodyElement(
        id=element.id,
        role="node",
        label=element.label or "",
        x=element.x,
        y=element.y,
        shape="rect",
        props={
            # `RosNode.role` is a publishing role, not a registry role — the
            # registry one is "node", above. Renamed so the two can never be
            # confused for each other in a prop lookup.
            "ros_role": element.role,
            "package": element.package or "",
            "executable": element.executable or "",
        },
    )


def _convert_ros_topic(element: RosTopic, context: _Context) -> RenderElement:
    source = context.positions.get(element.from_node_id)
    target = context.positions.get(element.to_node_id)
    if source is None or target is None:
        raise LegacyConversionError(
            f"ros_topic `{element.id}` 引用了不存在的节点："
            f"from={element.from_node_id!r} to={element.to_node_id!r}"
        )
    return LinkElement(
        id=element.id,
        role="topic",
        label=element.label or element.name,
        x=(source[0] + target[0]) / 2,
        y=(source[1] + target[1]) / 2,
        points=[_point(*source), _point(*target)],
        props={"message_type": element.message_type},
    )


def _convert_message_packet(element: MessagePacket, context: _Context) -> RenderElement:
    return TravelerElement(
        id=element.id,
        role="message",
        label=element.label or element.payload_label,
        x=element.x,
        y=element.y,
        path_id=element.topic_id,
        props={"state": element.state},
    )


def _convert(element: AnimatedElement, context: _Context) -> RenderElement:
    """Dispatch on the concrete frozen class. Exhaustive over `KNOWN_ELEMENT_TYPES`."""
    if isinstance(element, ManualObject):
        return _convert_manual_object(element, context)
    if isinstance(element, RobotCar):
        return _convert_robot_car(element, context)
    if isinstance(element, LidarSensor):
        return _convert_lidar_sensor(element, context)
    if isinstance(element, Obstacle):
        return _convert_obstacle(element, context)
    if isinstance(element, Circle):
        return _convert_circle(element, context)
    if isinstance(element, Label):
        return _convert_label(element, context)
    if isinstance(element, RosNode):
        return _convert_ros_node(element, context)
    if isinstance(element, RosTopic):
        return _convert_ros_topic(element, context)
    if isinstance(element, MessagePacket):
        return _convert_message_packet(element, context)
    # Never a silent drop: the name goes in the message so the fix is obvious.
    raise LegacyConversionError(
        f"元素 `{element.id}` 的类型 `{element.type}` 没有对应的 RenderSpec 图元；"
        f"已知类型：{'、'.join(KNOWN_ELEMENT_TYPES)}"
    )


def _convert_control(control: InteractionControl) -> RenderControl:
    if isinstance(control, SliderControl):
        return RenderControl(
            id=control.id,
            type="slider",
            label=control.label,
            target_property=control.target_property,
            min=control.min_value,
            max=control.max_value,
            default=control.default_value,
            step=control.step,
        )
    if isinstance(control, ToggleControl):
        return RenderControl(
            id=control.id,
            type="toggle",
            label=control.label,
            target_property=control.target_property,
            default=1.0 if control.default_value else 0.0,
        )
    if isinstance(control, ButtonControl):
        return RenderControl(
            id=control.id,
            type="button",
            label=control.label,
            target_property=control.target_property,
            action=control.action or None,
        )
    raise LegacyConversionError(
        f"控件 `{control.id}` 的类型 `{control.type}` 没有对应的 RenderSpec 控件"
    )


def _convert_step(step: TimelineStep) -> RenderStep:
    return RenderStep(
        id=step.id,
        title=step.title,
        narration=step.narration,
        highlights=list(step.focus_element_ids),
        # `actions` is an untyped `list[dict[str, Any]]` in the frozen baseline and
        # is empty in all three templates. Mapping it would need a typed
        # vocabulary nobody has; leaving it out is honest, and the test below
        # asserts every frozen scene really does leave it empty.
        states={},
    )


def _synthesise_safe_zone(
    scene: Scene, bodies: list[BodyElement], controls: list[RenderControl]
) -> ZoneElement | None:
    """Add the dashed safe-distance circle the baseline draws but never declares.

    See the module docstring. Returns `None` when the scene has no
    `scene.safe_distance` slider, or no vehicle to centre it on — both are
    legitimate scenes, not errors.
    """
    radius: float | None = None
    for control in controls:
        if control.target_property == "scene.safe_distance" and control.default is not None:
            radius = control.default
            break
    if radius is None:
        return None
    vehicle = next((body for body in bodies if body.role == "vehicle"), None)
    if vehicle is None:
        return None
    return ZoneElement(
        id="safe-zone",
        role="safe_distance",
        label="安全距离",
        x=vehicle.x,
        y=vehicle.y,
        radius=radius,
        anchor=vehicle.id,
        props={"radius": radius, "enabled": True},
    )


def scene_to_render_spec(scene: Scene, *, stage: RenderStage | None = None) -> RenderSpec:
    """Translate one frozen `Scene` into a `RenderSpec`.

    Deterministic and side-effect free: no LLM call, no layout algorithm, no
    clock. Called by `--render-sample` so the baseline can be drawn through the
    same player as everything else, which is what makes a three-way comparison
    between baseline, hand-written sample and generated output meaningful.
    """
    # `Scene.elements` is typed `list[SpecSerializable]`, a Protocol — so the
    # adapter narrows before it trusts anything, rather than reading `.x` off an
    # object that only promised `to_spec()`.
    animated = [element for element in scene.elements if isinstance(element, AnimatedElement)]
    if len(animated) != len(scene.elements):
        raise LegacyConversionError(
            f"场景 `{scene.title}` 里有 {len(scene.elements) - len(animated)} 个元素"
            "不是 AnimatedElement，没有 id/x/y 可供布局"
        )

    context = _Context(positions={item.id: (item.x, item.y) for item in animated})
    converted: list[RenderElement] = [_convert(element, context) for element in animated]

    bodies = [item for item in converted if isinstance(item, BodyElement)]
    controls = [_convert_control(control) for control in scene.controls]

    safe_zone = _synthesise_safe_zone(scene, bodies, controls)
    if safe_zone is not None:
        converted.append(safe_zone)

    template = scene.metadata.get("template")
    params: dict[str, PropValue] = {}
    for control in controls:
        if control.target_property.startswith("scene.") and control.default is not None:
            params[control.target_property.removeprefix("scene.")] = control.default

    render_scene = RenderScene(
        id=template if isinstance(template, str) else "scene-1",
        title=scene.title,
        preset=TEMPLATE_TO_PRESET.get(template if isinstance(template, str) else "", "generic"),
        attachment={
            item.id: item.anchor
            for item in converted
            if isinstance(item, (EmitterElement, ZoneElement)) and item.anchor is not None
        },
        elements=converted,
        steps=[_convert_step(step) for step in scene.timeline],
        controls=controls,
        params=params,
    )

    return RenderSpec(
        storyboard_id=template if isinstance(template, str) else "",
        title=scene.title,
        stage=stage if stage is not None else DEFAULT_STAGE,
        scenes=[render_scene],
    )
