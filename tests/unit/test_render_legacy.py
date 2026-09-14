"""Tests for the frozen-`Scene` -> `RenderSpec` adapter (checkpoint P2).

The adapter exists so the hand-written baseline can be drawn by the same player
as everything else. That only works if the translation is *complete* — an element
type quietly dropped produces a picture that is missing something and looks fine
— so most of what follows is completeness and determinism, not pixels.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, fields
from pathlib import Path

import pytest

from animate_agent.animation import elements as elements_module
from animate_agent.animation.elements import AnimatedElement, Scene
from animate_agent.animation.templates import (
    build_robot_obstacle_avoidance_scene,
    build_ros_pub_sub_scene,
)
from animate_agent.rendering.legacy import (
    FROZEN_TEMPLATES,
    KNOWN_ELEMENT_TYPES,
    LEGACY_TYPE_TO_ROLE,
    TEMPLATE_TO_PRESET,
    LegacyConversionError,
    scene_to_render_spec,
)
from animate_agent.rendering.models import BodyElement, EmitterElement, RenderSpec, ZoneElement
from animate_agent.rendering.registry import ROLE_TO_PRIMITIVE
from animate_agent.storyboard.models import StoryboardIR

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples" / "storyboards"


def _declared_element_types() -> set[str]:
    """Every `type` literal the frozen element classes declare, harvested live.

    Read off the dataclasses rather than a list kept in this file: the point of
    the test below is to notice when the baseline grows a class the adapter has
    never heard of.
    """
    found: set[str] = set()
    for name in dir(elements_module):
        candidate = getattr(elements_module, name)
        if not (isinstance(candidate, type) and issubclass(candidate, AnimatedElement)):
            continue
        if candidate is AnimatedElement:
            continue
        for field in fields(candidate):
            if field.name == "type" and field.default is not MISSING:
                assert isinstance(field.default, str)
                found.add(field.default)
    return found


def _frozen_scenes() -> list[Scene]:
    return [build_robot_obstacle_avoidance_scene(), build_ros_pub_sub_scene()]


def test_the_adapter_covers_every_type_the_baseline_can_emit() -> None:
    """A new element class in the frozen baseline must fail here, not silently vanish."""
    assert _declared_element_types() == set(KNOWN_ELEMENT_TYPES)


def test_every_renderable_template_has_a_preset() -> None:
    """`--render-template <name>` and the preset it labels the output with must agree."""
    assert set(FROZEN_TEMPLATES) == set(TEMPLATE_TO_PRESET)


@pytest.mark.parametrize("scene", _frozen_scenes(), ids=lambda scene: scene.title)
def test_every_frozen_scene_converts(scene: Scene) -> None:
    spec = scene_to_render_spec(scene)

    assert isinstance(spec, RenderSpec)
    assert len(spec.scenes) == 1
    assert spec.scenes[0].elements


def test_an_unknown_element_type_raises_with_the_type_name() -> None:
    """Never a silent drop — the message has to say which type was not handled."""

    class Invented(AnimatedElement):
        type: str = "invented_thing"

    scene = Scene(title="t", elements=[Invented(id="x", type="invented_thing", x=1, y=2)])

    with pytest.raises(LegacyConversionError, match="invented_thing"):
        scene_to_render_spec(scene)


def test_conversion_is_deterministic() -> None:
    """Same scene in, byte-identical spec out. A picture is only comparable if it repeats."""
    first = scene_to_render_spec(build_robot_obstacle_avoidance_scene())
    second = scene_to_render_spec(build_robot_obstacle_avoidance_scene())

    assert first.model_dump_json() == second.model_dump_json()


def test_every_frozen_scene_leaves_heading_at_zero() -> None:
    """Guards an ambiguity rather than a bug.

    `RobotCar.heading`'s unit is stated nowhere in the frozen source, and every
    frozen scene leaves it at the default — so degrees and radians are
    indistinguishable today. The adapter passes it through and the player reads
    degrees. The moment a nonzero heading appears, that guess becomes load-bearing
    and somebody has to answer the question for real; this test is what makes the
    moment announce itself instead of producing a subtly wrong picture.
    """
    for scene in _frozen_scenes():
        for element in scene.elements:
            assert getattr(element, "heading", 0) == 0, (
                f"{scene.title} 里的 `{element.id}` 给了非零 heading —— "
                "frozen 源码没有说明单位，先确认是度还是弧度再放开这条"
            )


def test_every_frozen_step_leaves_actions_empty() -> None:
    """The adapter drops `TimelineStep.actions`; dropping real data would be a lie."""
    for scene in _frozen_scenes():
        for step in scene.timeline:
            assert step.actions == [], (
                f"`{step.id}` 有 actions——适配器没搬它，得先给它一个类型化词汇表"
            )


def test_the_baseline_spec_carries_every_baseline_control() -> None:
    spec = scene_to_render_spec(build_robot_obstacle_avoidance_scene())
    scene = spec.scenes[0]

    assert {(control.type, control.target_property) for control in scene.controls} == {
        ("slider", "car.speed"),
        ("slider", "lidar.radius"),
        ("slider", "scene.safe_distance"),
        ("button", "scene"),
    }


def test_the_safe_zone_is_synthesised_and_anchored_to_the_vehicle() -> None:
    """The baseline draws this circle from a slider value; the spec has to declare it."""
    scene = scene_to_render_spec(build_robot_obstacle_avoidance_scene()).scenes[0]

    zone = next(item for item in scene.elements if item.id == "safe-zone")
    assert isinstance(zone, ZoneElement)
    assert zone.anchor == "car"
    assert zone.radius == 76  # the `safe-distance` slider's default
    assert scene.attachment["safe-zone"] == "car"
    assert scene.params["safe_distance"] == 76


def test_the_lidar_is_anchored_to_the_car_by_owner_id() -> None:
    scene = scene_to_render_spec(build_robot_obstacle_avoidance_scene()).scenes[0]
    car = next(item for item in scene.elements if item.id == "car")
    lidar = next(item for item in scene.elements if item.id == "lidar")

    assert isinstance(car, BodyElement)
    assert isinstance(lidar, EmitterElement)
    assert lidar.anchor == "car"
    assert (lidar.x, lidar.y) == (car.x, car.y)
    assert scene.attachment["lidar"] == "car"


def test_the_obstacle_is_a_parametric_star_not_a_glyph() -> None:
    """D1 allows a hand-written `d` string nowhere, and this shape does not need one.

    `app.js:696` draws 8 vertices alternating between `r` and `0.78r` — two
    numbers an atom computes exactly. Declaring it as a T2 glyph would mean a
    build-time normalisation script for something that is not an icon.
    """
    scene = scene_to_render_spec(build_robot_obstacle_avoidance_scene()).scenes[0]
    obstacle = next(item for item in scene.elements if item.id == "obstacle-1")

    assert isinstance(obstacle, BodyElement)
    assert obstacle.shape == "polygon"
    assert obstacle.glyph is None
    assert (obstacle.sides, obstacle.inner_ratio) == (8, 0.78)


def test_the_role_table_matches_what_the_adapter_produces() -> None:
    """The declared mapping and the converters must not drift apart.

    `LEGACY_TYPE_TO_ROLE` is documentation unless something holds it to the code.
    It earned that suspicion: the test suite and the adapter each had their own
    copy of this table and promptly disagreed about `circle`.
    """
    for scene in _frozen_scenes():
        produced = {
            element.id: element.role
            for element in scene_to_render_spec(scene).scenes[0].elements
        }
        for element in scene.elements:
            assert produced[element.id] == LEGACY_TYPE_TO_ROLE[element.type], element.id


def test_every_mapped_role_is_a_registered_role() -> None:
    unknown = sorted(set(LEGACY_TYPE_TO_ROLE.values()) - set(ROLE_TO_PRIMITIVE))

    assert unknown == []


def test_the_two_frozen_scenes_land_on_different_presets() -> None:
    """Otherwise the preset label carries no information."""
    robot = scene_to_render_spec(build_robot_obstacle_avoidance_scene()).scenes[0]
    ros = scene_to_render_spec(build_ros_pub_sub_scene()).scenes[0]

    assert robot.preset == "lane"
    assert ros.preset == "chain"


def test_the_ros_topic_edge_spans_its_two_nodes() -> None:
    """`ros_topic` is an edge: its geometry lives on the nodes it joins."""
    spec = scene_to_render_spec(build_ros_pub_sub_scene())
    scene = spec.scenes[0]
    positions = {item.id: (item.x, item.y) for item in scene.elements}
    topic = next(item for item in scene.elements if item.id == "cmd-topic")

    ends = {(point.x, point.y) for point in topic.points}  # type: ignore[attr-defined]
    assert ends == {positions["talker-node"], positions["base-node"]}


def test_the_ros_topic_raises_when_a_node_is_missing() -> None:
    from animate_agent.animation.elements import RosTopic

    scene = Scene(
        title="t",
        elements=[
            RosTopic(
                id="t1",
                x=0,
                y=0,
                name="/x",
                from_node_id="ghost",
                to_node_id="also-ghost",
                message_type="std_msgs/String",
            )
        ],
    )

    with pytest.raises(LegacyConversionError, match="ghost"):
        scene_to_render_spec(scene)


def test_the_legacy_robot_scene_and_the_hand_written_sample_agree() -> None:
    """The two descriptions of the baseline must not drift apart.

    `data/samples/storyboards/robot_obstacle_avoidance.json` is the quality
    baseline written in the *new* contract by hand; this adapter produces the same
    baseline in the new contract mechanically. Different routes, same scene — so
    the controls must match exactly and the narration must be word for word. If
    they diverge, one of the two has quietly stopped describing the baseline, and
    the three-way comparison would be measuring the wrong thing.
    """
    legacy = scene_to_render_spec(build_robot_obstacle_avoidance_scene()).scenes[0]
    sample = StoryboardIR.model_validate_json(
        (SAMPLES / "robot_obstacle_avoidance.json").read_text(encoding="utf-8")
    ).scenes[0]

    assert {
        (control.type, control.target_property) for control in legacy.controls
    } == {(control.type, control.target_property) for control in sample.controls}
    assert [step.narration for step in legacy.steps] == [
        step.description for step in sample.steps
    ]
    assert [step.id for step in legacy.steps] == [step.id for step in sample.steps]


def test_the_spec_round_trips_through_json_unchanged() -> None:
    """The player receives JSON, so the model's own serialisation must be lossless."""
    spec = scene_to_render_spec(build_robot_obstacle_avoidance_scene())
    dumped = spec.model_dump_json()

    assert RenderSpec.model_validate_json(dumped).model_dump_json() == dumped
    # And the discriminator survives, or the union could not be rebuilt on load.
    payload = json.loads(dumped)
    assert {item["kind"] for item in payload["scenes"][0]["elements"]} == {
        "body",
        "emitter",
        "zone",
    }
