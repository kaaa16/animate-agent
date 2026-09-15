"""Tests for the deterministic layout pass: a semantic scene becomes coordinates.

Layout is where the picture is decided (decision D3 — the model never supplies a
number), so the properties worth testing are the ones a person would otherwise
have to eye up on a canvas: that every object lands somewhere, that the same
input lands in the *same* place twice, that the arrangement reproduces the
quality baseline, and that a scene which cannot be arranged fails loudly instead
of producing a half-empty picture that reports success.
"""

from __future__ import annotations

import math
import re
from itertools import pairwise
from pathlib import Path

import pytest

from animate_agent.animation.elements import Obstacle, RobotCar
from animate_agent.animation.templates import build_robot_obstacle_avoidance_scene
from animate_agent.rendering import layout
from animate_agent.rendering.layout import (
    DEFAULT_OBSTACLE_RADIUS,
    LayoutError,
    layout_scene,
    layout_storyboard,
)
from animate_agent.rendering.models import (
    BodyElement,
    EmitterElement,
    LinkElement,
    RenderElement,
    RenderSpec,
    RenderStage,
    TravelerElement,
    ZoneElement,
)
from animate_agent.rendering.registry import (
    STAGE_RANGES,
    T1_PRIMITIVES,
    Glyph,
    GlyphPart,
)
from animate_agent.storyboard.models import (
    PropValue,
    StoryboardIR,
    StoryboardObject,
    StoryboardScene,
    StoryboardStep,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES = REPO_ROOT / "data" / "samples" / "storyboards"
BEHAVIORS_JS = REPO_ROOT / "frontend" / "player" / "behaviors.js"


def _sample() -> StoryboardIR:
    """The hand-written baseline storyboard — the layout layer's reference input."""
    return StoryboardIR.model_validate_json(
        (SAMPLES / "robot_obstacle_avoidance.json").read_text(encoding="utf-8")
    )


def _object(
    object_id: str,
    role: str,
    label: str = "对象",
    props: dict[str, PropValue] | None = None,
    **extra: PropValue,
) -> StoryboardObject:
    """A storyboard object. `props` is a dict because `from` is a Python keyword."""
    return StoryboardObject(id=object_id, role=role, label=label, props={**(props or {}), **extra})


def _scene(objects: list[StoryboardObject], **overrides: object) -> StoryboardScene:
    payload: dict[str, object] = {
        "id": "shot-1",
        "scene_type": "lane",
        "teaching_goal": "看懂这一拍",
        "lesson_scene_ids": ["scene-1"],
        "objects": objects,
        "steps": [
            StoryboardStep(
                id="beat",
                title="一拍",
                description="这一拍让画面发生一点变化。",
                highlights=["car"],
                key_points=["要点"],
            )
        ],
    }
    payload.update(overrides)
    return StoryboardScene.model_validate(payload)


def _car() -> StoryboardObject:
    return _object("car", "vehicle", "小车")


def _by_id() -> dict[str, RenderElement]:
    """The laid-out sample's elements, keyed by id. Used where position matters."""
    scene = layout_storyboard(_sample()).scenes[0]
    return {element.id: element for element in scene.elements}


# --------------------------------------------------------------------------
# The baseline: the picture this preset exists to reproduce
# --------------------------------------------------------------------------


def test_the_lane_ratios_are_the_baselines_own_proportions() -> None:
    """The acceptance criterion for M2, made exact.

    A laid-out `lane` scene and the translated baseline (`legacy.py`) are put
    side by side to answer "is the difference in the layout, or in the drawing?".
    That comparison is only worth anything if both describe the *same* picture,
    so the ratios are the baseline's own fractions rather than round decimals
    that land a few pixels away — a few pixels is exactly the kind of difference
    nobody can attribute to either layer.
    """
    baseline = build_robot_obstacle_avoidance_scene()
    baseline_car = next(item for item in baseline.elements if isinstance(item, RobotCar))
    baseline_obstacles = [item for item in baseline.elements if isinstance(item, Obstacle)]

    laid_out = _by_id()
    car = laid_out["car"]
    assert isinstance(car, BodyElement)
    assert (car.x, car.y) == pytest.approx((baseline_car.x, baseline_car.y))

    assert laid_out["obstacle-a"].x == pytest.approx(baseline_obstacles[0].x)
    assert laid_out["obstacle-b"].x == pytest.approx(baseline_obstacles[1].x)


def test_the_obstacles_alternate_sides_of_the_lane() -> None:
    """Otherwise the "compare left and right clearance" beat is a foregone conclusion.

    This is a deliberate departure from the baseline, which puts both obstacles
    above the lane (y=300 and y=210 against a lane at y=310) and therefore always
    swerves the same way.
    """
    laid_out = _by_id()
    car = laid_out["car"]

    assert laid_out["obstacle-a"].y < car.y
    assert laid_out["obstacle-b"].y > car.y


def test_the_car_can_escape_the_danger_envelope() -> None:
    """The one invariant that spans two layers, so it is checked against both.

    The obstacle offset is a layout constant; the distance the car can swerve is
    a player constant. Between them they decide whether the fourth beat *does*
    anything: too far and the obstacle stops being dangerous (no swerve), too
    near and no amount of swerving leaves the envelope (a permanent danger state
    that still animates, so nothing looks broken). Neither number is wrong on its
    own — only the inequality is — which is why it is asserted rather than
    reasoned about, and why `MAX_DODGE` is read out of the player's own source
    instead of being repeated here.
    """
    max_dodge = _player_constant("MAX_DODGE")
    scene = layout_storyboard(_sample()).scenes[0]
    laid_out = {element.id: element for element in scene.elements}
    lane_y = laid_out["car"].y
    safe_distance = float(scene.params["safe_distance"])

    for obstacle_id in ("obstacle-a", "obstacle-b"):
        offset = abs(laid_out[obstacle_id].y - lane_y)
        assert offset < safe_distance, (
            f"`{obstacle_id}` 离车道 {offset:.0f}px，已经超过安全距离 {safe_distance:.0f}px，"
            "小车根本不会进入危险态——这一拍就没有避让可看了"
        )
        assert offset + max_dodge > safe_distance, (
            f"`{obstacle_id}` 离车道 {offset:.0f}px，而小车最多只能侧移 {max_dodge:.0f}px，"
            f"加起来仍小于安全距离 {safe_distance:.0f}px：车永远出不了危险圈，"
            "画面照常在动，所以看不出坏"
        )


def _player_constant(name: str) -> float:
    if not BEHAVIORS_JS.is_file():
        pytest.skip(f"播放器源码不在：{BEHAVIORS_JS}")
    match = re.search(rf"const {name}\s*=\s*([0-9.]+)\s*;", BEHAVIORS_JS.read_text("utf-8"))
    assert match is not None, f"behaviors.js 里找不到常量 `{name}`；它被重命名了吗？"
    return float(match.group(1))


# --------------------------------------------------------------------------
# Completeness and determinism
# --------------------------------------------------------------------------


def test_every_object_becomes_exactly_one_element_in_storyboard_order() -> None:
    """A dropped object is a picture with a hole in it that reports success."""
    storyboard = _sample()
    scene = layout_storyboard(storyboard).scenes[0]

    assert [element.id for element in scene.elements] == [
        obj.id for obj in storyboard.scenes[0].objects
    ]


def test_every_element_is_a_registered_primitive() -> None:
    """The layout may only emit what the vocabulary says the renderer can draw."""
    known = {primitive.name for primitive in T1_PRIMITIVES}
    scene = layout_storyboard(_sample()).scenes[0]

    assert {element.kind for element in scene.elements} <= known


def test_layout_is_deterministic() -> None:
    """Same input, same frames — the property every comparison here rests on."""
    first = layout_storyboard(_sample()).model_dump_json()
    second = layout_storyboard(_sample()).model_dump_json()

    assert first == second


def test_the_laid_out_spec_round_trips_through_json_unchanged() -> None:
    """The player receives JSON: normalising it again must not change a byte."""
    dumped = layout_storyboard(_sample()).model_dump_json()

    assert RenderSpec.model_validate_json(dumped).model_dump_json() == dumped


def test_the_spec_carries_the_storyboard_identity() -> None:
    storyboard = _sample()
    spec = layout_storyboard(storyboard)
    scene = spec.scenes[0]

    assert (spec.storyboard_id, spec.lesson_id, spec.document_id) == (
        storyboard.storyboard_id,
        storyboard.lesson_id,
        storyboard.document_id,
    )
    assert (spec.title, spec.subject, spec.eyebrow) == (
        storyboard.title,
        storyboard.subject,
        storyboard.eyebrow,
    )
    assert scene.preset == storyboard.scenes[0].scene_type
    assert scene.teaching_goal == storyboard.scenes[0].teaching_goal


def test_steps_and_controls_survive_the_layout() -> None:
    storyboard = _sample()
    laid_out = layout_storyboard(storyboard).scenes[0]
    source = storyboard.scenes[0]

    assert [step.id for step in laid_out.steps] == [step.id for step in source.steps]
    assert [step.narration for step in laid_out.steps] == [
        step.description for step in source.steps
    ]
    assert [step.highlights for step in laid_out.steps] == [
        step.highlights for step in source.steps
    ]
    assert laid_out.steps[1].states == {"obstacle-a": {"danger": True}}
    assert [control.model_dump() for control in laid_out.controls] == [
        control.model_dump() for control in source.controls
    ]


def test_a_larger_stage_scales_the_lane_rather_than_cropping_it() -> None:
    """Every constant is a ratio, so the same scene composes on a bigger canvas."""
    small = layout_storyboard(_sample(), stage=RenderStage(width=960, height=600)).scenes[0]
    big = layout_storyboard(_sample(), stage=RenderStage(width=1920, height=1200)).scenes[0]

    small_car = next(element for element in small.elements if element.id == "car")
    big_car = next(element for element in big.elements if element.id == "car")
    assert (big_car.x, big_car.y) == pytest.approx((small_car.x * 2, small_car.y * 2))


# --------------------------------------------------------------------------
# Props: copied, never invented, never rewritten
# --------------------------------------------------------------------------


def test_props_are_copied_verbatim_and_not_shared_with_the_storyboard() -> None:
    """Layout reads props to *choose geometry* and must not rewrite one.

    Playback mutates props through `object_states`; a value layout baked in would
    be a second source of truth for the same number. The copy matters too — the
    spec must not alias the storyboard's own dict, or a later mutation of one
    would silently edit the other.
    """
    origin = _object("car", "vehicle", "小车", speed=1.0, danger=True, glyph="car")
    scene = _scene([origin, _object("obstacle-a", "obstacle", radius=40)])

    car = layout_scene(scene).elements[0]
    assert car.props == {"speed": 1.0, "danger": True, "glyph": "car"}
    assert car.props is not origin.props


def test_a_boolean_radius_is_not_read_as_a_number() -> None:
    """`bool` is an `int` subclass, so `radius: true` would silently become 1.0.

    A one-unit obstacle is not a small obstacle, it is an invisible one — the
    kind of wrong that survives every check downstream.
    """
    scene = _scene([_car(), _object("obstacle-a", "obstacle", radius=True)])
    obstacle = next(
        element for element in layout_scene(scene).elements if element.id == "obstacle-a"
    )

    assert isinstance(obstacle, BodyElement)
    assert obstacle.width == pytest.approx(DEFAULT_OBSTACLE_RADIUS * 2)


def test_the_lidar_keeps_its_declared_reach_fov_and_ray_count() -> None:
    lidar = _by_id()["lidar"]

    assert isinstance(lidar, EmitterElement)
    assert (lidar.radius, lidar.fov, lidar.rays) == (150.0, 270.0, 13)


def test_the_obstacle_is_a_parametric_star_not_a_glyph() -> None:
    """D1 rules out a hand-written `d` string, and this shape does not need one."""
    obstacle = _by_id()["obstacle-a"]

    assert isinstance(obstacle, BodyElement)
    assert (obstacle.shape, obstacle.sides, obstacle.inner_ratio) == ("polygon", 8, 0.78)


# --------------------------------------------------------------------------
# Mounting: an emitter or a zone rides on a body
# --------------------------------------------------------------------------


def test_mounted_elements_sit_on_their_anchor_and_record_the_relation() -> None:
    scene = layout_storyboard(_sample()).scenes[0]
    laid_out = {element.id: element for element in scene.elements}
    car = laid_out["car"]

    for mounted_id in ("lidar", "safe-zone"):
        mounted = laid_out[mounted_id]
        assert (mounted.x, mounted.y) == (car.x, car.y)
        assert scene.attachment[mounted_id] == "car"

    assert set(scene.attachment) == {"lidar", "safe-zone"}


def test_a_relation_pointing_at_something_without_a_position_raises() -> None:
    """An `of` aimed at another sensor has no coordinate to hang from.

    Drawing it at the origin is the tempting fallback and the worst one: the fan
    ends up in the corner of the canvas looking deliberate.
    """
    scene = _scene(
        [
            _car(),
            _object("obstacle-a", "obstacle"),
            _object("lidar", "sensor", of="car"),
            _object("lidar2", "sensor", of="lidar"),
        ]
    )

    with pytest.raises(LayoutError, match="lidar2"):
        layout_scene(scene)


def test_a_mounted_element_with_no_relation_at_all_raises() -> None:
    scene = _scene([_car(), _object("obstacle-a", "obstacle"), _object("lidar", "sensor")])

    with pytest.raises(LayoutError, match="lidar"):
        layout_scene(scene)


def test_the_safe_distance_zone_follows_its_scene_parameter() -> None:
    """The circle and the threshold are the same quantity; the lidar's is not.

    The sample declares no `radius` on the zone, so the value comes from
    `scene.safe_distance` — which is also what the slider moves. Without the
    binding the picture still changes when the slider moves, because the danger
    test reads the parameter; only the circle named 安全距离 sits still, and only
    a person looking at it would notice.
    """
    zone = _by_id()["safe-zone"]

    assert isinstance(zone, ZoneElement)
    assert zone.radius == pytest.approx(76.0)
    assert zone.binds == {"radius": "scene.safe_distance"}


def test_a_zone_that_declares_its_own_radius_is_not_bound_to_the_scene() -> None:
    """A lidar's reach and a danger threshold are both `radius`, and are not one knob."""
    ring_object = _object("ring", "threshold", of="car", radius=40)
    scene = _scene([_car(), _object("obstacle-a", "obstacle"), ring_object])

    ring = next(element for element in layout_scene(scene).elements if element.id == "ring")

    assert isinstance(ring, ZoneElement)
    assert ring.radius == pytest.approx(40.0)
    assert ring.binds == {}


def test_the_emitter_is_never_bound_to_a_scene_parameter() -> None:
    """Being mounted on the car is a position relation, not a value relation."""
    lidar = _by_id()["lidar"]

    assert isinstance(lidar, EmitterElement)
    assert lidar.binds == {}


# --------------------------------------------------------------------------
# Glyphs: requested by name, drawn only when the data exists
# --------------------------------------------------------------------------


def _car_with_glyph(scene_type: str = "lane") -> StoryboardScene:
    return _scene(
        [_object("car", "vehicle", "小车", glyph="car"), _object("obstacle-a", "obstacle")],
        scene_type=scene_type,
    )


def test_a_glyph_with_no_data_is_omitted_but_the_request_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The player refuses a glyph it has no data for, so layout must not name one.

    The requested name stays in `props` (props are copied verbatim), which is how
    the gap stays visible in the spec instead of vanishing.
    """
    monkeypatch.setattr(layout, "GLYPH_DATA_DIR", tmp_path / "absent")

    car = layout_scene(_car_with_glyph()).elements[0]
    assert isinstance(car, BodyElement)
    assert car.glyph is None
    assert car.props["glyph"] == "car"


def test_the_glyph_appears_as_soon_as_its_data_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No code change needed the day the assets land — this is that promise, tested."""
    glyph = Glyph(
        name="car",
        view_box=(0, 0, 24, 24),
        ink_box=(2, 5, 20, 15),
        domain="robotics",
        source="tabler:car",
        parts={"body": GlyphPart(d="M0 0h24v24H0z", mode="stroke")},
    )
    (tmp_path / "car.json").write_text(glyph.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(layout, "GLYPH_DATA_DIR", tmp_path)

    car = layout_scene(_car_with_glyph()).elements[0]
    assert isinstance(car, BodyElement)
    assert car.glyph == "car"


# --------------------------------------------------------------------------
# Failure: unplaceable scenes raise rather than being nudged into place
# --------------------------------------------------------------------------


def test_an_unimplemented_preset_raises_with_its_name() -> None:
    """Never falls back to `generic`: a scene that asked for a field and got a
    vertical list is a picture that lies about what it is.

    The name is invented rather than taken from the registry, because every
    preset the registry offers now has a placer — which is the point of the test
    below it. What is being pinned here is the *behaviour* on a miss, and that
    has to hold for the next preset someone adds as much as it did for `field`.
    """
    scene = _scene([_car(), _object("obstacle-a", "obstacle")], scene_type="lattice")

    with pytest.raises(LayoutError, match="lattice"):
        layout_scene(scene)


def test_every_preset_has_a_placer() -> None:
    """Adding a preset to the registry must not silently leave it unplaceable.

    This list used to be `{hub, field}` — the two that were owed. Both landed
    because the real chain asked for them: the projectile document picked
    `field` for 7 of its 8 scenes and the avoidance document picked `hub` for
    two, so `--render` failed on 2 of 3 sample documents until they existed.
    A preset offered by the vocabulary and absent here is a generation that
    fails for a reason no reader of the prompt could have predicted.
    """
    from animate_agent.rendering.layout import _PLACERS
    from animate_agent.rendering.registry import PRESET_BY_NAME

    assert set(PRESET_BY_NAME) - set(_PLACERS) == set()


def test_an_unimplemented_primitive_raises_with_its_name() -> None:
    """`region` and `wave` are the remainder, and they say so by name.

    The name matters more than the failure. This is the last gate — the validator
    has already had its say and the model has already had its retry — so whoever
    reads this is looking at a hand-written storyboard, and needs to be told
    which role to swap rather than left to grep the vocabulary.
    """
    scene = _scene(
        [_car(), _object("obstacle-a", "obstacle"), _object("blob", "area", bounded_by="car")]
    )

    with pytest.raises(LayoutError, match="region") as caught:
        layout_scene(scene)

    assert "zone" in str(caught.value), "错误里要给出能替代它的角色，不然读完还是不知道怎么改"


def test_every_drawable_primitive_reaches_a_converter() -> None:
    """`drawable=True` is a promise the dispatch below has to keep.

    The gap this closes is not the known one — `PENDING_PRIMITIVES` covers that
    and the validator rejects it before anything is drawn. It is the registry and
    `_to_element` disagreeing about a primitive neither of them thinks is in
    doubt: a new primitive lands in the vocabulary with the branch that draws it
    not yet written, the prompt offers it, the validator passes it, and the run
    dies at the same terminal step the whole `drawable` flag exists to prevent.
    Whichever of the two slides, this fails first and says which name.

    A `LayoutError` is an acceptable outcome — an empty box map means most
    converters have no geometry to work from. What is asserted is only that the
    failure is about the geometry and not about the registry.
    """
    host = _object("probe-host", "vehicle", "宿主")
    scene = _scene([host])
    stage = RenderStage(width=960, height=540)

    for primitive in T1_PRIMITIVES:
        if not primitive.drawable:
            continue
        relations = dict.fromkeys(primitive.required_relations, "probe-host")
        probe = _object("probe", primitive.roles[0], "探针", **relations)
        try:
            layout._to_element(probe, scene, {}, {}, stage)
        except LayoutError as exc:
            assert "不同步" not in str(exc), f"{primitive.name}: {exc}"


def test_an_unknown_role_raises_with_the_role() -> None:
    scene = _scene([_car(), _object("obstacle-a", "obstacle"), _object("thing", "wizard")])

    with pytest.raises(LayoutError, match="wizard"):
        layout_scene(scene)


def test_a_lane_with_no_obstacle_raises() -> None:
    with pytest.raises(LayoutError, match="obstacle"):
        layout_scene(_scene([_car()]))


def test_a_lane_with_nothing_on_it_raises() -> None:
    with pytest.raises(LayoutError, match="主体"):
        layout_scene(_scene([_object("obstacle-a", "obstacle")]))


def test_an_object_pushed_off_the_canvas_raises() -> None:
    """Illegible output has to fail generation, not reach a person to notice."""
    scene = _scene([_car()] + [_object(f"obstacle-{i}", "obstacle") for i in range(5)])

    with pytest.raises(LayoutError, match="画布外面"):
        layout_scene(scene)


def test_two_objects_landing_on_each_other_raise() -> None:
    """Two boxes in the same place cannot be told apart on screen."""
    scene = _scene(
        [
            _car(),
            _object("obstacle-a", "obstacle", radius=120),
            _object("obstacle-b", "obstacle", radius=120),
        ]
    )

    with pytest.raises(LayoutError, match="叠在一起"):
        layout_scene(scene)


# --------------------------------------------------------------------------
# `chain`: a left-to-right pipeline, and the two primitives that only it uses
# --------------------------------------------------------------------------


def _ros_sample() -> StoryboardIR:
    """The hand-written ROS storyboard — the `chain` preset's reference input."""
    return StoryboardIR.model_validate_json(
        (SAMPLES / "ros_pub_sub.json").read_text(encoding="utf-8")
    )


def test_the_nodes_sit_on_a_spine_in_storyboard_order() -> None:
    """Left to right, in the order the model wrote them — which is the reading order."""
    scene = layout_storyboard(_ros_sample()).scenes[0]
    laid_out = {element.id: element for element in scene.elements}

    assert laid_out["talker"].y == pytest.approx(laid_out["base"].y)
    assert laid_out["talker"].x < laid_out["base"].x


def test_a_chain_body_that_is_not_a_node_sits_below_the_spine() -> None:
    """The runtime the chain drives is not part of the chain."""
    scene = layout_storyboard(_ros_sample()).scenes[0]
    laid_out = {element.id: element for element in scene.elements}

    assert laid_out["runtime"].y > laid_out["talker"].y


def test_a_chain_participant_belongs_on_the_spine_whatever_its_role() -> None:
    """Found by running a real document, not by writing a fixture.

    The model roled a lidar *node* as `device`. Keying the spine off
    `role == "node"` put it on the lower row, and the link from it to the next
    node then cut diagonally across the whole picture — nothing errored, the
    diagram just stopped being readable. The structural question ("does a link
    touch it?") is the one that survives whatever the model calls things.
    """
    scene = _scene(
        [
            _object("a-node", "node"),
            _object("b-device", "device"),
            _object("edge-a", "topic", props={"from": "a-node", "to": "b-device"}),
            _object("runtime", "vehicle"),
        ],
        scene_type="chain",
    )
    laid_out = {element.id: element for element in layout_scene(scene).elements}

    assert laid_out["b-device"].y == pytest.approx(laid_out["a-node"].y)
    assert laid_out["runtime"].y > laid_out["a-node"].y


def test_a_link_spans_the_two_bodies_it_joins() -> None:
    """A link is an edge: its geometry lives on its ends, not on itself."""
    scene = layout_storyboard(_ros_sample()).scenes[0]
    laid_out = {element.id: element for element in scene.elements}
    link = laid_out["cmd-topic"]

    assert isinstance(link, LinkElement)
    ends = {(point.x, point.y) for point in link.points}
    assert ends == {
        (laid_out["talker"].x, laid_out["talker"].y),
        (laid_out["base"].x, laid_out["base"].y),
    }
    assert (link.x, link.y) == pytest.approx(
        ((laid_out["talker"].x + laid_out["base"].x) / 2, laid_out["talker"].y)
    )


def test_a_traveler_position_follows_from_its_path() -> None:
    """`progress` is a share of the path, so the coordinates are derived, not authored."""
    scene = _scene(
        [
            _object("a-node", "node"),
            _object("b-node", "node"),
            _object("edge", "topic", props={"from": "a-node", "to": "b-node"}),
            _object("token", "message", along="edge", progress=0.5),
        ],
        scene_type="chain",
    )
    laid_out = {element.id: element for element in layout_scene(scene).elements}
    token = laid_out["token"]

    assert isinstance(token, TravelerElement)
    assert token.x == pytest.approx((laid_out["a-node"].x + laid_out["b-node"].x) / 2)
    assert token.y == pytest.approx(laid_out["a-node"].y)


def test_a_traveler_starting_out_sits_at_the_publisher_end() -> None:
    """The sample's message is published, not already in flight."""
    scene = layout_storyboard(_ros_sample()).scenes[0]
    laid_out = {element.id: element for element in scene.elements}
    twist = laid_out["twist"]

    assert isinstance(twist, TravelerElement)
    assert (twist.x, twist.y) == (laid_out["talker"].x, laid_out["talker"].y)
    assert twist.path_id == "cmd-topic"


def test_a_traveler_does_not_depend_on_its_path_being_declared_first() -> None:
    """The model's ordering is a teaching order, not a dependency order."""
    scene = _scene(
        [
            _object("token", "message", along="edge", progress=1.0),
            _object("a-node", "node"),
            _object("b-node", "node"),
            _object("edge", "topic", props={"from": "a-node", "to": "b-node"}),
        ],
        scene_type="chain",
    )
    laid_out = {element.id: element for element in layout_scene(scene).elements}

    assert laid_out["token"].x == pytest.approx(laid_out["b-node"].x)


def test_a_chain_with_one_node_raises() -> None:
    scene = _scene([_object("only-node", "node")], scene_type="chain")

    with pytest.raises(LayoutError, match="only-node|两个节点"):
        layout_scene(scene)


def test_a_link_without_both_ends_raises() -> None:
    scene = _scene(
        [
            _object("a-node", "node"),
            _object("b-node", "node"),
            _object("edge", "topic", props={"from": "a-node"}),
        ],
        scene_type="chain",
    )

    with pytest.raises(LayoutError, match="edge"):
        layout_scene(scene)


def test_a_link_to_something_without_a_position_raises() -> None:
    """A link to a readout has nowhere to attach: a panel is not an endpoint."""
    scene = _scene(
        [
            _object("a-node", "node"),
            _object("b-node", "node"),
            _object("edge", "topic", props={"from": "a-node", "to": "hud"}),
            _object("hud", "hud"),
        ],
        scene_type="chain",
    )

    with pytest.raises(LayoutError, match="hud"):
        layout_scene(scene)


def test_a_traveler_along_something_that_is_not_a_link_raises() -> None:
    scene = _scene(
        [
            _object("a-node", "node"),
            _object("b-node", "node"),
            _object("token", "message", along="a-node"),
        ],
        scene_type="chain",
    )

    with pytest.raises(LayoutError, match="token"):
        layout_scene(scene)


# --------------------------------------------------------------------------
# `readout`: the shared text column
# --------------------------------------------------------------------------


def test_a_readout_falls_back_to_its_label_for_text() -> None:
    """A panel with no `text` says what the model called it, not nothing."""
    scene = _scene([_car(), _object("obstacle-a", "obstacle"), _object("hud", "hud", "决策")])
    hud = next(element for element in layout_scene(scene).elements if element.id == "hud")

    assert hud.text == "决策"


def test_a_readout_alignment_is_coerced_not_trusted() -> None:
    """An alignment the vocabulary never offered is a flourish, not a broken scene."""
    scene = _scene(
        [
            _car(),
            _object("obstacle-a", "obstacle"),
            _object("hud", "hud", text="x", align="diagonal"),
        ]
    )
    hud = next(element for element in layout_scene(scene).elements if element.id == "hud")

    assert hud.align == "left"


def test_readouts_stack_in_their_own_column() -> None:
    """They have no geometry to hang off, so they need a slot the objects don't use."""
    scene = _scene(
        [
            _car(),
            _object("obstacle-a", "obstacle"),
            _object("hud-1", "hud", text="一"),
            _object("hud-2", "hud", text="二"),
        ]
    )
    laid_out = {element.id: element for element in layout_scene(scene).elements}

    assert laid_out["hud-1"].x == pytest.approx(laid_out["hud-2"].x)
    assert laid_out["hud-1"].y < laid_out["hud-2"].y
    assert laid_out["hud-1"].x > laid_out["obstacle-a"].x


# --------------------------------------------------------------------------
# `generic`: the honest column
# --------------------------------------------------------------------------


def _generic_scene(count: int) -> StoryboardScene:
    return _scene(
        [_object(f"obj-{i}", "object", f"对象 {i}") for i in range(count)],
        scene_type="generic",
        steps=[
            StoryboardStep(
                id="beat",
                title="一拍",
                description="这一拍让画面发生一点变化。",
                highlights=[f"obj-{i}" for i in range(count)],
                key_points=["要点"],
            )
        ],
    )


def test_generic_stacks_objects_in_storyboard_order() -> None:
    scene = layout_scene(_generic_scene(4))
    positions = [(element.id, element.y) for element in scene.elements]

    assert [name for name, _ in positions] == ["obj-0", "obj-1", "obj-2", "obj-3"]
    assert [y for _, y in positions] == sorted(y for _, y in positions)


def test_generic_fits_the_largest_scene_its_preset_allows() -> None:
    """`generic` allows 8 bodies; the column has to hold them without overlapping."""
    scene = layout_scene(_generic_scene(8))

    assert len(scene.elements) == 8


def test_generic_refuses_more_than_it_can_hold() -> None:
    """Better a named failure than eight objects drawn on top of each other."""
    with pytest.raises(LayoutError):
        layout_scene(_generic_scene(12))


def test_many_readouts_shrink_to_fit_the_column() -> None:
    """Found by laying out a real storyboard, not by designing for it.

    The projectile document's 关键参数 scene is seven symbol cards. At the
    default panel height they cannot be stacked inside a 432px column, and the
    first version raised `LayoutError` over the overlap — failing a scene whose
    only crime was having more to say than most. `_column` already spaces evenly
    so the column survives any count; the panel height just has to follow it.
    """
    objects = [_car(), _object("obstacle-a", "obstacle")]
    objects += [_object(f"hud-{i}", "hud", f"卡片 {i}", text=f"t{i}") for i in range(7)]
    scene = layout_scene(_scene(objects))
    panels = [element for element in scene.elements if element.id.startswith("hud-")]

    assert len(panels) == 7
    gaps = [later.y - earlier.y for earlier, later in pairwise(panels)]
    assert min(gaps) + layout.FIT_TOLERANCE >= panels[0].height


# --------------------------------------------------------------------------
# `hub`: a centre and its leaves
# --------------------------------------------------------------------------


def _hub_scene(leaves: int = 3) -> StoryboardScene:
    return _scene(
        [_object("control", "node", "控制器")]
        + [_object(f"leaf-{i}", "endpoint", f"部件 {i}") for i in range(leaves)],
        scene_type="hub",
    )


def test_hub_puts_the_node_in_the_middle_and_the_leaves_around_it() -> None:
    scene = layout_scene(_hub_scene(4))
    laid_out = {element.id: element for element in scene.elements}
    centre = laid_out["control"]

    for index in range(4):
        leaf = laid_out[f"leaf-{index}"]
        assert math.hypot(leaf.x - centre.x, leaf.y - centre.y) > 100


def test_hub_leaves_are_spread_around_the_ring_not_bunched() -> None:
    """Every leaf gets its own arc of the ring, and no two sit on each other.

    Separation rather than equal spacing, because the ring is an ellipse: even
    *parametric* steps do not produce even *visual* angles, and a test that
    demanded them would be asserting a property the layout never claimed.
    """
    scene = layout_scene(_hub_scene(5))
    leaves = [element for element in scene.elements if element.id.startswith("leaf-")]
    centre = next(element for element in scene.elements if element.id == "control")

    angles = sorted(math.atan2(leaf.y - centre.y, leaf.x - centre.x) for leaf in leaves)
    gaps = [later - earlier for earlier, later in pairwise(angles)]
    gaps.append(2 * math.pi - angles[-1] + angles[0])

    assert min(gaps) > math.pi / len(leaves)
    for index, left in enumerate(leaves):
        for right in leaves[index + 1 :]:
            assert math.hypot(left.x - right.x, left.y - right.y) > 120


def test_hub_without_a_centre_raises() -> None:
    with pytest.raises(LayoutError, match="node"):
        layout_scene(_scene([_object("leaf", "endpoint")], scene_type="hub"))


def test_hub_with_no_leaf_raises() -> None:
    with pytest.raises(LayoutError, match="叶子"):
        layout_scene(_scene([_object("control", "node")], scene_type="hub"))


# --------------------------------------------------------------------------
# `field`: a plot, and the primitives that only it uses
# --------------------------------------------------------------------------


def _ball(**extra: PropValue) -> StoryboardObject:
    return _object("ball", "projectile", "抛体", **extra)


def _field(objects: list[StoryboardObject]) -> StoryboardScene:
    return _scene(objects, scene_type="field")


def test_field_puts_the_first_body_at_the_origin() -> None:
    scene = layout_scene(_field([_ball()]))
    ball = scene.elements[0]

    assert (ball.x, ball.y) == pytest.approx(
        (960 * layout.FIELD_ORIGIN_X_RATIO, 600 * layout.FIELD_ORIGIN_Y_RATIO)
    )


def test_field_spreads_comparison_bodies_instead_of_stacking_them() -> None:
    """Three projectiles at one point is a layout failure the scene never meant."""
    scene = layout_scene(
        _field(
            [
                _ball(),
                _object("oblique", "projectile", "斜抛"),
                _object("up", "projectile", "上抛"),
            ]
        )
    )
    xs = [element.x for element in scene.elements]

    assert xs == sorted(xs)
    assert len(set(xs)) == 3


def test_a_y_axis_runs_up_even_though_the_canvas_runs_down() -> None:
    """`+y` is down in canvas and up in physics. The layout owns that flip."""
    scene = layout_scene(
        _field(
            [
                _ball(),
                _object("ax", "x_axis", "x 轴", ticks=4),
                _object("ay", "y_axis", "y 轴", ticks=4),
            ]
        )
    )
    laid_out = {element.id: element for element in scene.elements}

    assert (laid_out["ax"].heading, laid_out["ay"].heading) == (0.0, -90.0)
    assert laid_out["ax"].ticks == 4


def test_a_vector_points_where_a_physics_angle_says() -> None:
    """`direction: 90` is straight up. Read as a canvas angle it points at the floor."""
    scene = layout_scene(
        _field([_ball(), _object("up", "velocity", "向上", magnitude=10, direction=90, of="ball")])
    )
    vector = next(element for element in scene.elements if element.id == "up")

    assert vector.dx == pytest.approx(0, abs=1e-6)
    assert vector.dy < 0


def test_vector_lengths_are_proportional_within_a_scene() -> None:
    """30 : 40 is a fact about the right triangle the lesson draws.

    Normalising per scene is what preserves it: drawn to a fixed length, the
    decomposition picture would say nothing about the components.
    """
    scene = layout_scene(
        _field(
            [
                _ball(),
                _object("vx", "velocity", "水平", magnitude=30, direction=0, of="ball"),
                _object("vy", "velocity", "竖直", magnitude=40, direction=90, of="ball"),
            ]
        )
    )
    lengths = {
        element.id: math.hypot(element.dx, element.dy)
        for element in scene.elements
        if element.id in ("vx", "vy")
    }

    assert lengths["vx"] / lengths["vy"] == pytest.approx(30 / 40)


def test_a_lone_vector_is_not_drawn_at_its_share_of_itself() -> None:
    """Scene-1 draws a 50 m/s velocity beside a 9.8 m/s² acceleration.

    Scaled literally the acceleration would be a fifth of the arrow and read as
    decoration. `VECTOR_MIN_FRACTION` is what stops the picture implying the two
    are the same quantity measured on one scale — which they are not, and the
    lesson is not about that.
    """
    scene = layout_scene(
        _field(
            [
                _ball(),
                _object("v0", "velocity", "初速度", magnitude=50, direction=45, of="ball"),
                _object("g", "force", "重力", magnitude=9.8, direction=-90, of="ball"),
            ]
        )
    )
    gravity = next(element for element in scene.elements if element.id == "g")

    assert math.hypot(gravity.dx, gravity.dy) == pytest.approx(
        layout.VECTOR_MAX_LENGTH * layout.VECTOR_MIN_FRACTION
    )
    assert gravity.dy > 0  # downwards, which is what -90 means in physics


def test_a_trajectory_is_a_parabola_starting_at_its_own_body() -> None:
    scene = layout_scene(
        _field([_ball(heading=45), _object("path", "trajectory", "轨迹", length=100, of="ball")])
    )
    ball = scene.elements[0]
    trace = next(element for element in scene.elements if element.id == "path")

    assert (trace.points[0].x, trace.points[0].y) == pytest.approx((ball.x, ball.y))
    assert min(point.y for point in trace.points) < ball.y
    assert trace.points[-1].y == pytest.approx(ball.y)


def _extent(heading: float) -> tuple[float, float]:
    """(range, apex height) of the trajectory drawn for one launch angle."""
    scene = layout_scene(
        _field([_ball(heading=heading), _object("path", "trajectory", "轨迹", of="ball")])
    )
    ball = scene.elements[0]
    trace = next(element for element in scene.elements if element.id == "path")
    return (
        max(point.x for point in trace.points) - ball.x,
        ball.y - min(point.y for point in trace.points),
    )


def test_complementary_angles_land_in_the_same_place() -> None:
    """θ and 90°−θ have equal range — one of the document's stated 注意事项.

    This is the assertion that caught the first version of the curve. Fitting
    each arc to the box made 30° and 60° both fill the slot, which looks fine
    and quietly denies the fact the lesson is built on.
    """
    shallow_range, _ = _extent(30)
    steep_range, _ = _extent(60)

    assert shallow_range == pytest.approx(steep_range)


def test_a_steeper_throw_comes_out_higher() -> None:
    for shallower, steeper in ((15, 45), (30, 60), (45, 75)):
        assert _extent(steeper)[1] > _extent(shallower)[1]


def test_forty_five_degrees_goes_furthest() -> None:
    """`R = v²sin2θ/g` peaks at 45°, which the document says outright."""
    ranges = [_extent(angle)[0] for angle in (15, 30, 45, 60, 75)]

    assert max(ranges) == pytest.approx(ranges[2])
    assert ranges[0] == pytest.approx(ranges[-1])
    assert ranges[1] == pytest.approx(ranges[-2])


def test_a_vertical_throw_has_no_range_at_all() -> None:
    """The 90° special case: 射程为 0, and the arc is a line straight up."""
    assert _extent(90)[0] == pytest.approx(0, abs=1.0)
    assert _extent(90)[1] > 0


def test_a_vertical_throw_draws_a_vertical_line() -> None:
    """90° collapses to the line that *is* 竖直上抛, rather than dividing by tan 90."""
    scene = layout_scene(
        _field([_ball(heading=90), _object("path", "trajectory", "轨迹", of="ball")])
    )
    ball = scene.elements[0]
    trace = next(element for element in scene.elements if element.id == "path")

    assert all(point.x == pytest.approx(ball.x) for point in trace.points)


def test_a_dimension_keeps_the_ids_of_what_it_measures() -> None:
    """The two ids are what let a callout follow the car it is measuring."""
    scene = layout_scene(
        _field(
            [
                _ball(),
                _object("far", "projectile", "远处"),
                _object("gap", "range", "距离", props={"from": "ball", "to": "far"}),
            ]
        )
    )
    gap = next(element for element in scene.elements if element.id == "gap")

    assert (gap.from_id, gap.to_id) == ("ball", "far")
    assert gap.start.x < gap.end.x


def test_the_callout_can_measure_between_two_bodies_in_a_lane() -> None:
    """The avoidance document measures car-to-obstacle, which is a `lane` scene."""
    scene = layout_scene(
        _scene(
            [
                _car(),
                _object("obstacle-a", "obstacle"),
                _object("gap", "range", "最近距离", props={"from": "car", "to": "obstacle-a"}),
            ]
        )
    )
    laid_out = {element.id: element for element in scene.elements}
    gap = laid_out["gap"]

    assert (gap.start.x, gap.start.y) == pytest.approx((laid_out["car"].x, laid_out["car"].y))
    assert gap.end.x == pytest.approx(laid_out["obstacle-a"].x)


def test_an_angle_falls_back_to_its_anchors_heading() -> None:
    scene = layout_scene(
        _field([_ball(heading=30), _object("theta", "angle", "发射角", of="ball")])
    )
    angle = next(element for element in scene.elements if element.id == "theta")

    assert (angle.from_degrees, angle.to_degrees) == (0.0, pytest.approx(30.0))


def test_an_angle_radius_is_pulled_into_a_drawable_range() -> None:
    """`radius` has no declared unit yet, so a document's 0.5 must not become a
    half-pixel arc nobody can see."""
    scene = layout_scene(
        _field([_ball(), _object("theta", "angle", "发射角", of="ball", radius=0.5)])
    )
    angle = next(element for element in scene.elements if element.id == "theta")

    assert angle.radius == layout.ANGLE_RADIUS_MIN


def test_an_undrawable_prop_value_is_clamped_rather_than_thrown() -> None:
    """The real chain wrote `fov: 0` for a lidar that had not started scanning.

    The element models are strict and the vocabulary declares no ranges, so that
    value reached `EmitterElement` and the Pydantic error escaped as a traceback
    naming a field and no object — from a document that was otherwise fine. A
    value the vocabulary permits and the drawing cannot express is a geometry
    problem, and it belongs in a shape the layout can handle.
    """
    scene = layout_scene(
        _scene(
            [
                _car(),
                _object("obstacle-a", "obstacle"),
                _object("fan", "sensor", "扫描扇区", of="car", fov=0, radius=150),
            ]
        )
    )
    fan = next(element for element in scene.elements if element.id == "fan")

    assert isinstance(fan, EmitterElement)
    assert fan.fov == layout.MIN_FOV


# --------------------------------------------------------------------------
# Units: a distance in the document is not a distance on the stage
# --------------------------------------------------------------------------


def _zoned(scene_type: str, radius: object) -> StoryboardScene:
    """A car with an obstacle and one threshold circle, in `scene_type`."""
    return _scene(
        [
            _car(),
            _object("obstacle-a", "obstacle"),
            _object("safe", "safe_distance", "安全距离", of="car", radius=radius),
        ],
        scene_type=scene_type,
    )


def test_a_radius_in_the_documents_units_is_held_to_the_stage() -> None:
    """`safe_zone.radius: 0.8` is eight tenths of a metre *and* of a pixel.

    The avoidance document wrote that, and every layer accepted it — a legal
    positive float is a legal radius, and nothing in the vocabulary said which
    unit it was in. The circle was invisible on a 960-pixel stage, and the only
    symptom was that the lesson about 安全距离 had no circle in it.
    """
    scene = layout_scene(_zoned("lane", 0.8))
    zone = next(element for element in scene.elements if element.id == "safe")
    low, _ = STAGE_RANGES[("zone", "radius")]

    assert zone.radius == low


def test_the_prop_carries_the_same_number_that_was_drawn() -> None:
    """The drawn circle and the reading gate must not disagree.

    `proximity_gate` reads the threshold out of the element's `props`, so leaving
    the raw 0.8 there while drawing a 24-pixel circle would mean the car reacts
    at one distance and the circle shows another — a label and a thing that
    disagree, which is the failure the `safe_distance` binding exists to prevent.
    """
    scene = layout_scene(_zoned("lane", 0.8))
    zone = next(element for element in scene.elements if element.id == "safe")

    assert zone.props["radius"] == zone.radius


def test_a_stage_sized_radius_is_left_alone() -> None:
    """The clamp is a backstop, not a policy: a legal value survives untouched."""
    scene = layout_scene(_zoned("lane", 150))
    zone = next(element for element in scene.elements if element.id == "safe")

    assert zone.radius == 150
    assert zone.props["radius"] == 150


def test_the_lane_and_the_gate_read_the_same_threshold() -> None:
    """Obstacle offset and danger threshold come from one number.

    `_lane_boxes` places obstacles at half the threshold off the lane, and
    `proximity_gate` fires when the nearest one is inside it — so the two have to
    be the same threshold or the car either reacts to nothing or is never able to
    escape. They used to read different sources (`scene.params` in the placer, the
    zone's own `radius` in the gate), which is why the avoidance run's car drove
    straight through everything.
    """
    zone = next(obj for obj in _zoned("lane", 120).objects if obj.id == "safe")
    scene = layout_scene(_zoned("lane", 120))
    laid_out = {element.id: element for element in scene.elements}
    obstacle = laid_out["obstacle-a"]

    offset = abs(obstacle.y - laid_out["car"].y)
    assert offset == pytest.approx(zone.props["radius"] * layout.LANE_OBSTACLE_OFFSET_RATIO)
    assert offset < zone.props["radius"]


# --------------------------------------------------------------------------
# field: the throw is baked so the player only interpolates
# --------------------------------------------------------------------------


def test_a_field_body_carries_the_path_it_flies() -> None:
    """The ball travels the same curve the trajectory draws.

    One curve, not two: `_flight` calls `_arc_points`, which is what `_to_trace`
    calls. A second implementation in JavaScript — evaluating the parabola per
    frame — is the drift this avoids, and decision D3 puts the arithmetic here.
    """
    scene = layout_scene(
        _field([_ball(heading=45), _object("path", "trajectory", "轨迹", of="ball")])
    )
    ball = next(element for element in scene.elements if element.id == "ball")
    trace = next(element for element in scene.elements if element.id == "path")

    assert len(ball.path) == len(trace.points)
    assert ball.path[0] == ball.path[0]
    assert (ball.path[0].x, ball.path[0].y) == pytest.approx((ball.x, ball.y))
    assert ball.duration > 0


def test_a_lane_body_has_no_flight_path() -> None:
    """`lane` carries a car along a corridor; it does not throw it."""
    scene = layout_scene(_scene([_car(), _object("obstacle-a", "obstacle")]))
    car = next(element for element in scene.elements if element.id == "car")

    assert car.path == []
    assert car.duration == 0.0


def test_a_steeper_throw_takes_longer() -> None:
    """Duration comes from the arc's length, so the picture stays physical.

    A 竖直上抛 and a 平抛 are the same launch at the same speed, so the shorter
    arc is genuinely over sooner. Baking one duration for both would have the
    ball crawl up a short vertical line while it races across a long flat one.
    """
    flat = layout_scene(_field([_ball(heading=5)]))
    steep = layout_scene(_field([_ball(heading=45)]))
    durations = {
        index: next(element for element in scene.elements if element.id == "ball").duration
        for index, scene in (("flat", flat), ("steep", steep))
    }

    assert durations["steep"] > durations["flat"]


# --------------------------------------------------------------------------
# Vectors and traces: the ratio the player needs, baked rather than retyped
# --------------------------------------------------------------------------


def test_a_vector_publishes_the_scale_that_resized_it() -> None:
    """A live `magnitude` can only resize the arrow if the ratio comes with it.

    Layout normalises lengths *within* a scene (`_vector_lengths`), so the player
    cannot multiply by a constant — and restating the normalisation in JavaScript
    would be two implementations of one rule. The baked ratio is what keeps
    `direction: 0 / 45 / 90` swinging the arrow instead of drawing it three times.
    """
    scene = layout_scene(
        _field([_ball(), _object("v", "velocity", "初速度", of="ball", magnitude=5)])
    )
    vector = next(element for element in scene.elements if element.id == "v")

    assert vector.length_scale == pytest.approx(5 * vector.length_scale / 5)
    assert vector.length_scale > 0
    assert math.hypot(vector.dx, vector.dy) == pytest.approx(5 * vector.length_scale)


def test_a_trace_publishes_the_length_that_means_all_of_it() -> None:
    """Same argument as the vector's ratio: the player must not retype the rule."""
    scene = layout_scene(
        _field([_ball(), _object("path", "trajectory", "轨迹", of="ball", length=200)])
    )
    trace = next(element for element in scene.elements if element.id == "path")

    assert trace.length_reference == layout.TRACE_LENGTH_REFERENCE


# --------------------------------------------------------------------------
# thresholds: which zone is whose, read off the relations
# --------------------------------------------------------------------------


def test_the_threshold_map_names_the_zone_a_body_wears() -> None:
    """Derived from the scene graph, not from a prop name.

    `behaviors.js` used to look for `safe_distance` on the *body*. That works for
    the hand-written baseline — it puts the number in `scene.params` — and fails
    silently for every document that roles a zone instead, which is what the
    avoidance run did: the gate `continue`d every frame and the car never went
    dangerous.
    """
    scene = layout_scene(_zoned("lane", 120))

    assert scene.thresholds == {"car": "safe"}


def test_a_coverage_zone_is_not_a_threshold() -> None:
    """Only the `safe_distance` role means "the distance this body reacts at".

    The avoidance document also has 左侧可通行空间 and 右侧可通行空间 circles on the
    same car. Folding those in would make the last one written win, which is the
    same "a role maps to exactly one primitive" hazard one layer up.
    """
    scene = layout_scene(
        _scene(
            [
                _car(),
                _object("obstacle-a", "obstacle"),
                _object("safe", "safe_distance", "安全距离", of="car", radius=120),
                _object("left", "coverage", "左侧空间", of="car", radius=120),
            ]
        )
    )

    assert scene.thresholds == {"car": "safe"}


def test_a_scene_with_no_threshold_zone_has_an_empty_map() -> None:
    """The baseline sets `scene.safe_distance`, so its gate reads the scene tier.

    Both paths survive: the map is empty here and `behaviors.js` falls through to
    `safe_distance` on the body, which walks down to `params`.
    """
    scene = layout_scene(_scene([_car(), _object("obstacle-a", "obstacle")]))

    assert scene.thresholds == {}
