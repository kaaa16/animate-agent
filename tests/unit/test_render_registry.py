"""Tests for the rendering vocabulary and its contract with the rest of the project."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import animate_agent.animation.elements as elements_module
from animate_agent.animation.templates import (
    build_robot_obstacle_avoidance_scene,
    build_ros_pub_sub_scene,
)
from animate_agent.rendering.legacy import LEGACY_TYPE_TO_ROLE
from animate_agent.rendering.registry import (
    BEHAVIORS,
    BUTTON_ACTIONS,
    CONSUMABLE_PROPS,
    PENDING_ALTERNATIVES,
    PENDING_PRIMITIVES,
    PRESET_BY_NAME,
    PRIMITIVE_BY_NAME,
    REQUIRED_RELATIONS,
    ROLE_TO_PRIMITIVE,
    STAGE_RANGES,
    T1_PRIMITIVES,
    T2_GLYPHS,
    Glyph,
    GlyphPart,
    render_vocabulary,
)
from animate_agent.storyboard.models import StoryboardStep


def _legacy_element_types() -> set[str]:
    """Every `type` literal the legacy element dataclasses declare."""
    types: set[str] = set()
    for name in dir(elements_module):
        fields: Any = getattr(getattr(elements_module, name), "__dataclass_fields__", None)
        if not isinstance(fields, dict) or "type" not in fields:
            continue
        default = fields["type"].default
        if isinstance(default, str):
            types.add(default)
    return types


def test_every_role_maps_to_exactly_one_primitive() -> None:
    seen: dict[str, str] = {}
    for primitive in T1_PRIMITIVES:
        for role in primitive.roles:
            assert role not in seen, f"角色 `{role}` 同时属于 {seen[role]} 和 {primitive.name}"
            seen[role] = primitive.name

    assert seen == ROLE_TO_PRIMITIVE


def test_legacy_mapping_covers_every_element_type() -> None:
    # Guards the baseline contract: a new legacy element type without a role
    # would silently become undrawable.
    assert set(LEGACY_TYPE_TO_ROLE) == _legacy_element_types()


def test_every_mapped_role_is_registered() -> None:
    unknown = sorted(set(LEGACY_TYPE_TO_ROLE.values()) - set(ROLE_TO_PRIMITIVE))

    assert unknown == []


def _baseline_targets() -> list[tuple[str, str, str]]:
    """(control type, target_property, action) for every baseline control."""
    scenes = [build_robot_obstacle_avoidance_scene(), build_ros_pub_sub_scene()]
    return [
        (control.type, control.target_property, getattr(control, "action", ""))
        for scene in scenes
        for control in scene.controls
    ]


def _role_of(scene: Any, element_id: str) -> str | None:
    for element in scene.elements:
        if element.id == element_id:
            return LEGACY_TYPE_TO_ROLE.get(element.type)
    return None


def test_baseline_controls_resolve_against_the_registry() -> None:
    """The three avoidance sliders and the two buttons must all be expressible.

    This is the machine-checkable form of the quality baseline: if the baseline
    can turn a knob the vocabulary cannot name, the new pipeline cannot reach
    the old animation's quality no matter how good the renderer is.
    """
    scene = build_robot_obstacle_avoidance_scene()
    for control in scene.controls:
        if control.type == "button":
            assert control.action in BUTTON_ACTIONS, control.action
            continue

        target, _, prop = control.target_property.partition(".")
        assert prop, control.target_property
        assert prop in CONSUMABLE_PROPS, f"`{prop}` 没有行为读取，控件会失效"

        if target == "scene":
            continue
        role = _role_of(scene, target)
        assert role is not None, target
        primitive = PRIMITIVE_BY_NAME[ROLE_TO_PRIMITIVE[role]]
        assert prop in primitive.props, f"{role} 没有属性 {prop}"


def test_baseline_ros_button_resolves() -> None:
    scene = build_ros_pub_sub_scene()
    [control] = scene.controls

    assert control.action in BUTTON_ACTIONS


def test_the_quality_baseline_satisfies_the_step_schema() -> None:
    """The baseline must pass the schema it is the yardstick for.

    It did not. The `description` floor was 20 characters and two of these four
    captions are shorter than that — so the pipeline was rejecting writing of
    exactly the quality it had been told to match. Running the baseline's own
    captions through `StoryboardStep` keeps that floor from drifting back above
    them.
    """
    scene = build_robot_obstacle_avoidance_scene()

    for step in scene.timeline:
        StoryboardStep(
            id="step-under-test",
            title=step.title,
            description=step.narration,
            highlights=["car"],
            key_points=["关键词"],
        )


def test_lidar_radius_and_safe_distance_stay_distinct() -> None:
    # Sensor reach and the decision threshold are different quantities; fusing
    # them would make "调雷达半径" and "调安全距离" the same action.
    assert "radius" in PRIMITIVE_BY_NAME["emitter"].props
    assert "safe_distance" in CONSUMABLE_PROPS
    assert "safe_distance" not in PRIMITIVE_BY_NAME["emitter"].props


def test_vocabulary_mentions_every_drawable_name() -> None:
    """Everything the model may write is named in the prompt it is given.

    Scoped to `drawable` because the rest of the registry is deliberately *not*
    offered — see `test_vocabulary_does_not_offer_a_primitive_that_cannot_be_drawn`.
    """
    vocabulary = render_vocabulary(("canvas_2d",))

    for primitive in T1_PRIMITIVES:
        if not primitive.drawable:
            continue
        assert f"`{primitive.name}`" in vocabulary
        for role in primitive.roles:
            assert f"`{role}`" in vocabulary
    for glyph in T2_GLYPHS:
        assert f"`{glyph.name}`" in vocabulary
    for behavior in BEHAVIORS:
        assert f"`{behavior.name}`" in vocabulary
    for action in BUTTON_ACTIONS:
        assert f"`{action}`" in vocabulary


def _offered_role_lines(vocabulary: str) -> list[str]:
    """The `role` list a model reads, before any of it is filtered.

    Structural on purpose. The previous version of the test above asked only
    whether the name appeared *somewhere* in the text, and `region`/`wave` went
    on appearing — in the list of names not to use. A name is offered by a line
    that offers it, not by the file containing the string.
    """
    return [
        line for line in vocabulary.splitlines() if line.startswith("- ") and "→ 角色只能是" in line
    ]


def test_vocabulary_does_not_offer_a_primitive_that_cannot_be_drawn() -> None:
    """The role list is a promise, so it may only hold what the layout can place.

    `region` and `wave` are registered, are in `RenderSpec`, and had no layout.
    Offering them is not a harmless overshoot: the failure they buy is terminal,
    because layout runs after the storyboard agent has returned — nothing is fed
    back to the model and nothing retries, so a document that reaches for a
    waveform produces no picture at all.

    Both halves are asserted. Excluded from the offer list is only half a fix;
    if the names were simply dropped the model would write `waveform`, be told
    it is not a registered role, and spend a retry guessing. It has to be told
    the name is real, unavailable, and what to write instead.
    """
    vocabulary = render_vocabulary(("canvas_2d",))
    offered = _offered_role_lines(vocabulary)

    for primitive in T1_PRIMITIVES:
        line = next((one for one in offered if one.startswith(f"- {primitive.name}（")), None)
        assert (line is not None) == primitive.drawable, (
            f"`{primitive.name}` 的 drawable={primitive.drawable}，"
            f"却{'仍然在' if line else '不在'}角色清单里"
        )

    for primitive in T1_PRIMITIVES:
        if primitive.drawable:
            continue
        forbidden = [
            line for line in vocabulary.splitlines() if line.startswith(f"- ~~{primitive.name}~~")
        ]
        assert len(forbidden) == 1, f"`{primitive.name}` 没有出现在「不要用」那一段里"
        for role in primitive.roles:
            assert f"`{role}`" in forbidden[0], role
        assert PENDING_ALTERNATIVES[primitive.name] in forbidden[0], primitive.name


def test_every_pending_primitive_says_what_to_use_instead() -> None:
    """An unavailable name with no alternative costs a retry and teaches nothing."""
    assert set(PENDING_ALTERNATIVES) == set(PENDING_PRIMITIVES)


# --------------------------------------------------------------------------
# live_props — the layer below `drawable`: declared is not the same as consumed
# --------------------------------------------------------------------------


def test_a_live_prop_is_a_declared_prop() -> None:
    """A beat can only move something the object is allowed to carry."""
    for primitive in T1_PRIMITIVES:
        assert set(primitive.live_props) <= set(primitive.props), primitive.name


def test_a_pending_primitive_promises_nothing() -> None:
    """Nothing draws it, so nothing about it can be live."""
    for primitive in T1_PRIMITIVES:
        if not primitive.drawable:
            assert primitive.live_props == (), primitive.name


def test_every_stage_range_names_a_real_prop() -> None:
    """A range for a prop that does not exist would be silently unreachable."""
    for primitive_name, prop in STAGE_RANGES:
        primitive = PRIMITIVE_BY_NAME[primitive_name]
        assert prop in primitive.props, f"{primitive_name}.{prop}"


def test_a_stage_ranged_prop_is_live() -> None:
    """Stating a unit for a prop no beat may move would be a unit with no reader.

    All three current entries are distances a beat can change, and that is not a
    coincidence: a distance is exactly the kind of thing a lesson walks from one
    value to another (安全距离 0.8m → 1.2m), so having a unit and being live go
    together. If a ranged prop turns out not to be live, this test is the place
    that says so out loud.
    """
    for primitive_name, prop in STAGE_RANGES:
        assert prop in PRIMITIVE_BY_NAME[primitive_name].live_props, primitive_name


def _prop_line(vocabulary: str, primitive_name: str) -> str:
    """The one prop line for `primitive_name`, before any of it is interpreted."""
    return next(
        line for line in vocabulary.splitlines() if line.startswith(f"- `{primitive_name}`：")
    )


def test_the_vocabulary_separates_what_a_beat_can_change() -> None:
    """The two groups have to be visibly different, because the rule differs.

    Setting `glyph` once on the object is correct and is what both hand-written
    samples do; setting it *per beat* is the defect `step_state_inert` rejects.
    A single undifferentiated prop list is what let the model write seven dead
    beats in the projectile document without ever being told there was a rule.
    """
    vocabulary = render_vocabulary()

    for primitive in T1_PRIMITIVES:
        if not primitive.drawable:
            continue
        line = _prop_line(vocabulary, primitive.name)
        live_part, _, static_part = line.partition("只能整体设置一次")

        for prop in primitive.live_props:
            assert f"`{prop}`" in live_part, f"{primitive.name}.{prop} 不在「可被节拍改变」里"
        for prop in primitive.props:
            if prop in primitive.live_props:
                continue
            assert f"`{prop}`" in static_part, f"{primitive.name}.{prop} 不在「只能整体设置一次」里"


def test_the_vocabulary_states_the_unit_of_a_distance() -> None:
    """The model cannot write pixels if it was never told the unit is pixels.

    Every ranged prop says its bounds on the same line as the prop, because a
    unit stated in a separate section is a unit the model reads after deciding
    what to write.
    """
    vocabulary = render_vocabulary()

    for (primitive_name, prop), (low, high) in STAGE_RANGES.items():
        line = _prop_line(vocabulary, primitive_name)
        assert "舞台像素" in line, primitive_name
        assert f"{low:g}~{high:g}" in line, f"{primitive_name}.{prop}"


def test_vocabulary_lists_consumable_props_as_control_targets() -> None:
    vocabulary = render_vocabulary()

    for prop in CONSUMABLE_PROPS:
        assert f"`{prop}`" in vocabulary


# --------------------------------------------------------------------------
# Required relations — the line between "declared" and "cannot be drawn without"
# --------------------------------------------------------------------------


def test_required_relations_are_declared_relations() -> None:
    """A primitive cannot require a relation it does not declare."""
    for primitive in T1_PRIMITIVES:
        assert set(primitive.required_relations) <= set(primitive.relations), primitive.name


def test_the_optional_relations_stay_optional() -> None:
    """Guards the other direction, which is the one that costs retries.

    `component_of` only applies to a component of a decomposition, `between` only
    to a measured angle. Requiring them would reject the plain case — the whole
    velocity vector, the un-measured angle — which is what most scenes want.
    """
    assert REQUIRED_RELATIONS["dimension"] == ("from", "to")
    assert REQUIRED_RELATIONS["vector"] == ("of",)
    assert "component_of" not in REQUIRED_RELATIONS["vector"]
    assert "between" not in REQUIRED_RELATIONS["angle"]


def test_the_vocabulary_marks_required_relations_as_mandatory() -> None:
    """A model can only obey a rule it was told about.

    This is the same failure D5 records, one layer down: the constraint was real
    and lived only in the code, so every retry spent on it was spent on a rule
    the model had never been given.
    """
    vocabulary = render_vocabulary()
    # A pending primitive has no prop row to mark — it is not offered at all, so
    # requiring its relations to be advertised would be requiring a promise the
    # prompt deliberately withholds. `region.bounded_by` is the current case.
    offered = {
        name: required
        for name, required in REQUIRED_RELATIONS.items()
        if name not in PENDING_PRIMITIVES
    }

    for name, required in offered.items():
        line = next(line for line in vocabulary.splitlines() if line.startswith(f"- `{name}`："))
        assert "必须填写" in line, name
        for relation in required:
            assert f"`{relation}`" in line, f"{name}.{relation}"


def test_the_fallback_preset_is_offered_last() -> None:
    """Order carries meaning here, so it is worth asserting.

    `generic` used to be described as "永远合法的兜底预设", and a real run picked
    it for 3 of the 5 scenes that `field` could serve. The model was optimising
    for "don't get rejected", and the wording made the fallback look like the
    safe choice. Offering the specific presets first is the structural half of
    that fix; the note's wording is the other half and is not assertable.
    """
    offered = [
        line.split("`")[1]
        for line in render_vocabulary().splitlines()
        if line.startswith("- `") and line.split("`")[1] in PRESET_BY_NAME
    ]

    assert offered[-1] == "generic"
    assert sorted(offered) == sorted(PRESET_BY_NAME)


# --------------------------------------------------------------------------
# Glyph schema — the locked contract from docs/storyboard-milestone.md (D1/D2)
# --------------------------------------------------------------------------


def _glyph(**part_overrides: Any) -> dict[str, Any]:
    part = {"d": "M0 0L1 1", "mode": "stroke", **part_overrides}
    return {
        "name": "car",
        "view_box": (0, 0, 24, 24),
        "domain": "robotics",
        "source": "tabler:car",
        "parts": {"body": part},
    }


def test_glyph_part_requires_an_explicit_mode() -> None:
    # Dropping `mode` is the silent failure that draws a stroke icon as a solid
    # black blob, so the schema must not let it default.
    with pytest.raises(ValidationError):
        GlyphPart.model_validate({"d": "M0 0L1 1"})


def test_glyph_part_defaults_to_nonzero_fill_rule() -> None:
    part = GlyphPart.model_validate({"d": "M0 0L1 1", "mode": "fill"})

    assert part.fill_rule == "nonzero"


def test_glyph_part_rejects_an_unknown_fill_rule() -> None:
    with pytest.raises(ValidationError):
        GlyphPart.model_validate({"d": "M0 0L1 1", "mode": "fill", "fill_rule": "banana"})


def test_glyph_part_carries_no_colour() -> None:
    # Colour belongs to the theme layer: `body.danger` flips at playback time
    # and a colour frozen into the asset cannot follow it (decision D1).
    with pytest.raises(ValidationError):
        GlyphPart.model_validate({"d": "M0 0L1 1", "mode": "stroke", "stroke": "#ff0000"})


def test_glyph_parts_are_flat_not_nested() -> None:
    nested = _glyph(parts={"body": {"d": "M0 0L1 1", "mode": "stroke"}})

    with pytest.raises(ValidationError):
        Glyph.model_validate(nested)


def test_glyph_requires_at_least_one_part() -> None:
    with pytest.raises(ValidationError):
        Glyph.model_validate({**_glyph(), "parts": {}})


def test_glyph_accepts_a_well_formed_asset() -> None:
    glyph = Glyph.model_validate(_glyph())

    assert glyph.parts["body"].mode == "stroke"
    assert glyph.view_box == (0, 0, 24, 24)


def test_spin_parts_may_declare_an_anchor() -> None:
    glyph = Glyph.model_validate(_glyph(anchor=(16, 18), spin=True))

    assert glyph.parts["body"].spin


# --------------------------------------------------------------------------
# registry.py <-> the player: two vocabularies that must not drift apart
# --------------------------------------------------------------------------

PLAYER_DIR = Path(__file__).resolve().parents[2] / "frontend" / "player"


def _player_registry_source() -> str:
    return (PLAYER_DIR / "registry.js").read_text(encoding="utf-8")


def _js_lists() -> tuple[set[str], set[str]]:
    """(kinds with a drawer, kinds declared but not yet drawn) as the JS states them.

    Parsed rather than imported: pytest must not need Node to run, and the thing
    worth checking is that the file *says* the right names. `tools/player_smoke.mjs`
    is the check that they actually work.
    """
    source = _player_registry_source()
    drawers = re.search(r"const DRAWERS = \{(.*?)\};", source, re.S)
    pending = re.search(r"export const PENDING_KINDS = \[(.*?)\];", source, re.S)
    assert drawers is not None, "registry.js 里找不到 DRAWERS 表"
    assert pending is not None, "registry.js 里找不到 PENDING_KINDS"
    return (
        set(re.findall(r"^\s*([a-z_]+):", drawers.group(1), re.M)),
        set(re.findall(r'"([a-z_]+)"', pending.group(1))),
    )


def _js_live_props() -> dict[str, list[str]]:
    """`registry.js`'s `LIVE_PROPS`, as the file states it. Parsed, not imported."""
    source = _player_registry_source()
    block = re.search(r"export const LIVE_PROPS = \{(.*?)\n\};", source, re.S)
    assert block is not None, "registry.js 里找不到 LIVE_PROPS"
    return {
        name: re.findall(r'"([a-z_]+)"', props)
        for name, props in re.findall(r"^\s*([a-z_]+): \[(.*?)\],", block.group(1), re.M | re.S)
    }


def test_the_player_reads_the_same_props_the_validator_allows() -> None:
    """`live_props` is a claim about the code that draws — this checks the code.

    The validator rejects a beat that sets a prop outside `Primitive.live_props`,
    and the prompt offers that list to the model as "可被节拍改变". Both of those
    are statements about `primitives.js`, made from Python. If a name is on the
    list that no drawer reads, the prompt is making a promise the player breaks:
    the model writes a beat, validation waves it through, and the picture is
    identical before and after.

    That is not hypothetical. `vector.direction` was registered, validated and
    written as `0 / 45 / 90` across three beats of the projectile document, and
    `drawVector` read `element.dx` — one arrow, three times. Twenty of the 29
    beats in the three sample documents were like this, and 6 of the 7 scenes
    came out as stills.

    Parsed rather than imported because pytest must not need Node, and because
    what is worth checking is what the two files *say*. `tools/player_smoke.mjs`
    is the check that the drawers actually work.
    """
    declared = {
        primitive.name: list(primitive.live_props)
        for primitive in T1_PRIMITIVES
        if primitive.live_props
    }

    assert _js_live_props() == declared


def test_the_player_accounts_for_every_registered_primitive() -> None:
    """A primitive the player has never heard of draws nothing, and says nothing.

    Between the two lists, every T1 name must be accounted for: either there is a
    drawer, or the player knows it is missing one. The failure this prevents is a
    primitive added to `registry.py` and forgotten here — which produces a spec
    that validates, ships, and renders an invisible object.
    """
    drawable, pending = _js_lists()

    assert drawable & pending == set(), "同一个图元既实现了又标成待实现"
    assert drawable | pending == {primitive.name for primitive in T1_PRIMITIVES}


def test_every_named_drawer_exists_in_primitives_js() -> None:
    """The table and the implementations must agree on the function names."""
    primitives = (PLAYER_DIR / "primitives.js").read_text(encoding="utf-8")

    for drawer in re.findall(r"^\s*[a-z_]+: (draw[A-Za-z]+),", _player_registry_source(), re.M):
        assert f"export function {drawer}(" in primitives, f"primitives.js 里没有 {drawer}"


def _player_sources() -> dict[str, str]:
    """Player JS with comments stripped.

    Stripping matters: the rule below is worth *documenting* in the code, and a
    first version of this test failed on the comment explaining that the player
    must not use that name. A check that forbids mentioning a thing forbids
    explaining why it is forbidden.

    The cost is that `//` inside a string literal would be stripped too. Nothing
    in the player holds such a string, and a slipped-through one would be caught
    by reading the file, which a two-line grep cannot replace.
    """
    sources = {}
    for path in sorted(PLAYER_DIR.glob("*.js")):
        source = re.sub(r"/\*.*?\*/", "", path.read_text(encoding="utf-8"), flags=re.S)
        sources[path.name] = re.sub(r"//[^\n]*", "", source)
    assert sources, "没找到播放器源码"
    return sources


def test_the_player_never_executes_spec_text() -> None:
    """Standing correction: the frontend consumes a spec, it never runs one.

    The baseline demo builds markup from storyboard strings
    (`frontend/demo/app.js:210`). That is safe there only because those strings
    are module-level literals; here they are model-authored, so the player must
    not adopt the pattern.
    """
    for name, source in _player_sources().items():
        for forbidden in ("eval(", "new Function(", "innerHTML", "document.write"):
            assert forbidden not in source, f"{name} 里出现了 {forbidden}"


def test_the_player_fetches_the_spec_exactly_once() -> None:
    """"Zero LLM calls at playback" also means no network at playback.

    One `fetch` for the spec file, and nothing else — a second call would be a
    per-frame request, which is the shape a hidden model call would take.
    """
    total = sum(source.count("fetch(") for source in _player_sources().values())

    assert total == 1, f"播放器源码里出现了 {total} 处 fetch("
