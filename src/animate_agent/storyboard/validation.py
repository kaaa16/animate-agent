"""Deterministic checks for a StoryboardIR.

Two deliberate differences from `knowledge/agent.py`'s validation, both because
this layer's schema is denser:

- it **collects every issue** instead of raising on the first one, so a retry
  costs one round trip instead of one per mistake;
- it resolves names against `rendering/registry.py`, so an unregistered role or
  an inert control is a hard failure. Nothing here degrades silently — a scene
  that cannot be drawn correctly must fail to generate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from animate_agent.documents.models import DocumentIR
from animate_agent.knowledge.models import LessonIR
from animate_agent.rendering.registry import (
    BUTTON_ACTIONS,
    CONSUMABLE_PROPS,
    KNOWN_OBJECT_PROPS,
    PENDING_ALTERNATIVES,
    PENDING_ROLES,
    PRESET_BY_NAME,
    PRIMITIVE_BY_NAME,
    ROLE_TO_PRIMITIVE,
    SPEED_PRESETS,
    T2_GLYPH_NAMES,
    TONE_NAMES,
    stage_range,
)
from animate_agent.storyboard.models import StoryboardIR, StoryboardScene

#: Whitespace and punctuation are stripped before comparing key points, because
#: the model is copying prose out of a lesson and cosmetic drift is not a defect.
_PUNCT = re.compile(r"[\s，。、；：！？,.;:!?\"'“”‘’（）()【】\[\]—\-…·]+")


@dataclass(frozen=True, slots=True)
class StoryboardLimits:
    """Thresholds the validator enforces, sourced from `agent:` in the YAML."""

    min_steps: int = 3
    max_steps: int = 7
    require_visual_objects: bool = True
    require_interactive_demo: bool = True
    allowed_renderers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One problem, addressed precisely enough that the model can act on it."""

    code: str
    where: str
    detail: str


def _normalize(text: str) -> str:
    return _PUNCT.sub("", text).casefold()


def _document_ref_ids(document: DocumentIR) -> frozenset[str]:
    ids: set[str] = set()
    for section in document.sections:
        ids.add(section.id)
        for block in section.blocks:
            ids.add(block.id)
    return frozenset(ids)


def validate_storyboard(
    storyboard: StoryboardIR,
    *,
    lesson: LessonIR | None = None,
    document: DocumentIR | None = None,
    limits: StoryboardLimits,
) -> list[ValidationIssue]:
    """Return every issue found, most structural first. Empty means valid.

    `lesson` and `document` are optional so a hand-authored sample — one with no
    course behind it — is still checkable on its own structure. When either is
    absent its checks are **skipped, not passed silently**: coverage against the
    course, and whether a `source_refs` id exists at all. The generation path
    always supplies both, so nothing that matters goes unchecked there.
    """
    issues: list[ValidationIssue] = []
    ref_ids = _document_ref_ids(document) if document is not None else None

    if lesson is not None and len(storyboard.scenes) > len(lesson.scenes):
        issues.append(
            ValidationIssue(
                "scene_count_exceeded",
                "scenes",
                f"storyboard 有 {len(storyboard.scenes)} 个场景，"
                f"多于 LessonIR 的 {len(lesson.scenes)} 个；"
                "场景数只能通过覆盖关系继承，不能自行增加",
            )
        )

    seen_scene_ids: set[str] = set()
    covered_lesson_ids: set[str] = set()

    for scene_index, scene in enumerate(storyboard.scenes):
        where = f"scenes[{scene_index}]"
        if scene.id in seen_scene_ids:
            issues.append(
                ValidationIssue("duplicate_id", f"{where}.id", f"场景 id `{scene.id}` 重复")
            )
        seen_scene_ids.add(scene.id)

        if lesson is not None:
            _check_lesson_links(scene, where, lesson, covered_lesson_ids, issues)
            _check_lesson_coverage(scene, where, lesson, ref_ids, issues)
        _check_preset(scene, where, issues)
        _check_renderer_hint(scene, where, limits, issues)
        _check_ids(scene, where, issues)
        _check_roles_and_props(
            scene, where, _acceptable_refs(scene, lesson, ref_ids), issues
        )
        _check_references(scene, where, issues)
        _check_required_relations(scene, where, issues)
        _check_required_props(scene, where, issues)
        _check_steps(scene, where, limits, issues)
        _check_controls(scene, where, issues)
        _check_orphans(scene, where, issues)

    if lesson is not None:
        _check_coverage(storyboard, lesson, covered_lesson_ids, issues)
    _check_demo_presence(storyboard, limits, issues)

    return issues


def _check_lesson_links(
    scene: StoryboardScene,
    where: str,
    lesson: LessonIR,
    covered_lesson_ids: set[str],
    issues: list[ValidationIssue],
) -> None:
    lesson_ids = {s.id for s in lesson.scenes}
    for index, ref in enumerate(scene.lesson_scene_ids):
        if ref not in lesson_ids:
            issues.append(
                ValidationIssue(
                    "lesson_scene_unknown",
                    f"{where}.lesson_scene_ids[{index}]",
                    f"`{ref}` 不在 LessonIR 中；合法值：{'、'.join(sorted(lesson_ids))}",
                )
            )
        else:
            covered_lesson_ids.add(ref)


def _check_coverage(
    storyboard: StoryboardIR,
    lesson: LessonIR,
    covered_lesson_ids: set[str],
    issues: list[ValidationIssue],
) -> None:
    missing = [s.id for s in lesson.scenes if s.id not in covered_lesson_ids]
    if missing:
        issues.append(
            ValidationIssue(
                "lesson_scene_uncovered",
                "scenes",
                f"这些课程场景没有任何 storyboard 场景覆盖：{'、'.join(missing)}",
            )
        )


def _acceptable_refs(
    scene: StoryboardScene,
    lesson: LessonIR | None,
    ref_ids: frozenset[str] | None,
) -> frozenset[str] | None:
    """Refs an object may carry, or None when there is nothing to check against.

    A ref is acceptable if the document contains it **or** the lesson cites it.
    The second clause is what keeps the two ref rules from contradicting each
    other: a lesson can cite a block the document does not have (a stale
    LessonIR, an edited document), and the prompt hands the model exactly those
    refs. Flagging them as invented punishes the model for obeying — and paired
    with the coverage rule it made the requirement unsatisfiable, so every retry
    was wasted. A ref in *neither* place is still flagged: that is the genuinely
    invented citation this check exists to catch.
    """
    if ref_ids is None:
        return None
    allowed = set(ref_ids)
    if lesson is not None:
        covered = set(scene.lesson_scene_ids)
        for lesson_scene in lesson.scenes:
            if lesson_scene.id in covered:
                allowed.update(lesson_scene.source_refs)
    return frozenset(allowed)


def _check_lesson_coverage(
    scene: StoryboardScene,
    where: str,
    lesson: LessonIR,
    ref_ids: frozenset[str] | None,
    issues: list[ValidationIssue],
) -> None:
    """Whatever this scene covers, it must carry the lesson's points and refs.

    Two separate failures, both of the "the lesson cited §3 but the picture never
    shows it" kind: a key point that lands in no beat is a fact the lesson taught
    and the animation silently skipped, and a source ref no object carries is
    the same gap one level down.
    """
    covered_ids = set(scene.lesson_scene_ids)
    covered = [s for s in lesson.scenes if s.id in covered_ids]
    if not covered:
        return  # lesson_scene_unknown already reported

    # Compare normalised, report the original wording — the model needs to see
    # the string it was supposed to copy verbatim.
    points = {_normalize(point): point for s in covered for point in s.key_points}
    present = {_normalize(point) for step in scene.steps for point in step.key_points}
    missing_points = [original for key, original in points.items() if key not in present]
    if missing_points:
        issues.append(
            ValidationIssue(
                "key_point_missing",
                f"{where}.steps",
                f"这些课程要点没有落在任何节拍里（要逐字照抄）：{'、'.join(sorted(missing_points))}",
            )
        )

    # Every ref the lesson cites must be carried, including one the document no
    # longer has: the storyboard mirrors the lesson, and whether the lesson's own
    # citations resolve is the knowledge layer's business. What makes that
    # satisfiable is `_acceptable_refs` — without it this rule and
    # `source_ref_unknown` contradict each other and no retry can ever pass.
    refs = [ref for s in covered for ref in s.source_refs]
    carried = {ref for obj in scene.objects for ref in obj.source_refs}
    missing_refs = sorted({ref for ref in refs if ref not in carried})
    if missing_refs:
        issues.append(
            ValidationIssue(
                "source_ref_gap",
                f"{where}.objects",
                f"课程引用了这些原文出处，但没有任何对象标注它们：{'、'.join(missing_refs)}",
            )
        )


def _check_preset(scene: StoryboardScene, where: str, issues: list[ValidationIssue]) -> None:
    preset = PRESET_BY_NAME.get(scene.scene_type)
    if preset is None:
        issues.append(
            ValidationIssue(
                "unknown_scene_type",
                f"{where}.scene_type",
                f"`{scene.scene_type}` 不是已注册预设；合法值：{'、'.join(sorted(PRESET_BY_NAME))}",
            )
        )
        return

    present_roles = {obj.role for obj in scene.objects}
    missing = [role for role in preset.required_roles if role not in present_roles]
    if missing:
        issues.append(
            ValidationIssue(
                "preset_role_missing",
                f"{where}.objects",
                f"预设 `{preset.name}` 需要角色 {'、'.join(missing)}，但场景里没有；"
                "换个预设，或把这些对象补上",
            )
        )

    body_count = sum(1 for obj in scene.objects if ROLE_TO_PRIMITIVE.get(obj.role) == "body")
    if body_count > preset.max_bodies:
        issues.append(
            ValidationIssue(
                "preset_capacity_exceeded",
                f"{where}.objects",
                f"预设 `{preset.name}` 最多容纳 {preset.max_bodies} 个对象，实际 {body_count} 个",
            )
        )


def _check_renderer_hint(
    scene: StoryboardScene,
    where: str,
    limits: StoryboardLimits,
    issues: list[ValidationIssue],
) -> None:
    if scene.renderer_hint is None or not limits.allowed_renderers:
        return
    if scene.renderer_hint not in limits.allowed_renderers:
        issues.append(
            ValidationIssue(
                "unknown_renderer_hint",
                f"{where}.renderer_hint",
                f"`{scene.renderer_hint}` 不在允许的渲染器里；合法值："
                f"{'、'.join(limits.allowed_renderers)}",
            )
        )


def _check_ids(scene: StoryboardScene, where: str, issues: list[ValidationIssue]) -> None:
    for label, ids in (
        ("objects", [obj.id for obj in scene.objects]),
        ("steps", [step.id for step in scene.steps]),
        ("controls", [control.id for control in scene.controls]),
    ):
        seen: set[str] = set()
        for index, value in enumerate(ids):
            if value in seen:
                issues.append(
                    ValidationIssue(
                        "duplicate_id",
                        f"{where}.{label}[{index}].id",
                        f"`{value}` 在同一场景内重复",
                    )
                )
            seen.add(value)


def _check_prop(
    prop: str,
    where: str,
    role: str,
    issues: list[ValidationIssue],
) -> None:
    primitive_name = ROLE_TO_PRIMITIVE.get(role)
    if primitive_name is None:
        return  # already reported as unknown_role
    primitive = PRIMITIVE_BY_NAME[primitive_name]
    allowed = (*primitive.props, *primitive.relations)
    if prop in allowed:
        return

    # Distinguish "you made this up" from "right name, wrong role" — the second
    # is the common case and the fix is different.
    hint = (
        f"`{prop}` 是别的图元上的属性"
        if prop in KNOWN_OBJECT_PROPS
        else f"`{prop}` 不是已注册的属性"
    )
    issues.append(
        ValidationIssue(
            "unknown_step_prop",
            where,
            f"{hint}；角色 `{role}`（图元 `{primitive_name}`）只能设 "
            f"{'、'.join(allowed) if allowed else '（无属性）'}",
        )
    )


def _check_tone(value: object, where: str, issues: list[ValidationIssue]) -> None:
    """A `tone` the player does not recognise draws as the ordinary colour.

    `toneColor` (`primitives.js`) answers an unknown name from its `default:`
    branch, and the default is the colour a readout draws in when nothing was
    asked of it. So `tone: "warning"` produces a panel that looks entirely
    deliberate, in a palette nobody chose, and the model never finds out that
    the word was `danger`. Nothing above this line can object: it is a string,
    `tone` is a real prop of a real primitive, and a beat may legally change it.

    Exactly the shape of `prop_out_of_range` one function down — legal to every
    layer that could have refused it, wrong only in what gets drawn — and the
    reason it is worth refusing rather than ignoring is the same: the fix is one
    word, and it belongs to the layer that is still allowed to retry.
    """
    if not isinstance(value, str) or value in TONE_NAMES:
        return
    issues.append(
        ValidationIssue(
            "unknown_tone",
            where,
            f"`{value}` 不是已注册的 tone；合法值：{'、'.join(TONE_NAMES)}。"
            "写错不会报错——播放器认不出就按默认色画，颜色和语义对不上，"
            "而画面上看不出这是写错了",
        )
    )


def _check_speed_preset(
    preset: str,
    primitive_name: str | None,
    prop: str,
    where: str,
    issues: list[ValidationIssue],
) -> None:
    """`speed` on a body the preset never moves is a number the player ignores.

    `behaviors.js` used to check only "is this a body with a numeric speed?" and
    carry it along +x at 90 pixels per speed-unit. The recorded avoidance
    document gives its 底盘 `speed: 60` in a `chain` scene, so the first thing
    the lesson showed was a link diagram with the chassis sliding off the right
    edge and wrapping — while the narration underneath said 四个组成部分依次亮相.
    That half is fixed at the source: the player asks the preset now
    (`SPEED_PRESETS`, mirrored in `registry.js`).

    What is left for this check is the other half, and it is a *different*
    complaint from the one the player was making. With the motion gone, a beat
    that sets `speed` on a `chain` node moves nothing at all — and
    `step_state_inert` cannot catch it, because `speed` really is a live prop of
    `body`. A model that writes it expecting a car to pull away gets a still
    frame and no explanation, so the explanation is owed here, one retry before
    the picture is drawn.

    Only bodies are checked. A `traveler` moves along its link in every preset,
    so its speed is read wherever it is written; the preset is what decides
    whether a *body* can move at all.
    """
    if prop != "speed" or primitive_name != "body" or preset in SPEED_PRESETS:
        return
    issues.append(
        ValidationIssue(
            "speed_outside_a_lane",
            where,
            f"预设 `{preset}` 里的 body 不靠 `speed` 移动，只有 "
            f"{'、'.join(sorted(SPEED_PRESETS))} 会——播放器不会因为这个数字"
            f"让对象动起来，写在这一拍里观众看不到任何变化。"
            f"要表达状态变化请改别的属性，或者把这一幕换成有通道的预设",
        )
    )


def _check_stage_range(
    prop: str,
    value: object,
    role: str,
    where: str,
    issues: list[ValidationIssue],
) -> None:
    """A rendered prop must be in the *renderer's* unit, not the document's.

    The failure this catches has no symptom at either layer that could catch it.
    `lidar_emitter.radius: 8` is eight metres, which is what the document says,
    and it is a legal positive float, so pydantic takes it, the validator takes
    it, and layout draws a fan eight pixels across — smaller than a full stop, on
    a 960-pixel stage. `radius: 0.8` is the same story at 0.8 pixels. `speed: 60`
    is the same story again in the other direction: 0.6 米每秒 read as a
    multiplier makes a car that crosses the lane five times a second. Nothing
    errors; the picture is just empty where the lesson says the interesting thing
    is, or a blur where it says the car, and the only way anyone finds out is by
    looking.

    Rejecting rather than silently rescaling, because the two meanings need
    different fixes: a model told "40~420 舞台像素" writes a legible number, while
    a model whose 8-metres value was quietly rewritten to 40 has learned nothing
    and will do it again on the next document.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    primitive_name = ROLE_TO_PRIMITIVE.get(role)
    if primitive_name is None:
        return  # already reported as unknown_role
    bounds = stage_range(primitive_name, prop)
    if bounds is None:
        return
    if bounds.low <= value <= bounds.high:
        return
    issues.append(
        ValidationIssue(
            "prop_out_of_range",
            where,
            f"`{prop}` 的单位是**渲染器的**（{bounds.unit}），不是文档里的物理量；"
            f"{value:g} 会照 {value:g} 画出来。"
            f"`{primitive_name}.{prop}` 要落在 {bounds.low:g}~{bounds.high:g} 之间"
            f"（{bounds.why}）",
        )
    )


def _check_roles_and_props(
    scene: StoryboardScene,
    where: str,
    ref_ids: frozenset[str] | None,
    issues: list[ValidationIssue],
) -> None:
    for obj_index, obj in enumerate(scene.objects):
        obj_where = f"{where}.objects[{obj_index}]"
        if obj.role not in ROLE_TO_PRIMITIVE:
            issues.append(
                ValidationIssue(
                    "unknown_role",
                    f"{obj_where}.role",
                    f"`{obj.role}` 不是已注册角色；合法值：{'、'.join(sorted(ROLE_TO_PRIMITIVE))}",
                )
            )
        elif obj.role in PENDING_ROLES:
            # The role is real, so `unknown_role` would be a lie. It is simply not
            # drawable yet, and this is the last moment anyone can do something
            # about it: after validation the storyboard agent returns, layout runs
            # with nobody left to retry, and the whole run dies having already
            # paid for the model call. Caught here it costs one retry and the
            # model picks the alternative.
            primitive = ROLE_TO_PRIMITIVE[obj.role]
            issues.append(
                ValidationIssue(
                    "primitive_not_drawable",
                    f"{obj_where}.role",
                    f"`{obj.role}`（属于图元 `{primitive}`）在契约里已登记，"
                    f"但布局层还没有实现它——写了会让整份分镜在最后一步作废、"
                    f"一张图都出不来。改用 "
                    f"{PENDING_ALTERNATIVES.get(primitive, '别的图元')}",
                )
            )

        for prop in obj.props:
            _check_prop(prop, f"{obj_where}.props.{prop}", obj.role, issues)
            _check_stage_range(prop, obj.props[prop], obj.role, f"{obj_where}.props.{prop}", issues)
            _check_speed_preset(
                scene.scene_type,
                ROLE_TO_PRIMITIVE.get(obj.role),
                prop,
                f"{obj_where}.props.{prop}",
                issues,
            )

        # Named, not looped: `_check_tone` refuses anything that is not one of
        # five words, and `readout.text` is a string too.
        if "tone" in obj.props:
            _check_tone(obj.props["tone"], f"{obj_where}.props.tone", issues)

        glyph = obj.props.get("glyph")
        if isinstance(glyph, str) and glyph not in T2_GLYPH_NAMES:
            issues.append(
                ValidationIssue(
                    "unknown_glyph",
                    f"{obj_where}.props.glyph",
                    f"`{glyph}` 不是已注册字形；合法值：{'、'.join(sorted(T2_GLYPH_NAMES))}",
                )
            )

        # Without a document there is no id space for a ref to be wrong about.
        # The structural checks above still ran.
        if ref_ids is not None:
            for ref_index, ref in enumerate(obj.source_refs):
                if ref not in ref_ids:
                    issues.append(
                        ValidationIssue(
                            "source_ref_unknown",
                            f"{obj_where}.source_refs[{ref_index}]",
                            f"`{ref}` 在源文档里不存在，像是编出来的引用",
                        )
                    )


def _check_references(scene: StoryboardScene, where: str, issues: list[ValidationIssue]) -> None:
    object_ids = {obj.id for obj in scene.objects}
    primitive_of = {obj.id: ROLE_TO_PRIMITIVE.get(obj.role) for obj in scene.objects}

    for obj_index, obj in enumerate(scene.objects):
        primitive_name = primitive_of[obj.id]
        if primitive_name is None:
            continue
        for relation in PRIMITIVE_BY_NAME[primitive_name].relations:
            value = obj.props.get(relation)
            if isinstance(value, str) and value not in object_ids:
                issues.append(
                    ValidationIssue(
                        "unresolved_ref",
                        f"{where}.objects[{obj_index}].props.{relation}",
                        f"`{value}` 不是本场景里的对象；合法值：{'、'.join(sorted(object_ids))}",
                    )
                )

        # `present` is not `usable`. Layout keys travelers off `_link_points`, so
        # an `along` naming a `trace` resolves as a reference and then raises —
        # after the storyboard agent has returned, so the run dies with nothing
        # fed back and nothing retried. The vocabulary note used to claim trace
        # was allowed, which made this a promise the layout broke.
        if primitive_name == "traveler":
            along = obj.props.get("along")
            path = next((one for one in scene.objects if one.id == along), None)
            if path is not None and primitive_of.get(path.id) != "link":
                issues.append(
                    ValidationIssue(
                        "relation_wrong_kind",
                        f"{where}.objects[{obj_index}].props.along",
                        f"`{along}` 的图元是 `{primitive_of.get(path.id)}`，"
                        f"但 `along` 必须指向一个 `link`——"
                        f"指向别的东西会让布局整单失败、一张图都出不来",
                    )
                )

    for step_index, step in enumerate(scene.steps):
        for ref_index, highlight in enumerate(step.highlights):
            if highlight not in object_ids:
                issues.append(
                    ValidationIssue(
                        "unresolved_ref",
                        f"{where}.steps[{step_index}].highlights[{ref_index}]",
                        f"高亮的 `{highlight}` 不是本场景里的对象；"
                        f"合法值：{'、'.join(sorted(object_ids))}",
                    )
                )
        for target in step.object_states:
            if target not in object_ids:
                issues.append(
                    ValidationIssue(
                        "unresolved_ref",
                        f"{where}.steps[{step_index}].object_states.{target}",
                        f"`{target}` 不是本场景里的对象；合法值：{'、'.join(sorted(object_ids))}",
                    )
                )


def _check_required_relations(
    scene: StoryboardScene, where: str, issues: list[ValidationIssue]
) -> None:
    """Every object must carry the relations its primitive cannot be drawn without.

    The missing half of `_check_references`. That one checks a reference which
    *was* written; nothing checked that a reference which *had to be* written is
    there. So a `dimension` with no `from`/`to` and a `trajectory` with no `of`
    passed every check and then had no geometry at all — `StoryboardIR` carries
    no coordinates (D3), so the layout pass has nothing to fall back on.

    **A validator that only rejects what it can see is blind to an omission**,
    and an omission is exactly how this class of object fails. The failure mode
    is not a wrong picture; it is a picture with a hole in it that the generator
    reported as success.
    """
    for obj_index, obj in enumerate(scene.objects):
        primitive_name = ROLE_TO_PRIMITIVE.get(obj.role)
        if primitive_name is None:
            continue  # unknown_role already reported
        primitive = PRIMITIVE_BY_NAME[primitive_name]
        missing = [
            relation
            for relation in primitive.required_relations
            if obj.props.get(relation) is None
        ]
        if not missing:
            continue
        issues.append(
            ValidationIssue(
                "required_relation_missing",
                f"{where}.objects[{obj_index}].props",
                f"角色 `{obj.role}`（图元 `{primitive_name}`）必须写出关系 "
                f"{'、'.join(f'`{rel}`' for rel in missing)}——"
                "填同一场景内另一个对象的 id。没有它布局层算不出这个对象画在哪，"
                f"画面会缺一块但不会报错。本场景合法 id："
                f"{'、'.join(sorted(obj2.id for obj2 in scene.objects))}",
            )
        )


def _check_required_props(
    scene: StoryboardScene,
    where: str,
    issues: list[ValidationIssue],
) -> None:
    """Every object must carry the props its drawing is meaningless without.

    The next question out from `_check_required_relations`, and a different kind
    of failure. That one catches an object with no geometry, so its symptom is a
    hole in the picture. This one catches an object that is drawn completely and
    teaches nothing: an `axis` with no `range` is a shaft, five evenly spaced
    notches and an arrowhead — a well-formed drawing of something that is not a
    coordinate axis. Three axes across the sample documents' descendants omitted
    it, nothing reported anything, and the only symptom was that a lesson about
    motion along axes had notched arrows on it instead.

    So the missing check here is not "is a thing absent" but "is a thing still
    what it claims to be", which is why it belongs in the validator rather than
    in a comment next to the drawer. `drawAxis` had the right sentence written
    above it the whole time. **A comment is not a gate.**
    """
    for obj_index, obj in enumerate(scene.objects):
        primitive_name = ROLE_TO_PRIMITIVE.get(obj.role)
        if primitive_name is None:
            continue  # unknown_role already reported
        primitive = PRIMITIVE_BY_NAME[primitive_name]
        missing = [
            prop for prop in primitive.required_props if obj.props.get(prop) is None
        ]
        if not missing:
            continue
        issues.append(
            ValidationIssue(
                "required_prop_missing",
                f"{where}.objects[{obj_index}].props",
                f"角色 `{obj.role}`（图元 `{primitive_name}`）必须写出属性 "
                f"{'、'.join(f'`{prop}`' for prop in missing)}——"
                "缺了它画面照样画得出来、也不会报错，只是画出来的不是它该有的样子"
                f"（{primitive.note}）",
            )
        )


def _check_steps(
    scene: StoryboardScene,
    where: str,
    limits: StoryboardLimits,
    issues: list[ValidationIssue],
) -> None:
    count = len(scene.steps)
    if count < limits.min_steps or count > limits.max_steps:
        issues.append(
            ValidationIssue(
                "step_count_invalid",
                f"{where}.steps",
                f"一个场景要有 {limits.min_steps}~{limits.max_steps} 个教学节拍，实际 {count} 个",
            )
        )

    for step_index, step in enumerate(scene.steps):
        step_where = f"{where}.steps[{step_index}]"
        for target, states in step.object_states.items():
            role = next((obj.role for obj in scene.objects if obj.id == target), None)
            if role is None:
                continue  # unresolved_ref already reported
            primitive_name = ROLE_TO_PRIMITIVE.get(role)
            primitive = PRIMITIVE_BY_NAME.get(primitive_name) if primitive_name else None

            for prop in states:
                prop_where = f"{step_where}.object_states.{target}.{prop}"
                if primitive is None:
                    continue  # unknown_role already reported for the object
                if prop not in (*primitive.props, *primitive.relations):
                    _check_prop(prop, prop_where, role, issues)
                    continue

                # The scene-level check has run on `obj.props` since the stage
                # ranges landed; this path was left out, and it is the one a
                # lesson actually takes. 降低速度 is a *beat* — `object_states:
                # {car: {speed: 25}}` — so the single most likely place for a
                # document's own unit to be copied in was the one place nothing
                # looked. Measured on the three recorded documents: every speed
                # they set is a beat state, and every one of them was outside
                # the range.
                _check_stage_range(prop, states[prop], role, prop_where, issues)
                _check_speed_preset(scene.scene_type, primitive_name, prop, prop_where, issues)

                if prop == "tone":
                    _check_tone(states[prop], prop_where, issues)

                # A beat is a visual beat. `StoryboardStep` already refuses a step
                # with neither a highlight nor a state; this refuses the harder
                # case — a state on a prop nothing reads. Measured across the three
                # sample documents before this rule existed: 20 of 29 states were
                # unread, and 6 of 7 scenes came out as stills.
                #
                # The model wrote those beats in good faith. `vector.direction` ran
                # 0/45/90 across the three beats explaining 平抛/斜抛/竖直上抛, and
                # it was a registered, validated prop — it was simply never
                # consumed. Caught here it costs one retry; caught nowhere it costs
                # a picture, and the person who notices is watching the screen.
                if prop not in primitive.live_props:
                    issues.append(
                        ValidationIssue(
                            "step_state_inert",
                            prop_where,
                            f"没有渲染代码读取 `{prop}`——写进这一拍不会让画面改变"
                            f"一点，这一拍等于一张幻灯片。角色 `{role}`"
                            f"（图元 `{primitive_name}`）在节拍里能改的属性："
                            f"{'、'.join(primitive.live_props) or '（无）'}；"
                            f"想整体设定它，写在这个对象的 `props` 里而不是节拍里",
                        )
                    )


def _check_controls(
    scene: StoryboardScene,
    where: str,
    issues: list[ValidationIssue],
) -> None:
    by_id = {obj.id: obj for obj in scene.objects}

    for index, control in enumerate(scene.controls):
        control_where = f"{where}.controls[{index}]"

        if control.type == "slider":
            if (
                control.min is not None
                and control.max is not None
                and control.min >= control.max
            ):
                issues.append(
                    ValidationIssue(
                        "control_range_invalid",
                        f"{control_where}.min",
                        f"min ({control.min}) 必须小于 max ({control.max})",
                    )
                )
            if (
                control.min is not None
                and control.max is not None
                and control.default is not None
                and not (control.min <= control.default <= control.max)
            ):
                issues.append(
                    ValidationIssue(
                        "control_range_invalid",
                        f"{control_where}.default",
                        f"default ({control.default}) 必须落在 [{control.min}, {control.max}] 内",
                    )
                )
            if control.step is not None and control.step <= 0:
                issues.append(
                    ValidationIssue(
                        "control_range_invalid",
                        f"{control_where}.step",
                        f"step ({control.step}) 必须大于 0",
                    )
                )

        if control.type == "button":
            # Buttons act, they do not set a value, so they are checked against
            # the action list instead of resolving a property target.
            if control.action not in BUTTON_ACTIONS:
                issues.append(
                    ValidationIssue(
                        "control_action_unknown",
                        f"{control_where}.action",
                        f"`{control.action}` 不是已注册动作；合法值：{'、'.join(BUTTON_ACTIONS)}",
                    )
                )
            continue

        target, _, prop = control.target_property.partition(".")
        if not prop:
            issues.append(
                ValidationIssue(
                    "control_target_unresolved",
                    f"{control_where}.target_property",
                    f"`{control.target_property}` 不是 `<对象id>.<属性>` 或 `scene.<属性>` 的形式",
                )
            )
            continue

        primitive_name: str | None = None
        if target != "scene":
            obj = by_id.get(target)
            if obj is None:
                issues.append(
                    ValidationIssue(
                        "control_target_unresolved",
                        f"{control_where}.target_property",
                        f"`{target}` 不是本场景里的对象；合法值：{'、'.join(sorted(by_id))}",
                    )
                )
                continue
            primitive_name = ROLE_TO_PRIMITIVE.get(obj.role)
            if primitive_name is not None and prop not in PRIMITIVE_BY_NAME[primitive_name].props:
                issues.append(
                    ValidationIssue(
                        "control_target_unresolved",
                        f"{control_where}.target_property",
                        f"角色 `{obj.role}`（图元 `{primitive_name}`）没有属性 `{prop}`；"
                        f"可绑定：{'、'.join(PRIMITIVE_BY_NAME[primitive_name].props)}",
                    )
                )
                continue

        if prop not in CONSUMABLE_PROPS:
            issues.append(
                ValidationIssue(
                    "control_target_unconsumed",
                    f"{control_where}.target_property",
                    f"没有任何行为读取 `{prop}`，这个控件只会改 UI 数字、不会改画面；"
                    f"可绑定的属性：{'、'.join(sorted(CONSUMABLE_PROPS))}",
                )
            )

        # The slider is the second road to the same number, and it is the one
        # that *looks* right: `car.speed` bound to a track labelled `0~120 cm/s`
        # is a completely sensible control for a document that talks in cm/s.
        # Its `unit` is the document's; the prop's is the renderer's; nothing
        # compared them, and the picture at the top of the track was a car
        # crossing the lane five times a second.
        #
        # Checked against the prop's range rather than the control's, because
        # the track has to lie inside what can be drawn — an out-of-range end is
        # a part of the slider where nothing readable is on screen.
        bounds = stage_range(primitive_name, prop) if primitive_name is not None else None
        if bounds is not None:
            outside = [
                f"{field}={value:g}"
                for field, value in (
                    ("min", control.min),
                    ("max", control.max),
                    ("default", control.default),
                )
                if value is not None and not (bounds.low <= value <= bounds.high)
            ]
            if outside:
                issues.append(
                    ValidationIssue(
                        "control_range_out_of_stage",
                        control_where,
                        f"`{control.target_property}` 用的是渲染器的单位（{bounds.unit}），"
                        f"可画范围 {bounds.low:g}~{bounds.high:g}；"
                        f"这个滑杆的 {'、'.join(outside)} 在范围外——"
                        f"拖到那一段，画面读不出来。"
                        f"滑杆的 `unit` 字段是给人看的，改它不影响画面",
                    )
                )


def _check_orphans(scene: StoryboardScene, where: str, issues: list[ValidationIssue]) -> None:
    referenced: set[str] = set()
    for step in scene.steps:
        referenced.update(step.highlights)
        referenced.update(step.object_states)
    for obj in scene.objects:
        primitive_name = ROLE_TO_PRIMITIVE.get(obj.role)
        if primitive_name is None:
            continue
        relations = PRIMITIVE_BY_NAME[primitive_name].relations
        for relation in relations:
            value = obj.props.get(relation)
            if isinstance(value, str):
                referenced.add(value)
        # An object that is *attached* to another one (a lidar on a car, a link
        # between two nodes) is anchored on screen by that relation, so it is not
        # an orphan even if no beat highlights it by name.
        if any(obj.props.get(rel) is not None for rel in relations):
            referenced.add(obj.id)

    for index, obj in enumerate(scene.objects):
        if obj.id not in referenced:
            issues.append(
                ValidationIssue(
                    "orphan_object",
                    f"{where}.objects[{index}]",
                    f"对象 `{obj.id}` 没有被任何节拍高亮，也没有被任何关系引用——画了但没教",
                )
            )


def _check_demo_presence(
    storyboard: StoryboardIR,
    limits: StoryboardLimits,
    issues: list[ValidationIssue],
) -> None:
    if limits.require_visual_objects and not any(scene.objects for scene in storyboard.scenes):
        issues.append(
            ValidationIssue("no_visual_objects", "scenes", "整个 storyboard 没有任何可视化对象")
        )
    if limits.require_interactive_demo and not any(
        scene.controls for scene in storyboard.scenes
    ):
        issues.append(
            ValidationIssue(
                "no_interactive_demo",
                "scenes",
                "整个 storyboard 没有任何可交互控件；至少一个场景要提供一个核心 demo",
            )
        )


def format_issues(issues: list[ValidationIssue]) -> str:
    """Render issues as feedback the model can act on, one line each."""
    if not issues:
        return ""
    lines = [f"共 {len(issues)} 处问题："]
    lines.extend(
        f"{index}. [{issue.code}] {issue.where}：{issue.detail}"
        for index, issue in enumerate(issues, 1)
    )
    return "\n".join(lines)
