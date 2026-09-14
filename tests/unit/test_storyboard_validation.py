"""Table-driven tests for the storyboard validation pass.

One known-good fixture plus one mutation per issue code. The fixture must
validate with zero issues — otherwise a red mutation test could mean the fixture
is broken rather than the check firing.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from animate_agent.documents.models import DocumentBlock, DocumentIR, DocumentSource, Section
from animate_agent.knowledge.models import LessonIR, LessonScene
from animate_agent.storyboard.models import StoryboardIR
from animate_agent.storyboard.validation import (
    StoryboardLimits,
    format_issues,
    validate_storyboard,
)

LIMITS = StoryboardLimits(
    min_steps=3,
    max_steps=7,
    require_visual_objects=True,
    require_interactive_demo=True,
    allowed_renderers=("canvas_2d", "svg_2d", "three_3d"),
)

NARRATION = (
    "机器人先看最近障碍物的距离。如果这个距离大于安全距离，机器人就保持当前速度继续直行；"
    "一旦距离小于安全距离，机器人会先降低速度，再比较左侧和右侧的可通行空间，"
    "选择更空旷的一侧转向绕行，从而安全通过。"
)


def _document() -> DocumentIR:
    return DocumentIR(
        document_id="doc1",
        title="移动机器人避障手册节选",
        source=DocumentSource(type="file"),
        sections=[
            Section(
                id="section-1",
                title="避障决策",
                level=1,
                blocks=[
                    DocumentBlock(id="section-1-block-1", type="paragraph", text="决策逻辑"),
                ],
            )
        ],
    )


def _lesson(*, extra_scene: bool = False) -> LessonIR:
    scenes = [
        LessonScene(
            id="scene-1",
            title="避障决策逻辑",
            objective="理解何时直行、何时避障",
            narration=NARRATION,
            key_points=["安全距离", "转向"],
            source_refs=["section-1-block-1"],
        )
    ]
    if extra_scene:
        scenes.append(
            LessonScene(
                id="scene-2",
                title="关键参数",
                objective="掌握雷达半径与速度",
                narration=NARRATION,
                key_points=["雷达半径", "控制循环"],
                source_refs=["section-1-block-1"],
            )
        )
    return LessonIR(
        lesson_id="lesson-doc1",
        document_id="doc1",
        title="移动机器人避障",
        subject="机器人",
        summary="基于安全距离的避障决策。",
        learning_objectives=["理解避障决策", "掌握关键参数"],
        scenes=scenes,
    )


def _storyboard_dict() -> dict[str, Any]:
    return {
        "storyboard_id": "storyboard-lesson-doc1",
        "lesson_id": "lesson-doc1",
        "document_id": "doc1",
        "title": "移动机器人避障",
        "subject": "机器人",
        "eyebrow": "避障 · 决策逻辑",
        "scenes": [
            {
                "id": "shot-1",
                "scene_type": "lane",
                "teaching_goal": "看懂距离到转向的决策链",
                "lesson_scene_ids": ["scene-1"],
                "objects": [
                    {
                        "id": "car",
                        "role": "vehicle",
                        "label": "小车",
                        "props": {"speed": 60, "heading": 0, "glyph": "car"},
                        "source_refs": ["section-1-block-1"],
                    },
                    {"id": "obstacle", "role": "obstacle", "label": "障碍物", "props": {}},
                    {
                        "id": "lidar",
                        "role": "sensor",
                        "label": "激光雷达",
                        "props": {"radius": 150, "of": "car"},
                    },
                    {
                        "id": "safe",
                        "role": "safe_distance",
                        "label": "安全距离",
                        "props": {"radius": 76, "of": "car"},
                    },
                ],
                "steps": [
                    {
                        "id": "step-1",
                        "title": "雷达扫描",
                        "description": "雷达向前方扇形区域发射多条测距射线，返回每个方向的距离",
                        "highlights": ["lidar"],
                        "key_points": ["安全距离"],
                    },
                    {
                        "id": "step-2",
                        "title": "阈值判断",
                        "description": "最近障碍物的距离低于安全距离，控制器进入避障状态",
                        "highlights": ["car"],
                        "object_states": {"car": {"danger": True}},
                        "key_points": ["转向"],
                    },
                    {
                        "id": "step-3",
                        "title": "转向绕行",
                        "description": "比较左侧与右侧的空旷程度，选择更安全的一侧转向绕行",
                        "highlights": ["obstacle"],
                        "key_points": ["转向"],
                    },
                ],
                "controls": [
                    {
                        "id": "speed",
                        "type": "slider",
                        "label": "车速",
                        "target_property": "car.speed",
                        "min": 0,
                        "max": 120,
                        "default": 60,
                        "step": 5,
                        "unit": "cm/s",
                    },
                    {
                        "id": "reset",
                        "type": "button",
                        "label": "重置",
                        "target_property": "scene",
                        "action": "reset_scene",
                    },
                ],
                "params": {},
                "renderer_hint": None,
            }
        ],
    }


def _scene(data: dict[str, Any]) -> dict[str, Any]:
    return data["scenes"][0]


def _codes(data: dict[str, Any], *, lesson: LessonIR | None = None) -> set[str]:
    storyboard = StoryboardIR.model_validate(data)
    issues = validate_storyboard(
        storyboard,
        lesson=lesson or _lesson(),
        document=_document(),
        limits=LIMITS,
    )
    return {issue.code for issue in issues}


def test_baseline_fixture_is_valid() -> None:
    # If this goes red, every mutation test below is meaningless.
    assert _codes(_storyboard_dict()) == set()


def test_baseline_fixture_reports_no_text() -> None:
    storyboard = StoryboardIR.model_validate(_storyboard_dict())
    issues = validate_storyboard(
        storyboard, lesson=_lesson(), document=_document(), limits=LIMITS
    )

    assert format_issues(issues) == ""


def _mutation(mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    data = copy.deepcopy(_storyboard_dict())
    mutate(data)
    return data


def test_unknown_scene_type() -> None:
    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["scene_type"] = "banana"

    assert "unknown_scene_type" in _codes(_mutation(mutate))


def test_primitive_not_drawable() -> None:
    """A registered role whose layout has not landed must be rejected *here*.

    `region` and `wave` are in the vocabulary and in `RenderSpec`, and nothing
    places them. The failure they used to buy was terminal: layout runs after the
    storyboard agent has returned, so there is no retry left and nothing is fed
    back — a document that reached for a waveform produced no picture at all, and
    the error named a primitive rather than the scene. Rejected at validation it
    costs one retry and the model writes `zone` instead.

    The companion assertion is the other half: this is *not* `unknown_role`. The
    name is real; pretending otherwise sends the model looking for a typo.
    """
    codes = _codes(
        _mutation(
            lambda data: _scene(data)["objects"].append(
                {
                    "id": "shade",
                    "role": "band",
                    "label": "阴影区间",
                    "props": {"bounded_by": "car"},
                }
            )
        )
    )

    assert "primitive_not_drawable" in codes
    assert "unknown_role" not in codes


def test_unknown_role() -> None:
    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["objects"][0]["role"] = "banana"

    assert "unknown_role" in _codes(_mutation(mutate))


def test_duplicate_id() -> None:
    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["objects"][1]["id"] = "car"

    assert "duplicate_id" in _codes(_mutation(mutate))


def test_unresolved_ref_in_highlights() -> None:
    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["steps"][0]["highlights"] = ["ghost"]

    assert "unresolved_ref" in _codes(_mutation(mutate))


def test_unresolved_ref_in_relation() -> None:
    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["objects"][2]["props"]["of"] = "ghost"

    assert "unresolved_ref" in _codes(_mutation(mutate))


def test_required_relation_missing_on_a_dimension() -> None:
    """The other half of `unresolved_ref`: a reference that was never written.

    A `dimension` with no `from`/`to` measures the distance between nothing, and
    `StoryboardIR` carries no coordinates to fall back on, so the layout pass has
    no geometry at all. It used to pass every check — which made it a picture
    with a hole in it, reported as success.
    """

    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["objects"].append(
            {"id": "range-r", "role": "range", "label": "射程 R", "props": {"label": "R"}}
        )

    assert "required_relation_missing" in _codes(_mutation(mutate))


def test_a_dimension_with_both_ends_is_accepted() -> None:
    """The requirement must be satisfiable, or the retry burns for nothing."""

    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["objects"].append(
            {
                "id": "range-r",
                "role": "range",
                "label": "射程 R",
                "props": {"label": "R", "from": "lidar", "to": "obstacle"},
            }
        )

    assert "required_relation_missing" not in _codes(_mutation(mutate))


def test_an_optional_relation_is_not_required() -> None:
    """`vector.component_of` only applies to a decomposed component.

    Flagging its absence would reject the plain case — the whole velocity
    vector — which is the one most scenes want. Required and optional live in
    the same tuple, so the line between them is worth a test.
    """

    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["objects"].append(
            {
                "id": "v0",
                "role": "velocity",
                "label": "初速度 v0",
                "props": {"magnitude": 20, "direction": 40, "of": "car"},
            }
        )

    codes = _codes(_mutation(mutate))

    assert "required_relation_missing" not in codes


def test_a_vector_without_a_subject_is_flagged() -> None:
    """`of` says what the arrow pushes on. Without it the arrow points at nothing."""

    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["objects"].append(
            {
                "id": "gravity",
                "role": "force",
                "label": "重力",
                "props": {"magnitude": 9.8, "direction": 90},
            }
        )

    assert "required_relation_missing" in _codes(_mutation(mutate))


def test_lesson_scene_unknown() -> None:
    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["lesson_scene_ids"] = ["scene-9"]

    assert "lesson_scene_unknown" in _codes(_mutation(mutate))


def test_lesson_scene_uncovered() -> None:
    # The storyboard must cover every lesson scene, not just some.
    codes = _codes(_storyboard_dict(), lesson=_lesson(extra_scene=True))

    assert "lesson_scene_uncovered" in codes


def test_key_point_missing() -> None:
    def mutate(data: dict[str, Any]) -> None:
        scene = _scene(data)
        for step in scene["steps"]:
            step["key_points"] = ["别的说法"]

    assert "key_point_missing" in _codes(_mutation(mutate))


def test_source_ref_gap() -> None:
    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["objects"][0]["source_refs"] = []

    assert "source_ref_gap" in _codes(_mutation(mutate))


def test_source_ref_unknown() -> None:
    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["objects"][0]["source_refs"] = ["section-999"]

    assert "source_ref_unknown" in _codes(_mutation(mutate))


def test_a_lesson_ref_the_document_lacks_is_not_treated_as_invented() -> None:
    """The two ref rules must not contradict each other.

    `_check_lesson_coverage` demands every lesson ref be carried by an object;
    `source_ref_unknown` rejects an object ref the document does not contain. If a
    lesson cites an id the document lacks, those two together were unsatisfiable —
    the model had to cite it and was forbidden from citing it — so every retry was
    wasted. Seen for real: a stale LessonIR paired with an edited document put 11
    such refs in the prompt, and the agent burned all three attempts on a
    requirement nothing could satisfy.

    Copying the lesson's refs is obeying, not inventing, so the object below ends
    clean on **both** rules. A ref in neither the document nor the lesson is still
    invented, and `test_source_ref_unknown` covers that case.
    """
    data = _storyboard_dict()
    lesson = _lesson()
    lesson.scenes[0].source_refs.append("section-9-block-9")  # the upstream defect
    _scene(data)["objects"][0]["source_refs"] = ["section-1-block-1", "section-9-block-9"]

    assert _codes(data, lesson=lesson) == set()


def test_orphan_object() -> None:
    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["steps"][2]["highlights"] = ["car"]

    assert "orphan_object" in _codes(_mutation(mutate))


def test_unknown_step_prop() -> None:
    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["steps"][1]["object_states"] = {"car": {"banana": 1}}

    assert "unknown_step_prop" in _codes(_mutation(mutate))


def test_unknown_glyph() -> None:
    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["objects"][0]["props"]["glyph"] = "banana"

    assert "unknown_glyph" in _codes(_mutation(mutate))


def test_control_target_unconsumed() -> None:
    # `danger` is an output a behaviour writes, not an input it reads: a slider
    # bound to it would move a number and change nothing on screen.
    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["controls"][0]["target_property"] = "car.danger"

    assert "control_target_unconsumed" in _codes(_mutation(mutate))


def test_control_target_unresolved() -> None:
    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["controls"][0]["target_property"] = "ghost.speed"

    assert "control_target_unresolved" in _codes(_mutation(mutate))


def test_control_range_invalid() -> None:
    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["controls"][0]["min"] = 120
        _scene(data)["controls"][0]["max"] = 0

    assert "control_range_invalid" in _codes(_mutation(mutate))


def test_control_action_unknown() -> None:
    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["controls"][1]["action"] = "banana"

    assert "control_action_unknown" in _codes(_mutation(mutate))


def test_step_count_invalid() -> None:
    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["steps"] = _scene(data)["steps"][:2]

    assert "step_count_invalid" in _codes(_mutation(mutate))


def test_preset_role_missing() -> None:
    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["objects"] = [
            obj for obj in _scene(data)["objects"] if obj["id"] != "obstacle"
        ]
        _scene(data)["steps"][2]["highlights"] = ["car"]

    assert "preset_role_missing" in _codes(_mutation(mutate))


def test_preset_capacity_exceeded() -> None:
    def mutate(data: dict[str, Any]) -> None:
        objects = _scene(data)["objects"]
        for index in range(6):
            objects.append(
                {"id": f"extra-{index}", "role": "obstacle", "label": "障碍", "props": {}}
            )

    assert "preset_capacity_exceeded" in _codes(_mutation(mutate))


def test_unknown_renderer_hint() -> None:
    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["renderer_hint"] = "banana"

    assert "unknown_renderer_hint" in _codes(_mutation(mutate))


def test_no_interactive_demo() -> None:
    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["controls"] = []

    assert "no_interactive_demo" in _codes(_mutation(mutate))


def test_scene_count_exceeded() -> None:
    def mutate(data: dict[str, Any]) -> None:
        extra = copy.deepcopy(_scene(data))
        extra["id"] = "shot-2"
        data["scenes"].append(extra)

    assert "scene_count_exceeded" in _codes(_mutation(mutate))


def test_format_issues_lists_every_problem() -> None:
    def mutate(data: dict[str, Any]) -> None:
        _scene(data)["scene_type"] = "banana"
        _scene(data)["objects"][0]["role"] = "banana"

    storyboard = StoryboardIR.model_validate(_mutation(mutate))
    issues = validate_storyboard(
        storyboard, lesson=_lesson(), document=_document(), limits=LIMITS
    )
    report = format_issues(issues)

    assert len(issues) >= 2
    assert "unknown_scene_type" in report
    assert "unknown_role" in report
