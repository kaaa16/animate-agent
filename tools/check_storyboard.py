"""Audit a *stored* StoryboardIR: does it validate, and can it actually be laid out?

Why this exists
---------------
`storyboard/validation.py` answers one question — is this storyboard *valid*? It
cannot answer the next one — is it *renderable*? The two differ, and the gap is
where silent quality loss lives. A `dimension` object with no `from` and no `to`
passes every check in the validator and then cannot be drawn, because there is
nothing to measure between. `validate_storyboard` checks that a relation which
**is** present resolves; it never checks that a **required** one is present.

This is the companion to `tools/inspect_storyboard_output.py`:

| tool | input | question |
|---|---|---|
| `inspect_storyboard_output.py` | a raw rejected response | what the model does *unaided* |
| `check_storyboard.py` (this) | the kept `storyboard-*.json` | is the kept artefact any good? |

Both are free: no LLM call, no network.

Usage
-----

    python -m tools.check_storyboard data/generated/s1-verify/storyboard-*.json \
        --lesson data/generated/s1-verify/lesson-*.json \
        --document data/generated/s1-run/document.json

`--document` is optional; without it the `source_refs` existence check is skipped.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

from animate_agent.documents.models import DocumentIR
from animate_agent.knowledge.models import LessonIR
from animate_agent.rendering.registry import PRESETS, PRIMITIVE_BY_NAME, ROLE_TO_PRIMITIVE
from animate_agent.storyboard.models import StoryboardIR, StoryboardScene
from animate_agent.storyboard.service import build_limits
from animate_agent.storyboard.validation import format_issues, validate_storyboard

EXIT_OK = 0
EXIT_MISSING = 1


def _eligible_presets(scene: StoryboardScene) -> list[str]:
    """Presets this scene satisfies, most specific first.

    Mirrors the validator's own rule rather than restating it: a preset is
    eligible when every `required_roles` entry is present and the count of
    **`body`-primitive** objects is within `max_bodies`. Counting all objects
    instead is the easy mistake — it makes an 8-object projectile scene look
    too big for `field`, when only one of those 8 is a body.
    """
    roles = {obj.role for obj in scene.objects}
    bodies = sum(1 for obj in scene.objects if ROLE_TO_PRIMITIVE.get(obj.role) == "body")
    return [
        preset.name
        for preset in PRESETS
        if all(role in roles for role in preset.required_roles) and bodies <= preset.max_bodies
    ]


def _unanchored(scene: StoryboardScene) -> list[str]:
    """Objects missing a relation their primitive cannot be drawn without.

    Derived from `Primitive.required_relations` — the same declaration the
    validator now enforces — so the scene-level roll-up below cannot disagree
    with the per-object `required_relation_missing` issues above it.

    This used to apply a weaker rule of its own ("declares relations, has none
    set"), because that was all the registry said. The weak version could not
    tell a missing `of` from a missing `component_of`, so it could not be turned
    into a rejection. Now that the registry states which relations are required,
    the tool and the validator ask the same question.
    """
    found: list[str] = []
    for obj in scene.objects:
        primitive = PRIMITIVE_BY_NAME.get(ROLE_TO_PRIMITIVE.get(obj.role, ""))
        if primitive is None or not primitive.required_relations:
            continue
        missing = [
            relation
            for relation in primitive.required_relations
            if obj.props.get(relation) is None
        ]
        if missing:
            found.append(f"{obj.id}({obj.role}, 缺 {'/'.join(missing)})")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("storyboard", type=Path, help="已落盘的 StoryboardIR JSON")
    parser.add_argument("--lesson", type=Path, required=True, help="对应的 LessonIR JSON")
    parser.add_argument("--document", type=Path, default=None, help="对应的 DocumentIR JSON")
    args = parser.parse_args(argv)

    for path in (args.storyboard, args.lesson):
        if not path.exists():
            print(f"找不到文件: {path}")
            return EXIT_MISSING

    storyboard = StoryboardIR.model_validate_json(args.storyboard.read_text(encoding="utf-8"))
    lesson = LessonIR.model_validate_json(args.lesson.read_text(encoding="utf-8"))
    document = (
        DocumentIR.model_validate_json(args.document.read_text(encoding="utf-8"))
        if args.document is not None
        else None
    )

    issues = validate_storyboard(
        storyboard, lesson=lesson, document=document, limits=build_limits()
    )
    print(f"校验台：{len(issues)} 处问题")
    for code, count in collections.Counter(issue.code for issue in issues).most_common():
        print(f"  {code}: {count} 处")
    if issues:
        print()
        print(format_issues(issues))

    print()
    print("逐场景（预设选择 / 布局就绪度）：")
    unused_domain_preset = 0
    for scene in storyboard.scenes:
        eligible = _eligible_presets(scene)
        alternatives = [name for name in eligible if name != scene.scene_type and name != "generic"]
        if scene.scene_type == "generic" and alternatives:
            unused_domain_preset += 1
        unanchored = _unanchored(scene)
        print(
            f"  {scene.id}: 选了 `{scene.scene_type}`；"
            f"可用 {eligible}；对象 {len(scene.objects)}；控件 {len(scene.controls)}"
        )
        if alternatives:
            print(f"      更具体的预设可用但未选：{alternatives}")
        if unanchored:
            print(f"      布局画不出来（缺必需关系）：{'、'.join(unanchored)}")

    print()
    if unused_domain_preset:
        print(
            f"⚠ {unused_domain_preset} 个场景选了 `generic` 而更具体的预设本来可用。"
            "校验台看不出这个差别——两个预设都是合法值——但画面看得出来：`field` 有坐标轴"
            "与轨迹槽位，`generic` 只有竖排对象加说明面板。词表已不再把 `generic` 描述成"
            "「兜底」（这正是它被当成安全选择的原因），所以再出现这一行就是回归，不是旧输出。"
        )
    else:
        print("没有「可用却没选」的领域预设。")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
