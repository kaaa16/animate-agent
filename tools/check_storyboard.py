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
from animate_agent.rendering.layout import carried_states
from animate_agent.rendering.registry import (
    FALLBACK_ROLES,
    PRESETS,
    PRIMITIVE_BY_NAME,
    ROLE_TO_PRIMITIVE,
)
from animate_agent.storyboard.models import (
    PropValue,
    StoryboardIR,
    StoryboardScene,
    StoryboardStep,
)
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


#: A storyboard whose fallback share reaches this is saying something about the
#: vocabulary rather than about itself. Measured over the 24 storyboards on disk,
#: 23 sit at or under 6% and the one at 58% is the JSON lesson, whose picture is
#: a column of grey boxes. The gap is wide enough that a threshold inside it is a
#: reading, not a fudge.
FALLBACK_SHARE_WARNING = 0.30

#: The same, for one prop carrying the whole picture. Over the same 24 files the
#: median storyboard spreads its changes over 6 props with the top one near a
#: third. Two clear it: 86% (`emphasis`, the JSON lesson) and 100% (`progress`, a
#: publish/subscribe lesson). The first is the problem this line exists to name.
#: The second is a link whose bar fills — a legitimate picture — so what gets
#: printed is an observation and not a complaint, and the explanation says so.
TOP_PROP_SHARE_WARNING = 0.80


def _visual_key(
    step: StoryboardStep, carried: dict[str, dict[str, PropValue]]
) -> tuple[object, ...]:
    """What the player draws a beat from, flattened into something comparable.

    `highlights` is a **set** — the order objects are named in is not a picture —
    and the states are a mapping of mappings. Sorting both is what makes "these
    two beats are the same frame" a question this can answer at all.

    `carried` rather than `step.object_states`: a state is kept until something
    changes it back, so the picture at beat *n* is what every beat up to and
    including *n* has written. Reading the beat's own mapping would call two
    frames different when the second one changes nothing — which is the defect
    this count exists to name. `layout.carried_states` computes it; the rule is
    not restated here.
    """
    return (
        tuple(sorted(step.highlights)),
        tuple(
            sorted((target, tuple(sorted(states.items()))) for target, states in carried.items())
        ),
    )


def _picture_report(storyboard: StoryboardIR) -> tuple[list[str], list[str]]:
    """The lines to print, and the paragraphs to print only if something tripped.

    Three numbers, none of them a pass/fail — and the one that *would* have been
    a pass/fail turned out to be already covered. `storyboard/validation.py`
    refuses a beat that changes nothing, and measured over the 24 storyboards on
    disk **0 of 339 consecutive pairs are visually identical**: the schema needs
    a highlight or a state, and `_check_steps` needs that state to be read, so
    "this beat changed nothing" never survives to reach here. A rule for it would
    be a rule that can never fire.

    What is left to measure is the opposite failure, and it is the one the person
    watching actually reports: a beat that changes something **nobody can see**.

    The sharpest reading is not how many props a film moves — the JSON lesson
    moves five and still lands 86% of them on `emphasis` — but the fallback
    share. A fallback role is a storyboard saying *the vocabulary has no word for
    the thing I am looking at*, which is a statement about this project rather
    than about the document in front of it.

    Neither number is refusable, and that is the point of printing them here
    instead of adding a check next door. Refusing a fallback role would cost the
    document its animation entirely — a wrong picture beats no picture — and
    refusing a lopsided prop mix would push the model to vary its output for the
    sake of varying it, which is how `generic` came to be described as a safe
    default and then chosen over `field` three times in five.
    """
    props: collections.Counter[str] = collections.Counter()
    for scene in storyboard.scenes:
        for step in scene.steps:
            for states in step.object_states.values():
                for prop in states:
                    props[prop] += 1

    roles = collections.Counter(obj.role for scene in storyboard.scenes for obj in scene.objects)
    cast = sum(roles.values())
    fallbacks = sum(count for role, count in roles.items() if role in FALLBACK_ROLES)

    pairs = 0
    identical = 0
    for scene in storyboard.scenes:
        keys = [
            _visual_key(step, held)
            for step, held in zip(scene.steps, carried_states(scene.steps), strict=True)
        ]
        for before, after in zip(keys, keys[1:], strict=False):
            pairs += 1
            identical += int(before == after)

    lines: list[str] = []
    notes: list[str] = []

    total = sum(props.values())
    if total:
        shown = "、".join(f"{name} {count}" for name, count in props.most_common(6))
        lines.append(f"  改画面用到的属性：{shown}")
        lines.append(f"      （共 {total} 处改动，{len(props)} 种属性）")
        top_prop, top_count = props.most_common(1)[0]
        share = 100 * top_count / total
        if top_count / total >= TOP_PROP_SHARE_WARNING:
            lines.append(f"      ⚠ `{top_prop}` 一种就占 {share:.0f}%，变化大多是同一个动作")
            notes.append(
                f"⚠ 全片 {total} 处改动里有 {share:.0f}% 压在 `{top_prop}` 一个属性上。"
                "属性种类的上限就是图元种类的上限：一个兜底盒子身上能改的、看得见的东西，"
                "基本只剩下 `emphasis`（弹一下、抖一下）。所以这一行常常是下面兜底角色那一行"
                "的另一种写法——两行一起出现时，读下面那行，它说的是原因。"
            )
    else:
        lines.append("  改画面用到的属性：（一处都没有）")

    if cast:
        fallback_name = "/".join(sorted(FALLBACK_ROLES))
        share = 100 * fallbacks / cast
        lines.append(f"  兜底角色 `{fallback_name}`：{fallbacks}/{cast} 个对象（{share:.0f}%）")
        if fallbacks / cast >= FALLBACK_SHARE_WARNING:
            lines.append(f"      ⚠ {share:.0f}% 的对象用了兜底角色，画面里多数东西没有专门的画法")
            notes.append(
                f"⚠ {cast} 个对象里有 {fallbacks} 个用了兜底角色 `{fallback_name}`。"
                "这个角色是词表里「这东西没有合适的角色」的落点。模型写它不是在偷懒，"
                "是在说「你给我的词里，没有一个说的是我看到的这个东西」——"
                "所以它说的不是这份产物，是图元不够用。\n"
                "  校验台拒不了它，也不该拒：拒了这份文档就一个画面都渲染不出来，"
                "那比画面难看更糟（见 `generic` 被描述成「安全兜底」之后发生了什么，"
                "`registry.py` 的 `PRESETS` 里记着）。它要的是往词表里加能画这类内容的东西，"
                "不是把模型逼去选一个不匹配的角色。"
            )
    lines.append(f"  相邻两拍画面完全一样：{identical} 对（共 {pairs} 对，越接近 0 越好）")

    return lines, notes


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
    #: Two ways a scene ends up `generic`, and they mean opposite things. The
    #: first is a regression — a domain preset it satisfied went unused. The
    #: second is an inventory problem: `generic` was the *only* eligible
    #: preset, so there was nothing to choose. The JSON lesson has five of the
    #: second and none of the first; reporting both as "chose generic" would
    #: send you looking for a model that picked wrong, when the model was right
    #: and the vocabulary was short.
    unused_domain_preset = 0
    forced_generic = 0
    for scene in storyboard.scenes:
        eligible = _eligible_presets(scene)
        alternatives = [name for name in eligible if name != scene.scene_type and name != "generic"]
        if scene.scene_type == "generic":
            if alternatives:
                unused_domain_preset += 1
            elif eligible == ["generic"]:
                forced_generic += 1
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
    print("画面变化（这份产物在屏幕上动了什么）：")
    picture_lines, picture_notes = _picture_report(storyboard)
    for line in picture_lines:
        print(line)
    for note in picture_notes:
        print()
        print(note)

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

    if forced_generic:
        print()
        print(
            f"⚠ {len(storyboard.scenes)} 个场景里有 {forced_generic} 个**除了 `generic` "
            "没有别的预设可选**——不是选错，是没得选。\n"
            "  预设是按「必需角色齐不齐」发的牌：`lane` 要 vehicle+obstacle，`chain` 要 node，"
            "`hub` 要 node+endpoint，`field` 要 projectile。一份讲键值对和数组的文档一个都不沾，"
            "所以每一幕都落在竖排。\n"
            "  这一行**不是**回归（上一行才是），也不该让模型换个预设来消掉——"
            "把 `chain` 硬套在「值可以是字符串和数字」上只会更难看。"
            "它和画面变化里那条兜底角色占比是同一件事的两种写法：**图元种类的账**，"
            "要还的是往词表里加东西，不是改提示词。"
        )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
