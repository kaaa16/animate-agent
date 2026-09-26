"""Prompt templates for the Storyboard Agent.

The vocabulary section is **generated from `rendering/registry.py`**, never
hand-copied. A name the renderer does not know must not be offered to the model,
and a name the renderer does know must not be forgotten here; rendering it means
the two cannot disagree, and a test asserts every registered name shows up.
"""

from __future__ import annotations

from typing import get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from animate_agent.knowledge.models import LessonIR
from animate_agent.rendering.registry import render_vocabulary
from animate_agent.storyboard.models import (
    StoryboardControl,
    StoryboardObject,
    StoryboardScene,
    StoryboardStep,
)

#: Fields the model authors, paired with the model that declares them. Only this
#: list lives here; the bounds are read off the declarations at call time, so a
#: changed bound cannot leave the prompt behind.
_CONSTRAINED_FIELDS: tuple[tuple[str, type[BaseModel], str], ...] = (
    ("scene.id", StoryboardScene, "id"),
    ("scene.teaching_goal", StoryboardScene, "teaching_goal"),
    ("scene.lesson_scene_ids", StoryboardScene, "lesson_scene_ids"),
    ("scene.objects", StoryboardScene, "objects"),
    ("scene.controls", StoryboardScene, "controls"),
    ("scene.renderer_hint", StoryboardScene, "renderer_hint"),
    ("object.id", StoryboardObject, "id"),
    ("object.label", StoryboardObject, "label"),
    ("step.id", StoryboardStep, "id"),
    ("step.title", StoryboardStep, "title"),
    ("step.description", StoryboardStep, "description"),
    ("step.highlights", StoryboardStep, "highlights"),
    ("step.key_points", StoryboardStep, "key_points"),
    ("control.id", StoryboardControl, "id"),
    ("control.label", StoryboardControl, "label"),
    ("control.target_property", StoryboardControl, "target_property"),
    ("control.unit", StoryboardControl, "unit"),
)


def _read_bounds(field: FieldInfo) -> tuple[int | None, int | None, str | None]:
    """Pull length bounds and a regex out of a field's constraint metadata.

    Duck-typed on attribute names rather than isinstance-checked against
    `annotated_types`: those attribute names have been stable across pydantic
    versions, the concrete classes have not.
    """
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    for meta in field.metadata:
        value = getattr(meta, "min_length", None)
        if value is not None:
            min_length = value
        value = getattr(meta, "max_length", None)
        if value is not None:
            max_length = value
        value = getattr(meta, "pattern", None)
        if value is not None:
            pattern = value
    return min_length, max_length, pattern


def render_field_constraints() -> str:
    """Render the schema's own bounds as prompt text.

    Generated from the models, for the same reason the vocabulary is generated
    from the registry: **a bound that lives only in the schema is a bound the
    model has never been told, and it will violate it.** That is how the first
    real run failed — every step came back with a description under a
    20-character floor the prompt had never mentioned.
    """
    lines = ["## 字段长度约束（由 schema 生成，请严格遵守）"]
    for label, model, field_name in _CONSTRAINED_FIELDS:
        field = model.model_fields[field_name]
        min_length, max_length, pattern = _read_bounds(field)
        unit = "个" if get_origin(field.annotation) is list else "字"
        if min_length is None:
            bound = f"最多 {max_length} {unit}"
        elif max_length is None:
            bound = f"至少 {min_length} {unit}"
        else:
            bound = f"{min_length}~{max_length} {unit}"
        if pattern is not None:
            bound += f"，必须匹配 `{pattern}`"
        lines.append(f"- `{label}`：{bound}")
    return "\n".join(lines)


STORYBOARD_SYSTEM_PROMPT = """你是一位动画分镜师。
用户会给你一份已经审核通过的教学课程大纲（LessonIR，包含若干场景，
每个场景有 id、标题、教学目标、讲解文案、关键词和原文出处）。

你的任务是把它转成一份「动画分镜」（StoryboardIR）——一个 JSON 对象，
描述每一幕画面上有哪些对象、按什么节拍动、有哪些可交互控件。
JSON 结构如下（不要输出 JSON 以外的任何文字）：

{
  "title": "课程标题",
  "subject": "学科/主题",
  "eyebrow": "画面上方的一行小字，例如「避障 · 决策逻辑」",
  "theme": "配色名，可以省略，见下方词表",
  "scenes": [
    {
      "id": "scene-1",
      "scene_type": "预设名，见下方词表",
      "teaching_goal": "这一幕要让学生看懂什么（一句话）",
      "lesson_scene_ids": ["被覆盖的课程场景 id，1~2 个"],
      "objects": [
        {
          "id": "car",
          "role": "角色名，见下方词表",
          "label": "画面上显示的名字",
          "props": { "speed": 60, "heading": 0, "glyph": "car" },
          "source_refs": ["section-2-block-1"]
        }
      ],
      "steps": [
        {
          "id": "step-1",
          "title": "节拍标题",
          "description": "这一拍屏幕上显示的那行字幕，一句话",
          "highlights": ["car"],
          "object_states": { "car": { "speed": 40 } },
          "key_points": ["关键词1", "关键词2"]
        }
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
          "unit": "cm/s"
        }
      ],
      "params": {},
      "renderer_hint": null
    }
  ]
}

硬性要求：

- **不要写任何坐标、像素尺寸、颜色、字号、时长、缓动。** 你只描述语义，
  画面怎么摆、多大多小、什么颜色，全部由渲染代码决定。写了也会被丢弃。
- **场景覆盖**：`lesson_scene_ids` 必须覆盖 LessonIR 里的**每一个**场景 id，
  不多不少。一个 storyboard 场景可以覆盖 1~2 个课程场景（内容单薄的相邻场景合并成一幕），
  但**不允许新增**课程里没有的场景。
- **节拍数量**：每个场景写 3~5 个 `steps`。
- **成片时长 60 秒**，超过 75 秒会被打回重写。
  时长是按字数推的：**每秒 6 个字**。所以整条片子的字幕加起来只有约 350 字，
  每个 `description` 写 9~14 字。换算给你一个直觉：
  10 幕 × 4 拍 × 9 字 = 360 字 = 60 秒；把每个 `description` 写成 30 字，
  同样结构就是 200 秒，直接不合格。
- **`description` 是屏幕上的字幕，不是讲解稿。** 它是 LessonIR 那条 narration 的
  **浓缩**，不是照抄——一条 narration 大约要切成 3~5 行字幕，每行只讲一件事。
  字数不够时宁可多说几拍，也不要在一拍里塞两件事。
- **每个节拍必须有视觉变化**：至少写一个 `highlights`，或至少改一个 `object_states`。
  只讲文字、画面上什么都没动的节拍不是节拍。
- **关键词覆盖**：每个场景各节拍的 `key_points` 合起来，必须**逐字**包含它所覆盖的
  那些课程场景的**全部** `key_points`。请直接照抄，不要改写、不要合并近义词。
- **出处覆盖**：每个场景各对象的 `source_refs` 合起来，必须覆盖它所覆盖的课程场景的
  全部 `source_refs`，且只能引用源文档里真实存在的 id。
- **对象不许闲置**：每个对象至少要被某个节拍 `highlights` 到，或被某个关系型属性
  （`of` / `from` / `to` / `along` / `bounded_by` 等）引用到。
- **画面不动的一幕，至少要有三样东西。** 会动的画面不受这条约束——车在开、
  抛体在飞，动画本身就是内容。但**没有东西会动的画面，画上的东西就是全部内容**：
  只有两三样东西的就成了一块留白，讲的道理再好也看不出来。三样之外，再写一样
  说得上话的：一个 `note` 把这一步的关键数字或结论写在画面上，一个 `bubble`、`verdict`
  或 `ordinal` 指着其中一个对象说一句，或者把这一步真正在比的那两样东西都写出来。
  **凑数的不算**：为了够数加一个什么也不说的对象，比画面空更糟。
- **讲到了才出现。** 一幕里的对象默认从第一拍起就都在画面上。想让某样东西
  **在讲它的那一拍才出现**，就在这个对象的 `props` 里写 `"visible": false`
  （`emitter`/`zone` 写 `"enabled": false`，同一个意思），
  再在讲它的那一拍用 `object_states` 把它改成 `true`——它会淡着浮上来。
  同一拍里 `highlights` 到它是正常的，那正是它登场的那一下。
  **写了 false 就必须有某一拍把它打开**，整幕都不打开的会被校验打回
  （说 `object_never_visible`）。
- **画面顺序跟着讲解顺序走。** 谁在上、谁在左，是按你旁白里**先讲谁**定的：
  一幕里同时有代码块和值卡片时，**第一拍先讲到的那一样占上半**，
  另一件排在下面；对象横排时也是先声明的在左边。
  所以按讲解顺序声明 `objects`，并且**第一拍就点出这一幕的主角**——
  第一拍点谁，谁就在上面，这不是装饰，是这一条的全部内容。
- **必需的关系一个都不能漏**：有些图元光靠自己画不出来——`dimension` 没有 `from`/`to`
  就不知道量的是哪两点，`trace` 没有 `of` 就不知道是谁的轨迹，`vector` 没有 `of`
  就不知道作用在谁身上。词表里标了「**必须填写**」的关系型属性必须写，
  填同一场景内另一个对象的 id。漏了会被校验打回，说 `required_relation_missing`。
- **交互 demo**：整个 storyboard 至少要有一个场景带 `controls`。
  控件的 `target_property` 只能绑定「被行为读取的属性」（见下方词表最后一段），
  否则它只会改 UI 数字、不会改画面，会被校验打回。
- 忠于课程大纲：只使用 LessonIR 里出现过的事实、数字和命名，不要自己补充。
- `title` / `subject` 会自动沿用课程大纲，你可以省略；`theme` 也可以省略
  （见下方词表，不写会按课程自动分一套）；其余字段必填。
- **`id` 只用小写 ASCII**：小写字母开头，之后只能是小写字母、数字、下划线、连字符。
  不要用大写字母、中文、空格或下标字符。物理量请写成 `v0` / `theta` / `range-r`，
  **不要**写 `V₀` / `θ` / `R`——这些会把 `id` 打回。中文请放到 `label` 里。

"""

# The constraint block is appended rather than retyped: see `render_field_constraints`.
STORYBOARD_SYSTEM_PROMPT = STORYBOARD_SYSTEM_PROMPT + "\n" + render_field_constraints() + "\n"


def build_storyboard_prompt(lesson: LessonIR, *, allowed_renderers: tuple[str, ...] = ()) -> str:
    """Serialize a LessonIR plus the generated vocabulary into the user message."""
    lines: list[str] = [
        f"课程标题：{lesson.title}",
        f"学科：{lesson.subject}",
        f"摘要：{lesson.summary}",
        "",
        "整体学习目标：",
    ]
    lines.extend(f"- {objective}" for objective in lesson.learning_objectives)
    lines.append("")

    lines.append("## 课程场景（必须全部覆盖）")
    for scene in lesson.scenes:
        lines.append(f"### [{scene.id}] {scene.title}")
        lines.append(f"教学目标：{scene.objective}")
        lines.append(f"讲解文案：{scene.narration}")
        lines.append(f"关键词（必须逐字出现在 key_points 里）：{'、'.join(scene.key_points)}")
        refs = "、".join(scene.source_refs)
        lines.append(f"原文出处（必须被某个对象的 source_refs 覆盖）：{refs}")
        lines.append("")

    lines.append("# 可用词表")
    lines.append("")
    lines.append(render_vocabulary(allowed_renderers))

    return "\n".join(lines)
