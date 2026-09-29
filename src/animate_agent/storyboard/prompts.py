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
from animate_agent.rendering.layout import SPEECH_CHARS_PER_SECOND, TRANSITION_SECONDS
from animate_agent.rendering.registry import render_vocabulary
from animate_agent.storyboard.models import (
    StoryboardControl,
    StoryboardObject,
    StoryboardScene,
    StoryboardStep,
)

#: How many beats a scene usually gets, and therefore the number the per-caption
#: character count is divided by.
#:
#: A target rather than a bound — `StoryboardLimits` allows three to five — but
#: it is the number the arithmetic needs, because the caption length on screen is
#: `narration ÷ (scenes × beats)`. Left to the model it averaged 3.67, which is
#: the difference between a 17-character caption and a 15-character one; small,
#: but it is the one lever that moves captions at all, so it is stated.
TARGET_STEPS_PER_SCENE = 4

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
- **节拍数量：每个场景以 4 个 `steps` 为主**（3~5 都行，四拍是常态）。
  四拍不是随手定的：一屏字幕 17 字上下，一幕四拍刚好是一段讲得完的话。
- **成片时长**由所有 `description` 加起来决定——**每秒 6 个字**，再加换幕的几秒。
  目标是一到两分钟，超了会被打回重写。
  **每个 `description` 该写多少字，下面的用户消息按这份课程的实际字数算给你了。**
  照那个数来：它比一个固定数字管用，因为课程一变它跟着变。
- **`description` 是屏幕上的字幕，一屏就是一个完整的意思。** 它是上面那条 narration
  切出来的：不是照抄，也不是压缩成一个标签——一条 narration 切成 3~5 屏，每屏把一件事
  讲完。**不要把一句话劈成两半**，半句占一屏，观众得等下一屏才知道你要说什么。
  多写「为什么」、少报「是什么」：这句话如果是画面上已经画着的东西，它不值得占一屏。
- **每个节拍必须有视觉变化**：至少写一个 `highlights`，或至少改一个 `object_states`。
  只讲文字、画面上什么都没动的节拍不是节拍。
- **`highlights` 和 `emphasis` 不是一回事。** `highlights` 是「看这里」——把这一拍
  要讲的对象圈出来；`emphasis` 是「它动了一下」——在 `object_states` 里给这个对象
  写一个动作名（可选值见下方词表）。两者可以落在同一拍上，那是最重的一下。
  **「动一下」要省着用**：同一个对象连着几拍都强调，观众就分不清哪一拍才是重点。
  按要表达的意思挑，不要一律用同一个：提醒注意用 `pulse`，表示「不对劲」用 `shake`，
  表示「它不稳」用 `wobble`，表示「它撑不住」用 `spring`，表示「就是它」用 `pop`，
  绕支点荡开用 `swing`（**只有 `arm` 角色有支点**，别放在别的东西上，那样它只会原地乱转）。
  **强调不会留到下一拍**——它是一次手势，不是状态；哪一拍要它就写在哪一拍，
  上一拍写过这一拍不写，它自己就停了。
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
  再在讲它的那一拍用 `object_states` 把它改成 `true`——它会按自己的出场方式冒出来
  （出场方式见下一条，不写就是淡着浮上来）。
  同一拍里 `highlights` 到它是正常的，那正是它登场的那一下。
  **写了 false 就必须有某一拍把它打开**，整幕都不打开的会被校验打回
  （说 `object_never_visible`）。
- **东西怎么出场，也是你定的。** 上一条管的是「什么时候出现」，这一条管的是
  「冒出来的那一下长什么样」：在这个对象的 `props` 里写 `"enter"`，可选值见下方词表。
  **一幕里不要都用同一种**——相邻的两三样换着来，观众才看得出它们是分别登场的，
  而不是整屏一起亮起来。按东西的来头挑：从下往上冒用 `rise`（默认）、
  从上面落下来用 `drop`（重物、压下来的结论）、由小涨到原大用 `zoom`（放大、聚焦、走近看）、
  一路只是淡入用 `fade`（底色、背景、本来就该安静的东西）、
  从左往右擦出来用 `wipe`（一段话、一张表被「读」出来）。
  **只有方框类的东西擦得出来**：线、箭头、坐标轴、令牌这类没有框的，
  写 `wipe` 会退回 `rise`，写了也不算错，只是看不出差别。
  还有一条要记住：**从头到尾 `visible` 都是 true 的对象，根本不会有出场那一下**——
  出场是「冒出来」的一部分，不冒出来就没有那一下。
- **先看这段内容是什么形状，再挑画法。** 有几类内容各有各的画法，
  **一律画成一排方框加箭头是错的**：
  - **「一层套一层」「谁在谁里面」「分了几层、分了几类」** → 写一个 `tree`，
    把层次**用两个空格的缩进写进它的 `text`**（源文里给了缩进的样子就照着抄）。
    缩进本身就是内容：外层顶格、里面那层缩进两格、再里面再缩两格，
    观众一眼看出谁装在谁里面。**不要**把这几层各写一个 `node` 串成 `chain`——
    那画出来是一排先后相承的方框，可它们之间**没有先后**。
  - **「一条一条列出来」「一份配置、一段代码、一串参数」** → 写一个 `code`，
    `text` 就是那几行本身，一行一行原样写。**不要**把这几行塞进某个对象的
    `label` 或令牌的 `state` 里——那是把一整块内容降级成一个名字。
  - **「这两种哪个对」「差在哪」「一边…另一边…」** → 两样东西都写 `option` 角色，
    这一幕就能选 `compare` 预设，它们会左右摆开。
  - **「一路经过某几站」** → 才是 `chain`。而且**这一幕里不要写 `traveler`**：
    一个令牌只跑得了一条线，它会停在半路，而字幕说它已经到了终点——
    要让路走完，就用 `link` 的 `active` 逐拍点亮下一段。
  - 一段话里既有结构又有过程时，按**先讲到的那一样**挑；两样都要讲，拆成两幕。
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


def caption_budget(lesson: LessonIR) -> tuple[int, int, float]:
    """How much one `description` should hold, for *this* lesson.

    Returns `(low, high, average)`, in characters. The band is a spread around the
    average rather than a rule: some beats carry a clause and some carry a whole
    sentence, and a model told to hit one number exactly writes uniformly, which
    is the monotony the rhythm instruction is there to avoid.

    The arithmetic is the whole of this round's fix, and it is one division:
    `narration ÷ (scenes × beats)`. What it replaces is a constant — 「每个
    description 写 9~14 字」 — which was correct for exactly one lesson: a
    350-character script cut into nine scenes of four beats. The lesson on disk
    has more characters than that, so the constant was already asking for captions
    shorter than the film could afford, and no test could see it because a
    character count in a prompt is not a thing anything reads back.
    """
    characters = sum(len(scene.narration) for scene in lesson.scenes)
    beats = max(len(lesson.scenes) * TARGET_STEPS_PER_SCENE, 1)
    average = characters / beats
    return max(round(average * 0.75), 1), max(round(average * 1.35), 2), average


def build_storyboard_prompt(
    lesson: LessonIR,
    *,
    allowed_renderers: tuple[str, ...] = (),
    script_mode: bool = False,
) -> str:
    """Serialize a LessonIR plus the generated vocabulary into the user message.

    The length arithmetic lives here rather than in the system prompt because it
    is a fact about *this* lesson: it is the lesson's own character count divided
    by the beats it will become. A number in the system prompt would be a constant
    again, and the constant is what this round is removing.
    """
    lines: list[str] = [
        f"课程标题：{lesson.title}",
        f"学科：{lesson.subject}",
        f"摘要：{lesson.summary}",
        "",
        "整体学习目标：",
    ]
    lines.extend(f"- {objective}" for objective in lesson.learning_objectives)
    lines.append("")

    low, high, average = caption_budget(lesson)
    total = sum(len(scene.narration) for scene in lesson.scenes)
    scenes = len(lesson.scenes)
    seconds = total / SPEECH_CHARS_PER_SECOND + max(scenes - 1, 0) * 2 * TRANSITION_SECONDS
    lines.append("## 这份稿子的字数账（照着算，不要自己估）")
    lines.append(f"- 全片讲解文案共 **{total} 字**，{scenes} 幕。")
    lines.append(
        f"- 每幕 {TARGET_STEPS_PER_SCENE} 拍就是 {scenes * TARGET_STEPS_PER_SCENE} 拍，"
        f"所以**每个 `description` 写 {low}~{high} 字**（平均 {average:.0f} 字）。"
    )
    lines.append(
        f"- 按每秒 6 个字、加上换幕，成片约 **{seconds:.0f} 秒**——这就是你要的时长，"
        "每一屏多几个字少几个字，最后都会加到这里。"
    )
    if script_mode:
        lines.append(
            "- **这份稿子是照着这条片子写的，字幕直接切它自己的句子。** "
            "不要改写措辞、不要换说法、不要为了长短把一句完整的话拆开或合并——"
            "观众要听到的是写稿人写下的那句话，你只决定它在第几屏出现。"
        )
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
