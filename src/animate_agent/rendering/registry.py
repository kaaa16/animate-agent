"""The semantic vocabulary a storyboard may use, and the limits of what can be drawn.

This module is the contract, not the renderer. It lives under `rendering/`
because the vocabulary is defined by *what can be drawn*: a role exists here
only if some primitive can express it, and a prop exists only if some primitive
or behaviour reads it. `storyboard/validation.py` and the Storyboard Agent
prompt are both generated from the tables below, which is what keeps the three
of them from drifting apart.

Layers (see `docs/storyboard-milestone.md`):

- **atoms**   — draw-layer shapes; never appear in the contract.
- **T1 primitives** — the 17 semantic primitives the model may reference.
- **macros**  — T2 glyphs and vector decomposition; they *expand into* T1
  primitives rather than being a new primitive kind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------
# Atoms — internal to the draw layer, not part of the contract.
# --------------------------------------------------------------------------

ATOMS: tuple[str, ...] = (
    "circle",
    "rect",
    "polygon",
    "capsule",
    "arc",
    "sector",
    "ring",
    "path",
)


#: How many rows a `tree` or a `code` block may hold.
#:
#: Declared here rather than in `layout.py`, where the row heights live, because
#: the two primitive notes below *print* it and a note is built at import time —
#: `layout` imports this module, so the number cannot travel the other way. That
#: is the same reason `StageRange` is a record here and `layout` only clamps with
#: it: the vocabulary states the limit and the drawing obeys it.
#:
#: The number is a fit, not a taste. A block's rows are 24~26 stage pixels apart
#: and the frame is 486 tall, so 16 rows is 380~416 of it — a block that fills the
#: stage and still leaves the caption strip alone. `layout._block_boxes` refuses
#: anything that does not fit the room it actually has; this is the ceiling the
#: model is *told*, so the refusal is a backstop rather than the first it hears
#: of it.
BLOCK_ROWS_MAX = 16


# --------------------------------------------------------------------------
# T1 — semantic primitives
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Primitive:
    """One drawable primitive and the semantic roles it can carry.

    A role belongs to exactly one primitive, so `role -> primitive` is a total
    function and an unknown role is a hard failure rather than a guess.
    `props` are scalars the model may set; `relations` hold another object's id
    and are resolved within the same scene.
    """

    name: str
    roles: tuple[str, ...]
    props: tuple[str, ...]
    relations: tuple[str, ...]
    note: str
    #: Relations that must be present on **every** object of this primitive.
    #:
    #: The criterion is one question: *can the layout pass compute where this
    #: object goes without it?* A `dimension` with no `from`/`to` has no two
    #: points to measure between, a `trace` with no `of` belongs to nobody, a
    #: `vector` with no `of` pushes nothing. None of those are lax drawings —
    #: they are objects with no geometry, and the layout layer has nothing to
    #: fall back on, because `StoryboardIR` carries no coordinates (decision D3).
    #:
    #: Declared here rather than derived, because "required" is a fact about the
    #: drawing, not something the name implies. `relations` stays the superset:
    #: `vector.component_of` and `angle.between` only apply to some objects of
    #: their primitive. A test asserts this is a subset of `relations`.
    required_relations: tuple[str, ...] = ()

    #: Props the drawing is **meaningless** without — the next question out from
    #: `required_relations`, asked the other way round.
    #:
    #: `required_relations` asks "can the layout compute where this goes". This
    #: asks "does the result still teach anything". An `axis` with no `range` has
    #: a complete, well-formed geometry: a shaft, five evenly spaced notches, an
    #: arrowhead. It just is not a coordinate axis — it is a notched arrow, and
    #: `drawAxis` says as much where it decides whether to label the ticks. The
    #: projectile document drew two of those and called them 坐标轴.
    #:
    #: So the failure this prevents is not a missing object but a plausible one,
    #: which is the harder kind to notice: nothing is blank, nothing throws, and
    #: the picture is simply not a picture of the thing the narration describes.
    #: Same criterion the project already applies to layout — `layout_error`
    #: exists because "可读性失败必须让生成失败，而不是交给人眼发现".
    required_props: tuple[str, ...] = ()

    #: Whether the pipeline can *draw* this primitive today.
    #:
    #: Not the same question as "is it in the contract". `RegionElement` and
    #: `WaveElement` are in `rendering/models.py` and stay there; what is missing
    #: is the layout pass that turns a semantic object into their points. A
    #: primitive may legitimately be declared before its layout lands.
    #:
    #: While `drawable` is false the name must not be **offered**: a name in the
    #: prompt is a promise, and this one is terminal when broken. Layout runs
    #: after the last LLM call, so a `band` that reaches it kills the run after
    #: the model has been paid for — with nothing fed back and nothing retried.
    #: Rejected in the validator instead, it becomes an ordinary issue: the model
    #: sees the error, picks `zone`, and a picture still comes out.
    drawable: bool = True

    #: Props some drawing code reads **during playback**, with the element's
    #: baked value as the fallback.
    #:
    #: `props` answers "may the model set this". This answers "does setting it
    #: do anything" — the layer below `drawable`, and the same failure one level
    #: down. A `vector` declared `direction` and had it validated and had the
    #: model write `0 / 45 / 90` across three beats, and drew the same arrow
    #: every time, because `drawVector` read the baked `dx`/`dy` and never asked
    #: `view.lookup`. Nothing errored; the picture was simply a still.
    #:
    #: Measured before this field existed: 20 of the 29 `object_states` in the
    #: three sample documents targeted a prop nothing reads. A beat whose only
    #: content is an unread prop is a slide, and no layer said so.
    #:
    #: Only `object_states` is gated on this. Setting `glyph` or `fov` once on
    #: the object is legitimate; setting it *per beat* and changing nothing is
    #: the defect. `frontend/player/registry.js` mirrors this table by hand and
    #: a test asserts the two are equal — a hand copy so that drift is a red
    #: test rather than a silent still (the `PENDING_KINDS` argument).
    live_props: tuple[str, ...] = ()


T1_PRIMITIVES: tuple[Primitive, ...] = (
    Primitive(
        name="body",
        roles=(
            "vehicle",
            "agent",
            "node",
            "endpoint",
            "obstacle",
            "device",
            "projectile",
            "object",
            "arm",
        ),
        props=("speed", "heading", "scale", "visible", "danger", "emphasis", "glyph"),
        relations=(),
        live_props=("speed", "heading", "scale", "visible", "danger", "emphasis"),
        note="基础形状由角色决定；`glyph` 是可选的实物图标，写了就画成那个图标"
        "（见下面的字形清单），不写就画基础形状。"
        "`object` 是兜底角色，只在确实没有合适角色时才用——留它是为了让每个被抽出的"
        "对象都有地方去，而不是被硬塞进某个不匹配的域角色。"
        "**`heading` 只转车头，不改路线**：`lane` 预设里 body 永远沿车道直行，"
        "`heading` 转的是它**看起来**朝哪，不是它往哪开。想让一个东西真的沿某个"
        "方向走，那是 `field` 预设的抛物线该干的事，不是靠 `heading` 掰。"
        "**`arm` 是绕支点转的角色**：它不绕自己的中心转，而是绕本体底边的中点转，"
        "所以 `heading` 在它身上是一根摆动的臂，不是一次自转。"
        "要不要用 `arm` 看内容里有没有「绕轴摆动」这件事（机械臂、单摆、指针、杠杆）；"
        "只是一根静止的杆子，用 `object` 就好。"
        "**`emphasis` 是「这一拍强调它」**：给一个 body 选一个强调动作，它就会在这一拍"
        "开头自己动一下（涨缩、抖、晃、摆…），一晃就归位，不改变它在画面里的位置，"
        "也不影响别的对象。适合「注意看这个」「关键就在这里」这一类节拍——"
        "比只写 `highlights`（只是发一下光）强。"
        "**`speed` 只在预设 `lane` 里有意义**：其余预设的 body 不靠它移动"
        "（`chain`/`hub`/`generic` 不动，`field` 是按抛物线飞），写了不会让画面变好，"
        "只会让这个对象莫名其妙地飘出去",
    ),
    Primitive(
        name="emitter",
        roles=("sensor", "transmitter", "source"),
        props=("radius", "fov", "enabled"),
        relations=("of",),
        required_relations=("of",),
        live_props=("radius", "enabled"),
        note="扇形 + N 条射线，其中一条高亮到命中点；radius 是探测距离（激光雷达半径）。"
        "必须挂在某个 body 上——storyboard 里没有坐标，离开母体就无处安放",
    ),
    Primitive(
        name="zone",
        roles=("threshold", "safe_distance", "coverage"),
        props=("radius", "enabled"),
        relations=("of",),
        required_relations=("of",),
        live_props=("radius", "enabled"),
        note="挂在 body 上的虚线扇区/圆/环",
    ),
    Primitive(
        name="link",
        roles=("topic", "edge", "flow", "relation"),
        props=("active",),
        relations=("from", "to"),
        required_relations=("from", "to"),
        live_props=("active",),
        note="两个 body 之间的连线，虚线↔实线；两端都必填，否则这条线不存在",
    ),
    Primitive(
        name="traveler",
        roles=("message", "request", "unit"),
        props=("speed", "progress", "state"),
        relations=("along",),
        required_relations=("along",),
        # `state` is live because the token wears it as a caption. The
        # hand-written ROS sample walks one message through 待发布 → 已发布 →
        # 已接收, which is the *whole lesson* — and until the drawer read this
        # prop, three beats of a publish/subscribe diagram showed the same token.
        live_props=("speed", "progress", "state"),
        # "或 trace" used to be here and was not true: `_to_traveler` keys off
        # `_link_points`, so an `along` naming a trace raised and killed the
        # whole run — after the last LLM call, with nothing fed back. The note
        # said a thing the layout refused, which is the same defect as offering
        # `wave`: a name in the prompt is a promise.
        note="沿 link 移动的小令牌；没有 along 就不知道它沿什么走，"
        "而 along **必须指向一个 link**（指向 trace 会让布局整单失败）。"
        "state 是令牌上那行说明（待发布/已发布/已接收），节拍改它画面就变",
    ),
    Primitive(
        name="trace",
        roles=("trail", "trajectory", "path"),
        props=("length", "visible"),
        relations=("of",),
        required_relations=("of",),
        live_props=("length", "visible"),
        note="折线：body 的历史轨迹，或按公式算出的曲线；`of` 表明是谁的轨迹",
    ),
    Primitive(
        name="vector",
        roles=("velocity", "force", "displacement"),
        props=("magnitude", "direction"),
        relations=("of", "component_of"),
        required_relations=("of",),
        live_props=("magnitude", "direction"),
        # `direction` and not `magnitude`: a magnitude the model forgot still
        # draws an arrow of the right *shape* at a scene-relative length, which
        # is what `_vector_lengths` does with every magnitude anyway. A direction
        # it forgot draws a specific wrong answer, because the layout has to pick
        # some angle and 0 means rightwards. There is no harmless default here.
        required_props=("direction",),
        note="箭头；`of` 是它作用在谁身上（必填），component_of 表达分解"
        "（v0 → v0cosθ + v0sinθ），只有分解出来的分量才需要。"
        "**`direction` 必须写**：不写会画成一支朝右的箭头，而重力的箭头朝右，"
        "是在物理课上给出一个错误答案——不是难看，是错",
    ),
    Primitive(
        name="axis",
        roles=("x_axis", "y_axis"),
        props=("range", "ticks", "origin"),
        relations=(),
        # `origin` is not here: no drawer reads it. `range` drives the tick
        # labels, which is what turns an unlabelled arrow into a coordinate axis.
        live_props=("range", "ticks"),
        # `range` is the whole difference between an axis and a notched arrow.
        # Three of the three axes in the sample documents' descendants omitted
        # it, and nothing said so: the drawing was complete and meaningless.
        required_props=("range",),
        note="带刻度与标签的坐标轴，可成对交于原点。"
        "**没有 `range` 它就只是一根带刻痕的箭头**——刻度数字由 `range` 给出，"
        "缺了它画面上不会有任何东西报错，只是这根轴不再表示任何量",
    ),
    Primitive(
        name="dimension",
        roles=("range", "height", "span"),
        props=("label",),
        relations=("from", "to"),
        required_relations=("from", "to"),
        live_props=("label",),
        note="两端带箭头的标注线 + 标签（射程 R、最大高度 H）；"
        "from/to 是它量的那两个点，缺一个就画不出标注线",
    ),
    Primitive(
        name="angle",
        roles=("angle", "bearing"),
        props=("degrees", "radius"),
        relations=("of", "between"),
        required_relations=("of",),
        live_props=("degrees", "radius"),
        note="角弧 + 标注（发射角 θ）；`of` 是角的顶点所在的对象。"
        "`between`（夹角的两条边）可选——prop 的取值是标量，装不下两个 id，"
        "真要表达「哪两条边之间」得扩展取值类型",
    ),
    Primitive(
        name="region",
        # `span` belongs to `dimension` — a role must map to exactly one
        # primitive, or the lookup silently keeps only the last one.
        roles=("area", "band"),
        props=("opacity",),
        relations=("bounded_by",),
        required_relations=("bounded_by",),
        note="半透明阴影区域，必须声明被什么界定",
        drawable=False,
    ),
    Primitive(
        name="wave",
        roles=("waveform", "envelope", "harmonic"),
        props=("amplitude", "frequency", "phase"),
        relations=(),
        note="正弦/包络曲线，可叠加",
        drawable=False,
    ),
    Primitive(
        name="readout",
        # `code` used to be one of these, and it moved to the `code` primitive
        # below when that one landed. A role must map to exactly one primitive —
        # `ROLE_TO_PRIMITIVE` is a plain dict, so a duplicate would keep whichever
        # entry came last and drop the other in silence, which is the `span`
        # argument two paragraphs down told from the third side. The name was
        # never used by a sample, so nothing on disk changed meaning: it was
        # handed a plain 14px panel then and it is handed a tokenised, line-numbered
        # one now.
        roles=("hud", "decision", "caption", "formula"),
        props=("text", "align", "tone"),
        relations=(),
        # `align` is not here, for the reason `axis.origin` is not: it is settled
        # once, at layout time (`_align_of`), and `drawReadout` reads the baked
        # value straight off the element rather than asking `view.lookup`. A
        # panel that re-aligns itself between beats is also a worse picture than
        # one that does not. It stays in `props`, where asking for it once is a
        # real request — the hand-written ROS sample does exactly that.
        #
        # It was on this list, and that mattered: the prop is real, it is
        # registered, the prompt offered it as 可被节拍改变, so `step_state_inert`
        # waved it through by design. The gate's own table held the defect the
        # gate exists to catch, and the JS↔Python drift test could not see it,
        # because both files said the same wrong thing.
        live_props=("text", "tone"),
        note="文本面板；公式在这里是构建期渲好的 path，不是可解析文本",
    ),
    # The only primitive that is drawn **on** another object rather than at a
    # place of its own. Everything else here is either a thing (`body`) or an
    # annotation of a relation between things (`dimension` between two, `angle`
    # between two, `vector` pushing one). A verdict is a judgement of *one*, and
    # the judgement is the whole content — a ✓ floating with nothing under it
    # teaches nobody anything.
    #
    # Which is why `of` is required rather than optional. It is not a nicety of
    # the drawing; it is the only thing that says where the mark goes, so it is
    # the `required_relations` question asked exactly as that field defines it.
    # The precedent is `trace.of` and `vector.of`.
    Primitive(
        name="verdict",
        roles=("verdict",),
        props=("mark", "text"),
        relations=("of",),
        required_relations=("of",),
        required_props=("mark",),
        live_props=("mark", "text"),
        note=(
            "对错标记：把一个 ✓ / ✗ / ! 钉在另一个对象上，"
            "讲「这样是对的、那样是错的」；`of` 指向被判断的那个对象"
        ),
    ),
    # The picture for 「这几种值有什么不同」: one value set large, its type on a
    # tag beneath it, several of them in a row.
    #
    # A primitive of its own rather than a large `readout`, because the *pair* is
    # the content. `42` alone says nothing about types and `数字` alone says
    # nothing about values; the card exists to put the two side by side, and that
    # comparison is the lesson. Both halves are therefore required, for the same
    # reason `verdict.mark` is: a card drawing only one of them is a card
    # answering a question the scene did not ask, and nothing on screen says so.
    Primitive(
        name="card",
        roles=("value",),
        props=("text", "type"),
        relations=(),
        required_props=("text", "type"),
        live_props=("text", "type"),
        note="值卡片：大字显示一个具体的值，下面挂一个类型标签；讲「这几种值有什么不同」时用它",
    ),
    # 嵌套结构树 —— 「谁在谁里面」的那张图。
    #
    # The third picture here whose content is *text*, and the reason it is not a
    # taller `readout`: an outline's shape **is** its content. Which node is under
    # which is carried by indentation, so the drawing has to read the text as a
    # structure rather than as a string — handed the same eight lines, a `readout`
    # draws eight lines and a `tree` draws a hierarchy.
    #
    # **缩进大纲, not a node-link diagram**, and that is a measurement rather than
    # a taste. A tidy tree's width is proportional to its *leaves*: six of them at
    # a 160-pixel node is the whole 960-pixel stage, and the failure mode is not
    # "crowded" but "this scene cannot be rendered at all". An outline's width is
    # bounded by its longest line, whatever its depth. It also shares every metric
    # with `code` — same monospace stack, same padding, same `_block_boxes` — so
    # the two are one piece of work rather than two.
    Primitive(
        name="tree",
        roles=("outline", "structure", "directory"),
        props=("text", "form", "focus", "tone"),
        relations=(),
        required_props=("text",),
        # `text` is deliberately absent, and it is the one prop on this primitive
        # a reader would expect to find here. A block's box is cut for the text
        # that is in it, at layout time; a beat that swapped in a longer tree
        # would draw it out over its own panel, and `drawReadout`'s grow-to-fit
        # has no counterpart here because two rows of a tree are not one row of a
        # caption. Swapping the structure a lesson is about is a second scene.
        #
        # `form` is absent for a second reason on top of the same one: the five
        # forms do not merely draw the same box differently, they *size* it
        # differently — `branch` puts a whole chain on one row and `boxes` spends
        # height per level — so a beat that changed the form would be drawing
        # into a panel cut for a different picture.
        live_props=("focus", "tone"),
        note=(
            "结构树：把「谁在谁里面」画出来。`text` 是一段多行文本，"
            "**一层缩进 2 个空格**，越靠右就越在里面；"
            "一行写成 `键: 值` 会分开上色（键一个颜色，值按类型）；"
            "行首可以写一个 `[图标名]`，这一行前面就带个小图标，可用名字见下。"
            "讲 JSON、配置文件、目录结构、文章大纲这类「一层套一层」的内容时用它。"
            "`form` 决定画成哪一种，见下面的合法值——"
            "**同一个结构换个画法观感差很远，挑最贴合你要讲的那件事的那种**。"
            f"**最多 {BLOCK_ROWS_MAX} 行**，`text` 和 `form` 都只能整体设置一次"
        ),
    ),
    # 代码块. The other half of the same typography, and the one place in this
    # vocabulary where the *words themselves* are the subject rather than a label
    # about something else.
    #
    # Why the tokenising happens at layout time and not in the player: the spec
    # carries only numbers and strings (decision D4), and colours are the theme's
    # (decision D1). So the layout pass splits each line into spans and names what
    # each one *is*, and the drawer's whole job is to turn a name into a colour.
    # That is the same split as `tone` and `mark` — semantics settled above, one
    # lookup below — and it is the arrangement Shiki calls "zero runtime".
    Primitive(
        name="code",
        roles=("listing", "config", "command"),
        props=("text", "language", "focus", "tone"),
        relations=(),
        required_props=("text",),
        live_props=("focus", "tone"),
        note=(
            "代码块：等宽排版的一段源码，左边带行号，按 `language` 分词上色。"
            "`text` 是多行文本，**每一行都原样画出来**（不折行、不改写）。"
            '`focus` 是被强调的行号（`"3"` 强调第 3 行，`"2-4"` 强调第 2~4 行），'
            "被点到的行会垫一条底色带——讲「看这一行」的那一拍就靠它。"
            "`language` 只影响上色，见下面的合法值。"
            f"**最多 {BLOCK_ROWS_MAX} 行**，`text` 只能整体设置一次"
        ),
    ),
)

#: role -> primitive name. Built from T1 so the two can never disagree.
ROLE_TO_PRIMITIVE: dict[str, str] = {
    role: primitive.name for primitive in T1_PRIMITIVES for role in primitive.roles
}

PRIMITIVE_BY_NAME: dict[str, Primitive] = {p.name: p for p in T1_PRIMITIVES}

#: Roles that exist so an extracted object has somewhere to go, not because the
#: vocabulary has anything to say about what it looks like. See the note on
#: `body` for why `object` is one.
#:
#: Declared here rather than only described in that note because a diagnostic
#: outside this module now counts them: `tools/check_storyboard.py` reports what
#: share of a storyboard's cast is a fallback, which is the closest thing this
#: project has to a measurement of "the vocabulary had no word for what this
#: document is about". A tool spelling `"object"` itself would go on reporting
#: zero the day a second fallback role is added.
FALLBACK_ROLES: frozenset[str] = frozenset({"object"})

#: Every prop and relation name any T1 primitive declares. Used to tell a
#: misspelled prop apart from one that is real but belongs to another role.
KNOWN_OBJECT_PROPS: frozenset[str] = frozenset(
    prop for p in T1_PRIMITIVES for prop in (*p.props, *p.relations)
)

#: primitive name -> the relations it cannot be drawn without. Derived from the
#: declarations above, so the validator and the prompt cannot disagree with them.
REQUIRED_RELATIONS: dict[str, tuple[str, ...]] = {
    primitive.name: primitive.required_relations
    for primitive in T1_PRIMITIVES
    if primitive.required_relations
}

#: primitive name -> the props it cannot be *meaningfully drawn* without. The
#: companion to `REQUIRED_RELATIONS`, one question further out: that one is about
#: geometry that cannot be computed, this one about a drawing that computes fine
#: and teaches nothing.
REQUIRED_PROPS: dict[str, tuple[str, ...]] = {
    primitive.name: primitive.required_props
    for primitive in T1_PRIMITIVES
    if primitive.required_props
}

#: primitive name -> the props a *beat* may move. Derived, so `step_state_inert`
#: and the prompt's two prop groups cannot disagree with `Primitive.live_props`.
LIVE_PROPS_BY_PRIMITIVE: dict[str, tuple[str, ...]] = {
    primitive.name: primitive.live_props for primitive in T1_PRIMITIVES if primitive.live_props
}

#: Every prop that some drawing code reads during playback.
LIVE_PROPS: frozenset[str] = frozenset(
    prop for primitive in T1_PRIMITIVES for prop in primitive.live_props
)


@dataclass(frozen=True, slots=True)
class StageRange:
    """The values a prop has to fall in for the picture to still say something.

    `unit` is what the vocabulary prints next to the bounds, and it is the
    whole reason this is a record rather than a `(low, high)` pair. The first
    three entries are distances and their unit is 舞台像素; `speed` is not a
    distance — it is a multiplier on `PIXELS_PER_SPEED` (`behaviors.js`), and
    printing 舞台像素 next to it would be the same class of error this table
    exists to correct, told from the other side.

    `why` records where the bounds came from. A bound a reader cannot trace is
    a bound that gets "helpfully" widened by the next person.
    """

    low: float
    high: float
    unit: str
    why: str

    def clamp(self, value: float) -> float:
        return min(max(value, self.low), self.high)


#: Props whose value is **in the renderer's own unit**, with the range that unit
#: is legible over on a 960×600 stage.
#:
#: This is the unit declaration the vocabulary never had, and it is why
#: `lidar_emitter.radius: 8` drew an 8-pixel fan and `safe_zone.radius: 0.8` an
#: 0.8-pixel circle: the model wrote metres, layout read pixels, and no layer
#: disagreed out loud. The model is not wrong to reach for metres — a 激光雷达
#: radius of 8 *is* eight metres — it was never told the drawing has its own
#: unit. The baseline's hand-written sample writes `150` and `76`, which is the
#: same statement made by someone who had read the renderer.
#:
#: `speed` is the same defect at the same scale, and it is the one that reached
#: a real run: the three generated documents write `speed: 60`, straight from a
#: document that says 0.6 米每秒, into a prop the player multiplies by 90. The
#: frozen baseline says what the number means — `RobotCar.speed: float = 1.0`,
#: its slider `0.4~2.2`, and `PIXELS_PER_SPEED = 90` documented as "stage pixels
#: travelled per second at `speed === 1`". So the bounds below are not mine:
#: they are the baseline's own slider, read off `templates.py` and the ROS
#: sample. A body at 60 is 5400 px/s — the 960-pixel lane, start to wrap, in
#: under a fifth of a second.
#:
#: `layout.py` clamps into these ranges as a backstop, and the validator rejects
#: a value outside them so the model gets to fix it rather than shipping an
#: invisible circle or a car that is a blur. Only props with a unit are listed:
#: `trace.length` and `vector.magnitude` are normalised within their own scene
#: and carry no absolute meaning to state a range for.
STAGE_RANGES: dict[tuple[str, str], StageRange] = {
    ("emitter", "radius"): StageRange(
        40.0, 420.0, "舞台像素", "低于 40 画成一个点；高于 420 扇形出画布"
    ),
    ("zone", "radius"): StageRange(
        24.0, 260.0, "舞台像素", "低于 24 圈比车还小；高于 260 圈住整个车道"
    ),
    ("angle", "radius"): StageRange(20.0, 120.0, "舞台像素", "弧的半径，太小读不出角度"),
    ("body", "speed"): StageRange(
        0.4, 2.2, "倍速", "基线自己的滑杆 0.4~2.2（`templates.py` 车速）；1 倍速 = 90 像素/秒"
    ),
    ("traveler", "speed"): StageRange(0.2, 1.5, "倍速", "手写 ROS 样例的滑杆 0.2~1.5（消息速度）"),
}


def stage_range(primitive: str, prop: str) -> StageRange | None:
    """The legible range for `prop`, or None when it carries no renderer unit."""
    return STAGE_RANGES.get((primitive, prop))


#: `tone` -> the short gloss that stands in for it in the prop list.
#:
#: A vocabulary in its own right, and until now an unstated one. `readout.tone`
#: was a registered prop, was a registered *live* prop — a beat could change it
#: — and its five legal names were printed nowhere the model could read them.
#: `toneColor` in `primitives.js` answers a name it does not know with its
#: `default:` branch, so `tone: "warning"` drew the ordinary colour, looked
#: entirely deliberate, and the model never learned the word was `danger`.
#:
#: The same sentence as `STAGE_RANGES`, one paragraph up and about a different
#: kind of value: a prop the model may set, whose legal values it was never
#: told. That table cost a picture nobody could see; this one cost an emphasis
#: nobody could see, which is the cheaper half of the same defect.
#:
#: Kept short deliberately — it prints inline on the prop line, and a gloss
#: that wraps is read as a paragraph rather than as a label. A test holds this
#: list equal to the `Tone` literal in `models.py`, and another holds it equal
#: to the `switch` in `toneColor`, so the three cannot drift.
TONE_GLOSSES: tuple[tuple[str, str], ...] = (
    ("normal", "默认"),
    ("accent", "强调"),
    ("danger", "危险"),
    ("success", "通过"),
    ("muted", "次要"),
)

TONE_NAMES: tuple[str, ...] = tuple(name for name, _ in TONE_GLOSSES)


#: `emphasis` -> the gloss that stands in for it in the prop list.
#:
#: The same defect as `tone`, one prop over, and found by asking what `tone`
#: was for. `tone` answers "which colour", `emphasis` answers "which motion" —
#: and until now the answer to the second was "none", because the player had no
#: motion vocabulary at all beyond the two that the *spec* carries (`heading`
#: and `speed`). A beat could say "this object is the thing to look at" and the
#: player's whole reply was a glow (`applyHighlight` in `primitives.js`).
#:
#: These are words, on purpose, and they are the model's to choose. The
#: alternative — letting the beat write an amplitude — is the side door
#: `prompts.py` closes: 不要写任何坐标、像素尺寸、颜色、字号、时长、缓动. An
#: amplitude and a duration *are* the presentation of an accent, so the beat
#: gets the word and `emphasis.js` owns the curve, the size and the length.
#:
#: `none` is listed rather than left implicit so that a later beat can *switch
#: an emphasis off*, which is the reason `tone` has `normal`. An absent prop and
#: an explicit `none` mean the same thing to the player; only one of them can be
#: written down.
EMPHASIS_GLOSSES: tuple[tuple[str, str], ...] = (
    ("none", "不强调"),
    ("pulse", "脉动：原地涨缩一下"),
    ("shake", "抖动：左右快速晃几下"),
    ("wobble", "摇晃：原地转着晃几下"),
    ("swing", "摆动：绕支点荡开再回来"),
    ("pop", "弹一下：涨大并微转，带回弹"),
    ("spring", "弹性：阻尼回弹，收尾带余震"),
)

EMPHASIS_NAMES: tuple[str, ...] = tuple(name for name, _ in EMPHASIS_GLOSSES)


#: The gloss that stands in for `vector.direction` in the prop list.
#:
#: The third instance of the same defect, and the one with the worst symptom so
#: far. A misspelled `tone` still draws, in the wrong colour; a misspelled
#: `emphasis` draws nothing. A *missing* `direction` draws a wrong answer — and
#: the answer it draws is always the same one, because `_number(obj, "direction",
#: 0.0)` turns "the model did not say" into 0, and 0 is rightwards.
#:
#: Measured, not supposed. Four earlier runs of `data/samples/projectile_motion.md`
#: wrote `direction` on every vector — `gravity: -90`, `vx: 0`, `vy: 90`,
#: `v0: 45` — and the fifth wrote none at all, because nothing in the vocabulary
#: had ever said the prop was there to write. The picture that came out had a
#: gravity arrow pointing off the right-hand edge of a physics lesson.
#:
#: The two numbers that are worth spelling out are the two the convention makes
#: non-obvious: `0` is horizontal-right and `90` is straight *up*, which is the
#: opposite of the canvas's y-down sense (`_to_canvas_angle` exists to reconcile
#: them), and gravity is `-90`. A gloss prints inline on the prop line, so the
#: rest of the convention is left to the words.
DIRECTION_GLOSS = "方向角：0 向右、90 向上、-90 向下（重力写 -90）"


#: `mark` -> the gloss that stands in for it in the prop list.
#:
#: The same shape again, and the one where the wrong word costs the most. `tone`
#: misspelled still draws; `emphasis` misspelled draws nothing; a `mark`
#: misspelled would draw the *opposite* — a ✗ is not an absent ✓, it is a claim
#: that the thing is wrong, and a lesson that says so is worse than one that says
#: nothing. So the three names are printed, and `unknown_mark` refuses anything
#: else while there is still a model to retry.
#:
#: `warn` is here rather than left out so that 「说不准」 is sayable. The
#: alternative is a model forced to pick a side it cannot defend, which is how a
#: two-valued vocabulary starts lying.
MARK_GLOSSES: tuple[tuple[str, str], ...] = (
    ("ok", "对：画一个绿色的勾"),
    ("bad", "错：画一个红色的叉"),
    ("warn", "存疑：画一个黄色的感叹号"),
)

MARK_NAMES: tuple[str, ...] = tuple(name for name, _ in MARK_GLOSSES)


#: `type` -> the gloss that stands in for it in the prop list.
#:
#: The six kinds of value a lesson about 数据类型 actually compares, and the
#: list is the whole point of the primitive: `card` exists to put six of these
#: side by side, and a card whose `type` nobody can read is a `readout` in a
#: bigger font. Printed for the reason the two above are.
#:
#: The names are the words a document about JSON already uses, not the Chinese
#: gloss — the gloss is what the *tag* draws, the name is what the model writes.
TYPE_GLOSSES: tuple[tuple[str, str], ...] = (
    ("string", "字符串"),
    ("number", "数字"),
    ("boolean", "布尔"),
    ("null", "空值"),
    ("object", "对象"),
    ("array", "数组"),
)

TYPE_NAMES: tuple[str, ...] = tuple(name for name, _ in TYPE_GLOSSES)


#: `language` -> the gloss that stands in for it in the prop list.
#:
#: A closed set of three, and the smallness is the point rather than a limitation
#: to apologise for. The layout pass tokenises with the standard library's own
#: `tokenize`, which knows exactly one language; `json` is the same tokeniser with
#: `true`/`false`/`null` read as literals, because JSON's punctuation happens to
#: be Python's. Nothing else can be coloured, and a `java` that silently drew as
#: plain text would be the `tone` defect again — a name the vocabulary offered,
#: whose effect was never wired up.
#:
#: So a lesson about another language writes `text` and gets its code drawn
#: uncoloured, which is what it would have got anyway, and the refusal says so
#: while there is still a model to retry.
CODE_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("python", "Python 代码：关键字、内置函数、字符串、数字、注释各自上色"),
    ("json", "JSON 片段：字符串、数字、true/false/null 各自上色"),
    ("text", "别的语言一律选它：原样画出来，不上色"),
)

CODE_LANGUAGE_NAMES: tuple[str, ...] = tuple(name for name, _ in CODE_LANGUAGES)


#: How a `tree` may be drawn, as a closed set with the reason each one exists.
#:
#: Five, and they are not five coats on one drawing. They answer two different
#: questions about the same rows, and which one a lesson picks is a claim about
#: what it is teaching:
#:
#: - `outline` steps depth along x one notch at a time, so the picture stays a
#:   *list*: it reads top to bottom, and a chain of eight nodes costs eight rows.
#: - `branch` puts every node of one depth in the same column and lets depth run
#:   to the right, so a chain of eight costs **one** row. That is the case the
#:   outline is worst at, and a real config file is usually exactly that shape.
#: - `mind` is `branch` in different clothes — rounded nodes and curving branches
#:   instead of right-angle connectors. Same geometry, so choosing between them
#:   is a question about how the content reads, not about what fits.
#: - `brace` keeps the outline's rows and wraps each run of siblings in a curly
#:   brace. It is the one form that *draws* 「这几个是一伙的」.
#: - `boxes` frames each level so its contents sit inside it — containment as
#:   geometry. The most direct picture of the idea, and the one with the tightest
#:   ceiling: a level costs about 36px of height, so four of them is most of the
#:   frame before a single row is drawn.
#:
#: What they have in common is the constraint that makes them a closed set:
#: **siblings go on `y` and depth goes on `x`**, so width is a function of how
#: deep the content goes and never of how much of it there is. An earlier round
#: rejected the node-link tree outright on the measurement that six leaves at
#: 160px is the whole 960 stage. That measurement is right for spreading leaves
#: across the available width and wrong for fixing each depth to a column — and
#: the stage is 960 wide with a block frame narrower still, so the distinction is
#: the difference between a form that cannot be drawn and one that can.
TREE_FORMS: tuple[tuple[str, str], ...] = (
    ("outline", "缩进大纲：一行一个节点，越深越靠右。最能装，也最像一页清单"),
    ("branch", "横向树：根在左边，一层往右走一格，节点之间连线。又深又窄的东西用它"),
    ("mind", "思维导图：和横向树同一个摆法，但节点是圆角块、分支是曲线。讲发散用它"),
    ("brace", "花括号分组：还是竖排的行，但同一层的兄弟被一个花括号括在一起"),
    ("boxes", "套盒子：每一层画成一个框，东西真装在框里。层数别多，四层就到顶"),
)

TREE_FORM_NAMES: tuple[str, ...] = tuple(name for name, _ in TREE_FORMS)


#: The icons a `tree` row may wear, as a closed set of **glyph names**.
#:
#: Written inside `text`, as a leading `[名字]` on the row:
#:
#:     器材
#:       [flask] 小球
#:       [clock] 计时器
#:
#: The names are the glyph names in `assets/glyphs/`, and that is the point
#: rather than laziness: a friendlier vocabulary (`folder`, `file`) would be a
#: second name for the same drawing plus a mapping to keep in step with it. The
#: set is small on purpose — it goes into the prompt, and most of the other
#: eighty-odd glyphs draw *things* (a car, a magnet, a wind turbine) rather than
#: marking a row of a structure.
#:
#: Why a token in `text` rather than a prop: `text` is one blob and the icons are
#: per row, so a prop would have to be a parallel list the model keeps in step
#: with the text it just wrote. That is a second thing to get wrong, and it gets
#: wrong in the worst way — every icon after the mistake lands one row off. A
#: leading token cannot drift, because it *is* the row.
#:
#: A row that opens with `[` and a name is read as an icon or refused; anything
#: else is text. That is what keeps a JSON line like `["a", "b"]` ordinary — the
#: character after its bracket is a quote, not a name.
TREE_ICONS: tuple[tuple[str, str], ...] = (
    ("container", "一个容器／文件夹"),
    ("package", "一个包"),
    ("sitemap", "一张结构图"),
    ("hierarchy", "一层套一层的关系"),
    ("database", "数据库"),
    ("report", "一份文档"),
    ("list-check", "一张清单"),
    ("clipboard-list", "一份记录"),
    ("network", "网络"),
    ("server", "服务器"),
    ("gear", "一项设置"),
    ("flask", "一次实验"),
    ("clock", "和时间有关"),
    ("warning", "需要注意的"),
)

TREE_ICON_NAMES: tuple[str, ...] = tuple(name for name, _ in TREE_ICONS)


#: The gloss that stands in for `focus` in the prop list.
#:
#: A *line range* written as a string, and the one prop in the vocabulary whose
#: legal values cannot be listed — they depend on how many lines the object's own
#: `text` has. So the shape is printed here and the bound is checked against the
#: text in `storyboard/validation.py` (`focus_out_of_range`), which is the same
#: division of labour as `STAGE_RANGES`: this module says what a legal value looks
#: like, the validator says whether *this* one is legal.
#:
#: Both block primitives carry it — a tree is walked through one node per beat
#: exactly as a listing is walked through one line — so it is glossed here, once.
FOCUS_GLOSS = '要强调的行：一个数（`"3"`）或一个区间（`"2-4"`），从 1 开始数'


#: The kinds a `CodeSpan` may be given, as a closed set.
#:
#: The names are Prism's, because the grouping they imply is the one that keeps a
#: syntax palette down to five colours: comment, keyword (and `literal`, which is
#: `true`/`false`/`null` — keywords to any reader), builtin, string, number. The
#: rest — `name`, `operator`, `text` — are deliberately *not* given colours of
#: their own: an operator is punctuation and punctuation in a sixth hue is noise,
#: which is the call VS Code's own defaults make. They exist in the set anyway
#: because the tokeniser produces them and a spec should say what it holds; the
#: drawer's `default:` is the ordinary text colour.
#:
#: Declared here rather than only in `models.py` so that the vocabulary's own
#: module can be the one place the set is listed. `CodeSpan.kind` is typed against
#: it and `codeColor` in `primitives.js` mirrors it by hand, held equal by a test —
#: the same trade `LIVE_PROPS` and `PENDING_KINDS` make.
CODE_KINDS: tuple[str, ...] = (
    "text",
    "name",
    "keyword",
    "builtin",
    "string",
    "number",
    "comment",
    "operator",
    "literal",
)


#: The two halves of `drawable`, derived so there is nowhere for a third answer
#: to hide. `PENDING_PRIMITIVES` is what the prompt excludes and the validator
#: rejects; `DRAWABLE_PRIMITIVES` is what `_to_element` must actually place.
#:
#: Roles are the thing the model writes, so this is also the set the validator
#: turns into role names — see `PENDING_ROLES`.
PENDING_PRIMITIVES: frozenset[str] = frozenset(
    primitive.name for primitive in T1_PRIMITIVES if not primitive.drawable
)
DRAWABLE_PRIMITIVES: frozenset[str] = frozenset(
    primitive.name for primitive in T1_PRIMITIVES if primitive.drawable
)

#: Roles whose primitive has no layout yet. Written as a set of *roles* because
#: the validator compares against `obj.role`, and because a role is what the
#: model has to change — the fix is `zone`, not "stop using primitives".
PENDING_ROLES: frozenset[str] = frozenset(
    role for primitive in T1_PRIMITIVES if not primitive.drawable for role in primitive.roles
)

#: Where a model that reached for a pending role should go instead. Kept beside
#: the pending set so the actionable half of the error cannot drift away from
#: the name it is about; a test asserts every pending primitive has a suggestion.
PENDING_ALTERNATIVES: dict[str, str] = {
    "region": "`zone`（挂在 body 上的虚线范围）或 `dimension`（带箭头的尺寸标注）",
    "wave": "`trace`（一条按算法采样出来的折线——它同样能画正弦）",
}


# --------------------------------------------------------------------------
# T2 — domain glyphs: data, not drawing functions
# --------------------------------------------------------------------------


class GlyphPart(BaseModel):
    """One named sub-shape of a glyph.

    `mode` and `fill_rule` are **required** and travel with the geometry, not
    with the theme: `Path2D` carries geometry only, so a stroke-only icon drawn
    with `ctx.fill()` is a solid black blob, and a subpath-drawn hole (wheel hub,
    vent) fills in when `evenodd` is dropped. Both fail *silently*.

    Colour is deliberately absent — it belongs to the theme layer, because
    `body.danger` flips at playback time and a frozen colour cannot follow it.
    `stroke_width` is only a default the theme may override.
    """

    model_config = ConfigDict(extra="forbid")

    d: str = Field(min_length=1)
    mode: Literal["stroke", "fill"]
    fill_rule: Literal["nonzero", "evenodd"] = "nonzero"
    stroke_width: float | None = Field(default=None, gt=0)
    #: Rotation centre in view-box coordinates, for parts that turn.
    anchor: tuple[float, float] | None = None
    spin: bool = False


class Glyph(BaseModel):
    """A domain glyph: named parts, each a path string in a normalised view box.

    `parts` is flat and one level deep on purpose. Nesting would force recursive
    transforms and a tree-walking loader that no target domain needs; the only
    reason `parts` exists at all is so a wheel can turn independently of the car
    body (see `docs/storyboard-milestone.md`, decision D2).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=32)
    view_box: tuple[float, float, float, float]
    #: The box around what the glyph actually draws, `[x, y, width, height]` in
    #: view-box units — the same shape as `view_box`, and not the same thing.
    #: `view_box` is the grid the icon set designed on and it is *padded*; fitting
    #: it draws the glyph smaller than the body that asked for it, by however much
    #: the set happens to pad. Required, and computed by `tools/build_glyphs.py`
    #: rather than declared, because a hand-typed number would go stale silently
    #: the moment the pinned upstream version moved the art.
    ink_box: tuple[float, float, float, float]
    domain: str = Field(min_length=1, max_length=24)
    #: Provenance, e.g. "tabler:car". Assets are build-time products of an icon
    #: set plus a normalisation script; hand-edited `d` strings are not allowed.
    source: str = Field(min_length=1, max_length=64)
    parts: dict[str, GlyphPart] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class GlyphDeclaration:
    """A glyph this project intends to ship, declared before its data exists."""

    name: str
    domain: str
    source: str
    note: str


#: Every name here has data in `assets/glyphs/`, and that is asserted rather
#: than intended: `tests/unit/test_render_glyphs.py` builds the set from the
#: directory and requires it to equal this one. The two can drift in a way that
#: nothing else would notice — `layout._glyph_to_draw` drops a requested glyph
#: it has no data for and draws the parametric shape instead, so a declared name
#: with no file is a promise to the model that the picture quietly breaks.
#:
#: Keep this small. Anything composable from T1 should be composed, not drawn —
#: the rule that used to be a comment here and is now enforced is that a glyph
#: earns its place by being a *shape*, and a regular polygon computed from two
#: numbers is not one. `obstacle_octagon` was declared on exactly that mistake:
#: its source was `hand:octagon`, and both `layout.py:162` and `models.py:99`
#: record the opposite decision (D1) — the baseline's obstacle is 8 vertices
#: alternating between `r` and `0.78r`, parametric on purpose.
#: `note` is the whole offer. It is printed to the model word for word, so it has
#: to say what the icon *is* and why a lesson would reach for it — the list used
#: to print bare names, and the real chain asked for a glyph zero times across
#: three documents while eleven written notes sat unread right here.
T2_GLYPHS: tuple[GlyphDeclaration, ...] = (
    # -- 机器人 ------------------------------------------------------------
    GlyphDeclaration(
        "car",
        "robotics",
        "tabler:car",
        "避障小车、移动平台的车身；车头朝右，`heading` 是从右开始转的",
    ),
    GlyphDeclaration(
        "lidar",
        "robotics",
        "tabler:radar",
        "激光雷达本体；扫描臂会自己转，讲「雷达在扫」的那一拍用它最省事",
    ),
    GlyphDeclaration(
        "robot",
        "robotics",
        "tabler:robot",
        "移动机器人本体；画人形或整机平台时用它",
    ),
    GlyphDeclaration("cpu", "robotics", "tabler:cpu", "控制器、计算节点、ROS 节点"),
    GlyphDeclaration("camera", "robotics", "tabler:camera", "摄像头、视觉传感器"),
    GlyphDeclaration("scan", "robotics", "tabler:scan", "扫描线、正在扫描的传感器"),
    GlyphDeclaration("gauge", "robotics", "tabler:gauge", "仪表、读数盘、被实时测量的量"),
    GlyphDeclaration("temperature", "robotics", "tabler:temperature", "温度传感器、温度读数"),
    # -- 载具 --------------------------------------------------------------
    GlyphDeclaration("drone", "robotics", "tabler:drone", "无人机、飞行平台"),
    GlyphDeclaration("truck", "robotics", "tabler:truck", "卡车、地面运输车"),
    GlyphDeclaration("plane", "robotics", "tabler:plane", "飞机、飞行器"),
    GlyphDeclaration("ship", "robotics", "tabler:ship", "船、水上载具"),
    # -- 网络与通信 --------------------------------------------------------
    GlyphDeclaration("antenna", "network", "tabler:antenna", "天线、无线收发端"),
    GlyphDeclaration("wifi", "network", "tabler:wifi", "WiFi 链路、一段无线连接"),
    GlyphDeclaration(
        "broadcast",
        "network",
        "tabler:broadcast",
        "广播、一对多发送；ROS 的 topic 广播用它",
    ),
    GlyphDeclaration("router", "network", "tabler:router", "路由器、网关"),
    GlyphDeclaration("network", "network", "tabler:network", "网络拓扑整体；单个节点用 cpu/server"),
    GlyphDeclaration("cloud", "cloud", "tabler:cloud", "云端服务、远端的算力"),
    GlyphDeclaration("server", "cloud", "tabler:server", "服务端、云端服务"),
    GlyphDeclaration("database", "cloud", "tabler:database", "数据库、存储"),
    GlyphDeclaration("package", "cloud", "tabler:package", "消息载荷、传输中的数据包"),
    GlyphDeclaration("brain", "cloud", "tabler:brain", "大脑、模型、智能体；讲 AI 或推理时用它"),
    GlyphDeclaration("chart", "cloud", "tabler:chart-line", "折线图、走势、数据在变化"),
    GlyphDeclaration("sitemap", "cloud", "tabler:sitemap", "层级结构图；讲「谁属于谁」时用它"),
    # -- 电与物理 ----------------------------------------------------------
    GlyphDeclaration("battery", "physics", "tabler:battery", "电池、储能单元"),
    GlyphDeclaration("bolt", "physics", "tabler:bolt", "闪电、放电；讲电能或能量释放"),
    GlyphDeclaration("magnet", "physics", "tabler:magnet", "磁铁、磁场源"),
    GlyphDeclaration("sine-wave", "physics", "tabler:wave-sine", "正弦波；讲波动、交流、简谐运动"),
    GlyphDeclaration("square-wave", "physics", "tabler:wave-square", "方波、数字信号、高低电平"),
    GlyphDeclaration("activity", "physics", "tabler:activity", "波形曲线、跳动中的量、生命体征"),
    GlyphDeclaration("propeller", "physics", "tabler:propeller", "螺旋桨、旋翼、推力"),
    GlyphDeclaration("wind", "physics", "tabler:wind", "风、气流"),
    # -- 科学与航天 --------------------------------------------------------
    GlyphDeclaration("atom", "science", "tabler:atom", "原子、分子结构"),
    GlyphDeclaration("rocket", "science", "tabler:rocket", "火箭、推进器"),
    GlyphDeclaration("satellite", "science", "tabler:satellite", "卫星、中继"),
    GlyphDeclaration("planet", "science", "tabler:planet", "星球、天体"),
    GlyphDeclaration("microscope", "science", "tabler:microscope", "显微镜、微观观测"),
    GlyphDeclaration("flask", "science", "tabler:flask", "烧瓶、实验器皿"),
    GlyphDeclaration("telescope", "science", "tabler:telescope", "望远镜、天文观测"),
    # -- 机械 --------------------------------------------------------------
    GlyphDeclaration("gear", "robotics", "tabler:settings", "齿轮、传动；讲动力怎么传下去"),
    GlyphDeclaration("wrench", "robotics", "tabler:tool", "工具、扳手；讲装配或维修"),
    # -- 状态与指示 --------------------------------------------------------
    GlyphDeclaration(
        "warning", "general", "tabler:alert-triangle", "警告标志、危险源、要避开的障碍"
    ),
    # The two halves of 对错对照, and the pair the `verdict` primitive draws. Both
    # are bare strokes on the same 24-grid as everything else — no enclosing
    # circle, because `drawVerdict` draws that itself in the semantic colour, and
    # a glyph with the circle baked in could not be recoloured.
    #
    # They are two glyphs rather than one because the *shape* is half of how a
    # verdict is read. WCAG 2.2 SC 1.4.1 (A) forbids colour as the only channel,
    # and a tick and a cross differ in shape before they differ in hue — so a
    # colour-blind viewer reads the same answer off the same picture.
    GlyphDeclaration("check", "general", "tabler:check", "表示「对、合法、通过」的对勾"),
    GlyphDeclaration("x", "general", "tabler:x", "表示「错、非法、不通过」的叉号"),
    GlyphDeclaration("lock", "general", "tabler:lock", "锁、加密、权限、被保护"),
    GlyphDeclaration("clock", "general", "tabler:clock", "时钟；讲时序、周期、延迟"),
    GlyphDeclaration("shield", "general", "tabler:shield", "防护、安全边界、容错"),
    # -- 工业制造 ----------------------------------------------------------
    # The round this set was short of. A real document about a sorting line named
    # six things and only two of them had a picture — and the model, told to draw
    # 六轴机械臂 with no arm to draw, reached for `robot` and put a humanoid on
    # the stage. `note` is the whole offer, so each of these says what the icon
    # *is* and when a lesson would reach for it, the same as the eleven above it
    # that went unread until they were written this way.
    GlyphDeclaration(
        "robot-arm",
        "robotics",
        "lucide:robot-arm",
        "六轴机械臂本体：底座加一节节的手臂；讲机械臂、关节、抓取时用它。"
        "**不是人形机器人**——那是 `robot`",
    ),
    GlyphDeclaration(
        "photo-sensor",
        "robotics",
        "tabler:photo-sensor",
        "光电传感器；讲到位检测、遮光触发、有无料",
    ),
    GlyphDeclaration(
        "assembly", "robotics", "tabler:assembly", "装配好的整机、成品件；讲组装与成品"
    ),
    GlyphDeclaration(
        "building-factory", "robotics", "tabler:building-factory", "厂房、生产车间、产线所在的建筑"
    ),
    GlyphDeclaration(
        "building-warehouse", "robotics", "tabler:building-warehouse", "仓库、立体库、存放区"
    ),
    GlyphDeclaration("forklift", "robotics", "tabler:forklift", "叉车、厂内搬运车；讲物流与搬运"),
    GlyphDeclaration("crane", "robotics", "tabler:crane", "起重机、吊装设备；讲吊运重物"),
    GlyphDeclaration(
        "container", "robotics", "tabler:container", "料箱、周转箱、集装箱；装工件或货物的容器"
    ),
    GlyphDeclaration(
        "circuit-motor", "robotics", "tabler:circuit-motor", "电机、马达；讲驱动与转动"
    ),
    GlyphDeclaration("engine", "robotics", "tabler:engine", "发动机、动力机；讲动力从哪里来"),
    GlyphDeclaration("drill", "robotics", "lucide:drill", "钻头、打孔加工；讲钻孔与切削"),
    GlyphDeclaration("hard-hat", "robotics", "lucide:hard-hat", "安全帽；讲作业防护与安全规范"),
    GlyphDeclaration(
        "cuboid", "robotics", "lucide:cuboid", "立方体、方块工件；讲一个具体的三维方料"
    ),
    GlyphDeclaration(
        "cylinder", "robotics", "lucide:cylinder", "圆柱体、圆柱形工件；讲轴、卷材、柱状件"
    ),
    # -- 通用教学 ----------------------------------------------------------
    GlyphDeclaration("hierarchy", "general", "tabler:hierarchy", "层级结构；讲上下级与隶属关系"),
    GlyphDeclaration("list-check", "general", "tabler:list-check", "带勾的清单；讲核对、逐条确认"),
    GlyphDeclaration("report", "general", "tabler:report", "报告、书面记录；讲汇报与结论"),
    GlyphDeclaration("license", "general", "tabler:license", "许可证、资质证书；讲准入与认证"),
    GlyphDeclaration("target", "general", "tabler:target", "靶心、目标；讲瞄准与达标"),
    GlyphDeclaration("route", "general", "tabler:route", "路线、路径规划；讲途经顺序"),
    GlyphDeclaration("scale", "general", "tabler:scale", "天平、秤；讲权衡、配比、称量"),
    GlyphDeclaration("compass", "general", "tabler:compass", "指南针；讲方向与定位"),
    GlyphDeclaration("milestone", "general", "lucide:milestone", "里程碑；讲阶段节点与进度"),
    GlyphDeclaration(
        "waypoints", "general", "lucide:waypoints", "途经点、节点序列；讲多步流程里的一个个停靠点"
    ),
    GlyphDeclaration(
        "clipboard-list", "general", "lucide:clipboard-list", "核对板、操作单；讲按单执行的步骤"
    ),
    GlyphDeclaration(
        "pencil-ruler", "general", "lucide:pencil-ruler", "尺规制图；讲设计、画图、量取尺寸"
    ),
    # -- 电子电路 ----------------------------------------------------------
    GlyphDeclaration("circuit-resistor", "physics", "tabler:circuit-resistor", "电阻元件"),
    GlyphDeclaration("circuit-capacitor", "physics", "tabler:circuit-capacitor", "电容元件"),
    GlyphDeclaration(
        "circuit-diode", "physics", "tabler:circuit-diode", "二极管；讲单向导通与整流"
    ),
    GlyphDeclaration("circuit-inductor", "physics", "tabler:circuit-inductor", "电感、线圈"),
    GlyphDeclaration("circuit-ammeter", "physics", "tabler:circuit-ammeter", "电流表；讲测量电流"),
    GlyphDeclaration(
        "circuit-voltmeter", "physics", "tabler:circuit-voltmeter", "电压表；讲测量电压"
    ),
    GlyphDeclaration(
        "circuit-switch-closed",
        "physics",
        "tabler:circuit-switch-closed",
        "闭合的开关；讲通断与控制回路",
    ),
    GlyphDeclaration(
        "circuit-ground", "physics", "tabler:circuit-ground", "接地；讲零电位与参考点"
    ),
    GlyphDeclaration("circuit-bulb", "physics", "tabler:circuit-bulb", "灯泡、用电器；讲负载"),
    GlyphDeclaration("plug", "physics", "tabler:plug", "插头、取电口；讲供电接入"),
    GlyphDeclaration(
        "circuit-board", "physics", "lucide:circuit-board", "电路板；讲整块板子与元器件集成"
    ),
    GlyphDeclaration("cable", "physics", "lucide:cable", "线缆、走线；讲连线与布线"),
    # -- 化工与能源 --------------------------------------------------------
    GlyphDeclaration("pipeline", "physics", "tabler:pipeline", "管道；讲输送与管路走向"),
    GlyphDeclaration("tank", "physics", "tabler:tank", "储罐、罐体；讲存放液体或气体"),
    GlyphDeclaration("barrel", "physics", "tabler:barrel", "油桶、桶装物料"),
    GlyphDeclaration(
        "building-wind-turbine", "physics", "tabler:building-wind-turbine", "风力发电机；讲风能发电"
    ),
    GlyphDeclaration("solar-panel", "physics", "tabler:solar-panel", "光伏板、太阳能电池板"),
    GlyphDeclaration("droplet", "physics", "tabler:droplet", "液滴、液体本身；讲流量与滴加"),
    GlyphDeclaration("flame", "physics", "tabler:flame", "火焰；讲燃烧、加热、高温"),
    GlyphDeclaration("recycle", "physics", "tabler:recycle", "循环、回收；讲闭环与再利用"),
    GlyphDeclaration("leaf", "general", "tabler:leaf", "叶子；讲环保、生态、自然"),
    GlyphDeclaration("beaker", "science", "lucide:beaker", "烧杯；讲化学实验、溶液、反应"),
    GlyphDeclaration("dna", "science", "lucide:dna", "DNA 双螺旋；讲分子、遗传、微观结构"),
)

T2_GLYPH_NAMES: frozenset[str] = frozenset(g.name for g in T2_GLYPHS)


# --------------------------------------------------------------------------
# Behaviours — what makes the picture move without another LLM call
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Behavior:
    """A deterministic local simulation step.

    `reads` is what makes "interactive parameters must change the result"
    machine-checkable: a control whose target no behaviour reads is rejected.
    """

    name: str
    reads: tuple[str, ...]
    note: str


BEHAVIORS: tuple[Behavior, ...] = (
    Behavior("linear_motion", ("speed", "heading"), "小车沿当前朝向匀速前进（lane）"),
    Behavior(
        "ballistic",
        ("speed", "heading"),
        "抛体沿布局算好的抛物线飞行（field）——弹道由代码烘好，播放期只做插值",
    ),
    Behavior(
        "proximity_gate",
        ("radius", "safe_distance"),
        "最近距离低于安全距离 → 置 danger，触发避障态",
    ),
    Behavior("clearance_choice", ("radius",), "比较左右空旷度，选择更空旷的一侧转向"),
    Behavior("hop_along_path", ("speed",), "令牌沿 link/trace 匀速移动"),
)

BEHAVIOR_BY_NAME: dict[str, Behavior] = {b.name: b for b in BEHAVIORS}

#: Props a control may bind to. A prop that nothing reads is an inert knob.
#: `safe_distance` is listed as a *prop* rather than read off a `zone` object
#: because the avoidance baseline exposes it as `scene.safe_distance` — the
#: threshold is a scene-level parameter, not a property of the circle drawn for
#: it. Both sliders in the baseline (`lidar.radius`, `scene.safe_distance`)
#: therefore resolve, and they stay distinct quantities.
CONSUMABLE_PROPS: frozenset[str] = frozenset(prop for b in BEHAVIORS for prop in b.reads)


#: What a `button` control is allowed to do. Buttons carry an action instead of
#: a property target: `reset_scene` restarts the timeline, it does not set a
#: value, so requiring `<object>.<prop>` on it would reject a valid control.
BUTTON_ACTIONS: tuple[str, ...] = (
    "reset_scene",
    "advance_timeline",
    "toggle_play",
)


# --------------------------------------------------------------------------
# Presets — named arrangements of primitives
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Preset:
    """A named slot layout. `required_roles` must be present or the scene fails.

    This is what stops a mismatched lesson from being forced into the avoidance
    scene: a preset that cannot be satisfied is rejected, not rendered wrong.
    """

    name: str
    required_roles: tuple[str, ...]
    max_bodies: int
    note: str

    #: The roles a *body* here may have. Empty means "any".
    #:
    #: `required_roles` says what the preset cannot be built without. This says
    #: what it has nowhere to put — the same question from the other side, and
    #: the side that produced a real wrong picture. A `field` scene's bodies
    #: were launched if they were not obstacles, so a 抛出点 and a 最高点 roled
    #: `object` each flew off along their own parabola, taking the two
    #: `dimension`s anchored to them with it.
    #:
    #: Declared here rather than only refused by `_field_boxes` because layout
    #: runs after the last model call: a refusal there is terminal, with nothing
    #: fed back and nothing retried. Here it is one retry and a better scene.
    body_roles: tuple[str, ...] = ()


PRESETS: tuple[Preset, ...] = (
    Preset("lane", ("vehicle", "obstacle"), 6, "一维通道上的避障：车、障碍、安全距离"),
    Preset("chain", ("node",), 8, "有向链路：ROS 节点 / API 调用链"),
    Preset("hub", ("node", "endpoint"), 8, "星形拓扑：一个中心与若干叶子"),
    Preset(
        "field",
        ("projectile",),
        4,
        "二维场：坐标轴 + 抛体轨迹 + 矢量分解。**只有 `projectile` 会被射出**——"
        "「抛出点」「最高点」这类静止标记它还没有位置可给（要落在轨迹或坐标轴上，"
        "得从抛物线推出来），写了会被打回",
        body_roles=("projectile",),
    ),
    # The preset `card` exists for, and the one that gives a document about data
    # types a `scene_type` to pick that is not `generic`. That matters more here
    # than the layout does: a row of cards is placed by `_card_boxes` under every
    # preset, so the *picture* was never the thing `generic` was costing. What it
    # was costing is a name — a scene whose only legal answer was 「竖排对象」,
    # which is exactly the reading `tools/check_storyboard.py` reports as
    # 「除了 `generic` 没有别的预设可选」.
    Preset(
        "values",
        ("value",),
        6,
        "并列对比：几张值卡片横排，每张是一个值加它的类型；"
        "讲「这几种值有什么不同」「哪种写法合法」时用它",
    ),
    # Described as the last resort, not as the safe default. It used to read
    # "永远合法的兜底预设", and the model — optimising for "don't get rejected" —
    # picked it for 3 of 5 scenes that `field` could have served. Nothing was
    # wrong with the model's output; the incentive in the wording was pointing
    # the wrong way. See `docs/storyboard-milestone.md`.
    Preset("generic", (), 8, "竖排对象 + 说明面板；只在没有任何更具体的预设可用时才选它"),
)

PRESET_BY_NAME: dict[str, Preset] = {p.name: p for p in PRESETS}

#: The presets whose bodies travel by `speed` — one of the five.
#:
#: `_flight` already says this in its own words: "`lane` carries a car along a
#: corridor, `chain` and `hub` hold their bodies still". `field` moves its
#: bodies too, but along a parabola this module baked, and it never reads
#: `speed` — a projectile's flight time is the arc's property, which is the
#: decision that put `duration` on the body in the first place. So there is
#: exactly one preset where the number means "how fast it goes", and the player
#: has to ask which one it is before it carries anything.
#:
#: It did not ask. `behaviors.js` checked only "is this a body with a numeric
#: speed?", and the recorded avoidance document gives its 底盘 `speed: 60` in a
#: `chain` scene — so the first thing on screen in that lesson was a link
#: diagram with the chassis sliding off the right edge and wrapping, forever.
#: Nothing errored; `speed` is a real prop of a real primitive, and the beat
#: that set it was reported as a working beat.
#:
#: Mirrored by hand in `registry.js` and held equal by a test, the same trade
#: `LIVE_PROPS` and `PENDING_KINDS` make.
SPEED_PRESETS: frozenset[str] = frozenset({"lane"})


# --------------------------------------------------------------------------
# Vocabulary rendering — the prompt is generated, never hand-copied
# --------------------------------------------------------------------------


def _prop_hint(primitive: str, prop: str) -> str:
    """The parenthetical after a prop name: its unit, or its legal values.

    Inline rather than in a footnote because the model reads the prop list when
    it writes the prop. A unit stated anywhere else is a unit said once, and the
    same goes for the five words `tone` accepts: this is the only place the
    vocabulary names them, and the only place the model is standing when it has
    to choose one.

    Keyed on the prop name rather than on `(primitive, prop)` because `tone`
    belongs to exactly one primitive, and if a second one ever declares it the
    gloss is still the right gloss. `emphasis` extends that from two props to
    three, `direction` to four, `mark` and `type` to six, and the two block
    props — `language` and `focus` — to eight, without changing the arrangement.
    `tone` is the one that proves the keying: it began on `readout` alone and is
    now on four primitives, and it has needed no edit here to follow them.

    `direction` and `focus` are the two that are not a *set* of words but a value,
    so they are glossed directly rather than through `_glossed` — same
    parentheses, same inline position, one entry instead of five.
    """
    bounds = stage_range(primitive, prop)
    if bounds is not None:
        return f"（{bounds.unit} {bounds.low:g}~{bounds.high:g}）"
    if prop == "tone":
        return _glossed(TONE_GLOSSES)
    if prop == "emphasis":
        return _glossed(EMPHASIS_GLOSSES)
    if prop == "direction":
        return f"（{DIRECTION_GLOSS}）"
    if prop == "mark":
        return _glossed(MARK_GLOSSES)
    if prop == "type":
        return _glossed(TYPE_GLOSSES)
    if prop == "language":
        return _glossed(CODE_LANGUAGES)
    if prop == "form":
        return _glossed(TREE_FORMS)
    if prop == "focus":
        return f"（{FOCUS_GLOSS}）"
    return ""


def _glossed(glosses: tuple[tuple[str, str], ...]) -> str:
    """A vocabulary rendered the one way, so the two of them cannot diverge."""
    return "（" + "、".join(f"{name} {gloss}" for name, gloss in glosses) + "）"


def render_vocabulary(renderers: tuple[str, ...] = ()) -> str:
    """Render the whole vocabulary as prompt text.

    The Agent prompt embeds this instead of restating the tables, so a name that
    exists in the registry is always offered to the model, and a test can assert
    the two never drift.
    """
    lines: list[str] = []

    lines.append("## 可用预设 scene_type（越靠前越具体）")
    lines.append(
        "**能用更具体的预设就必须用。** 先看这一幕的对象有哪些角色，"
        "再挑那个「必需角色全部满足」的预设。`generic` 排在最后——"
        "它只是**一个具体预设都用不上时**的最后选择。把它当默认值看，"
        "画面会退化成一张竖排列表：本来画得出来的坐标轴、轨迹、链路全都不会出现。"
    )
    # Ordered by specificity so the most descriptive preset is read first. The
    # ordering is the point, not decoration: `generic` advertised as "always
    # valid" got read as "always safe", and a real run took it for 3 of the 5
    # scenes that `field` could have served. Nothing was wrong with the model's
    # output — the incentive in the wording was pointing the wrong way.
    for preset in sorted(PRESETS, key=lambda item: -len(item.required_roles)):
        required = "、".join(preset.required_roles) if preset.required_roles else "无"
        lines.append(
            f"- `{preset.name}`：{preset.note}；必需角色 {required}；"
            f"最多 {preset.max_bodies} 个对象"
        )

    lines.append("")
    lines.append("## 可用角色 role（`role` 字段只能填冒号后面的名字）")
    lines.append("")
    # Said plainly because it was got wrong in practice: the format below puts the
    # primitive first, and a real run wrote `role: "vector"` / `role: "axis"` —
    # the *primitive* names — 14 times over. The two levels have to be visibly
    # different things, or the layout reads like one flat list of role names.
    lines.append(
        "冒号**前面**（`body`、`vector`、`axis`…）是**图元名**，不是角色名，"
        "绝对不能填进 `role`——填了会被校验打回。"
    )
    for primitive in T1_PRIMITIVES:
        if not primitive.drawable:
            continue
        roles = "、".join(f"`{role}`" for role in primitive.roles)
        lines.append(f"- {primitive.name}（{primitive.note}）→ 角色只能是：{roles}")

    # Named rather than omitted. A model that reaches for `waveform` and is told
    # "unknown role" will try again with another guess; one that is told the name
    # is real but not yet drawable, and what to write instead, gets it right the
    # first time. Silence here costs a retry and reads as an oversight.
    if PENDING_PRIMITIVES:
        lines.append("")
        lines.append(
            "**下面这些图元已在契约里登记，但布局层还没有实现。"
            "现在写它们的角色不会失败在提示词上，而会失败在最后一步的布局——"
            "整份分镜作废、一张图都出不来。一律不要用。**"
        )
        for primitive in T1_PRIMITIVES:
            if primitive.drawable:
                continue
            roles = "、".join(f"`{role}`" for role in primitive.roles)
            lines.append(
                f"- ~~{primitive.name}~~（角色 {roles}）→ 改用 "
                f"{PENDING_ALTERNATIVES.get(primitive.name, '别的图元')}"
            )

    lines.append("")
    lines.append("## 每个图元可设置的属性 prop")
    lines.append(
        "属性分两组。**`object_states` 里只能写「可被节拍改变」那一组**——"
        "另一组没有渲染代码读取，写进某一拍不会让画面动一下，"
        "那一拍就是一张幻灯片，校验台会打回。"
    )
    for primitive in T1_PRIMITIVES:
        if not primitive.drawable:
            continue
        live = primitive.live_props
        static = tuple(prop for prop in primitive.props if prop not in live)
        parts = []
        if live:
            live_text = "、".join(f"`{prop}`{_prop_hint(primitive.name, prop)}" for prop in live)
            parts.append(f"可被节拍改变：{live_text}")
        if static:
            # Glossed as well as the live group, and that is a fix rather than
            # symmetry for its own sake. `_prop_hint` was only ever asked about a
            # prop a beat could move, so a closed set of words on a prop set once
            # — `code.language`, the only one there has ever been — was printed
            # as a bare name. The names are the whole of what that prop needs to
            # say, and this is the one place the model is standing when it has to
            # choose one: the same argument the function's own docstring makes,
            # one group up.
            parts.append(
                "只能整体设置一次："
                + "、".join(f"`{prop}`{_prop_hint(primitive.name, prop)}" for prop in static)
            )
        if primitive.required_props:
            parts.append(
                "**必须填写**（缺了画面照样画得出来，只是画出来的不是它该有的样子）："
                + "、".join(f"`{prop}`" for prop in primitive.required_props)
            )
        mandatory = primitive.required_relations
        optional = tuple(rel for rel in primitive.relations if rel not in mandatory)
        if mandatory:
            parts.append(
                "**必须填写**的关系型属性（值是同一场景内另一个对象的 id）："
                + "、".join(f"`{rel}`" for rel in mandatory)
            )
        if optional:
            parts.append("可选关系型属性：" + "、".join(f"`{rel}`" for rel in optional))
        lines.append(f"- `{primitive.name}`：{'；'.join(parts) if parts else '（无）'}")

    lines.append("")
    lines.append("## 这些属性用的是渲染器的单位，不是文档里的物理单位")
    lines.append(
        "画布的坐标系是固定的：宽 960、高 600，单位是**舞台像素**，不是米、"
        "厘米、牛顿或秒。所以 `radius` 要写「这条射线在图上有多长」，"
        "不是「这颗激光雷达实际能测多远」。"
        "**文档里的物理量（8 米、0.8 米、0.6 米每秒）不能直接抄进来**——"
        "写 8 会被画成 8 个像素，比一个标点还小，屏幕上什么都看不见；"
        "`speed` 写 60 会让小车一秒跑完五个屏，糊成一道光。"
        "按文档的实际含义换算成一个占画布合理比例的值再写。"
    )
    for (primitive_name, prop), entry in sorted(STAGE_RANGES.items()):
        lines.append(
            f"- `{primitive_name}.{prop}`：{entry.low:g}~{entry.high:g} {entry.unit}（{entry.why}）"
        )

    lines.append("")
    lines.append("## 可用领域字形 glyph（只能用在 `body` 的 `glyph` 属性上）")
    # Printed as a bare name list until now, while every `note` below sat written
    # and unread. The measured cost: across the three sample documents the real
    # chain asked for a glyph zero times. The names were never the problem — a
    # model that has never been told what `package` is for has no reason to
    # prefer it to the rectangle it gets for free.
    # Every name in this paragraph is a **role**, and every one of them is
    # interpolated from the table rather than typed. It used to name three
    # *primitives* instead — `zone` / `readout` / `dimension` — in a sentence
    # whose whole job is to say what to use, and the model did as it was told:
    # one real run answered with `role: readout` twelve times, every one of them
    # an `unknown_role` the retry then had to spend a whole call on. The role
    # list was printed correctly a few lines above; the paragraph contradicted
    # it. See `test_the_glyph_paragraph_only_names_roles_you_can_write`.
    body_roles = "、".join(f"`{role}`" for role in PRIMITIVE_BY_NAME["body"].roles)
    lines.append(
        "字形是**一幅固定的实物线稿**：颜色、线宽、部件都改不了，只能按名字挑。"
        "该不该用的判据是「**这个对象在文档里是不是一个具体的实物**」——"
        "小车、雷达、控制器、服务器、数据包是；"
        "安全距离、空旷程度、判断结论不是，那些该用 "
        "`safe_distance` / `coverage` / `decision` / `range` 这类描述性的角色。"
        "不写不会错，只是画面更抽象；写对了，那一拍一眼就认得出画的是什么。"
        f"`glyph` 只对实物类角色有效：{body_roles}；"
        "`sensor`、`threshold`、`hud`、`topic` 这些描述性的角色没有这个属性，写了也不画。"
    )
    for glyph in T2_GLYPHS:
        lines.append(f"- `{glyph.name}`（{glyph.domain}）：{glyph.note}")

    lines.append("")
    lines.append("## 可用行为 behavior")
    for behavior in BEHAVIORS:
        reads = "、".join(f"`{prop}`" for prop in behavior.reads)
        lines.append(f"- `{behavior.name}`：{behavior.note}；读取 {reads}")

    lines.append("")
    lines.append("## 控件 target_property 的合法目标")
    lines.append(
        "只有被行为读取的属性可以做成控件："
        + "、".join(f"`{prop}`" for prop in sorted(CONSUMABLE_PROPS))
        + "。写成 `<对象id>.<属性>` 或 `scene.<属性>`。"
    )
    lines.append("")
    lines.append("## 按钮可用的 action")
    lines.append("、".join(f"`{action}`" for action in BUTTON_ACTIONS))

    if renderers:
        lines.append("")
        lines.append("## 可用 renderer_hint")
        lines.append("、".join(f"`{r}`" for r in renderers))

    return "\n".join(lines)
