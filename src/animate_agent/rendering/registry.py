"""The semantic vocabulary a storyboard may use, and the limits of what can be drawn.

This module is the contract, not the renderer. It lives under `rendering/`
because the vocabulary is defined by *what can be drawn*: a role exists here
only if some primitive can express it, and a prop exists only if some primitive
or behaviour reads it. `storyboard/validation.py` and the Storyboard Agent
prompt are both generated from the tables below, which is what keeps the three
of them from drifting apart.

Layers (see `docs/storyboard-milestone.md`):

- **atoms**   — draw-layer shapes; never appear in the contract.
- **T1 primitives** — the 13 semantic primitives the model may reference.
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
        ),
        props=("speed", "heading", "scale", "visible", "danger", "glyph"),
        relations=(),
        note="基础形状由 glyph 选（圆/矩形/多边形/胶囊）；glyph 可指向 T2 字形。"
        "`object` 是兜底角色，只在确实没有合适角色时才用——留它是为了让每个被抽出的"
        "对象都有地方去，而不是被硬塞进某个不匹配的域角色",
    ),
    Primitive(
        name="emitter",
        roles=("sensor", "transmitter", "source"),
        props=("radius", "fov", "enabled"),
        relations=("of",),
        required_relations=("of",),
        note="扇形 + N 条射线，其中一条高亮到命中点；radius 是探测距离（激光雷达半径）。"
        "必须挂在某个 body 上——storyboard 里没有坐标，离开母体就无处安放",
    ),
    Primitive(
        name="zone",
        roles=("threshold", "safe_distance", "coverage"),
        props=("radius", "enabled"),
        relations=("of",),
        required_relations=("of",),
        note="挂在 body 上的虚线扇区/圆/环",
    ),
    Primitive(
        name="link",
        roles=("topic", "edge", "flow", "relation"),
        props=("active",),
        relations=("from", "to"),
        required_relations=("from", "to"),
        note="两个 body 之间的连线，虚线↔实线；两端都必填，否则这条线不存在",
    ),
    Primitive(
        name="traveler",
        roles=("message", "request", "unit"),
        props=("speed", "progress", "state"),
        relations=("along",),
        required_relations=("along",),
        note="沿 link 或 trace 移动的小令牌；没有 along 就不知道它沿什么走",
    ),
    Primitive(
        name="trace",
        roles=("trail", "trajectory", "path"),
        props=("length", "visible"),
        relations=("of",),
        required_relations=("of",),
        note="折线：body 的历史轨迹，或按公式算出的曲线；`of` 表明是谁的轨迹",
    ),
    Primitive(
        name="vector",
        roles=("velocity", "force", "displacement"),
        props=("magnitude", "direction"),
        relations=("of", "component_of"),
        required_relations=("of",),
        note="箭头；`of` 是它作用在谁身上（必填），component_of 表达分解"
        "（v0 → v0cosθ + v0sinθ），只有分解出来的分量才需要",
    ),
    Primitive(
        name="axis",
        roles=("x_axis", "y_axis"),
        props=("range", "ticks", "origin"),
        relations=(),
        note="带刻度与标签的坐标轴，可成对交于原点",
    ),
    Primitive(
        name="dimension",
        roles=("range", "height", "span"),
        props=("label",),
        relations=("from", "to"),
        required_relations=("from", "to"),
        note="两端带箭头的标注线 + 标签（射程 R、最大高度 H）；"
        "from/to 是它量的那两个点，缺一个就画不出标注线",
    ),
    Primitive(
        name="angle",
        roles=("angle", "bearing"),
        props=("degrees", "radius"),
        relations=("of", "between"),
        required_relations=("of",),
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
        roles=("hud", "decision", "caption", "code", "formula"),
        props=("text", "align", "tone"),
        relations=(),
        note="文本面板；公式在这里是构建期渲好的 path，不是可解析文本",
    ),
)

#: role -> primitive name. Built from T1 so the two can never disagree.
ROLE_TO_PRIMITIVE: dict[str, str] = {
    role: primitive.name for primitive in T1_PRIMITIVES for role in primitive.roles
}

PRIMITIVE_BY_NAME: dict[str, Primitive] = {p.name: p for p in T1_PRIMITIVES}

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


#: Keep this small. Four target documents are expected to need 10~15 glyphs
#: total; anything that can be composed from T1 should be, rather than drawn.
T2_GLYPHS: tuple[GlyphDeclaration, ...] = (
    GlyphDeclaration("car", "robotics", "tabler:car", "避障小车的车身"),
    GlyphDeclaration("wheel", "robotics", "tabler:car-wheel", "可独立旋转的车轮"),
    GlyphDeclaration("steering_wheel", "robotics", "tabler:steering-wheel", "转向决策的视觉锚点"),
    GlyphDeclaration("lidar", "robotics", "tabler:radar", "激光雷达本体"),
    GlyphDeclaration("robot", "robotics", "tabler:robot", "移动机器人本体"),
    GlyphDeclaration("cpu", "robotics", "tabler:cpu", "控制器/计算节点"),
    GlyphDeclaration("server", "cloud", "tabler:server", "服务端"),
    GlyphDeclaration("package", "cloud", "tabler:package", "消息载荷"),
    GlyphDeclaration(
        "obstacle_octagon", "robotics", "hand:octagon", "八角障碍物，比通用多边形更像障碍"
    ),
    GlyphDeclaration("protractor", "physics", "game-icons:protractor", "角度标注"),
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
    Behavior("linear_motion", ("speed", "heading"), "小车/抛体沿当前朝向匀速前进"),
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


PRESETS: tuple[Preset, ...] = (
    Preset("lane", ("vehicle", "obstacle"), 6, "一维通道上的避障：车、障碍、安全距离"),
    Preset("chain", ("node",), 8, "有向链路：ROS 节点 / API 调用链"),
    Preset("hub", ("node", "endpoint"), 8, "星形拓扑：一个中心与若干叶子"),
    Preset("field", ("projectile",), 4, "二维场：坐标轴 + 抛体轨迹 + 矢量分解"),
    # Described as the last resort, not as the safe default. It used to read
    # "永远合法的兜底预设", and the model — optimising for "don't get rejected" —
    # picked it for 3 of 5 scenes that `field` could have served. Nothing was
    # wrong with the model's output; the incentive in the wording was pointing
    # the wrong way. See `docs/storyboard-milestone.md`.
    Preset("generic", (), 8, "竖排对象 + 说明面板；只在没有任何更具体的预设可用时才选它"),
)

PRESET_BY_NAME: dict[str, Preset] = {p.name: p for p in PRESETS}


# --------------------------------------------------------------------------
# Vocabulary rendering — the prompt is generated, never hand-copied
# --------------------------------------------------------------------------


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
    for primitive in T1_PRIMITIVES:
        if not primitive.drawable:
            continue
        parts = []
        if primitive.props:
            parts.append("、".join(f"`{prop}`" for prop in primitive.props))
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
    lines.append("## 可用领域字形 glyph（只能用在 `body` 的 `glyph` 属性上）")
    lines.append("、".join(f"`{g.name}`" for g in T2_GLYPHS))

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
