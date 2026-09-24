"""Build `assets/glyphs/*.json` — the T2 glyph data the player draws.

Why this exists
---------------
`Glyph.source` claims a glyph is "a build-time product of an icon set plus a
normalisation script", and the note beside it says hand-edited `d` strings are
not allowed (`rendering/registry.py`). This is that script. Uncommitted, those
JSONs are hand-edited `d` strings that went through extra steps, and the claim
is unverifiable.

Where the geometry comes from
-----------------------------
Two icon sets, each pinned by version and each recorded in the glyph's own
`source` field. The second one is not a fallback for when the first is down; it
is here because the first has no picture of a thing a lesson needed.

  - **Tabler Icons** (MIT) — the original set, and still every glyph but
    thirteen. A flat list of `<path>` elements on a 24x24 grid.
  - **Lucide** (ISC) — also 24x24, also stroke-only at width 2 with round caps,
    so the two mix without a seam. It is here for `robot-arm`, which Tabler does
    not have, and which is the only drawing of an industrial arm in any of the
    icon sets this project surveyed.

"Normalisation" is mostly *naming* — deciding which elements form one part and
which form another — plus turning the shapes that are not `<path>` into path
data. Lucide draws a robot arm's joint as `<circle>`, a cylinder as `<ellipse>`
and a clipboard's board as `<rect>`; the schema stores `d`, so a shape element
the parser does not understand is a part that goes missing with nothing to show
for it. `_shape_to_path` converts the shapes the pinned sets use and refuses the
rest by name.

Tabler marks its spacer path (`d="M0 0h24v24H0z"`) with `stroke="none"`. That
attribute is the set's own convention for "not artwork", so it is what gets
filtered — not a hard-coded `d` string, which would break on the next icon.

The output is measured, not just copied
---------------------------------------
`ink_box` is computed from the `d` strings by `tools.path_bbox.py` rather than
written down, because `--check` re-fetches upstream: a hand-typed number would
survive a version bump that moved the art, and the only symptom would be every
glyph drawn subtly the wrong size. The build also asserts the ink lies inside the
view box, which is what catches a parser or part-assignment mistake.

Element counts are pinned, not guessed
--------------------------------------
`GlyphSource.art_elements` is asserted. If upstream reorders or adds an element
the build fails loudly and a human re-checks the part assignment. The alternative
is a glyph that quietly becomes a different picture — and a part assigned to the
wrong element is invisible in the output JSON unless you already know what the
icon looks like.

Usage
-----
    python -m tools.build_glyphs            # write assets/glyphs/
    python -m tools.build_glyphs --check    # fail if the committed files differ

`--check` needs the network, so it is a tool and not a test: the test suite
asserts the other half instead — that the declared glyph set and the data on
disk are the same set, offline.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from tools.path_bbox import arcs, path_bbox

from animate_agent.rendering.registry import T2_GLYPHS


@dataclass(frozen=True, slots=True)
class IconSet:
    """One upstream icon set: where its SVGs live, and which release is pinned.

    Pinned so a rebuild is byte-identical. Bumping a version is a deliberate act
    that re-runs the element-count assertions below.
    """

    name: str
    version: str
    #: Formatted with `version` and `icon`.
    url: str

    def source(self, icon: str) -> str:
        """What goes in the glyph's `source` field, so it traces back to a cut."""
        return f"{self.name}:{icon}@{self.version}"


#: Keyed by the name a `GlyphSource` gives in its `set` field. Adding a third set
#: is a line here and nothing else — no other part of this file counts them, and
#: a set named in `GLYPH_SOURCES` but missing here fails in `fetch_icon`.
SETS: dict[str, IconSet] = {
    "tabler": IconSet(
        "tabler",
        "3.46.0",
        "https://unpkg.com/@tabler/icons@{version}/icons/outline/{icon}.svg",
    ),
    "lucide": IconSet(
        "lucide",
        "1.47.0",
        "https://unpkg.com/lucide-static@{version}/icons/{icon}.svg",
    ),
}

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "assets" / "glyphs"

#: `mode` for every shipped glyph. The outline set is stroke-only by
#: construction; a filled glyph would come from a different upstream directory
#: and would have to be added here deliberately.
STROKE = "stroke"


@dataclass(frozen=True, slots=True)
class GlyphSource:
    """One glyph's provenance and how its elements divide into named parts."""

    icon: str
    #: How many elements draw ink — not how many `<path>` tags there are. A
    #: `<circle>` is an element too, and counting tags would let an icon gain a
    #: shape without the count noticing, which is the drift this pins against.
    art_elements: int
    parts: dict[str, tuple[int, ...]]
    #: Parts that turn about their own arc centre. A list of names rather than a
    #: flag on each part, so "what moves in this glyph" is one line you can read.
    #: The anchor is measured from the geometry; see `_spin_anchor`.
    spin: tuple[str, ...] = ()
    #: Which set in `SETS` the icon comes from. Defaulted to the set that was
    #: here first, so the entries older than the second one were not touched.
    set: str = "tabler"


def _outline(count: int) -> dict[str, tuple[int, ...]]:
    """Every art element in one part, for a glyph that is a single picture."""
    return {"outline": tuple(range(count))}


#: The part names are what the player transforms independently, so they are a
#: decision, not a detail. Five of the six glyphs are one picture and get one
#: part; the car is the exception its `parts` field was designed for.
GLYPH_SOURCES: dict[str, GlyphSource] = {
    # Indices are Tabler's own path order: 0 and 1 are the two wheel circles
    # (`M5 17a2 ...` / `M15 17a2 ...`), 2 is the chassis. Front is the right-hand
    # one because the car faces +x, which is also the direction `body.heading`
    # rotates away from.
    #
    # Neither wheel spins, and the reason is geometric rather than an omission:
    # both are exact circles, so rotating one about its own centre is the identity
    # and the picture does not change by a pixel. The parts are split anyway
    # because a wheel that *did* have spokes would need it, and because merging
    # them back would be the thing that made it impossible.
    "car": GlyphSource("car", 3, {"body": (2,), "wheelRear": (0,), "wheelFront": (1,)}),
    # Tabler's radar is a 90-degree sweep sector plus the two range arcs, so path 0
    # is the arm and 1/2 are the dish it sweeps over. The arm's arcs are centred on
    # (12, 12) — the corner the two straight edges meet at — so it turns in place
    # rather than orbiting, which is what makes 雷达扫描 a moving picture.
    "lidar": GlyphSource("radar", 3, {"sweep": (0,), "outline": (1, 2)}, spin=("sweep",)),
    # Everything below is a single picture, so each takes `_outline` and gets one
    # part. That is not a shortcut: `tests/unit/test_render_glyphs.py` requires
    # exactly `["outline"]` of any glyph not named in its `MULTI_PART` table, so a
    # glyph that wants separate moving parts has to be declared as wanting them.
    # The left-hand key is the name the model writes; the string is the icon in
    # the set named by `set`, and the two differ where the set's word is not the
    # domain's (`gear` is `settings`, `sine-wave` is `wave-sine`). Element counts
    # are measured from the pinned release and asserted by `art_elements`, so an
    # upstream reorder fails loudly rather than quietly drawing a different
    # picture.
    # -- 机器人 ------------------------------------------------------------
    "robot": GlyphSource("robot", 9, _outline(9)),
    "cpu": GlyphSource("cpu", 10, _outline(10)),
    "camera": GlyphSource("camera", 2, _outline(2)),
    "scan": GlyphSource("scan", 5, _outline(5)),
    "gauge": GlyphSource("gauge", 4, _outline(4)),
    "temperature": GlyphSource("temperature", 2, _outline(2)),
    # -- 载具 --------------------------------------------------------------
    "drone": GlyphSource("drone", 9, _outline(9)),
    "truck": GlyphSource("truck", 3, _outline(3)),
    "plane": GlyphSource("plane", 1, _outline(1)),
    "ship": GlyphSource("ship", 4, _outline(4)),
    # -- 网络与通信 --------------------------------------------------------
    "antenna": GlyphSource("antenna", 6, _outline(6)),
    "wifi": GlyphSource("wifi", 4, _outline(4)),
    "broadcast": GlyphSource("broadcast", 3, _outline(3)),
    "router": GlyphSource("router", 6, _outline(6)),
    "network": GlyphSource("network", 8, _outline(8)),
    "cloud": GlyphSource("cloud", 1, _outline(1)),
    "server": GlyphSource("server", 4, _outline(4)),
    "database": GlyphSource("database", 3, _outline(3)),
    "package": GlyphSource("package", 5, _outline(5)),
    "brain": GlyphSource("brain", 6, _outline(6)),
    "chart": GlyphSource("chart-line", 2, _outline(2)),
    "sitemap": GlyphSource("sitemap", 5, _outline(5)),
    # -- 电与物理 ----------------------------------------------------------
    "battery": GlyphSource("battery", 1, _outline(1)),
    "bolt": GlyphSource("bolt", 1, _outline(1)),
    "magnet": GlyphSource("magnet", 3, _outline(3)),
    "sine-wave": GlyphSource("wave-sine", 1, _outline(1)),
    "square-wave": GlyphSource("wave-square", 1, _outline(1)),
    "activity": GlyphSource("activity", 1, _outline(1)),
    "propeller": GlyphSource("propeller", 4, _outline(4)),
    "wind": GlyphSource("wind", 3, _outline(3)),
    # -- 科学与航天 --------------------------------------------------------
    "atom": GlyphSource("atom", 3, _outline(3)),
    "rocket": GlyphSource("rocket", 3, _outline(3)),
    "satellite": GlyphSource("satellite", 6, _outline(6)),
    "planet": GlyphSource("planet", 2, _outline(2)),
    "microscope": GlyphSource("microscope", 7, _outline(7)),
    "flask": GlyphSource("flask", 3, _outline(3)),
    "telescope": GlyphSource("telescope", 4, _outline(4)),
    # -- 机械 --------------------------------------------------------------
    "gear": GlyphSource("settings", 2, _outline(2)),
    "wrench": GlyphSource("tool", 1, _outline(1)),
    # -- 状态与指示 --------------------------------------------------------
    "warning": GlyphSource("alert-triangle", 3, _outline(3)),
    # One element and two, measured from the pinned release. Both are bare
    # strokes: `drawVerdict` draws the disc behind them, in the colour the mark
    # asks for, so a glyph with its own circle would be one the drawer could not
    # recolour.
    "check": GlyphSource("check", 1, _outline(1)),
    "x": GlyphSource("x", 2, _outline(2)),
    "lock": GlyphSource("lock", 3, _outline(3)),
    "clock": GlyphSource("clock", 2, _outline(2)),
    "shield": GlyphSource("shield", 1, _outline(1)),
    # -- 工业制造 ----------------------------------------------------------
    # The round that this set was short of. A real document about a sorting line
    # named six things and only two of them had a picture; `robot-arm` had none
    # anywhere in Tabler, which is why a second set is in `SETS` at all.
    #
    # `robot-arm` is Lucide's, and it is the one entry whose five `<path>`s are
    # followed by a `<circle>` — the arm's base joint, and the reason
    # `_shape_to_path` exists. Dropping it would leave an arm floating above
    # nothing, and no part of the output would say so.
    "robot-arm": GlyphSource("robot-arm", 6, _outline(6), set="lucide"),
    "photo-sensor": GlyphSource("photo-sensor", 5, _outline(5)),
    "assembly": GlyphSource("assembly", 2, _outline(2)),
    "building-factory": GlyphSource("building-factory", 4, _outline(4)),
    "building-warehouse": GlyphSource("building-warehouse", 3, _outline(3)),
    "forklift": GlyphSource("forklift", 8, _outline(8)),
    "crane": GlyphSource("crane", 4, _outline(4)),
    "container": GlyphSource("container", 11, _outline(11)),
    "circuit-motor": GlyphSource("circuit-motor", 4, _outline(4)),
    "engine": GlyphSource("engine", 5, _outline(5)),
    "drill": GlyphSource("drill", 6, _outline(6), set="lucide"),
    "hard-hat": GlyphSource("hard-hat", 4, _outline(4), set="lucide"),
    "cuboid": GlyphSource("cuboid", 3, _outline(3), set="lucide"),
    "cylinder": GlyphSource("cylinder", 2, _outline(2), set="lucide"),
    # -- 通用教学 ----------------------------------------------------------
    "hierarchy": GlyphSource("hierarchy", 5, _outline(5)),
    "list-check": GlyphSource("list-check", 6, _outline(6)),
    "report": GlyphSource("report", 7, _outline(7)),
    "license": GlyphSource("license", 3, _outline(3)),
    "target": GlyphSource("target", 3, _outline(3)),
    "route": GlyphSource("route", 3, _outline(3)),
    "scale": GlyphSource("scale", 5, _outline(5)),
    "compass": GlyphSource("compass", 6, _outline(6)),
    "milestone": GlyphSource("milestone", 3, _outline(3), set="lucide"),
    "waypoints": GlyphSource("waypoints", 7, _outline(7), set="lucide"),
    "clipboard-list": GlyphSource("clipboard-list", 6, _outline(6), set="lucide"),
    "pencil-ruler": GlyphSource("pencil-ruler", 6, _outline(6), set="lucide"),
    # -- 电子电路 ----------------------------------------------------------
    "circuit-resistor": GlyphSource("circuit-resistor", 1, _outline(1)),
    "circuit-capacitor": GlyphSource("circuit-capacitor", 4, _outline(4)),
    "circuit-diode": GlyphSource("circuit-diode", 4, _outline(4)),
    "circuit-inductor": GlyphSource("circuit-inductor", 1, _outline(1)),
    "circuit-ammeter": GlyphSource("circuit-ammeter", 5, _outline(5)),
    "circuit-voltmeter": GlyphSource("circuit-voltmeter", 4, _outline(4)),
    "circuit-switch-closed": GlyphSource("circuit-switch-closed", 5, _outline(5)),
    "circuit-ground": GlyphSource("circuit-ground", 4, _outline(4)),
    "circuit-bulb": GlyphSource("circuit-bulb", 5, _outline(5)),
    "plug": GlyphSource("plug", 4, _outline(4)),
    "circuit-board": GlyphSource("circuit-board", 5, _outline(5), set="lucide"),
    "cable": GlyphSource("cable", 7, _outline(7), set="lucide"),
    # -- 化工与能源 --------------------------------------------------------
    "pipeline": GlyphSource("pipeline", 5, _outline(5)),
    "tank": GlyphSource("tank", 3, _outline(3)),
    "barrel": GlyphSource("barrel", 5, _outline(5)),
    "building-wind-turbine": GlyphSource("building-wind-turbine", 7, _outline(7)),
    "solar-panel": GlyphSource("solar-panel", 6, _outline(6)),
    "droplet": GlyphSource("droplet", 1, _outline(1)),
    "flame": GlyphSource("flame", 1, _outline(1)),
    "recycle": GlyphSource("recycle", 6, _outline(6)),
    "leaf": GlyphSource("leaf", 2, _outline(2)),
    "beaker": GlyphSource("beaker", 3, _outline(3), set="lucide"),
    "dna": GlyphSource("dna", 11, _outline(11), set="lucide"),
}

#: Every element that draws ink. `\b` keeps `<path>` from matching a longer tag,
#: and the alternation lists the multi-letter names first for the same reason.
_ELEMENT_RE = re.compile(
    r"<(polyline|polygon|ellipse|circle|rect|path|line)\b([^>]*?)/?>", re.DOTALL
)
_ATTR_RE = re.compile(r'([a-zA-Z][\w:-]*)\s*=\s*"([^"]*)"')
#: A leading relative moveto and its two coordinates, for `_absolute_start`.
_MOVETO_RE = re.compile(r"m\s*(-?[\d.]+)[\s,]+(-?[\d.]+)")
#: Whether what follows those coordinates is another bare number, which is the
#: repeat `_absolute_start` has to name explicitly.
_STILL_NUMBERS_RE = re.compile(r"[\s,]*[-+.\d]")
_D_RE = re.compile(r'\bd="([^"]*)"')
_SPACER_RE = re.compile(r'\bstroke="none"')
_VIEW_BOX_RE = re.compile(r'viewBox="([^"]*)"')
_STROKE_WIDTH_RE = re.compile(r'stroke-width="([^"]*)"')


class BuildError(RuntimeError):
    """A glyph could not be built from upstream as declared."""


def fetch_icon(icon: str, set_name: str) -> str:
    if set_name not in SETS:
        raise BuildError(
            f"`{set_name}` 不在 SETS 里。先把图标集登记进去，再让字形指向它——"
            f"现有的集合是 {'、'.join(sorted(SETS))}"
        )
    icon_set = SETS[set_name]
    url = icon_set.url.format(version=icon_set.version, icon=icon)
    request = urllib.request.Request(url, headers={"User-Agent": "animate-agent-build"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8")
    except urllib.error.URLError as error:  # pragma: no cover - network path
        raise BuildError(f"取不到 {url}：{error}") from error


def _num(value: float) -> str:
    """A derived coordinate, written the way the two sets write their own.

    Trailing zeros are stripped so the diameter of a 4-unit circle comes out as
    `8` and not `8.000000`. These strings land in files that `--check` compares
    byte-for-byte, and decimals nobody typed would make generated data look
    hand-edited — which is the one thing this build exists to make impossible.
    """
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _attrs(raw: str) -> dict[str, str]:
    return {name.lower(): value for name, value in _ATTR_RE.findall(raw)}


def _numbers(attrs: dict[str, str], element: str, icon: str, *names: str) -> list[float]:
    """Read required numeric attributes, refusing to default one that is absent.

    A defaulted `r` of 0 draws nothing and a defaulted `cx` of 0 draws the shape
    in the corner. Both ship, and neither reports anywhere downstream — so a
    missing attribute is treated as the parser and the file disagreeing, which is
    the only case worth stopping for.
    """
    values: list[float] = []
    for name in names:
        if name not in attrs:
            raise BuildError(f"`{icon}` 的 <{element}> 缺了 `{name}` 属性，量不出图形")
        try:
            values.append(float(attrs[name]))
        except ValueError as error:
            raise BuildError(
                f"`{icon}` 的 <{element}> 的 `{name}`={attrs[name]!r} 不是数字"
            ) from error
    return values


def _shape_to_path(element: str, attrs: dict[str, str], icon: str) -> str:
    """One non-`<path>` shape element rewritten as path data.

    Every branch is a shape the two pinned sets actually use: Lucide draws a
    robot arm's joint as `<circle>`, a cylinder as `<ellipse>`, and a clipboard
    or a hard hat as `<rect>`. The rest of the SVG shape vocabulary is refused by
    name rather than guessed at, because a dropped `<line>` is a missing whisker
    and nothing in the output would say so.
    """
    if element == "circle":
        cx, cy, r = _numbers(attrs, element, icon, "cx", "cy", "r")
        return (
            f"M{_num(cx - r)} {_num(cy)}"
            f"a{_num(r)} {_num(r)} 0 1 0 {_num(2 * r)} 0"
            f"a{_num(r)} {_num(r)} 0 1 0 {_num(-2 * r)} 0"
        )
    if element == "ellipse":
        cx, cy, rx, ry = _numbers(attrs, element, icon, "cx", "cy", "rx", "ry")
        return (
            f"M{_num(cx - rx)} {_num(cy)}"
            f"a{_num(rx)} {_num(ry)} 0 1 0 {_num(2 * rx)} 0"
            f"a{_num(rx)} {_num(ry)} 0 1 0 {_num(-2 * rx)} 0"
        )
    if element == "rect":
        # `x` and `y` may be omitted and mean 0, which the spec says rather than
        # this file: only the two the shape is nothing without are required.
        x = float(attrs.get("x", 0.0))
        y = float(attrs.get("y", 0.0))
        width, height = _numbers(attrs, element, icon, "width", "height")
        # One radius implies the other, and both clamp to half the side they
        # round. `rx="2"` on a 4-tall body is a stadium rather than an error, and
        # upstream relies on that.
        rx = min(float(attrs.get("rx", attrs.get("ry", 0.0))), width / 2)
        ry = min(float(attrs.get("ry", attrs.get("rx", 0.0))), height / 2)
        if rx <= 0 or ry <= 0:
            return f"M{_num(x)} {_num(y)}H{_num(x + width)}V{_num(y + height)}H{_num(x)}Z"
        return (
            f"M{_num(x + rx)} {_num(y)}"
            f"H{_num(x + width - rx)}"
            f"A{_num(rx)} {_num(ry)} 0 0 1 {_num(x + width)} {_num(y + ry)}"
            f"V{_num(y + height - ry)}"
            f"A{_num(rx)} {_num(ry)} 0 0 1 {_num(x + width - rx)} {_num(y + height)}"
            f"H{_num(x + rx)}"
            f"A{_num(rx)} {_num(ry)} 0 0 1 {_num(x)} {_num(y + height - ry)}"
            f"V{_num(y + ry)}"
            f"A{_num(rx)} {_num(ry)} 0 0 1 {_num(x + rx)} {_num(y)}Z"
        )
    if element == "line":
        x1, y1, x2, y2 = _numbers(attrs, element, icon, "x1", "y1", "x2", "y2")
        return f"M{_num(x1)} {_num(y1)}L{_num(x2)} {_num(y2)}"
    if element in {"polyline", "polygon"}:
        if "points" not in attrs:
            raise BuildError(f"`{icon}` 的 <{element}> 没有 `points` 属性")
        pairs = [pair for pair in re.split(r"[\s,]+", attrs["points"].strip()) if pair]
        if len(pairs) < 4 or len(pairs) % 2:
            raise BuildError(f"`{icon}` 的 <{element}> 的 points 不是成对的坐标")
        head = f"M{_num(float(pairs[0]))} {_num(float(pairs[1]))}"
        rest = [
            f"L{_num(float(pairs[index]))} {_num(float(pairs[index + 1]))}"
            for index in range(2, len(pairs), 2)
        ]
        close = "Z" if element == "polygon" else ""
        return head + "".join(rest) + close
    raise BuildError(
        f"`{icon}` 里有 <{element}>，这个脚本还不认识它。"
        f"要么在 `_shape_to_path` 里补一条转换，要么换一个图标——"
        f"装作没看见会让这个形状从画面上消失，而且不报错"
    )


def _absolute_start(d: str) -> str:
    """Rewrite a path element's leading relative `m` as an absolute `M`.

    Every `<path>` element is its own subpath, so it starts at the origin and
    `m 3 4` draws exactly what `M 3 4` draws — *until* two elements are joined
    into one `d`, at which point the second one's `m` is measured from wherever
    the first one ended. Lucide writes some of its elements that way and Tabler
    starts every one with an absolute `M`, so this is what makes joining them
    safe rather than a difference the two sets happen to have.

    The case alone is not the whole fix, and forgetting that is how the first
    attempt at this went wrong. A command letter that is followed by bare numbers
    **repeats**, and which command it repeats depends on the letter's case: after
    `m` the repeat is a relative `l`, after `M` it is an absolute `L`. So
    `m14 7 1.75-3.767` — move to (14, 7), then draw 1.75 right and 3.767 up —
    becomes `M14 7L1.75 -3.767` if only the case is changed, and the arm's second
    segment lands at y = -4. Naming the repeat keeps what the element meant.
    """
    stripped = d.lstrip()
    if not stripped.startswith("m"):
        return d
    head = _MOVETO_RE.match(stripped)
    if head is None:
        raise BuildError(f"`{d}` 以 m 开头却读不出起点坐标，量不出图形")
    rest = stripped[head.end() :]
    repeated = "l" if _STILL_NUMBERS_RE.match(rest) else ""
    return "M" + stripped[1 : head.end()] + repeated + rest


def art_elements(svg: str, *, icon: str, expected: int) -> list[str]:
    """The path data of every element that draws ink, in document order.

    Each returned string is **self-contained**: joined into one `d` by the
    caller, it still draws what its own element drew. See `_absolute_start` for
    why that needs saying.

    Tabler's spacer path carries its own `stroke="none"`; everything that is
    actually drawn inherits stroke from the `<svg>` element and has no style
    attributes of its own.

    A `<path>` with no `d` is skipped rather than refused, which is what this did
    before the second set arrived. The pinned count below catches it either way:
    the element would simply be missing from the list.
    """
    found: list[str] = []
    for element, raw in _ELEMENT_RE.findall(svg):
        if _SPACER_RE.search(raw):
            continue
        if element == "path":
            match = _D_RE.search(raw)
            if match:
                found.append(_absolute_start(match.group(1)))
            continue
        found.append(_shape_to_path(element, _attrs(raw), icon))
    if len(found) != expected:
        raise BuildError(
            f"`{icon}` 有 {len(found)} 个图形元素，脚本记的是 {expected} 个。"
            f"上游改过这个图标——请核对 part 分配后再改 `art_elements`"
        )
    return found


def build_glyph(name: str, source: GlyphSource) -> dict[str, object]:
    declaration = next((g for g in T2_GLYPHS if g.name == name), None)
    if declaration is None:
        raise BuildError(
            f"`{name}` 不在 registry 的 T2_GLYPHS 里。先声明再构建——"
            f"数据比声明多一个，等于多一个模型看不见、却画得出来的字形"
        )

    svg = fetch_icon(source.icon, source.set)
    paths = art_elements(svg, icon=source.icon, expected=source.art_elements)

    assigned = [index for indices in source.parts.values() for index in indices]
    if sorted(assigned) != list(range(source.art_elements)):
        raise BuildError(
            f"`{name}` 的 part 分配没有恰好盖住每个图形元素：得到 {sorted(assigned)}，"
            f"应当是 {list(range(source.art_elements))}"
        )

    view_box = _VIEW_BOX_RE.search(svg)
    stroke_width = _STROKE_WIDTH_RE.search(svg)
    if view_box is None or stroke_width is None:
        raise BuildError(f"`{source.icon}` 的 <svg> 少了 viewBox 或 stroke-width")

    parts: dict[str, dict[str, object]] = {}
    for part, indices in source.parts.items():
        geometry = " ".join(paths[index] for index in indices)
        entry: dict[str, object] = {
            "d": geometry,
            "mode": STROKE,
            "fill_rule": "nonzero",
            "stroke_width": float(stroke_width.group(1)),
        }
        if part in source.spin:
            # `spin` and `anchor` travel together or not at all. A part marked as
            # turning with no centre to turn about is a part the player cannot
            # place, and the failure would be an icon that sits still — which
            # looks exactly like an icon that is meant to.
            entry["anchor"] = list(_spin_anchor(name, part, geometry))
            entry["spin"] = True
        parts[part] = entry
    box = [float(value) for value in view_box.group(1).split()]
    ink = _ink_box(parts)
    _check_ink_inside_view_box(name, ink, box)

    return {
        "name": name,
        "view_box": box,
        "ink_box": ink,
        "domain": declaration.domain,
        "source": SETS[source.set].source(source.icon),
        "parts": parts,
    }


def _spin_anchor(name: str, part: str, geometry: str) -> tuple[float, float]:
    """The point a turning part turns about: the centre of its largest arc.

    Measured rather than declared, for the same reason as `ink_box` — an anchor
    typed next to geometry that can move is two things that silently stop
    agreeing, and the symptom here is an arm that orbits its icon instead of
    swinging inside it.

    It is also the honest rule rather than a convenience: a part turns about the
    centre of the arc it is drawn from. The radar's arm is drawn from a r=9 arc
    centred on (12, 12), and the car's wheels from r=2 arcs centred on
    themselves — so both would derive correctly, which is the check that the rule
    is a rule and not a fit to one example.

    A `spin` part with no arc has no derivable centre, and the build fails rather
    than picking a default. Silently defaulting to the ink box's middle would
    turn a sweep into something that wobbles, and nothing downstream could tell
    that apart from a design choice.
    """
    found = list(arcs(geometry))
    if not found:
        raise BuildError(
            f"`{name}` 的 part `{part}` 标了 spin，但它的路径里没有圆弧，推不出旋转中心。"
            f"要么给它一段圆弧，要么别让它转——转一个没有圆心的东西只会让它乱飞"
        )
    largest = max(found, key=lambda arc: arc.radius)
    return (round(largest.cx, 4), round(largest.cy, 4))


def _ink_box(parts: dict[str, dict[str, object]]) -> list[float]:
    """The box around what the glyph actually draws, as `[x, y, width, height]`.

    Same shape as `view_box`, and the two answer different questions: `view_box`
    is the grid the icon was designed on, `ink_box` is where the ink landed on
    it. Tabler pads — the car draws in x 3..21, y 6..19 of a 24x24 grid — and
    fitting the grid into a body shrinks the car by exactly that padding. The
    player fits this instead.

    Stroke counts as ink: a stroke straddles its path, so it reaches half the
    stroke width past the geometry on every side. That is 1 unit each way on a
    24-unit grid, which is the same order as the padding being reclaimed.
    """
    boxes = []
    for part in parts.values():
        stroke = float(part["stroke_width"] or 0.0)  # type: ignore[arg-type]
        pad = stroke / 2 if part["mode"] == STROKE else 0.0
        boxes.append(path_bbox(str(part["d"]), pad=pad))

    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    return [min_x, min_y, max_x - min_x, max_y - min_y]


def _check_ink_inside_view_box(name: str, ink: list[float], box: list[float]) -> None:
    """Ink outside the grid means the parser or the part assignment is wrong.

    Cheap to check and the failure it guards is invisible: a glyph that draws
    outside its view box still renders, just clipped or offset, and nobody
    notices until someone compares it against the icon set by eye.
    """
    tolerance = 0.5
    left, top, ink_width, ink_height = ink
    view_width, view_height = box[2], box[3]
    if (
        left < box[0] - tolerance
        or top < box[1] - tolerance
        or left + ink_width > box[0] + view_width + tolerance
        or top + ink_height > box[1] + view_height + tolerance
    ):
        raise BuildError(
            f"`{name}` 的墨迹边界 {ink} 超出了 view_box {box}；"
            f"要么 part 分配错了，要么 path 解析错了"
        )


def render(glyph: dict[str, object]) -> str:
    """The exact bytes of one glyph file. Sorted and LF-only, so it is stable."""
    return json.dumps(glyph, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def check_declared_and_built_match() -> None:
    declared = {g.name for g in T2_GLYPHS}
    built = set(GLYPH_SOURCES)
    if declared != built:
        only_declared = sorted(declared - built)
        only_built = sorted(built - declared)
        raise BuildError(
            "registry 的声明与本脚本能构建的字形对不上："
            f"只声明没得建 {only_declared}，只建没声明 {only_built}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="不写文件，只比对已提交的数据是否与重新构建的结果一致",
    )
    args = parser.parse_args(argv)

    try:
        check_declared_and_built_match()
        built = {name: render(build_glyph(name, src)) for name, src in GLYPH_SOURCES.items()}
    except BuildError as error:
        print(f"构建失败：{error}", file=sys.stderr)
        return 1

    if args.check:
        stale = []
        for name, text in sorted(built.items()):
            path = OUT_DIR / f"{name}.json"
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(name)
        if stale:
            print(
                f"assets/glyphs/ 与重新构建的结果不一致：{'、'.join(stale)}；"
                f"跑一次 `python -m tools.build_glyphs`",
                file=sys.stderr,
            )
            return 1
        print(f"assets/glyphs/ 是最新的（{len(built)} 个字形）")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in sorted(built.items()):
        path = OUT_DIR / f"{name}.json"
        # `newline=""` so Python does not translate to CRLF on Windows: these
        # files are compared byte-for-byte by `--check`.
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
    print(f"写出 {len(built)} 个字形到 {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
