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
Tabler Icons (MIT), pinned by version. Upstream is a flat list of `<path>`
elements on a 24x24 grid — already the coordinate system the glyph schema wants,
so "normalisation" here is mostly *naming*: deciding which paths form one part
and which form another.

Tabler marks its spacer path (`d="M0 0h24v24H0z"`) with `stroke="none"`. That
attribute is the set's own convention for "not artwork", so it is what gets
filtered — not a hard-coded `d` string, which would break on the next icon.

The output is measured, not just copied
---------------------------------------
`ink_box` is computed from the `d` strings by `tools.path_bbox.py` rather than
written down, because `--check` re-fetches upstream: a hand-typed number would
survive a `TABLER_VERSION` bump that moved the art, and the only symptom would be
every glyph drawn subtly the wrong size. The build also asserts the ink lies
inside the view box, which is what catches a parser or part-assignment mistake.

Path counts are pinned, not guessed
-----------------------------------
`GlyphSource.art_paths` is asserted. If upstream reorders or adds a path the
build fails loudly and a human re-checks the part assignment. The alternative is
a glyph that quietly becomes a different picture — and a part assigned to the
wrong path is invisible in the output JSON unless you already know what the icon
looks like.

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

#: Pinned so a rebuild is byte-identical. Bumping this is a deliberate act that
#: re-runs the path-count assertions below.
TABLER_VERSION = "3.46.0"
ICON_URL = "https://unpkg.com/@tabler/icons@{version}/icons/outline/{icon}.svg"

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "assets" / "glyphs"

#: `mode` for every shipped glyph. The outline set is stroke-only by
#: construction; a filled glyph would come from a different upstream directory
#: and would have to be added here deliberately.
STROKE = "stroke"


@dataclass(frozen=True, slots=True)
class GlyphSource:
    """One glyph's provenance and how its paths divide into named parts."""

    icon: str
    art_paths: int
    parts: dict[str, tuple[int, ...]]
    #: Parts that turn about their own arc centre. A list of names rather than a
    #: flag on each part, so "what moves in this glyph" is one line you can read.
    #: The anchor is measured from the geometry; see `_spin_anchor`.
    spin: tuple[str, ...] = ()


def _outline(count: int) -> dict[str, tuple[int, ...]]:
    """Every path in one part, for a glyph that is a single picture."""
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
    "robot": GlyphSource("robot", 9, _outline(9)),
    "cpu": GlyphSource("cpu", 10, _outline(10)),
    "server": GlyphSource("server", 4, _outline(4)),
    "package": GlyphSource("package", 5, _outline(5)),
}

_PATH_RE = re.compile(r"<path\b([^>]*?)/?>", re.DOTALL)
_D_RE = re.compile(r'\bd="([^"]*)"')
_SPACER_RE = re.compile(r'\bstroke="none"')
_VIEW_BOX_RE = re.compile(r'viewBox="([^"]*)"')
_STROKE_WIDTH_RE = re.compile(r'stroke-width="([^"]*)"')


class BuildError(RuntimeError):
    """A glyph could not be built from upstream as declared."""


def fetch_icon(icon: str) -> str:
    url = ICON_URL.format(version=TABLER_VERSION, icon=icon)
    request = urllib.request.Request(url, headers={"User-Agent": "animate-agent-build"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8")
    except urllib.error.URLError as error:  # pragma: no cover - network path
        raise BuildError(f"取不到 {url}：{error}") from error


def art_paths(svg: str, *, icon: str, expected: int) -> list[str]:
    """The `<path>` elements that are artwork, in document order.

    Tabler's spacer path carries its own `stroke="none"`; everything that is
    actually drawn inherits stroke from the `<svg>` element and has no style
    attributes of its own.
    """
    found = [
        match.group(1)
        for attrs in _PATH_RE.findall(svg)
        if not _SPACER_RE.search(attrs)
        for match in [_D_RE.search(attrs)]
        if match
    ]
    if len(found) != expected:
        raise BuildError(
            f"`{icon}` 有 {len(found)} 条图形路径，脚本记的是 {expected} 条。"
            f"上游改过这个图标——请核对 part 分配后再改 `art_paths`"
        )
    return found


def build_glyph(name: str, source: GlyphSource) -> dict[str, object]:
    declaration = next((g for g in T2_GLYPHS if g.name == name), None)
    if declaration is None:
        raise BuildError(
            f"`{name}` 不在 registry 的 T2_GLYPHS 里。先声明再构建——"
            f"数据比声明多一个，等于多一个模型看不见、却画得出来的字形"
        )

    svg = fetch_icon(source.icon)
    paths = art_paths(svg, icon=source.icon, expected=source.art_paths)

    assigned = [index for indices in source.parts.values() for index in indices]
    if sorted(assigned) != list(range(source.art_paths)):
        raise BuildError(
            f"`{name}` 的 part 分配没有恰好盖住每条路径：得到 {sorted(assigned)}，"
            f"应当是 {list(range(source.art_paths))}"
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
        "source": f"tabler:{source.icon}@{TABLER_VERSION}",
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
