"""The shipped glyph data, checked against the registry that promises it exists.

`registry.py` says every declared glyph has data in `assets/glyphs/`, and that the
two sets are equal. This is the test that makes that sentence true rather than
aspirational. Nothing else would notice the gap: `layout._glyph_to_draw` drops a
requested glyph it has no data for and draws the parametric shape instead, so a
declared-but-absent name is a promise the prompt makes to the model and the
picture quietly breaks.

No network here on purpose. `tools/build_glyphs.py` is what talks to the icon set,
and it needs the network, so it is a tool: this file checks the half that can be
checked offline — that what is committed is well-formed and self-consistent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.path_bbox import arcs, path_bbox

from animate_agent.rendering.registry import T2_GLYPHS, Glyph

GLYPH_DIR = Path(__file__).resolve().parents[2] / "assets" / "glyphs"

#: A glyph drawn outside the grid it declares is clipped or offset depending on
#: the renderer, and looks deliberate either way.
INK_TOLERANCE = 0.5


def _glyph_files() -> list[Path]:
    return sorted(GLYPH_DIR.glob("*.json"))


def _load(path: Path) -> Glyph:
    return Glyph.model_validate_json(path.read_text(encoding="utf-8"))


def test_the_declared_set_is_the_shipped_set() -> None:
    """Both directions, because the two failures cost different things.

    A declared name with no file is a promise the picture breaks. A file with no
    declaration is a glyph the model is never offered and `build_glyphs.py`
    refuses to rebuild — dead weight that looks like a working asset.
    """
    declared = {declaration.name for declaration in T2_GLYPHS}
    shipped = {path.stem for path in _glyph_files()}

    assert declared - shipped == set(), "声明了却没有数据，模型照着写就会掉进兜底形状"
    assert shipped - declared == set(), "有数据却没有声明，构建脚本不肯重建它"


def test_the_directory_is_not_empty() -> None:
    # Guards the vacuous pass: if the directory went missing, the set equality
    # above would only hold for an equally empty declaration list.
    assert _glyph_files(), f"{GLYPH_DIR} 里一个字形都没有"


@pytest.mark.parametrize("path", _glyph_files(), ids=lambda path: path.stem)
def test_every_shipped_glyph_matches_the_schema(path: Path) -> None:
    glyph = _load(path)

    assert glyph.name == path.stem, "文件名与 name 字段对不上，按名取用时就会错"
    assert glyph.source.startswith("tabler:"), glyph.source


@pytest.mark.parametrize("path", _glyph_files(), ids=lambda path: path.stem)
def test_every_glyph_draws_inside_its_own_view_box(path: Path) -> None:
    """The ink box and the design grid have to agree about where the art is.

    `build_glyphs.py` asserts this too, but that assertion needs the network. This
    is the offline half, and it is the one that runs on a hand-edited file: ink
    outside the grid means the geometry and the box came from different icons, or
    that the box was measured against a version of the art that has since moved.
    """
    glyph = _load(path)
    view_x, view_y, view_width, view_height = glyph.view_box
    ink_x, ink_y, ink_width, ink_height = glyph.ink_box

    assert ink_width > 0 and ink_height > 0, "空墨迹量不出边界，玩家会算出一个 NaN 缩放"
    assert ink_x >= view_x - INK_TOLERANCE, path.stem
    assert ink_y >= view_y - INK_TOLERANCE, path.stem
    assert ink_x + ink_width <= view_x + view_width + INK_TOLERANCE, path.stem
    assert ink_y + ink_height <= view_y + view_height + INK_TOLERANCE, path.stem


#: The glyphs that are more than one picture, and what their parts are.
#:
#: Everything else is one part called `outline`. A name only appears here because
#: splitting it was a decision: merging a car and its wheels, or a radar and its
#: arm, into a single path is the thing `parts` exists to prevent — and so is
#: splitting something that has no reason to move on its own. A new entry is
#: meant to be noticed.
MULTI_PART: dict[str, set[str]] = {
    "car": {"body", "wheelFront", "wheelRear"},
    "lidar": {"outline", "sweep"},
}


@pytest.mark.parametrize("path", _glyph_files(), ids=lambda path: path.stem)
def test_a_glyph_is_one_picture_unless_it_is_named_above(path: Path) -> None:
    glyph = _load(path)
    expected = MULTI_PART.get(glyph.name)
    if expected is None:
        assert list(glyph.parts) == ["outline"], f"{path.stem} 的 part 划分没有说明理由"
        return

    assert set(glyph.parts) == expected


def test_a_turning_part_carries_the_point_it_turns_about() -> None:
    """`spin` without `anchor` is a promise the player cannot keep.

    The player throws on that pair rather than drawing the part still, because a
    part that was meant to turn and does not is indistinguishable on screen from
    one that was never meant to. This is the data half of the same rule.
    """
    for path in _glyph_files():
        for name, part in _load(path).parts.items():
            if not part.spin:
                continue
            assert part.anchor is not None, f"{path.stem}.{name} 标了 spin 却没有 anchor"


def test_the_lidar_arm_turns_and_the_car_wheels_do_not() -> None:
    """The named reader, and the named non-reader, so neither drifts silently.

    The lidar's arm is what `anchor`/`spin` were for: Tabler's radar path 0 is a
    90-degree sector, so rotating it about the corner the straight edges meet at
    is a picture that changes. The car's wheels are the case that *looks* like it
    should turn and must not be made to: both are exact circles, so rotating one
    about its own centre is the identity — code that would run every frame and
    move nothing, which is the same inert shape as a prop no drawer reads.

    Asserting the negative is the unusual half and the deliberate one. A wheel
    marked `spin` would look like a feature in the data and be invisible in the
    picture, and nothing else here would notice.
    """
    lidar = _load(GLYPH_DIR / "lidar.json")
    car = _load(GLYPH_DIR / "car.json")

    assert lidar.parts["sweep"].spin
    assert lidar.parts["sweep"].anchor == (12.0, 12.0)
    assert not lidar.parts["outline"].spin

    for name, part in car.parts.items():
        assert not part.spin, f"车字形的 {name} 不该转：正圆绕圆心转是恒等变换"


def test_the_anchor_is_the_centre_of_the_arc_the_part_is_drawn_from() -> None:
    """The rule the build uses, restated here against the shipped data.

    `tools/build_glyphs.py` derives an anchor as the centre of the part's largest
    arc rather than taking a number someone typed beside the geometry. This
    checks the committed files still satisfy that — so an anchor that was
    hand-edited, or measured against a version of the art that has since moved,
    fails here rather than as an arm that orbits its own icon.

    The two cases are both checked because one of them is not in the data: the
    car's wheels satisfy the rule too, and asserting that is what separates "a
    rule" from "a fit to the one example that needed it".
    """
    lidar = _load(GLYPH_DIR / "lidar.json")
    sweep = list(arcs(lidar.parts["sweep"].d))
    assert sweep, "扫描臂没有圆弧，推不出圆心"
    largest = max(sweep, key=lambda arc: arc.radius)

    assert lidar.parts["sweep"].anchor == pytest.approx((largest.cx, largest.cy))
    # Both of the arm's arcs share a centre — a radius-1 hub arc and the radius-9
    # sweep — which is why "the largest arc" is not a choice between two answers.
    assert {(round(arc.cx, 6), round(arc.cy, 6)) for arc in sweep} == {(12.0, 12.0)}

    car = _load(GLYPH_DIR / "car.json")
    for name in ("wheelFront", "wheelRear"):
        wheel = max(arcs(car.parts[name].d), key=lambda arc: arc.radius)
        min_x, min_y, max_x, max_y = path_bbox(car.parts[name].d)
        assert (wheel.cx, wheel.cy) == ((min_x + max_x) / 2, (min_y + max_y) / 2), name


PLAYER_DIR = Path(__file__).resolve().parents[2] / "frontend" / "player"


def test_the_player_reads_the_spin_fields() -> None:
    """The data has to reach the drawing code, not just the schema.

    `anchor` and `spin` were in `GlyphPart` from the first commit with nothing
    emitting them and nothing reading them — a declared field is not a feature,
    which is the failure this whole round is about. This is the cheap half of the
    check; `tools/player_smoke.mjs` is the half that drives the drawer and
    compares two frames of the arm, and it is the one that would catch a reader
    that read the fields and then did nothing with them.
    """
    source = (PLAYER_DIR / "glyphs.js").read_text(encoding="utf-8")

    assert "part.spin" in source
    assert "part.anchor" in source
    # A turn rather than a nudge: translation would also make two frames differ.
    assert "ctx.rotate(" in source
