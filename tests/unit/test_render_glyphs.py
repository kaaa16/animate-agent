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
from tools.path_bbox import path_bbox

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


@pytest.mark.parametrize("path", _glyph_files(), ids=lambda path: path.stem)
def test_every_glyph_has_a_part_named_outline_or_a_reason_not_to(path: Path) -> None:
    """`outline` is the convention for a glyph that is one picture.

    The car is the exception, and it is the case the schema's `parts` exists for:
    its wheels have to turn without the chassis turning. Any *new* exception will
    fail here, which is the point — a second multi-part glyph should be a
    decision someone made on purpose, not something that appeared.
    """
    glyph = _load(path)
    if glyph.name == "car":
        assert set(glyph.parts) == {"body", "wheelFront", "wheelRear"}
        return

    assert list(glyph.parts) == ["outline"], f"{path.stem} 的 part 划分没有说明理由"


def test_a_wheel_turns_about_a_centre_that_can_be_measured() -> None:
    """The anchor a wheel spins about is derivable, so it is not typed anywhere.

    `GlyphPart.anchor` is a declared field with no reader yet: the schema has it,
    nothing emits it, nothing consumes it. That is the shape this project keeps
    catching, and the difference here is that it is named as pending rather than
    assumed done. What this test pins is the half that must not be hand-written
    when the reader arrives — a wheel is a circle, so its centre is the middle of
    its own bounding box, and typing `[17, 17]` next to a computed `d` is how the
    two silently stop agreeing.

    Today the assertion is just that the derivation works and gives a circle.
    When the wheel turns, the anchor is this number, not a new one.
    """
    car = _load(GLYPH_DIR / "car.json")

    centres = {}
    for name, part in car.parts.items():
        if not name.startswith("wheel"):
            continue
        min_x, min_y, max_x, max_y = path_bbox(part.d)
        centres[name] = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        # A circle is as wide as it is tall — the property `anchor` relies on to
        # be a single point rather than two.
        assert max_x - min_x == pytest.approx(max_y - min_y), name

    assert centres == {"wheelFront": (17.0, 17.0), "wheelRear": (7.0, 17.0)}
    # Both wheels share an axis, which is what makes the car sit level.
    assert len({y for _, y in centres.values()}) == 1
