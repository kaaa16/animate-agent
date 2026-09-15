"""`tools/path_bbox.py` measured against boxes traced by hand from the `d` string.

Why a hand trace is the right oracle
------------------------------------
Every other way of checking this module is circular. Comparing it to itself, or to
`ink_box` as committed, only proves the build is reproducible — and the failure
that matters is exactly the one where it is *reproducibly wrong*. So the expected
numbers below were worked out by walking the path grammar on paper:

    M5 17  -> (5,17)   h-2 -> (3,17)   v-6 -> (3,11)   l2 -5 -> (5,6)
    h9     -> (14,6)   l4 5 -> (18,11) h1  -> (19,11)  a2 2 0 0 1 2 2 -> (21,13)
    v4     -> (21,17)  h-2 -> (19,17)  m-4 0 -> (15,17) h-6 -> (9,17)
    m-6 -6 -> (3,11)   h15 -> (18,11)  m-6 0 -> (12,11) v-5 -> (12,6)

giving x 3..21 and y 6..17 — so the car's chassis alone is (3, 6, 21, 17), and
the wheels reach y 19. These are the numbers the assertions use.

What is *not* asserted
----------------------
Exact equality on curves. The module flattens rather than solving for extrema, so
a curve's true extreme can sit a few ten-thousandths outside the sampled bound.
The tests below use a tolerance for anything involving a curve and exact equality
for the straight-line cases, which is the honest split: the module is documented
as a measurement for positioning, not a geometric predicate.
"""

from __future__ import annotations

import pytest
from tools.path_bbox import PathError, path_bbox, path_points

#: The slack allowed where a curve is involved. See the module docstring of
#: `tools/path_bbox.py`: a quarter-arc of radius 2 sampled 32 ways is inside by
#: about 6e-4, so this is three orders of magnitude of headroom, not a fudge.
CURVE_TOLERANCE = 0.01

CAR_CHASSIS = "M5 17h-2v-6l2 -5h9l4 5h1a2 2 0 0 1 2 2v4h-2m-4 0h-6m-6 -6h15m-6 0v-5"
CAR_WHEEL_FRONT = "M15 17a2 2 0 1 0 4 0a2 2 0 1 0 -4 0"


# --------------------------------------------------------------------------
# Straight-line commands — exact, since nothing here is flattened
# --------------------------------------------------------------------------


def test_absolute_and_relative_lines_agree() -> None:
    assert path_bbox("M0 0L10 0L10 5L0 5Z") == (0.0, 0.0, 10.0, 5.0)
    assert path_bbox("m0 0l10 0l0 5l-10 0z") == (0.0, 0.0, 10.0, 5.0)


def test_horizontal_and_vertical_commands_move_one_axis() -> None:
    assert path_bbox("M3 7H9V2H3Z") == (3.0, 2.0, 9.0, 7.0)


def test_an_implicit_repeat_is_the_same_command_again() -> None:
    """`L 1 1 2 2` is two segments, not one with four arguments."""
    assert path_bbox("M0 0L1 1 2 2") == (0.0, 0.0, 2.0, 2.0)


def test_after_a_move_the_implicit_command_is_a_line() -> None:
    """The rule that is easy to get wrong, and silent when it is.

    SVG says a coordinate pair following `M` is an implicit `L`, so `M0 0 5 5`
    travels. Reading it as another `M` gives the same box here — which is why the
    assertion below also checks a case where it would differ: the second pair must
    not reset the subpath start, so a later `Z` returns to (0,0) rather than (5,5).
    """
    assert path_bbox("M0 0 5 5") == path_bbox("M0 0L5 5")
    assert path_bbox("M0 0 5 5L20 20Z") == (0.0, 0.0, 20.0, 20.0)


def test_close_path_returns_to_where_the_subpath_started() -> None:
    # Without tracking the subpath start, the closing line has no far end and the
    # box comes out as just the points that were written down.
    assert path_bbox("M4 4L9 9L1 9Z") == (1.0, 4.0, 9.0, 9.0)


def test_pad_grows_the_box_on_every_side() -> None:
    assert path_bbox("M0 0L10 0L10 5L0 5Z", pad=1.0) == (-1.0, -1.0, 11.0, 6.0)


# --------------------------------------------------------------------------
# Curves — sampled, so compared with a tolerance
# --------------------------------------------------------------------------


def test_the_car_chassis_is_where_the_hand_trace_puts_it() -> None:
    """The oracle test. These four numbers were traced by hand from the `d`."""
    min_x, min_y, max_x, max_y = path_bbox(CAR_CHASSIS)

    assert min_x == pytest.approx(3.0)
    assert min_y == pytest.approx(6.0)
    assert max_x == pytest.approx(21.0)
    assert max_y == pytest.approx(17.0)


def test_an_arc_does_not_escape_the_box_its_endpoints_describe() -> None:
    """The corner radius on the car's nose bulges *inward*, not outward.

    Worth pinning because the opposite is common: an arc that sweeps the long way
    round reaches past both endpoints, and a module that only looked at endpoints
    would report a box smaller than the ink.
    """
    box = path_bbox(CAR_CHASSIS)
    # The chassis already reaches x 21 at the arc's end; the arc must not add to it.
    assert box[2] == pytest.approx(21.0, abs=CURVE_TOLERANCE)


def test_a_half_circle_arc_reaches_past_its_endpoints() -> None:
    """The case the test above does *not* cover, so both behaviours are fixed.

    From (0,0) to (10,0) with radius 5 and `large-arc=0 sweep=1`, the arc bulges
    to y -5. An endpoint-only implementation reports y 0 and the glyph is fitted
    as if it were flat.
    """
    min_x, min_y, max_x, max_y = path_bbox("M0 0A5 5 0 0 1 10 0")

    assert (min_x, max_x) == (pytest.approx(0.0), pytest.approx(10.0))
    assert min_y == pytest.approx(-5.0, abs=CURVE_TOLERANCE)
    assert max_y == pytest.approx(0.0, abs=CURVE_TOLERANCE)


def test_a_wheel_is_a_square_box_because_a_circle_is_round() -> None:
    """Both wheel paths are full circles, so the box is square and centred.

    This is the derivation `GlyphPart.anchor` will rest on: a wheel spins about
    the middle of its own box, so the anchor is measurable and does not have to be
    typed next to the geometry it has to agree with.
    """
    min_x, min_y, max_x, max_y = path_bbox(CAR_WHEEL_FRONT)

    assert round(max_x - min_x, 3) == round(max_y - min_y, 3) == 4.0
    assert ((min_x + max_x) / 2, (min_y + max_y) / 2) == (17.0, 17.0)


def test_zero_radius_arc_is_a_line() -> None:
    """The spec's own degenerate case; a division by a zero radius would raise."""
    assert path_bbox("M0 0A0 0 0 0 1 10 10") == (0.0, 0.0, 10.0, 10.0)


def test_radii_too_small_to_reach_are_scaled_up_like_a_renderer_does() -> None:
    # SVG grows impossible radii rather than rejecting the path, so the box has
    # to agree with what would actually be drawn.
    box = path_bbox("M0 0A1 1 0 0 1 10 0")

    assert box[2] >= 9.0


def test_reflected_control_points_are_tracked_for_s_and_t() -> None:
    """`S` and `T` reuse the previous curve's control point, mirrored.

    A module that treated the missing control point as the current point would
    flatten the curve and report a box that is too small — the same failure as
    looking only at endpoints.
    """
    # A smooth continuation: the second curve mirrors the first and keeps rising.
    box = path_bbox("M0 0C0 -4 4 -4 4 0S8 4 8 0")

    assert box[0] == pytest.approx(0.0)
    assert box[2] == pytest.approx(8.0)


# --------------------------------------------------------------------------
# Failure — a `d` this module cannot read says so
# --------------------------------------------------------------------------


def test_an_empty_path_is_an_error_not_an_empty_box() -> None:
    """A zero-size box would fit as a scale of infinity and draw nothing.

    The caller cannot tell that apart from a glyph that legitimately draws
    nothing, so the module refuses instead.
    """
    with pytest.raises(PathError):
        path_bbox("")


def test_too_few_arguments_names_the_command_and_the_path() -> None:
    with pytest.raises(PathError, match="M5"):
        path_bbox("M5 5L1")


def test_a_path_starting_with_a_number_is_an_error() -> None:
    with pytest.raises(PathError):
        path_bbox("5 5L1 1")


def test_points_can_be_walked_without_a_box() -> None:
    # The walk is the useful part; the box is a fold over it.
    assert list(path_points("M0 0L1 1")) == [(0.0, 0.0), (1.0, 1.0)]
