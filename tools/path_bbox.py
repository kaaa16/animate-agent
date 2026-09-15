"""The tight box around what an SVG path string actually draws.

Why this exists
---------------
A glyph's `view_box` is a *design grid*, not a measurement. Tabler draws every
icon on a 24x24 grid and pads the art inside it — the car's ink spans x 3..21,
y 6..19, so a third of the width and a fifth of the height are empty. Fitting the
view box into a body therefore draws the car smaller than the body it was given,
by exactly the amount the icon set happens to pad.

The fix is to fit the ink. That needs the box around the ink, and it has to be
computed rather than typed: `--check` re-fetches upstream, so a hand-written
number survives a `TABLER_VERSION` bump that moves the art, and nothing notices.
`Glyph.ink_box` is that computed number.

Accuracy
--------
Curves are **flattened**, not solved: each segment is sampled and the box is
taken over the samples. So the result is a hair *inside* the true bound — for the
sagitta of a quarter-arc of radius 2 sampled 32 ways that is 6e-4 units. This
number positions a drawing; it is not a geometric predicate, and an analytic
extrema solve would buy four decimal places nobody can see for a page of
derivative roots.

The stroke counts as ink. A stroke is centred on the path, so it reaches
`stroke_width / 2` past the geometry on every side — which for a 2-unit stroke on
a 24-unit grid is 4% of the icon, precisely the order of padding this module
exists to stop wasting.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator

#: How many chords approximate one curve. See the accuracy note in the module
#: docstring — this is chosen for "well under a pixel", not for exactness.
CURVE_SAMPLES = 24
ARC_SAMPLES = 32

#: Argument count per command. `Z` is here with zero so the table is the full
#: grammar rather than the grammar minus the case that needs special handling.
_ARITY = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}

_TOKEN_RE = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?")


class PathError(ValueError):
    """A `d` string this module cannot read."""


def _numbers(tokens: list[str], position: int, count: int, *, letter: str, d: str) -> list[float]:
    chunk = tokens[position : position + count]
    if len(chunk) < count or any(
        not token[0].isdigit() and token[0] not in "-." for token in chunk
    ):
        raise PathError(f"`{d}` 里 `{letter}` 的参数不够：需要 {count} 个，只找到 {chunk}")
    return [float(token) for token in chunk]


def commands(d: str) -> Iterator[tuple[str, list[float]]]:
    """Walk `d` as (command letter, arguments) pairs, implicit repetition included.

    SVG lets one command letter carry several argument groups — `L 1 1 2 2` is two
    line segments — and after an `M` the implicit command is `L`, not `M`. Both
    rules are handled here so callers see one group at a time.
    """
    tokens = _TOKEN_RE.findall(d)
    position = 0
    letter = ""
    while position < len(tokens):
        token = tokens[position]
        if token[0].isalpha():
            letter = token
            position += 1
            if letter in "Zz":
                # Zero arguments, so there is nothing to repeat: a following
                # number would have no command to belong to.
                yield letter, []
                letter = ""
                continue
        elif not letter:
            # Before the implicit-repeat branch, not after: `"" in "Mm"` is `True`
            # in Python, so a `d` starting with a number would otherwise be read
            # as an implicit `L` and decoded into a box for a path nobody wrote.
            raise PathError(f"`{d}` 以数字开头，没有命令字母")
        elif letter in "Mm":
            letter = "l" if letter == "m" else "L"
        arguments = _numbers(tokens, position, _ARITY[letter.upper()], letter=letter, d=d)
        position += _ARITY[letter.upper()]
        yield letter, arguments


def _cubic(
    start: tuple[float, float],
    control1: tuple[float, float],
    control2: tuple[float, float],
    end: tuple[float, float],
) -> list[tuple[float, float]]:
    x0, y0 = start
    return [
        (
            (1 - t) ** 3 * x0
            + 3 * (1 - t) ** 2 * t * control1[0]
            + 3 * (1 - t) * t**2 * control2[0]
            + t**3 * end[0],
            (1 - t) ** 3 * y0
            + 3 * (1 - t) ** 2 * t * control1[1]
            + 3 * (1 - t) * t**2 * control2[1]
            + t**3 * end[1],
        )
        for t in (index / CURVE_SAMPLES for index in range(1, CURVE_SAMPLES + 1))
    ]


def _quadratic(
    start: tuple[float, float], control: tuple[float, float], end: tuple[float, float]
) -> list[tuple[float, float]]:
    x0, y0 = start
    return [
        (
            (1 - t) ** 2 * x0 + 2 * (1 - t) * t * control[0] + t**2 * end[0],
            (1 - t) ** 2 * y0 + 2 * (1 - t) * t * control[1] + t**2 * end[1],
        )
        for t in (index / CURVE_SAMPLES for index in range(1, CURVE_SAMPLES + 1))
    ]


def _arc(
    start: tuple[float, float],
    rx: float,
    ry: float,
    rotation: float,
    large_arc: int,
    sweep: int,
    end: tuple[float, float],
) -> list[tuple[float, float]]:
    """An `A` command as sampled points, per the endpoint-to-centre conversion in
    SVG 1.1 appendix F.6.5."""
    if rx == 0 or ry == 0:
        return [end]
    x1, y1 = start
    x2, y2 = end
    rx, ry = abs(rx), abs(ry)
    phi = math.radians(rotation % 360)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)

    half_x, half_y = (x1 - x2) / 2, (y1 - y2) / 2
    x1p = cos_phi * half_x + sin_phi * half_y
    y1p = -sin_phi * half_x + cos_phi * half_y

    # Radii too small to reach the endpoint are scaled up rather than rejected;
    # this is what a renderer does, so the box must agree with it.
    lam = (x1p / rx) ** 2 + (y1p / ry) ** 2
    if lam > 1:
        scale = math.sqrt(lam)
        rx, ry = rx * scale, ry * scale

    denominator = (rx * y1p) ** 2 + (ry * x1p) ** 2
    numerator = (rx * ry) ** 2 - (rx * y1p) ** 2 - (ry * x1p) ** 2
    factor = 0.0 if denominator == 0 else math.sqrt(max(0.0, numerator / denominator))
    if large_arc == sweep:
        factor = -factor
    cxp = factor * rx * y1p / ry
    cyp = -factor * ry * x1p / rx
    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2

    def angle(ux: float, uy: float, vx: float, vy: float) -> float:
        norm = math.hypot(ux, uy) * math.hypot(vx, vy)
        if norm == 0:
            return 0.0
        found = math.acos(max(-1.0, min(1.0, (ux * vx + uy * vy) / norm)))
        return -found if ux * vy - uy * vx < 0 else found

    ux, uy = (x1p - cxp) / rx, (y1p - cyp) / ry
    vx, vy = (-x1p - cxp) / rx, (-y1p - cyp) / ry
    theta = angle(1.0, 0.0, ux, uy)
    delta = angle(ux, uy, vx, vy)
    if not sweep and delta > 0:
        delta -= 2 * math.pi
    elif sweep and delta < 0:
        delta += 2 * math.pi

    return [
        (
            cx + rx * math.cos(t) * cos_phi - ry * math.sin(t) * sin_phi,
            cy + rx * math.cos(t) * sin_phi + ry * math.sin(t) * cos_phi,
        )
        for t in (theta + delta * index / ARC_SAMPLES for index in range(1, ARC_SAMPLES + 1))
    ]


def path_points(d: str) -> Iterator[tuple[float, float]]:
    """Every point on the path that can bound it: endpoints and sampled curves."""
    x = y = 0.0
    start_x = start_y = 0.0
    #: Reflection source for `S`/`T`, which reuse the previous curve's control
    #: point. `None` means "the previous command was not that curve kind", which
    #: the spec resolves by coinciding the control point with the current point.
    cubic_control: tuple[float, float] | None = None
    quad_control: tuple[float, float] | None = None

    for letter, args in commands(d):
        upper = letter.upper()
        relative = letter.islower()
        here = (x, y)

        if upper == "M":
            x, y = (x + args[0], y + args[1]) if relative else (args[0], args[1])
            start_x, start_y = x, y
            yield x, y
        elif upper == "L":
            x, y = (x + args[0], y + args[1]) if relative else (args[0], args[1])
            yield x, y
        elif upper == "H":
            x = x + args[0] if relative else args[0]
            yield x, y
        elif upper == "V":
            y = y + args[0] if relative else args[0]
            yield x, y
        elif upper == "C":
            points = [
                (args[0] + x, args[1] + y) if relative else (args[0], args[1]),
                (args[2] + x, args[3] + y) if relative else (args[2], args[3]),
                (args[4] + x, args[5] + y) if relative else (args[4], args[5]),
            ]
            yield from _cubic(here, points[0], points[1], points[2])
            x, y = points[2]
            cubic_control = points[1]
        elif upper == "S":
            control1 = (
                here
                if cubic_control is None
                else (2 * x - cubic_control[0], 2 * y - cubic_control[1])
            )
            points = [
                (args[0] + x, args[1] + y) if relative else (args[0], args[1]),
                (args[2] + x, args[3] + y) if relative else (args[2], args[3]),
            ]
            yield from _cubic(here, control1, points[0], points[1])
            x, y = points[1]
            cubic_control = points[0]
        elif upper == "Q":
            points = [
                (args[0] + x, args[1] + y) if relative else (args[0], args[1]),
                (args[2] + x, args[3] + y) if relative else (args[2], args[3]),
            ]
            yield from _quadratic(here, points[0], points[1])
            x, y = points[1]
            quad_control = points[0]
        elif upper == "T":
            control = (
                here if quad_control is None else (2 * x - quad_control[0], 2 * y - quad_control[1])
            )
            end = (args[0] + x, args[1] + y) if relative else (args[0], args[1])
            yield from _quadratic(here, control, end)
            x, y = end
            quad_control = control
        elif upper == "A":
            end = (args[5] + x, args[6] + y) if relative else (args[5], args[6])
            yield from _arc(here, args[0], args[1], args[2], int(args[3]), int(args[4]), end)
            x, y = end
        elif upper == "Z":
            x, y = start_x, start_y
            yield x, y

        if upper not in "CS":
            cubic_control = None
        if upper not in "QT":
            quad_control = None


def path_bbox(d: str, *, pad: float = 0.0) -> tuple[float, float, float, float]:
    """`(min_x, min_y, max_x, max_y)` of what `d` draws, plus `pad` on every side."""
    points = list(path_points(d))
    if not points:
        raise PathError(f"`{d}` 上没有任何点——空路径量不出边界")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
