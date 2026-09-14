"""Lay out a *stored* StoryboardIR and persist the RenderSpec — no LLM call.

Why this exists
---------------
`animate-agent <doc> --render` is the real path, and it costs two LLM calls
every time you want to know whether the picture changed. That is the right
price for finding out what the model does, and the wrong price for finding out
what `layout.py` does — the storyboard is already on disk, and only the
deterministic half of the pipeline needs to run again.

So this is the cheap loop: read the artefact back, lay it out, write the spec.
It answers exactly one question, *can this storyboard be arranged, and where do
the objects land* — and it answers it the same way the real run would, because
it calls the same `layout_storyboard`.

Usage
-----

    python -m tools.layout_storyboard data/generated/m5-*/storyboard-*.json \\
        --output-dir data/generated/m5-layout

Point a browser at `frontend/player/index.html?spec=<the written file>` to look
at the result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from animate_agent.rendering.layout import LayoutError, layout_storyboard
from animate_agent.rendering.models import RenderSpec
from animate_agent.rendering.registry import ROLE_TO_PRIMITIVE
from animate_agent.storyboard.models import StoryboardIR

EXIT_OK = 0
EXIT_MISSING = 1
EXIT_LAYOUT_FAILED = 2


def _describe(scene_id: str, spec: RenderSpec) -> list[str]:
    """One line per element, in the order the player will draw them.

    Printing the primitive as well as the role is the point: the two are
    different levels and a reader who confuses them reads `axis` as a role. The
    coordinates are what a comparison against the baseline is actually made of.
    """
    scene = next(item for item in spec.scenes if item.id == scene_id)
    lines = [f"  {scene.id}（预设 {scene.preset}，{len(scene.elements)} 个元素）"]
    for element in scene.elements:
        role = element.role
        primitive = ROLE_TO_PRIMITIVE.get(role, "?")
        geometry = ""
        if hasattr(element, "points"):
            geometry = f"，{len(element.points)} 个点"
        elif hasattr(element, "length"):
            geometry = f"，长 {element.length:.0f} @ {element.heading:.0f}°"
        elif hasattr(element, "dx"):
            geometry = f"，Δ({element.dx:.0f}, {element.dy:.0f})"
        elif hasattr(element, "radius"):
            geometry = f"，半径 {element.radius:.0f}"
        lines.append(
            f"    {element.id:<18} {primitive:<10} ({role:<12}) "
            f"({element.x:7.1f}, {element.y:6.1f}){geometry}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("storyboard", type=Path, help="已落盘的 StoryboardIR JSON")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated"),
        help="RenderSpec 落盘目录，默认 data/generated",
    )
    args = parser.parse_args(argv)

    if not args.storyboard.exists():
        print(f"找不到文件: {args.storyboard}", file=sys.stderr)
        return EXIT_MISSING

    storyboard = StoryboardIR.model_validate_json(args.storyboard.read_text(encoding="utf-8"))
    try:
        spec = layout_storyboard(storyboard)
    except LayoutError as exc:
        print(f"布局失败: {exc}", file=sys.stderr)
        return EXIT_LAYOUT_FAILED

    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"render-{storyboard.storyboard_id}.json"
    destination.write_text(spec.model_dump_json(indent=2), encoding="utf-8")

    for scene in spec.scenes:
        print("\n".join(_describe(scene.id, spec)))
    print(f"已写入 {destination}", file=sys.stderr)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
