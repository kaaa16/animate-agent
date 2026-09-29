"""Lay a StoryboardIR out into a RenderSpec, and put it where a URL can find it.

The player is handed a **URL**, not a value: `frontend/player/player.js` reads
`?spec=` and fetches it. So "where the file is" and "what it is called" is a
contract between whoever writes the spec and whoever plays it — and until the
HTTP API needed to write the same file, that contract lived as three copies
inside `cli.py`. A convention with three copies is one rename away from the
player fetching a file that is not there, and the symptom of that is a picture
that never appears, which reads like a layout bug.

`output_dir` is required and keyword-only throughout, on purpose: shared code
carries no cwd-relative default, so every caller has to say where it is writing.
`knowledge/service.py:14` and `storyboard/service.py:16` both default to
`Path("data/generated")`, which is right for a CLI run from the repository root
and wrong for a server — the API writes what it is about to serve, so it passes
an absolute path. A default here would have let those two disagree in silence.
"""

from __future__ import annotations

from pathlib import Path

from animate_agent.rendering.layout import layout_storyboard
from animate_agent.rendering.models import RenderSpec, RenderStage
from animate_agent.storyboard.models import StoryboardIR


def render_spec_path(render_key: str, *, output_dir: Path) -> Path:
    """Where the RenderSpec called `render_key` lives. The only place that names it.

    `render_key` is the storyboard id for a generated storyboard and the preset
    name for `--render-template`; both are just "the thing this spec is a
    rendering of", which is exactly what the player is told to fetch.
    """
    return output_dir / f"render-{render_key}.json"


def write_render_spec(spec: RenderSpec, destination: Path) -> None:
    """Persist a RenderSpec, creating its directory.

    Three details have to agree with every other producer of a spec — that the
    directory exists, that the encoding is UTF-8, and that the indent is 2 — and
    none of them is interesting enough to be worth reading three times.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(spec.model_dump_json(indent=2), encoding="utf-8")


def render_storyboard(
    storyboard: StoryboardIR,
    *,
    output_dir: Path,
    stage: RenderStage | None = None,
) -> RenderSpec:
    """Lay out `storyboard`, write the spec into `output_dir`, and return it.

    Raises `LayoutError` (a `ValueError`) when the picture does not fit. Layout is
    deterministic, so that refusal is reproducible — and the StoryboardIR is the
    only thing that says why, which is why every caller is expected to keep it.
    """
    spec = layout_storyboard(storyboard, stage=stage)
    write_render_spec(spec, render_spec_path(storyboard.storyboard_id, output_dir=output_dir))
    return spec
