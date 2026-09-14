"""Tests for the hand-authored storyboard samples in `data/samples/storyboards/`.

These samples exist because the pipeline's first real output could not be
obtained, and because the quality baseline — the robot obstacle-avoidance
animation — has to be expressible in the new contract *before* any layout code
is written against it. A sample that stops validating is the alarm for that.

They are hand-written, so they are **not** evidence of what the model produces.
They are evidence that the contract can hold what the baseline draws.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from animate_agent.animation.templates import build_robot_obstacle_avoidance_scene
from animate_agent.storyboard.models import StoryboardIR
from animate_agent.storyboard.validation import (
    StoryboardLimits,
    format_issues,
    validate_storyboard,
)

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "samples" / "storyboards"

# As configured in `config/app.example.yaml`.
LIMITS = StoryboardLimits(
    min_steps=3,
    max_steps=7,
    require_visual_objects=True,
    require_interactive_demo=True,
    allowed_renderers=("canvas_2d", "svg_2d", "three_3d"),
)


def _sample(name: str) -> StoryboardIR:
    raw = json.loads((SAMPLES_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return StoryboardIR.model_validate(raw)


def _sample_names() -> list[str]:
    return sorted(path.stem for path in SAMPLES_DIR.glob("*.json"))


def test_the_samples_directory_is_not_empty() -> None:
    # Guards against a glob that silently matches nothing and makes every
    # parametrised case below vacuous.
    assert _sample_names()


@pytest.mark.parametrize("name", _sample_names())
def test_every_sample_validates(name: str) -> None:
    issues = validate_storyboard(_sample(name), limits=LIMITS)

    assert issues == [], format_issues(issues)


def test_the_baseline_sample_carries_every_baseline_control() -> None:
    """The baseline turns four knobs; all four must survive the translation.

    This is the quality directive made checkable at the StoryboardIR level: if
    the contract cannot name what the old animation could do, no renderer will
    reach its quality. `test_baseline_controls_resolve_against_the_registry`
    already checks this against the registry; this checks it end to end.
    """
    [scene] = _sample("robot_obstacle_avoidance").scenes
    baseline = build_robot_obstacle_avoidance_scene()

    translated = {(control.type, control.target_property) for control in scene.controls}
    expected = {
        (control.type, control.target_property) for control in baseline.controls
    }

    assert translated == expected


def test_the_baseline_sample_reuses_the_baseline_wording() -> None:
    """The step captions are the baseline's own narration, verbatim.

    Deliberate: it is what proved the `description` floor had been set above
    the writing the project is trying to match (see D5).
    """
    [scene] = _sample("robot_obstacle_avoidance").scenes
    baseline = build_robot_obstacle_avoidance_scene()

    assert [step.description for step in scene.steps] == [
        step.narration for step in baseline.timeline
    ]


def test_the_baseline_sample_anchors_every_object() -> None:
    """Every object is either highlighted by a beat, or anchored by a relation.

    `orphan_object` rejects anything that is neither — "drawn but never taught".
    The baseline's `obstacle-2` is purely scenic and carries no relation, so it
    would fail; the sample therefore highlights both obstacles in the final beat,
    which is also the honest reading (comparing left against right clearance is a
    statement about both of them).

    Recorded as a test because it is a real tension between scenic decoration and
    the orphan rule, and the next person to hit it should find this first.
    """
    [scene] = _sample("robot_obstacle_avoidance").scenes
    highlighted = {name for step in scene.steps for name in step.highlights}
    anchored = {obj.id for obj in scene.objects if obj.props.get("of") is not None}

    assert {obj.id for obj in scene.objects} - highlighted - anchored == set()
    assert "obstacle-b" in highlighted
