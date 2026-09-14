"""Tests for the Storyboard prompt's generated constraint block.

The prompt teaches mostly by example, so before this block existed every bound
that lived only in the schema was a bound the model had never been told — and
it violated one. The first real run came back with 33 of 33 steps under a
`description` floor the prompt never mentioned, and two object ids that failed
a pattern the prompt never mentioned either.
"""

from __future__ import annotations

import re

import pytest

from animate_agent.storyboard.models import StoryboardIR, StoryboardStep
from animate_agent.storyboard.prompts import (
    _CONSTRAINED_FIELDS,
    STORYBOARD_SYSTEM_PROMPT,
    _read_bounds,
    render_field_constraints,
)


def test_every_constrained_field_is_listed() -> None:
    block = render_field_constraints()

    for label, _, _ in _CONSTRAINED_FIELDS:
        assert f"`{label}`" in block, label


def test_the_block_reaches_the_system_prompt() -> None:
    assert render_field_constraints() in STORYBOARD_SYSTEM_PROMPT


def test_each_rendered_line_states_exactly_the_declared_bounds() -> None:
    """The block is generated, so it cannot lag a changed bound.

    Compared as a set of numbers rather than against the expected wording, so
    this asserts the *content* without re-implementing the formatting. Any extra
    number on the line fails too, which is how a bound that is no longer in the
    schema would be caught.
    """
    block = render_field_constraints()

    for label, model, field_name in _CONSTRAINED_FIELDS:
        min_length, max_length, _ = _read_bounds(model.model_fields[field_name])
        line = next(line for line in block.splitlines() if line.startswith(f"- `{label}`"))
        # The id regex carries digits of its own; they are not bounds.
        bounds_text = line.split("，必须匹配")[0]
        declared = {bound for bound in (min_length, max_length) if bound is not None}

        assert {int(n) for n in re.findall(r"\d+", bounds_text)} == declared, line


def test_the_id_pattern_is_stated() -> None:
    """Physics quantities are what broke this: `V₀` / `θ` / `R` all fail the
    pattern, and nothing in the prompt said ids were ASCII-only."""
    _, _, pattern = _read_bounds(StoryboardStep.model_fields["id"])

    assert pattern is not None
    assert pattern in STORYBOARD_SYSTEM_PROMPT


def test_the_description_floor_is_stated() -> None:
    min_length, max_length, _ = _read_bounds(StoryboardStep.model_fields["description"])

    assert f"{min_length}~{max_length}" in STORYBOARD_SYSTEM_PROMPT


def test_derived_fields_are_not_offered_to_the_model() -> None:
    """The agent assigns these itself; stating their bounds invites the model to write them."""
    listed = {label for label, _, _ in _CONSTRAINED_FIELDS}

    assert not listed & {"storyboard_id", "lesson_id", "document_id", "title", "subject"}


def test_read_bounds_returns_nothing_for_an_unconstrained_field() -> None:
    min_length, max_length, pattern = _read_bounds(StoryboardIR.model_fields["eyebrow"])

    assert (min_length, max_length, pattern) == (None, None, None)


@pytest.mark.parametrize(
    ("label", "model", "field_name"),
    _CONSTRAINED_FIELDS,
    ids=[label for label, _, _ in _CONSTRAINED_FIELDS],
)
def test_every_listed_field_exists(label: str, model: type, field_name: str) -> None:
    # The table above is hand-written; the lookups are not. A renamed field has
    # to fail loudly here rather than 500-ing inside prompt construction.
    assert field_name in model.model_fields, label
