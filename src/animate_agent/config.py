"""Application configuration loaded from YAML."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field


class KnowledgeSettings(BaseModel):
    """Tunable parameters for the Knowledge Agent."""

    model_config = ConfigDict(extra="forbid")

    # 12, not 8. A one-minute explainer is about ten pictures, not six: the
    # reference `anything2explainer` fits 44 shots into 275 seconds — one every
    # 6.2 — so a 60-second cut of the same density is ten. The old 8 was chosen
    # when nothing had measured how long a picture may stay on screen, and the
    # answer turned out to be "not long" (see `SPEECH_CHARS_PER_SECOND`).
    max_scenes: int = Field(default=12, ge=1)
    max_retries: int = Field(default=3, ge=1)
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    require_full_coverage: bool = True
    # Costs a second LLM call per generation; see knowledge/fidelity.py.
    verify_fidelity: bool = False


class AnimationSettings(BaseModel):
    """Renderer selection, consumed by the storyboard layer and the API mounts."""

    model_config = ConfigDict(extra="forbid")

    default_renderer: str = "canvas_2d"
    allowed_renderers: tuple[str, ...] = ("canvas_2d", "svg_2d", "three_3d")


class StoryboardSettings(BaseModel):
    """Tunable parameters for the Storyboard Agent and its validation pass.

    Read from **two** YAML blocks: the step limits and demo requirements were
    already reserved under `agent:`, and only the model-call knobs are new under
    `storyboard:`. Splitting them keeps the reservation from being restated.
    """

    model_config = ConfigDict(extra="forbid")

    min_storyboard_steps: int = Field(default=3, ge=1, le=7)
    # 5, not 7. The 3..7 shot rule was written when a beat was 3.8~7.0 seconds
    # with a floor nobody could edit below; at four beats a scene that put the
    # shortest possible video at 45 seconds and the longest at 196. A scene now
    # wants to be about 6 seconds — three or four beats of a second and a half —
    # and the ceiling that expresses that is 5.
    max_storyboard_steps: int = Field(default=5, ge=3, le=7)
    # The whole video, in seconds, and how far over it a storyboard may still
    # land.
    #
    # This is the only number in the pipeline that is about the *finished thing*
    # rather than about one layer, and that is the point: every other threshold
    # here is a local rule, and a model can satisfy all of them locally while
    # producing three and a half minutes. `validation._check_total_duration`
    # multiplies it by the band and refuses a storyboard that goes over, which is
    # what turns "make a one-minute video" into something a retry can act on.
    #
    # One-sided on purpose — see that check's docstring for why a floor here
    # would invalidate the hand-authored samples and every test fixture.
    target_seconds: float = Field(default=60.0, gt=0)
    target_band: float = Field(default=0.25, gt=0, le=1)
    require_visual_objects: bool = True
    require_interactive_demo: bool = True
    max_retries: int = Field(default=3, ge=1)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


DEFAULT_CONFIG_PATH = Path("config/app.example.yaml")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return {}
    return cast("dict[str, Any]", raw)


def _read_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name, {})
    if not isinstance(section, dict):
        return {}
    return cast("dict[str, Any]", section)


@lru_cache(maxsize=1)
def load_knowledge_settings(path: Path = DEFAULT_CONFIG_PATH) -> KnowledgeSettings:
    """Load the `knowledge` section from the YAML config, falling back to defaults."""
    return KnowledgeSettings.model_validate(_read_section(_read_yaml(path), "knowledge"))


@lru_cache(maxsize=1)
def load_animation_settings(path: Path = DEFAULT_CONFIG_PATH) -> AnimationSettings:
    """Load the `animation` section from the YAML config, falling back to defaults."""
    return AnimationSettings.model_validate(_read_section(_read_yaml(path), "animation"))


@lru_cache(maxsize=1)
def load_storyboard_settings(path: Path = DEFAULT_CONFIG_PATH) -> StoryboardSettings:
    """Load the `agent` + `storyboard` sections, falling back to defaults."""
    raw = _read_yaml(path)
    merged = {**_read_section(raw, "agent"), **_read_section(raw, "storyboard")}
    return StoryboardSettings.model_validate(merged)
