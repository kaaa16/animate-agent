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

    max_scenes: int = Field(default=10, ge=1)
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
    max_storyboard_steps: int = Field(default=7, ge=3, le=7)
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
