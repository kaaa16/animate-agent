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


DEFAULT_CONFIG_PATH = Path("config/app.example.yaml")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return {}
    return cast("dict[str, Any]", raw)


@lru_cache(maxsize=1)
def load_knowledge_settings(path: Path = DEFAULT_CONFIG_PATH) -> KnowledgeSettings:
    """Load the `knowledge` section from the YAML config, falling back to defaults."""
    raw = _read_yaml(path)
    knowledge = raw.get("knowledge", {})
    if not isinstance(knowledge, dict):
        knowledge = {}
    return KnowledgeSettings.model_validate(knowledge)
