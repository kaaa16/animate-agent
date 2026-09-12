"""Tests for YAML application configuration loading."""

from pathlib import Path

import pytest

from animate_agent.config import KnowledgeSettings, load_knowledge_settings


def test_load_knowledge_settings_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(
        "knowledge:\n  max_scenes: 7\n  max_retries: 2\n  temperature: 0.8\n",
        encoding="utf-8",
    )

    settings = load_knowledge_settings(path)

    assert settings.max_scenes == 7
    assert settings.max_retries == 2
    assert settings.temperature == 0.8


def test_load_knowledge_settings_defaults_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text("app:\n  env: development\n", encoding="utf-8")

    settings = load_knowledge_settings(path)

    assert settings == KnowledgeSettings()


def test_knowledge_settings_validates_range() -> None:
    with pytest.raises(ValueError):
        KnowledgeSettings(max_scenes=0)


def test_knowledge_settings_requires_coverage_by_default() -> None:
    assert KnowledgeSettings().require_full_coverage is True


def test_load_knowledge_settings_reads_coverage_flag(tmp_path: Path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text("knowledge:\n  require_full_coverage: false\n", encoding="utf-8")

    settings = load_knowledge_settings(path)

    assert settings.require_full_coverage is False
