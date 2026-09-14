"""Tests for YAML application configuration loading."""

from pathlib import Path

import pytest

from animate_agent.config import (
    AnimationSettings,
    KnowledgeSettings,
    StoryboardSettings,
    load_animation_settings,
    load_knowledge_settings,
    load_storyboard_settings,
)


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


def test_load_storyboard_settings_merges_both_blocks(tmp_path: Path) -> None:
    # The step limits were already reserved under `agent:`; only the model-call
    # knobs are new, and the loader has to read both.
    path = tmp_path / "app.yaml"
    path.write_text(
        "agent:\n"
        "  max_storyboard_steps: 6\n"
        "  require_interactive_demo: false\n"
        "storyboard:\n"
        "  min_storyboard_steps: 4\n"
        "  temperature: 0.5\n"
        "  max_retries: 2\n",
        encoding="utf-8",
    )

    settings = load_storyboard_settings(path)

    assert settings.max_storyboard_steps == 6
    assert settings.min_storyboard_steps == 4
    assert settings.temperature == 0.5
    assert settings.max_retries == 2
    assert settings.require_interactive_demo is False


def test_load_storyboard_settings_defaults_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text("app:\n  env: development\n", encoding="utf-8")

    assert load_storyboard_settings(path) == StoryboardSettings()


def test_storyboard_settings_rejects_out_of_range_step_count() -> None:
    with pytest.raises(ValueError):
        StoryboardSettings(max_storyboard_steps=12)


def test_load_animation_settings_reads_renderers(tmp_path: Path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(
        "animation:\n  default_renderer: svg_2d\n  allowed_renderers:\n    - svg_2d\n",
        encoding="utf-8",
    )

    settings = load_animation_settings(path)

    assert settings.default_renderer == "svg_2d"
    assert settings.allowed_renderers == ("svg_2d",)


def test_load_animation_settings_defaults_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text("app:\n  env: development\n", encoding="utf-8")

    assert load_animation_settings(path) == AnimationSettings()
