"""Storyboard Agent orchestration and persistence."""

from __future__ import annotations

import json
from pathlib import Path

from animate_agent.config import load_animation_settings, load_storyboard_settings
from animate_agent.documents.models import DocumentIR
from animate_agent.knowledge.models import LessonIR
from animate_agent.llm import LLMClient, load_llm_config
from animate_agent.storyboard.agent import StoryboardAgent
from animate_agent.storyboard.models import StoryboardIR
from animate_agent.storyboard.validation import StoryboardLimits

DEFAULT_GENERATED_DIR = Path("data/generated")


def build_limits() -> StoryboardLimits:
    """Read the storyboard thresholds from the YAML config."""
    storyboard_settings = load_storyboard_settings()
    animation_settings = load_animation_settings()
    return StoryboardLimits(
        min_steps=storyboard_settings.min_storyboard_steps,
        max_steps=storyboard_settings.max_storyboard_steps,
        require_visual_objects=storyboard_settings.require_visual_objects,
        require_interactive_demo=storyboard_settings.require_interactive_demo,
        allowed_renderers=animation_settings.allowed_renderers,
    )


async def generate_storyboard(
    lesson: LessonIR,
    document: DocumentIR,
    *,
    agent: StoryboardAgent | None = None,
    output_dir: Path = DEFAULT_GENERATED_DIR,
) -> StoryboardIR:
    """Run the Storyboard Agent over a LessonIR and persist the resulting StoryboardIR."""
    llm: LLMClient | None = None
    owns_client = False
    if agent is None:
        storyboard_settings = load_storyboard_settings()
        animation_settings = load_animation_settings()
        llm = LLMClient(load_llm_config())
        agent = StoryboardAgent(
            llm,
            limits=build_limits(),
            allowed_renderers=animation_settings.allowed_renderers,
            max_retries=storyboard_settings.max_retries,
            temperature=storyboard_settings.temperature,
            debug_dir=output_dir,
        )
        owns_client = True
    try:
        storyboard = await agent.generate(lesson, document)
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{storyboard.storyboard_id}.json"
        destination.write_text(
            json.dumps(storyboard.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return storyboard
    finally:
        if owns_client and llm is not None:
            await llm.aclose()
