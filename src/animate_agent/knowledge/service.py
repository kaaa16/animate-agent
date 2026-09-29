"""Knowledge Agent orchestration and persistence."""

from __future__ import annotations

import json
from pathlib import Path

from animate_agent.config import load_knowledge_settings, load_storyboard_settings
from animate_agent.documents.models import DocumentIR
from animate_agent.knowledge.agent import KnowledgeAgent
from animate_agent.knowledge.models import LessonIR
from animate_agent.llm import LLMClient, load_llm_config

DEFAULT_GENERATED_DIR = Path("data/generated")


async def generate_lesson(
    document: DocumentIR,
    *,
    agent: KnowledgeAgent | None = None,
    output_dir: Path = DEFAULT_GENERATED_DIR,
    script_mode: bool | None = None,
) -> LessonIR:
    """Run the Knowledge Agent over a DocumentIR and persist the resulting LessonIR.

    `script_mode` defaults to whatever the YAML says; passing it explicitly is
    what a caller does when it knows better than the config — the API's form
    field, the CLI's `--script`. `None` rather than `False` so that "not
    specified" stays distinguishable from "specified as off", which matters the
    moment the config default is anything but off.
    """
    llm: LLMClient | None = None
    owns_client = False
    if agent is None:
        settings = load_knowledge_settings()
        # The film's length is stated once, under `storyboard:` — the word budget
        # this layer writes to is derived from it, so it is read from there
        # rather than restated here. Two copies of "the film is 90 seconds" would
        # be two numbers to keep in step, and the one that drifted would be the
        # one nobody was reading.
        storyboard_settings = load_storyboard_settings()
        llm = LLMClient(load_llm_config())
        agent = KnowledgeAgent(
            llm,
            max_retries=settings.max_retries,
            max_scenes=settings.max_scenes,
            temperature=settings.temperature,
            require_full_coverage=settings.require_full_coverage,
            verify_fidelity=settings.verify_fidelity,
            script_mode=settings.script_mode if script_mode is None else script_mode,
            target_seconds=storyboard_settings.target_seconds,
            target_band=storyboard_settings.target_band,
        )
        owns_client = True
    try:
        lesson = await agent.generate(document)
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{lesson.lesson_id}.json"
        destination.write_text(
            json.dumps(lesson.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return lesson
    finally:
        if owns_client and llm is not None:
            await llm.aclose()
