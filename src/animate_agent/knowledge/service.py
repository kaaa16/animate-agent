"""Knowledge Agent orchestration and persistence."""

from __future__ import annotations

import json
from pathlib import Path

from animate_agent.config import load_knowledge_settings
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
) -> LessonIR:
    """Run the Knowledge Agent over a DocumentIR and persist the resulting LessonIR."""
    llm: LLMClient | None = None
    owns_client = False
    if agent is None:
        settings = load_knowledge_settings()
        llm = LLMClient(load_llm_config())
        agent = KnowledgeAgent(
            llm,
            max_retries=settings.max_retries,
            max_scenes=settings.max_scenes,
            temperature=settings.temperature,
            require_full_coverage=settings.require_full_coverage,
            verify_fidelity=settings.verify_fidelity,
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
