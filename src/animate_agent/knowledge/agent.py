"""Knowledge Agent: turn a DocumentIR into a validated LessonIR."""

from __future__ import annotations

import json
import re
from typing import Any, cast

from animate_agent.documents.models import DocumentIR
from animate_agent.knowledge.fidelity import (
    FIDELITY_SYSTEM_PROMPT,
    FidelityReport,
    build_fidelity_prompt,
)
from animate_agent.knowledge.models import LessonIR
from animate_agent.knowledge.prompts import KNOWLEDGE_SYSTEM_PROMPT, build_knowledge_prompt
from animate_agent.llm import LLMClient

DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_SCENES = 10
DEFAULT_TEMPERATURE = 0.4
DEFAULT_REQUIRE_FULL_COVERAGE = True
DEFAULT_VERIFY_FIDELITY = False


def _extract_json(raw: str) -> dict[str, Any]:
    """Extract a JSON object from an LLM response, tolerating fences and prose."""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    parsed: Any = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"LLM 输出不是 JSON 对象: {type(parsed)!r}")
    return cast("dict[str, Any]", parsed)


def _derive_lesson_id(document: DocumentIR) -> str:
    return f"lesson-{document.document_id}"


def _collect_ids(document: DocumentIR) -> set[str]:
    """Collect every section id and block id in the document for source_ref validation."""
    ids = {section.id for section in document.sections}
    for section in document.sections:
        ids.update(block.id for block in section.blocks)
    return ids


def _uncovered_ids(document: DocumentIR, scenes: list[Any]) -> list[str]:
    """Return document ids that no scene references, in stable order.

    A section is covered once every one of its blocks is covered — the blocks
    are its content, so referencing the section id itself is not required.
    A blockless section has no other content, so its own id must be referenced.
    """
    referenced: set[str] = set()
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        refs = scene.get("source_refs")
        if isinstance(refs, list):
            referenced.update(ref for ref in refs if isinstance(ref, str))

    missing: list[str] = []
    for section in document.sections:
        block_ids = {block.id for block in section.blocks}
        if block_ids:
            missing.extend(sorted(block_ids - referenced))
        elif section.id not in referenced:
            missing.append(section.id)
    return sorted(missing)


class KnowledgeAgent:
    """Extract a LessonIR from a DocumentIR via a single, validated LLM call."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_scenes: int = DEFAULT_MAX_SCENES,
        temperature: float = DEFAULT_TEMPERATURE,
        require_full_coverage: bool = DEFAULT_REQUIRE_FULL_COVERAGE,
        verify_fidelity: bool = DEFAULT_VERIFY_FIDELITY,
    ) -> None:
        self._llm = llm
        self._max_retries = max_retries
        self._max_scenes = max_scenes
        self._temperature = temperature
        self._require_full_coverage = require_full_coverage
        self._verify_fidelity = verify_fidelity

    async def generate(self, document: DocumentIR) -> LessonIR:
        user_prompt = build_knowledge_prompt(document)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": KNOWLEDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        last_error = ""
        for _attempt in range(1, self._max_retries + 1):
            raw = await self._llm.chat(messages, temperature=self._temperature, max_tokens=8192)
            try:
                data = _extract_json(raw)
                lesson = self._validate(document, data)
                if self._verify_fidelity:
                    report = await self._check_fidelity(document, lesson)
                    if report.has_findings:
                        raise ValueError(f"保真核对未通过——{report.describe()}")
                return lesson
            except ValueError as exc:
                last_error = str(exc)
                messages = [
                    {"role": "system", "content": KNOWLEDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"{user_prompt}\n\n"
                            f"上一次输出校验失败：{last_error}\n"
                            "请重新输出一个符合 schema 的完整 JSON 对象。"
                        ),
                    },
                ]
        raise ValueError(f"Knowledge Agent 多次重试仍无法生成有效 LessonIR：{last_error}")

    async def _check_fidelity(self, document: DocumentIR, lesson: LessonIR) -> FidelityReport:
        """Ask a second pass whether the lesson actually taught the source."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": FIDELITY_SYSTEM_PROMPT},
            {"role": "user", "content": build_fidelity_prompt(document, lesson)},
        ]
        raw = await self._llm.chat(messages, temperature=0.0, max_tokens=2048)
        return FidelityReport.model_validate(_extract_json(raw))

    def _validate(self, document: DocumentIR, data: dict[str, Any]) -> LessonIR:
        data = dict(data)
        data["lesson_id"] = _derive_lesson_id(document)
        data["document_id"] = document.document_id
        scenes = data.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            raise ValueError("输出缺少非空的 scenes 列表")
        if len(scenes) > self._max_scenes:
            raise ValueError(f"scenes 数量 {len(scenes)} 超过上限 {self._max_scenes}")
        valid_ids = _collect_ids(document)
        for index, scene in enumerate(scenes, start=1):
            if not isinstance(scene, dict):
                raise ValueError(f"scenes[{index - 1}] 不是对象")
            scene.setdefault("id", f"scene-{index}")
            refs = scene.get("source_refs")
            if isinstance(refs, list):
                bad = [ref for ref in refs if ref not in valid_ids]
                if bad:
                    raise ValueError(
                        f"scenes[{index - 1}].source_refs 引用了不存在的 id: {bad}"
                    )
        if self._require_full_coverage:
            missing = _uncovered_ids(document, scenes)
            if missing:
                raise ValueError(
                    f"以下原文内容没有被任何场景的 source_refs 覆盖（疑似遗漏）：{missing}。"
                    "请把遗漏的内容补进相应场景，或调整 source_refs。"
                )
        return LessonIR.model_validate(data)
