"""Knowledge Agent: turn a DocumentIR into a validated LessonIR."""

from __future__ import annotations

import json
import re
from typing import Any, cast

from animate_agent.documents.models import DocumentIR
from animate_agent.knowledge.models import LessonIR
from animate_agent.knowledge.prompts import KNOWLEDGE_SYSTEM_PROMPT, build_knowledge_prompt
from animate_agent.llm import LLMClient

DEFAULT_MAX_RETRIES = 3


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


class KnowledgeAgent:
    """Extract a LessonIR from a DocumentIR via a single, validated LLM call."""

    def __init__(self, llm: LLMClient, *, max_retries: int = DEFAULT_MAX_RETRIES) -> None:
        self._llm = llm
        self._max_retries = max_retries

    async def generate(self, document: DocumentIR) -> LessonIR:
        user_prompt = build_knowledge_prompt(document)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": KNOWLEDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        last_error = ""
        for _attempt in range(1, self._max_retries + 1):
            raw = await self._llm.chat(messages, temperature=0.4, max_tokens=8192)
            try:
                data = _extract_json(raw)
                return self._validate(document, data)
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

    def _validate(self, document: DocumentIR, data: dict[str, Any]) -> LessonIR:
        data = dict(data)
        data["lesson_id"] = _derive_lesson_id(document)
        data["document_id"] = document.document_id
        scenes = data.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            raise ValueError("输出缺少非空的 scenes 列表")
        for index, scene in enumerate(scenes, start=1):
            if not isinstance(scene, dict):
                raise ValueError(f"scenes[{index - 1}] 不是对象")
            scene.setdefault("id", f"scene-{index}")
        return LessonIR.model_validate(data)
