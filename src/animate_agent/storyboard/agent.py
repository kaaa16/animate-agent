"""Storyboard Agent: turn a LessonIR into a validated StoryboardIR.

Structurally the same single-call-with-retry shape as `knowledge/agent.py`, with
one difference: validation **collects** issues rather than raising on the first,
so one retry round trip carries every mistake at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from animate_agent.documents.models import DocumentIR
from animate_agent.json_utils import extract_json_object
from animate_agent.knowledge.models import LessonIR
from animate_agent.llm import DEFAULT_MAX_TOKENS, LLMClient
from animate_agent.storyboard.models import StoryboardIR
from animate_agent.storyboard.prompts import STORYBOARD_SYSTEM_PROMPT, build_storyboard_prompt
from animate_agent.storyboard.validation import (
    StoryboardLimits,
    format_issues,
    validate_storyboard,
)

DEFAULT_MAX_RETRIES = 3
# Lower than the Knowledge Agent's 0.4: this layer is schema-dense and the
# creative work was already done upstream.
DEFAULT_TEMPERATURE = 0.3


def _derive_storyboard_id(lesson: LessonIR) -> str:
    return f"storyboard-{lesson.lesson_id}"


def apply_identity(lesson: LessonIR, data: dict[str, Any]) -> dict[str, Any]:
    """Stamp the derived identity onto a model payload.

    Deliberately separate from `StoryboardAgent._validate` so the offline
    inspector (`tools/inspect_storyboard_output.py`) can reproduce exactly what a
    real run would have checked, without spending an LLM call to get there.
    """
    data = dict(data)
    data["storyboard_id"] = _derive_storyboard_id(lesson)
    data["lesson_id"] = lesson.lesson_id
    data["document_id"] = lesson.document_id
    # Assigned, not defaulted: the course already has a title, and letting the
    # model overwrite it would silently rename the lesson downstream.
    data["title"] = lesson.title
    data["subject"] = lesson.subject
    return data


def _format_validation_error(exc: ValidationError) -> str:
    """Compact pydantic errors — the full repr is unreadable in a retry prompt."""
    lines = [f"共 {exc.error_count()} 处 schema 不符："]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"- {location}：{error['msg']}")
    return "\n".join(lines)


class StoryboardAgent:
    """Plan a StoryboardIR from a LessonIR via a single, validated LLM call."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        limits: StoryboardLimits,
        allowed_renderers: tuple[str, ...] = (),
        max_retries: int = DEFAULT_MAX_RETRIES,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        debug_dir: Path | None = None,
    ) -> None:
        self._llm = llm
        self._limits = limits
        self._allowed_renderers = allowed_renderers
        self._max_retries = max_retries
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._debug_dir = debug_dir

    def _dump_raw(self, lesson: LessonIR, attempt: int, raw: str) -> None:
        """Persist a rejected response so the next failure is diagnosable.

        Without this the only trace of a bad response is pydantic's complaint
        about it, which says what was wrong but never what was *there* — so the
        fix gets designed against a guess. Attempts are numbered: comparing
        attempt 1 with attempt 3 shows whether the retry feedback landed.
        """
        if self._debug_dir is None:
            return
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        name = f"{_derive_storyboard_id(lesson)}-raw-attempt-{attempt}.txt"
        (self._debug_dir / name).write_text(raw, encoding="utf-8")

    async def generate(self, lesson: LessonIR, document: DocumentIR) -> StoryboardIR:
        user_prompt = build_storyboard_prompt(lesson, allowed_renderers=self._allowed_renderers)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": STORYBOARD_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        last_error = ""
        for attempt in range(1, self._max_retries + 1):
            raw = await self._llm.chat(
                messages, temperature=self._temperature, max_tokens=self._max_tokens
            )
            try:
                data = extract_json_object(raw)
                return self._validate(lesson, document, data)
            except ValueError as exc:
                last_error = str(exc)
                self._dump_raw(lesson, attempt, raw)
                messages = [
                    {"role": "system", "content": STORYBOARD_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"{user_prompt}\n\n"
                            f"上一次输出校验失败：{last_error}\n"
                            "请重新输出一个符合 schema 的完整 JSON 对象。"
                        ),
                    },
                ]
        where = f"（模型原始输出已落盘到 {self._debug_dir}）" if self._debug_dir else ""
        raise ValueError(
            f"Storyboard Agent 多次重试仍无法生成有效 StoryboardIR：{last_error}{where}"
        )

    def _validate(
        self,
        lesson: LessonIR,
        document: DocumentIR,
        data: dict[str, Any],
    ) -> StoryboardIR:
        # Identity and course-level fields are derived, not authored: the model
        # should not be able to rename the course or invent an id.
        data = apply_identity(lesson, data)

        try:
            storyboard = StoryboardIR.model_validate(data)
        except ValidationError as exc:
            raise ValueError(_format_validation_error(exc)) from exc

        issues = validate_storyboard(
            storyboard, lesson=lesson, document=document, limits=self._limits
        )
        if issues:
            raise ValueError(format_issues(issues))
        return storyboard
