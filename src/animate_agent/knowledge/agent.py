"""Knowledge Agent: turn a DocumentIR into a validated LessonIR."""

from __future__ import annotations

from typing import Any

from animate_agent.documents.models import DocumentIR
from animate_agent.json_utils import extract_json_object
from animate_agent.knowledge.fidelity import (
    FIDELITY_SYSTEM_PROMPT,
    FidelityReport,
    build_fidelity_prompt,
)
from animate_agent.knowledge.models import LessonIR
from animate_agent.knowledge.prompts import (
    DEFAULT_FILM_SECONDS,
    build_knowledge_prompt,
    build_knowledge_system_prompt,
    narration_bounds,
)
from animate_agent.llm import DEFAULT_MAX_TOKENS, LLMClient, retry_suffix
from animate_agent.rendering.layout import SPEECH_CHARS_PER_SECOND, TRANSITION_SECONDS

DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_SCENES = 10
DEFAULT_TEMPERATURE = 0.4
DEFAULT_REQUIRE_FULL_COVERAGE = True
DEFAULT_VERIFY_FIDELITY = False
DEFAULT_SCRIPT_MODE = False
DEFAULT_TARGET_BAND = 0.35


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
        script_mode: bool = DEFAULT_SCRIPT_MODE,
        target_seconds: float | None = None,
        target_band: float = DEFAULT_TARGET_BAND,
    ) -> None:
        self._llm = llm
        self._max_retries = max_retries
        self._max_scenes = max_scenes
        self._temperature = temperature
        self._require_full_coverage = require_full_coverage
        self._verify_fidelity = verify_fidelity
        self._script_mode = script_mode
        # The film's length, which this layer does not measure and cannot — it is
        # stated once, under `storyboard:` in the YAML, and read here because the
        # word budget is derived from it. Two copies of "the film is 90 seconds"
        # would be two numbers to keep in step, and the one that drifted would be
        # the one nobody was reading.
        #
        # `None` means there is no film to check against, and that is the default
        # on purpose — the opposite of the default one layer down. A *ceiling* is
        # safe to apply to everything, because a fixture too short for the film
        # passes it; a *floor* refuses every one of them. The tests and the tools
        # build this agent by hand around three-scene lessons that are twenty
        # seconds of very good picture, and a floor applied to those would turn
        # the whole suite red for being right. `service.generate_lesson` is the
        # one caller that has a film in mind, and it says so.
        self._target_seconds = target_seconds
        self._target_band = target_band

    async def generate(self, document: DocumentIR) -> LessonIR:
        # The prompt always has a length to aim at, even when there is no check to
        # go with it: writing towards nothing is how a model writes three
        # sentences and stops. `or` rather than a conditional because the only
        # falsey non-None value it could see is 0, and a zero-second film is not a
        # configuration anyone means.
        system_prompt = build_knowledge_system_prompt(
            script_mode=self._script_mode,
            target_seconds=self._target_seconds or DEFAULT_FILM_SECONDS,
        )
        user_prompt = build_knowledge_prompt(document)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_error = ""
        for _attempt in range(1, self._max_retries + 1):
            try:
                # The `chat` call is inside the `try` on purpose: a truncated
                # response raises from here, and it is the one failure the retry
                # loop can actually fix. Placing the call above the `try` — as
                # this was — lets it escape the loop and fail the whole run.
                # `LLMBudgetExhaustedError` still escapes, because it is a
                # `RuntimeError` and no retry can make the same budget fit.
                raw = await self._llm.chat(
                    messages,
                    temperature=self._temperature,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    label="知识层",
                )
                data = extract_json_object(raw)
                lesson = self._validate(document, data)
                if self._verify_fidelity:
                    report = await self._check_fidelity(document, lesson)
                    if report.has_findings:
                        raise ValueError(f"保真核对未通过——{report.describe()}")
                return lesson
            except ValueError as exc:
                last_error = str(exc)
                messages = [
                    {"role": "system", "content": system_prompt},
                    # `retry_suffix` picks the wording from *why* the attempt
                    # failed: a truncated body needs "写短一点", a malformed one
                    # needs "照 schema 改" — and asking for the wrong one is how
                    # a retry burns a full call to repeat the same mistake.
                    {"role": "user", "content": f"{user_prompt}\n\n{retry_suffix(exc)}"},
                ]
        raise ValueError(f"Knowledge Agent 多次重试仍无法生成有效 LessonIR：{last_error}")

    async def _check_fidelity(self, document: DocumentIR, lesson: LessonIR) -> FidelityReport:
        """Ask a second pass whether the lesson actually taught the source."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": FIDELITY_SYSTEM_PROMPT},
            {"role": "user", "content": build_fidelity_prompt(document, lesson)},
        ]
        raw = await self._llm.chat(messages, temperature=0.0, max_tokens=2048)
        return FidelityReport.model_validate(extract_json_object(raw))

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
        lesson = LessonIR.model_validate(data)
        self._check_narration_length(lesson)
        return lesson

    def _check_narration_length(self, lesson: LessonIR) -> None:
        """Refuse a lesson whose script cannot make a film of the right length.

        A per-scene band and a whole-script budget, in that order — and both
        belong to this layer rather than to the storyboard's.

        **Why the floor is not in `validation._check_total_duration`, which is the
        only thing in the pipeline that measures a finished film.** That check
        measures the sum of the beats' captions, and those captions are *cut* from
        this narration: the corpus on disk runs 368 characters of narration into
        342 of captions and never the other way. A storyboard handed a thin lesson
        can shorten it; it cannot lengthen it without inventing facts, which is
        the one thing `knowledge/fidelity.py` exists to forbid. A floor there
        would refuse the film, ask for a retry, refuse it again three times and
        throw the run away — while the writing that could have fixed it was two
        layers upstream and never heard about it. The ceiling is checked in both
        places because both can act on it. The floor only here.

        Both ends are conservative in the direction that matters. Against the
        ceiling the scene changes are counted for the most scenes the storyboard
        could possibly draw — one per lesson scene, and `lesson_scene_ids` only
        ever merges, never splits — because every transition only makes the film
        longer. Against the floor they are not counted at all, for the same
        reason read the other way: a film is never shorter than its own narration.
        """
        low, high = narration_bounds(self._script_mode)
        kind = "稿子" if self._script_mode else "文档"
        for scene in lesson.scenes:
            length = len(scene.narration)
            if low <= length <= high:
                continue
            if length < low:
                raise ValueError(
                    f"场景 `{scene.id}` 的 narration 只有 {length} 字，{kind}模式下每幕要 "
                    f"{low}~{high} 字。太短的一幕讲不完一件事——它现在多半只是把画面上"
                    "已经写着的东西念了一遍。把这一段的结论说出来：原文写了「与物体质量无关」，"
                    "就要把这句话说出来，而不是只留一个「与质量无关」的标签"
                )
            raise ValueError(
                f"场景 `{scene.id}` 的 narration 有 {length} 字，{kind}模式下每幕最多 "
                f"{high} 字。一幕讲两件事就拆成两幕——多一幕不占时间，字多才占时间。"
                "先看这一幕是不是塞了两个概念，把它们分开，各写一句"
            )

        if self._target_seconds is None:
            # No film to check against: see `__init__` on why that is the default
            # here and not one layer down.
            return

        characters = sum(len(scene.narration) for scene in lesson.scenes)
        scenes = len(lesson.scenes)
        spoken = characters / SPEECH_CHARS_PER_SECOND
        transitions = max(scenes - 1, 0) * 2 * TRANSITION_SECONDS
        ceiling = self._target_seconds * (1 + self._target_band)
        floor = self._target_seconds * (1 - self._target_band)
        # What the budget is in characters, having given the scene changes their
        # seconds first.
        ceiling_chars = (ceiling - transitions) * SPEECH_CHARS_PER_SECOND
        floor_chars = floor * SPEECH_CHARS_PER_SECOND
        if characters > ceiling_chars:
            raise ValueError(
                f"全片 narration 共 {characters} 字，按每秒 6 个字是 {spoken:.0f} 秒；"
                f"{scenes} 幕最多有 {scenes - 1} 次换幕，再加 {transitions:.1f} 秒，"
                f"成片就是 {spoken + transitions:.0f} 秒，超过了 {ceiling:.0f} 秒的上限"
                f"（目标 {self._target_seconds:.0f} 秒）。这个幕数下最多只能写 "
                f"{ceiling_chars:.0f} 字，现在多了 {characters - ceiling_chars:.0f} 字。"
                "只有「少说」能压时长：先删掉那些只是在念画面上已经画出来的东西的句子，"
                "再把每句话收紧，还不够就减场景数——幕多不占时间，字多才占时间"
            )
        if characters < floor_chars:
            raise ValueError(
                f"全片 narration 只有 {characters} 字，按每秒 6 个字是 {spoken:.0f} 秒，"
                f"不到 {floor:.1f} 秒的下限（目标 {self._target_seconds:.0f} 秒）。"
                f"片子短只有一个原因：说得太少。{scenes} 幕平均每幕要写到 "
                f"{floor_chars / scenes:.0f} 字上下，现在平均 {characters / scenes:.0f} 字。"
                "先看是不是有场景只丢下一个小标题就过去了——原文给了定义和结论，"
                "就要把结论说出来，而不是只写个标签。不要加废话凑字数："
                "画面上已经画出来的东西不用再念一遍"
            )
