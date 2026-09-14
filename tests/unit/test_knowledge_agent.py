"""Tests for the Knowledge Agent (DocumentIR -> LessonIR)."""

import asyncio
import json

import httpx
import pytest

from animate_agent.documents.models import DocumentBlock, DocumentIR, DocumentSource, Section
from animate_agent.json_utils import extract_json_object
from animate_agent.knowledge.agent import KnowledgeAgent
from animate_agent.knowledge.fidelity import FidelityReport, build_fidelity_prompt
from animate_agent.knowledge.models import LessonIR
from animate_agent.llm import LLMClient, LLMConfig

VALID_LESSON = {
    "title": "力学基础",
    "subject": "物理",
    "summary": "讲解力学的基本概念",
    "learning_objectives": ["理解牛顿定律", "掌握抛物运动"],
    "scenes": [
        {
            "title": "牛顿第一定律",
            "objective": "理解惯性",
            "narration": (
                "物体在不受任何外力作用的时候，会一直保持静止，或者保持匀速直线运动状态，"
                "这就是牛顿第一定律，也叫做惯性定律。生活中汽车急刹车时人会向前倾，"
                "正是惯性在起作用，所以乘车时一定要系好安全带。"
            ),
            "key_points": ["惯性", "匀速直线运动"],
            "source_refs": ["section-1"],
        },
        {
            "title": "牛顿第二定律",
            "objective": "理解 F=ma",
            "narration": (
                "力是改变物体运动状态的原因。物体加速度的大小跟作用力成正比，"
                "跟物体的质量成反比，加速度的方向跟作用力的方向相同，"
                "这就是牛顿第二定律，写成公式就是 F 等于 ma。"
            ),
            "key_points": ["力", "加速度"],
            "source_refs": ["section-1-block-1"],
        },
        {
            "title": "牛顿第三定律",
            "objective": "理解作用力与反作用力",
            "narration": (
                "两个物体之间的作用力和反作用力，总是大小相等、方向相反，"
                "并且作用在同一条直线上，这就是牛顿第三定律。比如你用力推墙，"
                "墙同时也用同样大的力推你，所以划船时向后划水，船才会向前走。"
            ),
            "key_points": ["作用力", "反作用力"],
            "source_refs": ["section-1", "section-1-block-1"],
        },
    ],
}


def _make_document() -> DocumentIR:
    return DocumentIR(
        document_id="doc1",
        title="测试文档",
        source=DocumentSource(type="url", url="https://example.com"),
        sections=[
            Section(
                id="section-1",
                title="章节",
                level=1,
                blocks=[DocumentBlock(id="section-1-block-1", type="paragraph", text="内容")],
            )
        ],
    )


def _run_generate(
    document: DocumentIR,
    responses: list[str],
    *,
    max_scenes: int = 10,
    verify_fidelity: bool = False,
) -> LessonIR:
    async def run() -> LessonIR:
        idx = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal idx
            content = responses[idx] if idx < len(responses) else responses[-1]
            idx += 1
            payload = {"choices": [{"message": {"content": content}}]}
            return httpx.Response(200, json=payload, request=request)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            llm = LLMClient(
                LLMConfig(base_url="http://test", api_key="k", model="m"),
                client=client,
            )
            agent = KnowledgeAgent(
                llm, max_scenes=max_scenes, verify_fidelity=verify_fidelity
            )
            return await agent.generate(document)

    return asyncio.run(run())


def test_extract_json_strips_fence() -> None:
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_extracts_from_prose() -> None:
    assert extract_json_object('结果如下：{"a": 1} 完成') == {"a": 1}


def _complete_lesson(**overrides) -> dict:
    scenes = [
        {**scene, "id": f"scene-{index}"}
        for index, scene in enumerate(VALID_LESSON["scenes"], start=1)
    ]
    data = {
        **VALID_LESSON,
        "lesson_id": "l1",
        "document_id": "d1",
        "scenes": scenes,
    }
    data.update(overrides)
    return data


def test_lesson_ir_rejects_extra_field() -> None:
    with pytest.raises(ValueError):
        LessonIR.model_validate(_complete_lesson(extra=1))


def test_lesson_ir_rejects_empty_scenes() -> None:
    with pytest.raises(ValueError):
        LessonIR.model_validate(_complete_lesson(scenes=[]))


def test_lesson_ir_accepts_single_scene() -> None:
    single = [{**VALID_LESSON["scenes"][0], "id": "scene-1"}]
    lesson = LessonIR.model_validate(_complete_lesson(scenes=single))

    assert len(lesson.scenes) == 1


def test_generate_returns_valid_lesson() -> None:
    lesson = _run_generate(_make_document(), [json.dumps(VALID_LESSON, ensure_ascii=False)])

    assert isinstance(lesson, LessonIR)
    assert lesson.lesson_id == "lesson-doc1"
    assert lesson.document_id == "doc1"
    assert [scene.id for scene in lesson.scenes] == ["scene-1", "scene-2", "scene-3"]
    assert lesson.scenes[0].title == "牛顿第一定律"


def test_generate_retries_then_succeeds() -> None:
    valid = json.dumps(VALID_LESSON, ensure_ascii=False)
    lesson = _run_generate(_make_document(), ["not json at all", valid])

    assert lesson.lesson_id == "lesson-doc1"
    assert len(lesson.scenes) == 3


def test_generate_raises_after_retries() -> None:
    with pytest.raises(ValueError):
        _run_generate(_make_document(), ["bad", "bad", "bad"])


def test_generate_rejects_missing_scenes() -> None:
    payload = {k: v for k, v in VALID_LESSON.items() if k != "scenes"}
    with pytest.raises(ValueError):
        _run_generate(_make_document(), [json.dumps(payload, ensure_ascii=False)])


def test_generate_rejects_invalid_source_ref() -> None:
    scene = {**VALID_LESSON["scenes"][0], "source_refs": ["section-999"]}
    bad = {**VALID_LESSON, "scenes": [scene, *VALID_LESSON["scenes"][1:]]}
    with pytest.raises(ValueError):
        _run_generate(_make_document(), [json.dumps(bad, ensure_ascii=False)])


def test_generate_rejects_too_many_scenes() -> None:
    with pytest.raises(ValueError):
        _run_generate(
            _make_document(),
            [json.dumps(VALID_LESSON, ensure_ascii=False)],
            max_scenes=2,
        )


CLEAN_REPORT = json.dumps({"dropped_facts": [], "added_claims": []})
DIRTY_REPORT = json.dumps(
    {"dropped_facts": ["t 的定义被压掉了"], "added_claims": ["「化曲为直」原文没有"]},
    ensure_ascii=False,
)


def test_fidelity_report_flags_findings() -> None:
    assert not FidelityReport().has_findings

    dirty = FidelityReport(dropped_facts=["漏了 t"], added_claims=["编了修辞"])
    assert dirty.has_findings
    assert "遗漏了原文内容" in dirty.describe()
    assert "出现了原文没有的内容" in dirty.describe()


def test_build_fidelity_prompt_shows_both_sides() -> None:
    prompt = build_fidelity_prompt(_make_document(), LessonIR.model_validate(_complete_lesson()))

    assert "测试文档" in prompt
    assert "牛顿第一定律" in prompt


def test_fidelity_check_is_off_by_default() -> None:
    # With the check off, only the lesson response is consumed and returned.
    lesson = _run_generate(_make_document(), [json.dumps(VALID_LESSON, ensure_ascii=False)])

    assert len(lesson.scenes) == 3


def test_fidelity_check_returns_lesson_on_clean_report() -> None:
    lesson = _run_generate(
        _make_document(),
        [json.dumps(VALID_LESSON, ensure_ascii=False), CLEAN_REPORT],
        verify_fidelity=True,
    )

    assert lesson.lesson_id == "lesson-doc1"


def test_fidelity_check_retries_after_findings() -> None:
    valid = json.dumps(VALID_LESSON, ensure_ascii=False)
    lesson = _run_generate(
        _make_document(),
        [valid, DIRTY_REPORT, valid, CLEAN_REPORT],
        verify_fidelity=True,
    )

    assert len(lesson.scenes) == 3


def test_fidelity_check_raises_when_never_clean() -> None:
    valid = json.dumps(VALID_LESSON, ensure_ascii=False)
    with pytest.raises(ValueError, match="保真核对未通过"):
        _run_generate(
            _make_document(),
            [valid, DIRTY_REPORT] * 3,
            verify_fidelity=True,
        )


def _scene_with_key_points(count: int) -> dict:
    return {
        **VALID_LESSON["scenes"][0],
        "id": "scene-1",
        "key_points": [f"要点{i}" for i in range(count)],
    }


def test_lesson_scene_allows_key_points_up_to_the_cap() -> None:
    lesson = LessonIR.model_validate(_complete_lesson(scenes=[_scene_with_key_points(8)]))

    assert len(lesson.scenes[0].key_points) == 8


def test_lesson_scene_rejects_key_points_beyond_the_cap() -> None:
    with pytest.raises(ValueError):
        LessonIR.model_validate(_complete_lesson(scenes=[_scene_with_key_points(9)]))


def test_lesson_ir_rejects_short_narration() -> None:
    scene = {**VALID_LESSON["scenes"][0], "id": "scene-1", "narration": "太短了。"}
    with pytest.raises(ValueError):
        LessonIR.model_validate(_complete_lesson(scenes=[scene]))


def test_generate_rejects_uncovered_content() -> None:
    document = _make_document()
    document.sections.append(
        Section(
            id="section-2",
            title="没人引用的章节",
            level=1,
            blocks=[
                DocumentBlock(id="section-2-block-1", type="paragraph", text="被漏掉的内容"),
            ],
        )
    )

    with pytest.raises(ValueError, match="没有被任何场景"):
        _run_generate(document, [json.dumps(VALID_LESSON, ensure_ascii=False)])


def test_coverage_allows_unreferenced_section_when_its_blocks_are_covered() -> None:
    # VALID_LESSON references every block of section-1 but never the bare
    # section id itself; that must still count as covered.
    lesson = _run_generate(_make_document(), [json.dumps(VALID_LESSON, ensure_ascii=False)])

    assert len(lesson.scenes) == 3


def test_generate_rejects_uncovered_blockless_section() -> None:
    document = _make_document()
    document.sections.append(Section(id="section-empty", title="空章节", level=1))

    with pytest.raises(ValueError, match="没有被任何场景"):
        _run_generate(document, [json.dumps(VALID_LESSON, ensure_ascii=False)])
