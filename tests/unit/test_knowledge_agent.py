"""Tests for the Knowledge Agent (DocumentIR -> LessonIR)."""

import asyncio
import json

import httpx
import pytest

from animate_agent.documents.models import DocumentBlock, DocumentIR, DocumentSource, Section
from animate_agent.knowledge.agent import KnowledgeAgent, _extract_json
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
            "narration": "物体在不受外力时保持静止或匀速直线运动。",
            "key_points": ["惯性", "匀速直线运动"],
            "source_refs": ["section-1"],
        },
        {
            "title": "牛顿第二定律",
            "objective": "理解 F=ma",
            "narration": "力是改变物体运动状态的原因，加速度与力成正比。",
            "key_points": ["力", "加速度"],
            "source_refs": ["section-1-block-1"],
        },
        {
            "title": "牛顿第三定律",
            "objective": "理解作用力与反作用力",
            "narration": "作用力和反作用力大小相等、方向相反。",
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


def _run_generate(document: DocumentIR, responses: list[str], *, max_scenes: int = 10) -> LessonIR:
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
            agent = KnowledgeAgent(llm, max_scenes=max_scenes)
            return await agent.generate(document)

    return asyncio.run(run())


def test_extract_json_strips_fence() -> None:
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_extracts_from_prose() -> None:
    assert _extract_json('结果如下：{"a": 1} 完成') == {"a": 1}


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
