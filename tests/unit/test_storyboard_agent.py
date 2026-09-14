"""Tests for the Storyboard Agent (LessonIR -> StoryboardIR)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from animate_agent.documents.models import DocumentBlock, DocumentIR, DocumentSource, Section
from animate_agent.knowledge.models import LessonIR, LessonScene
from animate_agent.llm import LLMClient, LLMConfig
from animate_agent.storyboard.agent import StoryboardAgent
from animate_agent.storyboard.models import StoryboardIR
from animate_agent.storyboard.validation import StoryboardLimits

LIMITS = StoryboardLimits(min_steps=3, max_steps=7)

NARRATION = (
    "机器人先看最近障碍物的距离。如果这个距离大于安全距离，机器人就保持当前速度继续直行；"
    "一旦距离小于安全距离，机器人会先降低速度，再比较左侧和右侧的可通行空间，"
    "选择更空旷的一侧转向绕行，从而安全通过。"
)

VALID_STORYBOARD: dict[str, Any] = {
    "title": "移动机器人避障",
    "subject": "机器人",
    "eyebrow": "避障 · 决策逻辑",
    "scenes": [
        {
            "id": "shot-1",
            "scene_type": "lane",
            "teaching_goal": "看懂距离到转向的决策链",
            "lesson_scene_ids": ["scene-1"],
            "objects": [
                {
                    "id": "car",
                    "role": "vehicle",
                    "label": "小车",
                    "props": {"speed": 60, "heading": 0},
                    "source_refs": ["section-1-block-1"],
                },
                {"id": "obstacle", "role": "obstacle", "label": "障碍物", "props": {}},
            ],
            "steps": [
                {
                    "id": "step-1",
                    "title": "雷达扫描",
                    "description": "雷达向前方扇形区域发射多条测距射线，返回每个方向的距离",
                    "highlights": ["car"],
                    "key_points": ["安全距离"],
                },
                {
                    "id": "step-2",
                    "title": "阈值判断",
                    "description": "最近障碍物的距离低于安全距离，控制器进入避障状态",
                    "highlights": ["car"],
                    "object_states": {"car": {"danger": True}},
                    "key_points": ["安全距离", "转向"],
                },
                {
                    "id": "step-3",
                    "title": "转向绕行",
                    "description": "比较左侧与右侧的空旷程度，选择更安全的一侧转向绕行",
                    "highlights": ["obstacle"],
                    "key_points": ["转向"],
                },
            ],
            "controls": [
                {
                    "id": "speed",
                    "type": "slider",
                    "label": "车速",
                    "target_property": "car.speed",
                    "min": 0,
                    "max": 120,
                    "default": 60,
                    "step": 5,
                }
            ],
            "params": {},
        }
    ],
}


def _document() -> DocumentIR:
    return DocumentIR(
        document_id="doc1",
        title="移动机器人避障手册节选",
        source=DocumentSource(type="file"),
        sections=[
            Section(
                id="section-1",
                title="避障决策",
                level=1,
                blocks=[
                    DocumentBlock(id="section-1-block-1", type="paragraph", text="决策逻辑")
                ],
            )
        ],
    )


def _lesson() -> LessonIR:
    return LessonIR(
        lesson_id="lesson-doc1",
        document_id="doc1",
        title="移动机器人避障",
        subject="机器人",
        summary="基于安全距离的避障决策。",
        learning_objectives=["理解避障决策", "掌握关键参数"],
        scenes=[
            LessonScene(
                id="scene-1",
                title="避障决策逻辑",
                objective="理解何时直行、何时避障",
                narration=NARRATION,
                key_points=["安全距离", "转向"],
                source_refs=["section-1-block-1"],
            )
        ],
    )


def _run_generate(
    responses: list[str],
    *,
    captured: list[dict[str, Any]] | None = None,
    debug_dir: Path | None = None,
) -> StoryboardIR:
    async def run() -> StoryboardIR:
        index = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal index
            content = responses[index] if index < len(responses) else responses[-1]
            index += 1
            if captured is not None:
                captured.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": content}}]},
                request=request,
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            llm = LLMClient(
                LLMConfig(base_url="http://test", api_key="k", model="m"),
                client=client,
            )
            agent = StoryboardAgent(llm, limits=LIMITS, debug_dir=debug_dir)
            return await agent.generate(_lesson(), _document())

    return asyncio.run(run())


def test_generate_returns_valid_storyboard() -> None:
    storyboard = _run_generate([json.dumps(VALID_STORYBOARD, ensure_ascii=False)])

    assert isinstance(storyboard, StoryboardIR)
    assert [scene.id for scene in storyboard.scenes] == ["shot-1"]


def test_generate_derives_identity_from_the_lesson() -> None:
    # The model must not be able to rename the course or invent ids.
    payload = {
        **VALID_STORYBOARD,
        "storyboard_id": "hacked",
        "lesson_id": "hacked",
        "document_id": "hacked",
        "title": "hacked",
        "subject": "hacked",
    }
    storyboard = _run_generate([json.dumps(payload, ensure_ascii=False)])

    assert storyboard.storyboard_id == "storyboard-lesson-doc1"
    assert storyboard.lesson_id == "lesson-doc1"
    assert storyboard.document_id == "doc1"
    assert storyboard.title == "移动机器人避障"
    assert storyboard.subject == "机器人"


def test_generate_retries_then_succeeds() -> None:
    valid = json.dumps(VALID_STORYBOARD, ensure_ascii=False)
    storyboard = _run_generate(["not json at all", valid])

    assert len(storyboard.scenes) == 1


def test_generate_raises_after_retries() -> None:
    with pytest.raises(ValueError, match="多次重试"):
        _run_generate(["bad", "bad", "bad"])


def test_rejected_responses_are_dumped_when_a_debug_dir_is_given(tmp_path: Path) -> None:
    # The only other trace of a bad response is pydantic's complaint about it,
    # which says what was wrong but never what was there. Attempts are numbered
    # so attempt 1 vs attempt 3 shows whether the retry feedback landed.
    with pytest.raises(ValueError):
        _run_generate(["bad", "bad", "bad"], debug_dir=tmp_path)

    dumps = sorted(path.name for path in tmp_path.iterdir())
    assert dumps == [f"storyboard-lesson-doc1-raw-attempt-{n}.txt" for n in (1, 2, 3)]
    assert (tmp_path / dumps[0]).read_text(encoding="utf-8") == "bad"


def test_the_failure_message_points_at_the_dump(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="原始输出已落盘"):
        _run_generate(["bad", "bad", "bad"], debug_dir=tmp_path)


def test_retry_prompt_carries_the_validation_issues() -> None:
    # The retry is only useful if it says what was wrong.
    broken = json.loads(json.dumps(VALID_STORYBOARD))
    broken["scenes"][0]["objects"][0]["role"] = "banana"
    captured: list[dict[str, Any]] = []

    with pytest.raises(ValueError):
        _run_generate(
            [json.dumps(broken, ensure_ascii=False)] * 3,
            captured=captured,
        )

    retry_content = captured[1]["messages"][1]["content"]
    assert "上一次输出校验失败" in retry_content
    assert "unknown_role" in retry_content


def test_generate_rejects_an_unregistered_role() -> None:
    broken = json.loads(json.dumps(VALID_STORYBOARD))
    broken["scenes"][0]["objects"][0]["role"] = "banana"

    with pytest.raises(ValueError, match="unknown_role"):
        _run_generate([json.dumps(broken, ensure_ascii=False)] * 3)


def test_generate_rejects_an_inert_control() -> None:
    broken = json.loads(json.dumps(VALID_STORYBOARD))
    broken["scenes"][0]["controls"][0]["target_property"] = "car.danger"

    with pytest.raises(ValueError, match="control_target_unconsumed"):
        _run_generate([json.dumps(broken, ensure_ascii=False)] * 3)


def test_generate_rejects_a_scene_that_teaches_nothing_visually() -> None:
    broken = json.loads(json.dumps(VALID_STORYBOARD))
    broken["scenes"][0]["steps"][0] = {
        "id": "step-1",
        "title": "只有文字",
        "description": "这一步只写了文字说明，画面上什么都没有动",
        "key_points": ["安全距离"],
    }

    with pytest.raises(ValueError, match="高亮"):
        _run_generate([json.dumps(broken, ensure_ascii=False)] * 3)
