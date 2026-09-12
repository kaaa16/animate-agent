"""Tests for the lesson-generation API endpoints."""

from fastapi.testclient import TestClient

from animate_agent import api
from animate_agent.documents.models import DocumentIR, DocumentSource
from animate_agent.knowledge.models import LessonIR, LessonScene


def _expected_lesson() -> LessonIR:
    return LessonIR(
        lesson_id="lesson-doc1",
        document_id="doc1",
        title="力学基础",
        subject="物理",
        summary="讲解力学",
        learning_objectives=["理解牛顿定律", "掌握抛物运动"],
        scenes=[
            LessonScene(
                id="scene-1",
                title="牛顿第一定律",
                objective="理解惯性",
                narration=(
                    "物体在不受任何外力作用的时候，会一直保持静止，或者保持匀速直线运动状态，"
                    "这就是牛顿第一定律，也叫做惯性定律。生活中汽车急刹车时人会向前倾，"
                    "正是惯性在起作用，所以乘车时一定要系好安全带。"
                ),
                key_points=["惯性", "匀速运动"],
                source_refs=["section-1"],
            ),
            LessonScene(
                id="scene-2",
                title="牛顿第二定律",
                objective="理解 F=ma",
                narration=(
                    "力是改变物体运动状态的原因。物体加速度的大小跟作用力成正比，"
                    "跟物体的质量成反比，加速度的方向跟作用力的方向相同。"
                    "质量越大，同样的力产生的加速度就越小，写成公式就是 F 等于 ma。"
                ),
                key_points=["力", "加速度"],
                source_refs=["section-1"],
            ),
            LessonScene(
                id="scene-3",
                title="牛顿第三定律",
                objective="理解作用力与反作用力",
                narration=(
                    "两个物体之间的作用力和反作用力，总是大小相等、方向相反，"
                    "并且作用在同一条直线上，它们总是成对出现。比如你用力推墙，"
                    "墙同时也用同样大的力推你，所以划船时向后划水，船才会向前走。"
                ),
                key_points=["作用力", "反作用力"],
                source_refs=["section-1"],
            ),
        ],
    )


def _fake_document() -> DocumentIR:
    return DocumentIR(
        document_id="doc1",
        title="测试文档",
        source=DocumentSource(type="url", url="https://example.com"),
    )


def test_lessons_from_url_api(monkeypatch) -> None:
    expected = _expected_lesson()

    async def fake_ingest_url(url: str) -> DocumentIR:
        assert url == "https://example.com/docs"
        return _fake_document()

    async def fake_generate_lesson(document: DocumentIR) -> LessonIR:
        assert document.document_id == "doc1"
        return expected

    monkeypatch.setattr(api, "ingest_url", fake_ingest_url)
    monkeypatch.setattr(api, "generate_lesson", fake_generate_lesson)

    response = TestClient(api.app).post(
        "/api/lessons/from-url",
        json={"url": "https://example.com/docs"},
    )

    assert response.status_code == 200
    assert LessonIR.model_validate(response.json()) == expected


def test_lessons_from_file_api(monkeypatch) -> None:
    expected = _expected_lesson()

    def fake_ingest_file(path) -> DocumentIR:
        return _fake_document()

    async def fake_generate_lesson(document: DocumentIR) -> LessonIR:
        return expected

    monkeypatch.setattr(api, "ingest_file", fake_ingest_file)
    monkeypatch.setattr(api, "generate_lesson", fake_generate_lesson)

    response = TestClient(api.app).post(
        "/api/lessons/from-file",
        files={"file": ("sample.docx", b"dummy", "application/octet-stream")},
    )

    assert response.status_code == 200
    assert LessonIR.model_validate(response.json()) == expected


def test_lessons_from_file_rejects_unknown_extension() -> None:
    response = TestClient(api.app).post(
        "/api/lessons/from-file",
        files={"file": ("sample.txt", b"dummy", "text/plain")},
    )

    assert response.status_code == 400
