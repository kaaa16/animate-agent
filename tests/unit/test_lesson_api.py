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
        learning_objectives=["理解牛顿定律"],
        scenes=[
            LessonScene(
                id="scene-1",
                title="牛顿定律",
                objective="理解 F=ma",
                narration="力改变运动状态。",
                key_points=["力"],
                source_refs=["section-1"],
            )
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
