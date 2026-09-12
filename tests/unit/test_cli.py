"""Tests for the `animate-agent` command-line entry point."""

import json
from pathlib import Path

import pytest

from animate_agent import cli
from animate_agent.documents.models import DocumentIR
from animate_agent.knowledge.models import LessonIR, LessonScene

SAMPLE = "# 力学基础\n\n物体在不受外力时保持静止。\n"


def _write_sample(tmp_path: Path) -> Path:
    path = tmp_path / "sample.md"
    path.write_text(SAMPLE, encoding="utf-8")
    return path


def test_ingest_only_prints_document_ir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_sample(tmp_path)

    exit_code = cli.main([str(path), "--ingest-only"])

    assert exit_code == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    document = DocumentIR.model_validate(payload)
    assert document.source.type == "file"
    assert document.sections[0].title == "力学基础"


def test_missing_key_reports_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("DEEPSEEK_KEY", raising=False)
    path = _write_sample(tmp_path)

    exit_code = cli.main([str(path)])

    assert exit_code == cli.EXIT_MISSING_KEY
    assert "DEEPSEEK_KEY" in capsys.readouterr().err


def test_missing_file_reports_clearly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main([str(tmp_path / "nope.md")])

    assert exit_code == cli.EXIT_INGEST_FAILED
    assert "找不到文件" in capsys.readouterr().err


def test_unsupported_extension_reports_clearly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "sample.xyz"
    path.write_text("hello", encoding="utf-8")

    exit_code = cli.main([str(path)])

    assert exit_code == cli.EXIT_INGEST_FAILED
    assert "不支持的文件格式" in capsys.readouterr().err


def test_runs_full_chain_and_prints_lesson(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DEEPSEEK_KEY", "test-key")
    path = _write_sample(tmp_path)
    expected = LessonIR(
        lesson_id="lesson-x",
        document_id="x",
        title="力学基础",
        subject="物理",
        summary="摘要",
        learning_objectives=["目标一", "目标二"],
        scenes=[
            LessonScene(
                id="scene-1",
                title="惯性",
                objective="理解惯性",
                narration=(
                    "物体在不受任何外力作用的时候，会一直保持静止，或者保持匀速直线运动状态，"
                    "这就是牛顿第一定律，也叫做惯性定律。生活中汽车急刹车时人会向前倾，"
                    "正是惯性在起作用，所以乘车时一定要系好安全带。"
                ),
                key_points=["惯性", "静止"],
                source_refs=["section-1-block-1"],
            )
        ],
    )

    async def fake_generate_lesson(document: DocumentIR) -> LessonIR:
        assert document.sections[0].title == "力学基础"
        return expected

    monkeypatch.setattr(cli, "generate_lesson", fake_generate_lesson)

    exit_code = cli.main([str(path)])
    captured = capsys.readouterr()

    assert exit_code == cli.EXIT_OK
    assert LessonIR.model_validate(json.loads(captured.out)) == expected
    assert "scenes=1" in captured.err
