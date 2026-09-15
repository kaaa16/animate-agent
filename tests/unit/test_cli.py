"""Tests for the `animate-agent` command-line entry point."""

import json
from pathlib import Path

import pytest

from animate_agent import cli, llm
from animate_agent.documents.models import DocumentIR
from animate_agent.knowledge.models import LessonIR, LessonScene
from animate_agent.rendering.models import RenderSpec
from animate_agent.storyboard.models import (
    StoryboardIR,
    StoryboardObject,
    StoryboardScene,
    StoryboardStep,
)

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
    # Clearing the variable is no longer enough to mean "there is no key": since
    # the loader landed, a `.env` at the repository root is a second source, so
    # on a developer's machine this test would sail past the check and dial the
    # API for real — failing, and charging for it. Point the loader at a file
    # that is not there, so "no key" is a property of the test and not of
    # whoever happens to be running it.
    monkeypatch.setattr(llm, "ENV_FILE", tmp_path / "absent.env")
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


def _lesson() -> LessonIR:
    return LessonIR(
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


def _storyboard() -> StoryboardIR:
    return StoryboardIR(
        storyboard_id="storyboard-lesson-x",
        lesson_id="lesson-x",
        document_id="x",
        title="力学基础",
        subject="物理",
        scenes=[
            StoryboardScene(
                id="shot-1",
                scene_type="generic",
                teaching_goal="理解惯性定律",
                lesson_scene_ids=["scene-1"],
                objects=[StoryboardObject(id="body-1", role="object", label="物体")],
                steps=[
                    StoryboardStep(
                        id="step-1",
                        title="保持静止",
                        description="物体保持在原来的位置静止不动。",
                        highlights=["body-1"],
                        key_points=["惯性"],
                    )
                ],
            )
        ],
    )


def test_storyboard_from_reuses_the_lesson_and_skips_the_knowledge_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Iterating on the storyboard prompt must not pay for a knowledge call, and
    # reusing the lesson keeps the outline fixed so a change in the output is
    # attributable to the prompt rather than to a regenerated lesson.
    monkeypatch.setenv("DEEPSEEK_KEY", "test-key")
    path = _write_sample(tmp_path)
    lesson_path = tmp_path / "lesson.json"
    lesson_path.write_text(_lesson().model_dump_json(), encoding="utf-8")
    calls: list[str] = []

    async def forbidden_generate_lesson(document: DocumentIR) -> LessonIR:
        calls.append("knowledge")
        raise AssertionError("--storyboard-from 不应该调用 Knowledge Agent")

    async def fake_generate_storyboard(
        lesson: LessonIR,
        document: DocumentIR,
        output_dir: Path | None = None,
    ) -> StoryboardIR:
        calls.append("storyboard")
        assert lesson.lesson_id == "lesson-x"
        return _storyboard()

    monkeypatch.setattr(cli, "generate_lesson", forbidden_generate_lesson)
    monkeypatch.setattr(cli, "generate_storyboard", fake_generate_storyboard)

    exit_code = cli.main([str(path), "--storyboard-from", str(lesson_path)])
    captured = capsys.readouterr()

    assert exit_code == cli.EXIT_OK
    assert calls == ["storyboard"]
    assert StoryboardIR.model_validate(json.loads(captured.out)) == _storyboard()


def test_storyboard_from_missing_file_reports_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DEEPSEEK_KEY", "test-key")

    exit_code = cli.main(
        [str(_write_sample(tmp_path)), "--storyboard-from", str(tmp_path / "nope.json")]
    )

    assert exit_code == cli.EXIT_INGEST_FAILED
    assert "找不到 LessonIR 文件" in capsys.readouterr().err


def test_render_template_needs_no_document_and_no_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Checkpoint P2's entry point: the baseline reaches the player with zero LLM.

    `load_llm_config` is booby-trapped rather than merely unset, so this fails if
    the render path ever starts consulting the model configuration — which is
    exactly the property that makes a baseline picture trustworthy.
    """

    def _explode() -> None:
        raise AssertionError("--render-template 不该碰 LLM 配置")

    monkeypatch.setattr(cli, "load_llm_config", _explode)

    exit_code = cli.main(
        ["--render-template", "robot_obstacle_avoidance", "--output-dir", str(tmp_path)]
    )

    assert exit_code == cli.EXIT_OK
    written = tmp_path / "render-robot_obstacle_avoidance.json"
    spec = RenderSpec.model_validate_json(written.read_text(encoding="utf-8"))
    assert spec.scenes[0].preset == "lane"
    assert {element.id for element in spec.scenes[0].elements} == {
        "car",
        "lidar",
        "obstacle-1",
        "obstacle-2",
        "safe-zone",
    }
    assert RenderSpec.model_validate(json.loads(capsys.readouterr().out)) == spec


def test_a_missing_path_says_what_to_do_instead(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main([])

    assert exit_code == cli.EXIT_INGEST_FAILED
    assert "--render-template" in capsys.readouterr().err
