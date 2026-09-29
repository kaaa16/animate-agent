"""Command-line entry point: document file -> DocumentIR -> LessonIR.

Turns the whole ingestion/Knowledge-Agent chain into one reproducible command,
so a document can be run end-to-end without going through the HTTP API:

    animate-agent path/to/doc.md
    animate-agent path/to/doc.md --ingest-only      # no LLM call, no API key needed
    animate-agent path/to/doc.md --storyboard       # continue on to StoryboardIR
    animate-agent path/to/doc.md --render           # all the way to a drawable RenderSpec
    animate-agent path/to/doc.md --storyboard-from data/generated/lesson-x.json
    animate-agent --render-template robot_obstacle_avoidance   # no document, no LLM
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import httpx

from animate_agent.documents.models import DocumentIR
from animate_agent.documents.service import ingest_file
from animate_agent.knowledge.models import LessonIR
from animate_agent.knowledge.service import generate_lesson
from animate_agent.llm import LLMBudgetExhaustedError, load_llm_config
from animate_agent.rendering.layout import LayoutError, layout_storyboard
from animate_agent.rendering.legacy import FROZEN_TEMPLATES, scene_to_render_spec
from animate_agent.rendering.models import RenderSpec
from animate_agent.rendering.service import (
    render_spec_path,
    render_storyboard,
    write_render_spec,
)
from animate_agent.speech.engine import EngineUnavailable, SpeechError
from animate_agent.speech.service import SpeechReport, attach_speech
from animate_agent.storyboard.models import StoryboardIR
from animate_agent.storyboard.service import DEFAULT_GENERATED_DIR, generate_storyboard

EXIT_OK = 0
EXIT_GENERATION_FAILED = 1
EXIT_INGEST_FAILED = 2
EXIT_MISSING_KEY = 3
#: edge-tts is an extra (`pip install -e ".[speech]"`) and the pipeline is
#: otherwise installable offline, so "you have not installed the voice" is its
#: own answer rather than a generation failure.
EXIT_MISSING_ENGINE = 4

#: Hand-written StoryboardIR fixtures. They are the validator's clean samples and
#: the layout layer's reference input — the picture a preset has to reproduce.
SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "samples" / "storyboards"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="animate-agent",
        description="把文档转成 LessonIR（DocumentIR -> Knowledge Agent -> LessonIR）。",
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help="输入文档路径（.pptx/.docx/.pdf/.md/.txt）；用 --render-template 时可省略。",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="只做摄取并打印 DocumentIR，不调用 LLM（无需 API key）。",
    )
    parser.add_argument(
        "--storyboard",
        action="store_true",
        help="摄取后继续跑 Storyboard Agent，输出 StoryboardIR（比默认多一次 LLM 调用）。",
    )
    parser.add_argument(
        "--storyboard-from",
        type=Path,
        default=None,
        help="跳过 Knowledge Agent，复用已有的 LessonIR JSON 直接跑 Storyboard Agent。"
        "迭代 storyboard 层时用它：省掉一次 LLM 调用，且课程大纲保持不变，"
        "这样 prompt 改动才是唯一变量。",
    )
    parser.add_argument(
        "--render-template",
        choices=sorted(FROZEN_TEMPLATES),
        default=None,
        help="把 animation/templates.py 里手写的场景翻成 RenderSpec 并落盘。"
        "不读文档、不调 LLM、不需要 API key——这是检查点 P2 的入口："
        "让那份手写的基线画面走与生成物**同一个播放器**，三方对照才是同口径的。",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="跑完整链路并落到一张能直接播放的 RenderSpec（隐含 --storyboard）。"
        "这是检查点 M4 的入口：一份真文档、一条命令、一帧真画面。"
        "布局失败会**失败**，不静默交出一张缺东西的图。",
    )
    parser.add_argument(
        "--render-sample",
        default=None,
        metavar="NAME",
        help="读 data/samples/storyboards/<NAME>.json（手写的 StoryboardIR），"
        "过布局层落成 RenderSpec。不读文档、不调 LLM、不需要 API key——"
        "这是检查点 P1 的入口：手写语义走完整路径，用来隔离「模型生成得对不对」"
        "与「布局画得对不对」这两个问题。",
    )
    parser.add_argument(
        "--script",
        action="store_true",
        default=None,
        help="稿子模式：输入是**为了做这条片子写的稿子**，不是随手拿到的文档。"
        "知识层照原话和顺序走、不做压缩，片长跟着稿子（仍然受一到两分钟那条线管）。"
        "不给这个参数时按 config/app.example.yaml 的 knowledge.script_mode 走，默认关。",
    )
    parser.add_argument(
        "--speak",
        action="store_true",
        help="给成片配音（隐含 --render）。每一拍合成一个 mp3，音色见 --voice。"
        '需要联网，并且要先装依赖：pip install -e ".[speech]"。'
        "合成完会把实测时长写回 spec，所以播放器从此按音频走，不再按字数公式。",
    )
    parser.add_argument(
        "--speak-from",
        type=Path,
        default=None,
        metavar="PATH",
        help="给一份**已经落盘的** RenderSpec 补配音，就地改写它。"
        "对一份跑好的片子换音色、或者补上第一次忘了配的音，用它不必重跑 LLM。",
    )
    parser.add_argument(
        "--voice",
        action="append",
        default=None,
        metavar="ID",
        help="配音音色，可重复给两次表示两个都要（多花一倍合成时间）。"
        "不给就按 config 的 speech.default_voice。可用值见 speech/voices.py。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="生成物落盘目录，默认 data/generated。",
    )
    return parser


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


async def _run(args: argparse.Namespace) -> int:
    if args.render_template is not None:
        return _render_template(args)
    if args.render_sample is not None:
        return _render_sample(args)
    if args.speak_from is not None:
        return await _speak_from(args)
    if args.path is None:
        print(
            "缺少输入文档路径。若只想看画面：基线用 --render-template，"
            "手写分镜用 --render-sample <NAME>。",
            file=sys.stderr,
        )
        return EXIT_INGEST_FAILED

    try:
        document = ingest_file(args.path)
    except FileNotFoundError:
        print(f"找不到文件: {args.path}", file=sys.stderr)
        return EXIT_INGEST_FAILED
    except ValueError as exc:
        print(f"摄取失败: {exc}", file=sys.stderr)
        return EXIT_INGEST_FAILED

    if args.ingest_only:
        _print_json(document.model_dump(mode="json"))
        return EXIT_OK

    if not load_llm_config().api_key:
        print(
            "未配置 DEEPSEEK_KEY，无法调用 Knowledge Agent。"
            "请在 .env 或环境变量中设置，或先用 --ingest-only 检查摄取结果。",
            file=sys.stderr,
        )
        return EXIT_MISSING_KEY

    # `--render` is the storyboard step plus a layout, so asking for a picture
    # implies asking for the IR it is drawn from. `--speak` goes further still —
    # speech is attached to a spec — so it implies the same two.
    wants_storyboard = (
        args.storyboard or args.storyboard_from is not None or args.render or args.speak
    )
    # And speech needs a spec to attach itself to. `--speak` says `--render` by
    # implication rather than by requiring both on the command line, the way
    # `--render` says `--storyboard`.
    wants_render = args.render or args.speak

    try:
        if args.storyboard_from is not None:
            lesson = _load_lesson(args.storyboard_from)
        else:
            lesson = await _generate_lesson(document, args.output_dir, args.script)
        if wants_storyboard:
            # `args.script` and not the lesson's own history: a storyboard built
            # from a `--storyboard-from` file has no way to know which mode wrote
            # it, and the flag is the caller saying what it wants *now*.
            storyboard = await _generate_storyboard(lesson, document, args.output_dir, args.script)
    except FileNotFoundError:
        print(f"找不到 LessonIR 文件: {args.storyboard_from}", file=sys.stderr)
        return EXIT_INGEST_FAILED
    except LLMBudgetExhaustedError as exc:
        print(f"调用 LLM 失败: {exc}", file=sys.stderr)
        return EXIT_GENERATION_FAILED
    except httpx.HTTPError as exc:
        print(f"调用 LLM 失败: {exc}", file=sys.stderr)
        return EXIT_GENERATION_FAILED
    except ValueError as exc:
        print(f"生成失败: {exc}", file=sys.stderr)
        return EXIT_GENERATION_FAILED

    print(f"lesson_id={lesson.lesson_id} scenes={len(lesson.scenes)}", file=sys.stderr)
    if not wants_storyboard:
        _print_json(lesson.model_dump(mode="json"))
        return EXIT_OK
    if not wants_render:
        _print_json(storyboard.model_dump(mode="json"))
        return EXIT_OK

    output_dir = args.output_dir or DEFAULT_GENERATED_DIR
    try:
        spec = render_storyboard(storyboard, output_dir=output_dir)
    except LayoutError as exc:
        # The storyboard is already on disk and is the only way to find out why
        # the layout refused. Say so rather than leaving a bare traceback — and
        # name the directory actually in use, which is not the default whenever
        # `--output-dir` was passed. Pointing at the wrong one sends a person
        # looking for a file that was never written there.
        print(f"布局失败: {exc}", file=sys.stderr)
        print(
            f"StoryboardIR 已经落盘（{output_dir}/{storyboard.storyboard_id}.json）"
            "——布局失败的唯一线索，别丢掉它。",
            file=sys.stderr,
        )
        return EXIT_GENERATION_FAILED

    destination = render_spec_path(storyboard.storyboard_id, output_dir=output_dir)
    print(f"已写入 {destination}", file=sys.stderr)

    if args.speak:
        code = await _speak_into(
            spec, output_dir=output_dir, voices=args.voice, destination=destination
        )
        if code != EXIT_OK:
            return code

    _print_json(spec.model_dump(mode="json"))
    return EXIT_OK


def _report_speech(report: SpeechReport) -> None:
    """Say what the speech stage did, on stderr, where the other progress lines go."""
    print(
        f"配音完成: {report.beats} 拍 / 音色 {'、'.join(report.voices)} / "
        f"新合成 {report.synthesised} 段、复用 {report.reused} 段 / "
        f"成片 {report.seconds:.1f} 秒（{report.seconds / 60:.2f} 分钟）",
        file=sys.stderr,
    )
    if report.warning is not None:
        print(f"注意: {report.warning}", file=sys.stderr)


async def _speak_into(
    spec: RenderSpec, *, output_dir: Path, voices: list[str] | None, destination: Path
) -> int:
    """Attach speech to `spec`, rewrite it, and report. Returns an exit code.

    Writes the spec itself rather than leaving it to the caller, because the
    spec on disk and the spec in memory have to be the same object's dump — the
    file was already written once by `render_storyboard`, and the timings in it
    are now wrong.
    """
    try:
        report = await attach_speech(spec, output_dir=output_dir, voices=voices)
    except EngineUnavailable as exc:
        print(f"配不了音: {exc}", file=sys.stderr)
        return EXIT_MISSING_ENGINE
    except (SpeechError, ValueError) as exc:
        # The unvoiced spec is already on disk and still playable. Say so, rather
        # than leaving a person with a half-configured film and no way to tell.
        print(f"配音失败: {exc}", file=sys.stderr)
        print(f"没有配音的 {destination} 仍然可以播。", file=sys.stderr)
        return EXIT_GENERATION_FAILED

    write_render_spec(spec, destination)
    _report_speech(report)
    return EXIT_OK


async def _speak_from(args: argparse.Namespace) -> int:
    """Attach speech to a spec that is already on disk, rewriting it in place.

    The road for a film that was rendered before there was any voice, and for
    trying a second voice on a film that already has one: both are a spec on
    disk and a synthesis pass, with no model call anywhere.
    """
    source: Path = args.speak_from
    if not source.is_file():
        print(f"找不到 spec：{source}", file=sys.stderr)
        return EXIT_INGEST_FAILED

    # `utf-8-sig` for the same reason as `--render-sample`: written by this CLI,
    # but the file a person edits between runs, and a BOM read as plain utf-8
    # surfaces as "expected value at line 1 column 1".
    spec = RenderSpec.model_validate_json(source.read_text(encoding="utf-8-sig"))
    # The clips belong next to the spec, which is where its relative `src` points
    # — not in the default directory, which is right only when the spec is there
    # too. `--output-dir` still wins when it is given.
    output_dir = args.output_dir or source.parent
    code = await _speak_into(spec, output_dir=output_dir, voices=args.voice, destination=source)
    if code != EXIT_OK:
        return code

    print(f"已改写 {source}", file=sys.stderr)
    _print_json(spec.model_dump(mode="json"))
    return EXIT_OK


def _render_template(args: argparse.Namespace) -> int:
    """Translate a frozen `Scene` and persist the resulting `RenderSpec`.

    Writes to disk as well as stdout because the player fetches a spec by URL —
    `--render-template` is how a picture reaches the browser without an LLM call.
    """
    spec = scene_to_render_spec(FROZEN_TEMPLATES[args.render_template]())
    output_dir = args.output_dir or DEFAULT_GENERATED_DIR
    destination = render_spec_path(args.render_template, output_dir=output_dir)
    write_render_spec(spec, destination)

    print(f"已写入 {destination}", file=sys.stderr)
    _print_json(spec.model_dump(mode="json"))
    return EXIT_OK


def _render_sample(args: argparse.Namespace) -> int:
    """Lay out a hand-written StoryboardIR and persist the `RenderSpec`.

    The counterpart to `_render_template`, and the other half of the three-way
    comparison: `--render-template` shows the baseline *translated*, this shows
    the new contract *laid out*. Same player, same stage, so a difference
    between the two pictures is a difference in the layout, not in the drawing.
    """
    source = SAMPLES_DIR / f"{args.render_sample}.json"
    if not source.exists():
        available = (
            sorted(path.stem for path in SAMPLES_DIR.glob("*.json")) if SAMPLES_DIR.is_dir() else []
        )
        print(
            f"找不到样例 {source}；可用：{'、'.join(available) or '（无）'}",
            file=sys.stderr,
        )
        return EXIT_INGEST_FAILED

    # `utf-8-sig`: these samples are hand-written, and a BOM read as plain `utf-8`
    # surfaces as "Invalid JSON: expected value at line 1 column 1", which names
    # everything except the actual cause.
    storyboard = StoryboardIR.model_validate_json(source.read_text(encoding="utf-8-sig"))
    try:
        # `layout_storyboard`, not `render_storyboard`: the latter also writes the
        # spec under `render_spec_path`, and this command's file is named
        # differently (see below). Calling it here would write the picture twice
        # and leave behind a `render-<storyboard_id>.json` nobody asked for.
        spec = layout_storyboard(storyboard)
    except LayoutError as exc:
        print(f"布局失败: {exc}", file=sys.stderr)
        return EXIT_GENERATION_FAILED

    output_dir = args.output_dir or DEFAULT_GENERATED_DIR
    # `render-sample-`, deliberately not `render_spec_path`: `--render-sample` and
    # `--render-template` can be given the same key, and both are checkpoint P2's
    # acceptance fixtures. One shared filename would let either silently
    # overwrite the other's picture, which is the one thing a baseline used for
    # comparison must never do.
    destination = output_dir / f"render-sample-{args.render_sample}.json"
    write_render_spec(spec, destination)

    print(f"已写入 {destination}", file=sys.stderr)
    _print_json(spec.model_dump(mode="json"))
    return EXIT_OK


def _load_lesson(path: Path) -> LessonIR:
    """Read a LessonIR previously written by this CLI.

    Reusing one keeps the course outline fixed across storyboard-prompt
    iterations, so a change in the output is attributable to the prompt.
    """
    # Tolerates a BOM for the same reason as `--render-sample`: this file is
    # written by the CLI, but it is also the one a person edits by hand between
    # prompt iterations, which is the whole point of the flag.
    return LessonIR.model_validate_json(path.read_text(encoding="utf-8-sig"))


async def _generate_lesson(
    document: DocumentIR, output_dir: Path | None, script_mode: bool | None
) -> LessonIR:
    if output_dir is None:
        return await generate_lesson(document, script_mode=script_mode)
    return await generate_lesson(document, output_dir=output_dir, script_mode=script_mode)


async def _generate_storyboard(
    lesson: LessonIR,
    document: DocumentIR,
    output_dir: Path | None,
    script_mode: bool | None,
) -> StoryboardIR:
    if output_dir is None:
        return await generate_storyboard(lesson, document, script_mode=script_mode)
    return await generate_storyboard(
        lesson, document, output_dir=output_dir, script_mode=script_mode
    )


def main(argv: list[str] | None = None) -> int:
    # The per-call timing line (`llm._log_call`) is INFO, and INFO goes nowhere
    # without a handler — Python's fallback shows WARNING and above only. That
    # line is the only thing that says *which* of the two model calls is costing
    # the minutes, so the entry point that runs the pipeline turns it on.
    # `%(message)s` and no timestamp: the line already names its own stage, and
    # the added prefix would be the same four words on every line.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
