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
from animate_agent.storyboard.models import StoryboardIR
from animate_agent.storyboard.service import DEFAULT_GENERATED_DIR, generate_storyboard

EXIT_OK = 0
EXIT_GENERATION_FAILED = 1
EXIT_INGEST_FAILED = 2
EXIT_MISSING_KEY = 3

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
    # implies asking for the IR it is drawn from.
    wants_storyboard = args.storyboard or args.storyboard_from is not None or args.render

    try:
        if args.storyboard_from is not None:
            lesson = _load_lesson(args.storyboard_from)
        else:
            lesson = await _generate_lesson(document, args.output_dir)
        if wants_storyboard:
            storyboard = await _generate_storyboard(lesson, document, args.output_dir)
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
    if not args.render:
        _print_json(storyboard.model_dump(mode="json"))
        return EXIT_OK

    try:
        spec = layout_storyboard(storyboard)
    except LayoutError as exc:
        # The storyboard is already on disk and is the only way to find out why
        # the layout refused. Say so rather than leaving a bare traceback.
        print(f"布局失败: {exc}", file=sys.stderr)
        print(
            f"StoryboardIR 已经落盘（{DEFAULT_GENERATED_DIR}/"
            f"{storyboard.storyboard_id}.json）——布局失败的唯一线索，别丢掉它。",
            file=sys.stderr,
        )
        return EXIT_GENERATION_FAILED

    output_dir = args.output_dir or DEFAULT_GENERATED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"render-{storyboard.storyboard_id}.json"
    destination.write_text(spec.model_dump_json(indent=2), encoding="utf-8")

    print(f"已写入 {destination}", file=sys.stderr)
    _print_json(spec.model_dump(mode="json"))
    return EXIT_OK


def _render_template(args: argparse.Namespace) -> int:
    """Translate a frozen `Scene` and persist the resulting `RenderSpec`.

    Writes to disk as well as stdout because the player fetches a spec by URL —
    `--render-template` is how a picture reaches the browser without an LLM call.
    """
    spec = scene_to_render_spec(FROZEN_TEMPLATES[args.render_template]())
    output_dir = args.output_dir or DEFAULT_GENERATED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"render-{args.render_template}.json"
    destination.write_text(spec.model_dump_json(indent=2), encoding="utf-8")

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

    storyboard = StoryboardIR.model_validate_json(source.read_text(encoding="utf-8"))
    try:
        spec = layout_storyboard(storyboard)
    except LayoutError as exc:
        print(f"布局失败: {exc}", file=sys.stderr)
        return EXIT_GENERATION_FAILED

    output_dir = args.output_dir or DEFAULT_GENERATED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"render-sample-{args.render_sample}.json"
    destination.write_text(spec.model_dump_json(indent=2), encoding="utf-8")

    print(f"已写入 {destination}", file=sys.stderr)
    _print_json(spec.model_dump(mode="json"))
    return EXIT_OK


def _load_lesson(path: Path) -> LessonIR:
    """Read a LessonIR previously written by this CLI.

    Reusing one keeps the course outline fixed across storyboard-prompt
    iterations, so a change in the output is attributable to the prompt.
    """
    return LessonIR.model_validate_json(path.read_text(encoding="utf-8"))


async def _generate_lesson(document: DocumentIR, output_dir: Path | None) -> LessonIR:
    if output_dir is None:
        return await generate_lesson(document)
    return await generate_lesson(document, output_dir=output_dir)


async def _generate_storyboard(
    lesson: LessonIR,
    document: DocumentIR,
    output_dir: Path | None,
) -> StoryboardIR:
    if output_dir is None:
        return await generate_storyboard(lesson, document)
    return await generate_storyboard(lesson, document, output_dir=output_dir)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
