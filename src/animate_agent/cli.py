"""Command-line entry point: document file -> DocumentIR -> LessonIR.

Turns the whole ingestion/Knowledge-Agent chain into one reproducible command,
so a document can be run end-to-end without going through the HTTP API:

    animate-agent path/to/doc.md
    animate-agent path/to/doc.md --ingest-only      # no LLM call, no API key needed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

from animate_agent.documents.service import ingest_file
from animate_agent.knowledge.service import generate_lesson
from animate_agent.llm import load_llm_config

EXIT_OK = 0
EXIT_GENERATION_FAILED = 1
EXIT_INGEST_FAILED = 2
EXIT_MISSING_KEY = 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="animate-agent",
        description="把文档转成 LessonIR（DocumentIR -> Knowledge Agent -> LessonIR）。",
    )
    parser.add_argument("path", type=Path, help="输入文档路径（.pptx/.docx/.pdf/.md/.txt）")
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="只做摄取并打印 DocumentIR，不调用 LLM（无需 API key）。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="LessonIR 落盘目录，默认 data/generated。",
    )
    return parser


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


async def _run(args: argparse.Namespace) -> int:
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

    try:
        if args.output_dir is not None:
            lesson = await generate_lesson(document, output_dir=args.output_dir)
        else:
            lesson = await generate_lesson(document)
    except httpx.HTTPError as exc:
        print(f"调用 LLM 失败: {exc}", file=sys.stderr)
        return EXIT_GENERATION_FAILED
    except ValueError as exc:
        print(f"生成 LessonIR 失败: {exc}", file=sys.stderr)
        return EXIT_GENERATION_FAILED

    print(
        f"lesson_id={lesson.lesson_id} scenes={len(lesson.scenes)}",
        file=sys.stderr,
    )
    _print_json(lesson.model_dump(mode="json"))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
