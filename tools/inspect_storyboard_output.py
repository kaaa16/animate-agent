"""Feed an un-retouched model response through the storyboard checks.

Why this exists
---------------

The Storyboard Agent retries with the validator's feedback, so the StoryboardIR
it finally returns is a *repaired* one — it says nothing about what the model
does on its own. The interesting artefact is **attempt 1**: the response the
model produced with no feedback at all. Everything the validator flags there is
a constraint the prompt failed to teach.

`StoryboardAgent(debug_dir=...)` writes every rejected response to disk, numbered
by attempt, as a `…-attempt-N-raw.txt` / `…-attempt-N-error.txt` pair. The error
file is the authoritative record of what the validator said *during that run*;
this script re-derives it from the raw text. The difference between the two is
the point: replaying an old attempt through a *newer* validator answers "would
today's rules have rejected what the model wrote before the rules changed?", and
that is measurable without spending an LLM call.

Usage
-----

    python -m tools.inspect_storyboard_output \
        data/generated/s1-run/storyboard-lesson-985e1285ffd6ef6b-attempt-1-raw.txt \
        --lesson data/generated/lesson-985e1285ffd6ef6b.json

Add `--document` to also check that every `source_refs` id really exists; without
it that one check is skipped (see `validate_storyboard`).
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from pydantic import ValidationError

from animate_agent.documents.models import DocumentIR
from animate_agent.json_utils import extract_json_object
from animate_agent.knowledge.models import LessonIR
from animate_agent.storyboard.agent import apply_identity
from animate_agent.storyboard.models import StoryboardIR
from animate_agent.storyboard.service import build_limits
from animate_agent.storyboard.validation import format_issues, validate_storyboard

EXIT_OK = 0
EXIT_UNREADABLE = 1


def _report_schema_errors(exc: ValidationError) -> None:
    print(f"schema 不符：共 {exc.error_count()} 处")
    counts: collections.Counter[str] = collections.Counter()
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        print(f"  - {location}：{error['msg']}")
        # Group by the *field*, not the whole path: 31 `steps[N].description`
        # errors are one problem, and the count is what says so.
        field = location.rsplit(".", 1)[-1]
        counts[field] += 1
    print("\n按字段归并：")
    for field, count in counts.most_common():
        print(f"  {field}: {count} 处")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw", type=Path, help="模型原始输出（.txt 或 .json）")
    parser.add_argument("--lesson", type=Path, required=True, help="对应的 LessonIR JSON")
    parser.add_argument("--document", type=Path, default=None, help="对应的 DocumentIR JSON")
    args = parser.parse_args(argv)

    lesson = LessonIR.model_validate_json(args.lesson.read_text(encoding="utf-8"))
    document = (
        DocumentIR.model_validate_json(args.document.read_text(encoding="utf-8"))
        if args.document is not None
        else None
    )

    try:
        data = extract_json_object(args.raw.read_text(encoding="utf-8"))
    except ValueError as exc:
        print(f"连 JSON 都没抽出来（模型可能压根没产出正文）：{exc}", file=sys.stderr)
        return EXIT_UNREADABLE
    except FileNotFoundError:
        print(f"找不到文件: {args.raw}", file=sys.stderr)
        return EXIT_UNREADABLE

    try:
        storyboard = StoryboardIR.model_validate(apply_identity(lesson, data))
    except ValidationError as exc:
        _report_schema_errors(exc)
        return EXIT_OK

    print("schema 通过。")
    issues = validate_storyboard(
        storyboard, lesson=lesson, document=document, limits=build_limits()
    )
    if not issues:
        print("校验台也全过——这一份输出是干净的。")
        return EXIT_OK

    print(format_issues(issues))
    print("\n按 code 归并：")
    for code, count in collections.Counter(issue.code for issue in issues).most_common():
        print(f"  {code}: {count} 处")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
