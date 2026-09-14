"""Shared helpers for reading structured data out of LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any, cast


def extract_json_object(raw: str) -> dict[str, Any]:
    """Extract a JSON object from an LLM response, tolerating fences and prose."""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    parsed: Any = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"LLM 输出不是 JSON 对象: {type(parsed)!r}")
    return cast("dict[str, Any]", parsed)
