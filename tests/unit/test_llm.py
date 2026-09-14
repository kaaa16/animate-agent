"""Tests for the shared LLM client's failure handling."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from animate_agent.llm import DEFAULT_MAX_TOKENS, LLMBudgetExhaustedError, LLMClient, LLMConfig


def _chat(payload: dict[str, Any], *, max_tokens: int = 8192) -> str:
    async def run() -> str:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload, request=request)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            llm = LLMClient(
                LLMConfig(base_url="http://test", api_key="k", model="m"),
                client=client,
            )
            return await llm.chat(
                [{"role": "user", "content": "hi"}],
                temperature=0.3,
                max_tokens=max_tokens,
            )

    return asyncio.run(run())


def _choice(content: Any, finish_reason: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}


def test_returns_content() -> None:
    assert _chat(_choice("hello", "stop")) == "hello"


def test_empty_content_with_length_finish_reason_is_a_budget_error() -> None:
    # The reasoning model spends the whole budget on its hidden reasoning before
    # writing anything, so the body comes back empty with HTTP 200. Without this
    # check it surfaces downstream as "Expecting value: line 1 column 1".
    with pytest.raises(LLMBudgetExhaustedError, match="max_tokens"):
        _chat(_choice("", "length"), max_tokens=8192)


def test_budget_error_is_not_a_value_error() -> None:
    # The agents retry on ValueError. Retrying at the same budget cannot succeed,
    # so this must escape the retry loop rather than buy two more slow calls.
    assert not issubclass(LLMBudgetExhaustedError, ValueError)


def test_empty_content_for_other_reasons_is_a_plain_failure() -> None:
    with pytest.raises(ValueError, match="空正文"):
        _chat(_choice("", "stop"))


def test_whitespace_only_content_counts_as_empty() -> None:
    with pytest.raises(ValueError, match="空正文"):
        _chat(_choice("   \n ", "stop"))


def test_missing_choices_raises() -> None:
    with pytest.raises(ValueError, match="no choices"):
        _chat({"choices": []})


def test_non_string_content_raises() -> None:
    with pytest.raises(ValueError, match="non-string"):
        _chat(_choice(None, "stop"))


def test_default_budget_leaves_room_for_the_reasoning_pass() -> None:
    # The storyboard prompt alone was observed spending ~23k tokens of hidden
    # reasoning; a smaller shared default reintroduces the empty-content bug.
    assert DEFAULT_MAX_TOKENS >= 32768
