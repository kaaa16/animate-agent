"""Tests for the shared LLM client's failure handling."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from animate_agent import llm
from animate_agent.llm import (
    DEFAULT_MAX_TOKENS,
    ENV_FILE,
    LLMBudgetExhaustedError,
    LLMClient,
    LLMConfig,
    load_env_file,
)


@pytest.fixture(autouse=True)
def _restore_environ() -> Iterator[None]:
    """Undo what `load_env_file` writes straight into `os.environ`.

    The function exists to mutate the process environment, so `monkeypatch`
    cannot unwind it — monkeypatch only reverts its own edits. Without this, a
    key one test loaded stays set for every test that runs after it, and the
    suite turns order-dependent in a way nothing announces.
    """
    before = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(before)


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


def test_the_default_env_file_sits_at_the_repository_root() -> None:
    # Resolved from the module, not the working directory: `animate-agent` has to
    # find the same key whether it is invoked from the repo root or from anywhere
    # else. Two levels up from both `src/animate_agent/` and `tests/unit/`.
    repository_root = Path(__file__).resolve().parents[2]

    assert repository_root / ".env" == ENV_FILE


def test_env_file_fills_names_that_are_not_already_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("DEEPSEEK_KEY=sk-from-file\n", encoding="utf-8")

    assert load_env_file(path) == ["DEEPSEEK_KEY"]
    assert os.environ["DEEPSEEK_KEY"] == "sk-from-file"


def test_a_real_environment_variable_beats_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A deliberate `$env:DEEPSEEK_KEY` must win. Otherwise a stale file left in
    # the working copy silently overrides the key someone just exported, and the
    # run talks to the wrong account with no sign of it.
    monkeypatch.setenv("DEEPSEEK_KEY", "sk-from-env")
    path = tmp_path / ".env"
    path.write_text("DEEPSEEK_KEY=sk-from-file\n", encoding="utf-8")

    assert load_env_file(path) == []
    assert os.environ["DEEPSEEK_KEY"] == "sk-from-env"


def test_env_file_parses_assignments_and_ignores_everything_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("DEEPSEEK_KEY", "ALPHA", "BETA"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "# DEEPSEEK_KEY=sk-commented-out\n"
        "\n"
        "   \n"
        "ALPHA=1\n"
        "export BETA=2\n"
        "this line assigns nothing\n",
        encoding="utf-8",
    )

    assert load_env_file(path) == ["ALPHA", "BETA"]
    assert os.environ["ALPHA"] == "1"
    # `export NAME=value` is what most docs show and what gets copy-pasted;
    # keeping the name as `export BETA` would set a variable nobody reads while
    # the real one silently never loads.
    assert os.environ["BETA"] == "2"
    assert "DEEPSEEK_KEY" not in os.environ


def test_quotes_around_a_value_are_not_part_of_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text('DEEPSEEK_KEY="sk-quoted"\n', encoding="utf-8")

    load_env_file(path)

    assert os.environ["DEEPSEEK_KEY"] == "sk-quoted"


def test_a_byte_order_mark_does_not_break_the_first_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Windows editors still offer "UTF-8 with BOM". Decoding as plain `utf-8`
    # leaves the mark glued to the first name, so it parses as U+FEFF +
    # "DEEPSEEK_KEY" — a variable nobody reads, and the real key never loads.
    # Silently, which is the whole failure mode this loader exists to remove.
    monkeypatch.delenv("DEEPSEEK_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_bytes(b"\xef\xbb\xbfDEEPSEEK_KEY=sk-after-bom\n")

    assert load_env_file(path) == ["DEEPSEEK_KEY"]
    assert os.environ["DEEPSEEK_KEY"] == "sk-after-bom"
    # No name anywhere may carry the mark: that is the shape of the bug, and it
    # is invisible in a diff. Built from its code point so this file stays ASCII
    # and no invisible character ends up in the source.
    assert not any(name.startswith(chr(0xFEFF)) for name in os.environ)


def test_a_missing_env_file_is_not_an_error(tmp_path: Path) -> None:
    # Exporting the variable directly is a normal way to work, so a missing file
    # must not be a fault the caller has to guard against.
    assert load_env_file(tmp_path / "absent.env") == []


def test_load_llm_config_falls_back_to_the_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The CLI's "请在 .env 或环境变量中设置" promise. Until this wiring existed the
    # message named a file that nothing in `src/` ever opened.
    monkeypatch.delenv("DEEPSEEK_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("DEEPSEEK_KEY=sk-from-file\n", encoding="utf-8")
    monkeypatch.setattr(llm, "ENV_FILE", path)

    assert llm.load_llm_config().api_key == "sk-from-file"
