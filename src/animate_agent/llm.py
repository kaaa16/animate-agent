"""Shared OpenAI-compatible LLM client used by the agent stages."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import httpx

#: Output budget for one chat call. The configured endpoint serves a *reasoning*
#: model: `max_tokens` covers the hidden reasoning plus the visible answer, and
#: on the storyboard prompt the reasoning alone runs to ~23k tokens. A tight
#: budget is spent entirely on reasoning, which returns HTTP 200 with an empty
#: body — not an error the transport can flag.
DEFAULT_MAX_TOKENS = 32768


class LLMBudgetExhaustedError(RuntimeError):
    """The model spent its whole output budget on reasoning and never answered.

    Deliberately not a `ValueError`: the agents retry on `ValueError`, and
    retrying at the same budget cannot succeed — it just burns another slow,
    expensive reasoning call.
    """


@dataclass(frozen=True)
class LLMConfig:
    """Connection settings for an OpenAI-compatible chat-completions endpoint."""

    base_url: str
    api_key: str
    model: str


class LLMClient:
    """Thin async client for OpenAI-compatible chat completions (DeepSeek, Moonshot)."""

    def __init__(self, config: LLMConfig, *, client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Send one chat request and return the assistant message content."""
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self._config.api_key}"}
        url = self._config.base_url.rstrip("/") + "/chat/completions"

        response = await self._client.post(url, json=payload, headers=headers)
        if response.status_code == 429:
            await asyncio.sleep(3.0)
            response = await self._client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices")
        if not choices:
            raise ValueError(f"LLM returned no choices: {data}")

        choice = choices[0]
        content = choice.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError(f"LLM returned non-string content: {content!r}")
        if not content.strip():
            # Left undetected, an exhausted budget surfaces further down as
            # "Expecting value: line 1 column 1", which blames the JSON instead
            # of the budget and sends you looking in the wrong place.
            if choice.get("finish_reason") == "length":
                raise LLMBudgetExhaustedError(
                    f"模型把 max_tokens({max_tokens}) 全部用在推理上，没有产出正文。"
                    "请调大 max_tokens——用同样的预算重试没有意义。"
                )
            raise ValueError(f"LLM 返回空正文（finish_reason={choice.get('finish_reason')}）")
        return content

    async def aclose(self) -> None:
        """Close the underlying client when this object owns it."""
        if self._owns_client:
            await self._client.aclose()


def load_llm_config() -> LLMConfig:
    """Build an LLMConfig from environment variables (DeepSeek defaults)."""
    base_url = os.environ.get("DEEPSEEK_BASE", "https://api.deepseek.com")
    api_key = os.environ.get("DEEPSEEK_KEY", "")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
    return LLMConfig(base_url=base_url, api_key=api_key, model=model)
