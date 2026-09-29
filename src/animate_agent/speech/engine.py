"""The edge-tts adapter, and the two things about it that fail invisibly.

**One.** edge-tts is imported inside a function and nowhere else. Two reasons, and
the second is the one that is easy to miss. The obvious one is that a machine
without the package must still be able to import this module — `speech` is part
of the package, the test suite imports it, and `edge-tts` is an extra rather than
a dependency. The subtle one is `mypy --strict` with `warn_unused_ignores`: a
plain `import edge_tts` is `import-not-found` where the package is absent and
needs a `# type: ignore`, and on a machine where it *is* present that same
comment is an unused ignore, which is itself an error. The two states cannot both
be satisfied by one import line. `importlib.import_module` has no import to
check, and satisfies both.

**Two.** `boundary="WordBoundary"` has to be passed explicitly. edge-tts 7.2
changed the default, and the new default returns **no boundary events at all** —
not a coarser kind, none. A caller who does not ask gets a stream of audio and
nothing else, so `last_word_end` stays at zero and every beat silently falls back
to the character formula. Nothing raises, nothing looks wrong, and the film is
about a second short per beat for reasons no error message mentions. Hence the
count below and the exception when it is zero: the failure mode is "everything
looks fine", which is the one worth spending an exception on.

The other trap lives in the same loop — see `TICKS_PER_SECOND`.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass
from typing import Any, Protocol

#: edge-tts reports a word's `offset` and `duration` in 100-nanosecond ticks,
#: which is Azure's wire unit and not milliseconds. Dividing by 1000 instead
#: turns a four-second beat into eleven hours, at which point the player holds
#: the first beat of the film for ever and no test can see it — the fake engine
#: builds its own `seconds` and never touches this line.
TICKS_PER_SECOND = 10_000_000


class SpeechError(RuntimeError):
    """The speech stage could not do its job."""


class EngineUnavailable(SpeechError):
    """edge-tts is not installed, or the package metadata is unreadable."""


@dataclass(frozen=True)
class Clip:
    """One synthesized line: the bytes, and when the voice stopped speaking.

    `last_word_end` is measured from the start of *this* clip, so it includes the
    lead-in silence. It is deliberately not the clip's length — the tail the
    encoder pads on is three times the pause a beat should get.
    """

    audio: bytes
    last_word_end: float


class SpeechEngine(Protocol):
    """What `service` needs from an engine, and the whole of what it needs.

    A `Protocol` rather than a base class so that `fakes.DeterministicSpeechEngine`
    is a test double by construction and not by inheritance — and so that a future
    engine (a paid API, a local model) is a new class rather than a new branch.
    """

    #: The encoding, as part of the cache key. Two engines that produce different
    #: bytes for the same text must not share a clip file.
    fmt: str

    @property
    def engine_id(self) -> str:
        """Version of the thing whose *boundaries* were measured. Cache key part."""
        ...

    async def synthesize(self, text: str, voice: str, rate: str) -> Clip:
        """Read `text` in `voice` at `rate`, and say when it stopped."""
        ...


def _edge_tts() -> Any:
    """The edge-tts module, or `EngineUnavailable`.

    The only place in the package that names it. `Any` and not a stub: the return
    value is handed straight back to this module's own code, and inventing a
    protocol for a third-party library's `Communicate` would be a second thing to
    keep in step with it.
    """
    try:
        return importlib.import_module("edge_tts")
    except ImportError as error:
        raise EngineUnavailable(
            '未安装 edge-tts。装上它再配配音：pip install -e ".[speech]"'
        ) from error


def engine_version() -> str:
    """The installed edge-tts version, or `""` when it cannot be read.

    Never raises. This is a cache key component, and a cache that refuses to
    work because it cannot name itself is worse than one that occasionally
    re-synthesizes.
    """
    try:
        return importlib.metadata.version("edge-tts")
    except importlib.metadata.PackageNotFoundError:
        return ""


class EdgeTTSEngine:
    """Synthesizes through Microsoft Edge's read-aloud endpoint, via edge-tts.

    Free, needs no key and no account, and reads Chinese well — which is why the
    reference project uses it and why the pipeline's `SPEECH_CHARS_PER_SECOND`
    is calibrated against it. The cost is that it is an unofficial client of a
    service that is not documented as one; edge-tts' own README pins the version
    for exactly that reason.
    """

    fmt = "audio-24khz-48kbitrate-mono-mp3"

    def __init__(self) -> None:
        # Fail here rather than on the first beat, so that a missing extra is one
        # error at the start instead of one per clip.
        _edge_tts()

    @property
    def engine_id(self) -> str:
        return engine_version()

    async def synthesize(self, text: str, voice: str, rate: str) -> Clip:
        edge_tts = _edge_tts()
        audio = bytearray()
        last_word_end = 0.0
        boundaries = 0

        communicate = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
        async for chunk in communicate.stream():
            kind = chunk.get("type")
            if kind == "audio":
                audio.extend(chunk["data"])
            elif kind == "WordBoundary":
                boundaries += 1
                end = (chunk["offset"] + chunk["duration"]) / TICKS_PER_SECOND
                last_word_end = max(last_word_end, end)

        if not audio:
            raise SpeechError(f"edge-tts 对《{text[:20]}》一个字节的音频都没返回")
        if boundaries == 0:
            # See the module docstring: this is the failure that looks like
            # success. Without it every beat quietly uses the character formula.
            raise SpeechError(
                f"edge-tts 对《{text[:20]}》没有返回任何词边界——"
                '`boundary="WordBoundary"` 是不是没传？'
                "没有边界就量不出这一拍该停多久，只能悄悄退回字数公式"
            )
        return Clip(audio=bytes(audio), last_word_end=last_word_end)
