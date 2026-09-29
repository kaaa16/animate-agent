"""A speech engine that needs no network, for tests and for offline runs.

It lives in the source tree rather than under `tests/` for two reasons, one
practical and one honest. Practically, `tests/unit/` has no `conftest.py` and no
`__init__.py`, so a double shared by two test files has nowhere natural to sit;
here it is a normal import. Honestly, it has a second real use: it runs the whole
speech stage — file layout, cache hits, spec rewriting, the duration report —
without touching Microsoft, which is what makes those testable at all.

It is **not** a production switch. Nothing in `config`, `cli` or `api` can select
it; the only way to get one is to pass it to `attach_speech` as `engine=`, which
is what the tests do. A `speech.engine: fake` setting would be a way to generate a
film that looks voiced and is not, and that is not a setting worth having.

Its `fmt` is its own, so a cache warmed by this engine can never be mistaken for
one warmed by the real thing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from animate_agent.speech.engine import Clip
from animate_agent.speech.timing import HEAD_SILENCE_SECONDS, TAIL_PAUSE_SECONDS

#: Characters a second, matching `layout.SPEECH_CHARS_PER_SECOND`. Spelled out
#: rather than imported: `rendering` and `speech` are separate layers and a
#: double that reaches into the thing it is standing in for is a double that
#: cannot disagree with it, which is the one job a double has.
CHARS_PER_SECOND = 6.0


@dataclass(frozen=True)
class SynthesisCall:
    """One request the fake was asked to serve, in the order it was asked."""

    voice: str
    text: str


class DeterministicSpeechEngine:
    """A `SpeechEngine` whose answers are a pure function of its input.

    `last_word_end` is `HEAD_SILENCE_SECONDS + len(text) / CHARS_PER_SECOND`,
    which is the shape the real clips have (the lead-in is inside the measurement
    and the tail is not), so a film run through this engine comes out at the
    length a real one does rather than at a suspiciously round number.
    """

    fmt = "audio/fake"

    def __init__(
        self,
        delay: Callable[[int], float] | None = None,
        speech_end: Callable[[str], float] | None = None,
    ) -> None:
        #: Seconds to wait before answering, given the zero-based call index.
        #: `None` means answer immediately. The service synthesizes concurrently,
        #: so a `delay` that *decreases* with the index makes the calls complete
        #: in the reverse of the order they were issued — which is how the test
        #: that a spec does not depend on completion order gets its reversal.
        self._delay = delay
        #: When the last word stops, given the text. `None` uses the length
        #: formula below, which is what a real clip roughly looks like. A caller
        #: overrides it to place a measurement deliberately — the rounding test
        #: needs a value on a boundary, and no natural line length lands there.
        #:
        #: `last_word_end` is a float with four decimal places of meaning,
        #: because that is what the engine's word boundaries are: the real one
        #: reports 100-nanosecond ticks, so a beat's length is a number no
        #: hand-written fixture would produce.
        self._speech_end = speech_end
        self.calls: list[SynthesisCall] = []

    @property
    def engine_id(self) -> str:
        return "fake-1"

    async def synthesize(self, text: str, voice: str, rate: str) -> Clip:
        index = len(self.calls)
        self.calls.append(SynthesisCall(voice=voice, text=text))

        if self._delay is not None:
            await asyncio.sleep(self._delay(index))

        if self._speech_end is not None:
            seconds = self._speech_end(text)
        else:
            seconds = HEAD_SILENCE_SECONDS + len(text) / CHARS_PER_SECOND
        return Clip(audio=b"ID3FAKE" + text.encode("utf-8"), last_word_end=seconds)

    def called_for(self) -> list[tuple[str, str]]:
        """`(voice, text)` for every request served, as a list a test can compare."""
        return [(call.voice, call.text) for call in self.calls]


def fake_seconds(text: str) -> float:
    """What `DeterministicSpeechEngine` will report for `text`, tail included.

    Exported so a test can assert against the number the service *should* have
    written without re-deriving the arithmetic — the two would drift silently if
    the service and the test each did the sum themselves.
    """
    return HEAD_SILENCE_SECONDS + len(text) / CHARS_PER_SECOND + TAIL_PAUSE_SECONDS
