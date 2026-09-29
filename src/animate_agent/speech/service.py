"""Attach a voice track to a finished RenderSpec.

The pipeline's last stage, and the only one that **rewrites** an artifact rather
than producing the next one: a spec comes in, the same spec goes out with
`speech` filled in and `duration` moved from the character formula to a measured
number. One artifact, one file, one URL — which is why the API's response shape
does not change and why the player needs no second request to find out whether
there is audio.

**Nothing here decides to run.** There is no `if settings.enabled: return` at the
top, because the only callers are the two places that already asked for speech —
`cli --speak` and the API's form field — and a check inside a function whose
every caller already decided would hide that decision rather than make it. The
"off by default" property lives in the wiring, and it is tested there: with no
`speak` field, `attach_speech` is never called at all.

**Determinism.** Two runs over the same spec must produce byte-identical files,
and three things are load-bearing:

- a cache hit takes `speech_end` from the sidecar and never re-measures, so a
  boundary that lands a millisecond differently cannot reach the spec;
- every duration is rounded to 2dp, so no float tail reaches the JSON;
- the results are collected into a dict keyed by position, built from
  `asyncio.gather`'s return value — which is in submission order — rather than
  from completion order. That last one is the quietest: `gather` is ordered, but
  a callback or `as_completed` would not be, and a spec whose `speech` dicts
  come out in network-latency order is a spec that changes between runs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from animate_agent.config import SpeechSettings, load_speech_settings, load_storyboard_settings
from animate_agent.rendering.layout import TRANSITION_SECONDS
from animate_agent.rendering.models import RenderSpec, RenderSpeech, RenderStep
from animate_agent.speech.cache import (
    SPEECH_END_FIELD,
    clip_key,
    clip_paths,
    read_sidecar,
    relative_src,
    write_sidecar,
)
from animate_agent.speech.engine import EdgeTTSEngine, SpeechEngine
from animate_agent.speech.pronunciation import pronounced
from animate_agent.speech.timing import seconds_from_last_word
from animate_agent.speech.voices import VOICE_IDS, is_voice


@dataclass(frozen=True)
class SpeechReport:
    """What the speech stage did, for a caller to print or ignore."""

    beats: int
    voices: tuple[str, ...]
    synthesised: int
    reused: int
    seconds: float
    #: Set when the voiced film lands outside the storyboard layer's own band.
    #: A message, not a refusal — see `duration_warning`.
    warning: str | None = None


@dataclass
class _Beat:
    """One step, and the cache key of its clip in each chosen voice."""

    step: RenderStep
    text: str
    keys: dict[str, str]


def film_seconds(spec: RenderSpec) -> float:
    """How long the film runs with the durations it currently carries.

    The same sum `storyboard.validation._check_total_duration` makes, over the
    spec rather than the storyboard — beats plus the fades between scenes. It is
    written out here rather than imported from there because the two are
    measuring different objects, and the validator's version also has to answer
    whether the numbers were *legal*.
    """
    beats = sum(step.duration for scene in spec.scenes for step in scene.steps)
    transitions = max(len(spec.scenes) - 1, 0) * 2 * TRANSITION_SECONDS
    return beats + transitions


def duration_warning(seconds: float, *, target_seconds: float, band: float) -> str | None:
    """Say so when the voiced film misses the band the storyboard layer enforced.

    The storyboard's check runs before there is any audio, on the assumption that
    a beat holds `字数 ÷ 6`. It does not: it holds that *plus* the silence at each
    end of the clip, which across a real film is about 0.42 seconds a beat — a
    dozen seconds on a twenty-eight beat storyboard. So the validator can pass a
    film at 118 seconds that plays for 130.

    The honest place to fix that is the word budget, two layers up, which is why
    this warns instead of raising: by the time the number is knowable, sixty
    syntheses have been paid for and the only way to act on it is to start over.
    A message that a person can read is worth more than a refusal they cannot use.
    """
    ceiling = target_seconds * (1 + band)
    floor = target_seconds * (1 - band)
    if floor <= seconds <= ceiling:
        return None
    where = "长" if seconds > ceiling else "短"
    return (
        f"配音之后全片 {seconds:.1f} 秒，比目标 {target_seconds:.0f} 秒{where}出了 "
        f"{floor:.0f}~{ceiling:.0f} 秒的带子。"
        "原因是分镜层量时长时按「字数÷6」算，每拍还另有约 0.42 秒的头尾静音量不到。"
        "要收紧就调 `storyboard.target_seconds`，或者把字幕写短一点。"
    )


async def attach_speech(
    spec: RenderSpec,
    *,
    output_dir: Path,
    voices: Sequence[str] | None = None,
    engine: SpeechEngine | None = None,
    settings: SpeechSettings | None = None,
) -> SpeechReport:
    """Synthesize a track for every beat of `spec` and fill it in.

    **Mutates `spec` in place** and returns a report. In place because the spec
    is the artifact: there is one file, one URL, and a second copy that the
    caller has to remember to write is a second thing that can go missing.

    `voices` defaults to the configured default — one voice. Two is available and
    deliberately not the default: it doubles the synthesis time, which on a
    thirty-six beat film is about half a minute, and the viewer can only listen
    to one at a time.
    """
    settings = settings or load_speech_settings()
    chosen: tuple[str, ...] = tuple(dict.fromkeys(voices)) if voices else (settings.default_voice,)
    for voice in chosen:
        if not is_voice(voice):
            raise ValueError(f"不认识的音色：{voice}。可用的有：{'、'.join(VOICE_IDS)}")
    engine = engine or EdgeTTSEngine()
    primary = chosen[0]

    beats: list[_Beat] = []
    for scene in spec.scenes:
        for step in scene.steps:
            # The *spoken* text, not the caption. `pronunciation` exists because
            # a polyphone cannot be corrected in the SSML this service accepts —
            # see its module docstring. The caption keeps `step.narration`, and
            # so does everything else; only the reading is rewritten, which is
            # why the key below commits to this string and not to the caption.
            text = pronounced(step.narration.strip())
            if not text:
                # A beat with nothing to say keeps whatever `duration` it had and
                # gets no track. Nothing to synthesize is not an error.
                continue
            beats.append(
                _Beat(
                    step=step,
                    text=text,
                    keys={
                        voice: clip_key(
                            voice=voice,
                            rate=settings.rate,
                            text=text,
                            engine=engine.engine_id,
                            fmt=engine.fmt,
                        )
                        for voice in chosen
                    },
                )
            )

    semaphore = asyncio.Semaphore(settings.concurrency)
    counters = {"synthesised": 0, "reused": 0}

    async def measure(index: int, beat: _Beat, voice: str) -> tuple[int, str, float]:
        """When the voice stops on this beat, from cache or from the engine."""
        key = beat.keys[voice]
        mp3, sidecar = clip_paths(output_dir, voice, key)
        cached = read_sidecar(sidecar)
        if cached is not None and mp3.is_file() and SPEECH_END_FIELD in cached:
            counters["reused"] += 1
            return index, voice, float(cached[SPEECH_END_FIELD])

        async with semaphore:
            clip = await engine.synthesize(beat.text, voice, settings.rate)
        # Quantized here, **once**, and this is the number that goes in the
        # sidecar — not a rounded copy of it. Whoever computes a length and
        # whoever remembers it have to be holding the same value, or a warm run
        # and a cold one disagree; the reference project's four-second clip comes
        # back with a word boundary good to ten millionths of a second, and two
        # real runs of this spec differed on nine beats out of thirty-six before
        # the rounding moved here.
        #
        # A millisecond is far finer than anything downstream can see and coarse
        # enough that the file is legible.
        speech_end = round(clip.last_word_end, 3)
        mp3.parent.mkdir(parents=True, exist_ok=True)
        mp3.write_bytes(clip.audio)
        write_sidecar(
            sidecar,
            speech_end=speech_end,
            engine=engine.engine_id,
            voice=voice,
            rate=settings.rate,
            text=beat.text,
        )
        counters["synthesised"] += 1
        return index, voice, speech_end

    gathered = await asyncio.gather(
        *(measure(index, beat, voice) for index, beat in enumerate(beats) for voice in chosen)
    )
    # `gather` returns in the order it was given, whatever order the calls
    # finished in. Keyed by position so that the assembly below reads back in
    # beat order and not in the order the network happened to answer.
    measured = {(index, voice): end for index, voice, end in gathered}

    for index, beat in enumerate(beats):
        track = {
            voice: RenderSpeech(
                src=relative_src(voice, beat.keys[voice]),
                seconds=round(seconds_from_last_word(measured[(index, voice)]), 2),
            )
            for voice in chosen
        }
        beat.step.speech = track
        # The generated voice's length, not the longest of them. A viewer who
        # switches to a slower voice gets that voice's own duration from
        # `speech`, so nothing is ever cut off — and the film does not pay for a
        # difference no one is listening to.
        beat.step.duration = track[primary].seconds
    spec.voice = primary

    storyboard_settings = load_storyboard_settings()
    seconds = film_seconds(spec)
    return SpeechReport(
        beats=len(beats),
        voices=chosen,
        synthesised=counters["synthesised"],
        reused=counters["reused"],
        seconds=seconds,
        warning=duration_warning(
            seconds,
            target_seconds=storyboard_settings.target_seconds,
            band=storyboard_settings.target_band,
        ),
    )
