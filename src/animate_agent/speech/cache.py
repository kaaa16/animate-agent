"""Content-addressed clips: the key, the two files, and the sidecar.

Naming a clip after its *content* rather than after its position is what makes a
second run nearly free. A beat that says the same thing in the same voice is the
same mp3 whether it is beat 3 of this film or beat 11 of a film made last week,
and a lesson that repeats a sentence — 「记住一句话就够了」 — pays for it once.
Rewriting one caption costs one synthesis, not a whole film.

The sidecar is the part that looks redundant and is not. `seconds` cannot be
recovered from the mp3 without decoding it, and it **must not be re-measured**:
the second run would get a word boundary a few milliseconds from the first and
write that into the spec, and a spec that changes between two runs of the same
input is not a contract. So the number is frozen next to the bytes that produced
it, and a hit means the measurement is never taken twice.

The engine version is in the key because the thing being measured is not the
audio but the *boundaries*, and those are a property of the client. edge-tts
changed its default boundary type in 7.2 — silently, so that a caller who did not
ask for `WordBoundary` got no boundaries at all — which is exactly the sort of
change that would leave a cache full of clips whose `seconds` came from the old
semantics. Bumping the version invalidates the lot.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

#: Where clips live under a generated-output directory. The spec refers to these
#: by a path *relative to itself*, so this name is also half of `RenderSpeech.src`.
AUDIO_DIR_NAME = "audio"

#: Hex characters of the digest kept in a filename. Twenty is 80 bits — far more
#: than enough to keep a few thousand clips of one lesson apart, and short enough
#: that the spec stays readable.
_KEY_LENGTH = 20


def clip_key(*, voice: str, rate: str, text: str, engine: str, fmt: str) -> str:
    """The content address of one clip.

    Every input that can change the bytes or the measured length is in here, so
    that two different requests cannot land on one file: the voice and the rate
    change the audio, the text is what is being said, `engine` covers a client
    whose boundary semantics moved, and `fmt` covers the encoding.

    The separator is `\\x1f`, the ASCII unit separator, rather than `|` or `:` —
    both of which can appear in the text of a lesson and would let two different
    requests hash the same string.
    """
    material = "\x1f".join((voice, rate, text, engine, fmt))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:_KEY_LENGTH]


def audio_dir(output_dir: Path, voice: str) -> Path:
    """Where one voice's clips go."""
    return output_dir / AUDIO_DIR_NAME / voice


def clip_paths(output_dir: Path, voice: str, key: str) -> tuple[Path, Path]:
    """Return `(mp3, sidecar)` for one clip."""
    directory = audio_dir(output_dir, voice)
    return directory / f"{key}.mp3", directory / f"{key}.json"


def relative_src(voice: str, key: str) -> str:
    """What goes in `RenderSpeech.src` — relative to the spec, not to the site.

    `data/generated` is mounted twice (`/specs` and `/data`) and `--output-dir`
    puts it somewhere else again, so a spec that named a rooted URL would be
    right in exactly one of those places.
    """
    return f"{AUDIO_DIR_NAME}/{voice}/{key}.mp3"


def read_sidecar(path: Path) -> dict[str, Any] | None:
    """The sidecar's contents, or `None` if it is missing or unreadable.

    A truncated or hand-edited sidecar reads as a miss rather than as an error:
    the worst it costs is one synthesis, and a cache that can take the whole run
    down is a cache nobody leaves on.
    """
    try:
        loaded: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


#: The one field `service` reads back.
SPEECH_END_FIELD = "speech_end"


def write_sidecar(
    path: Path, *, speech_end: float, engine: str, voice: str, rate: str, text: str
) -> None:
    """Freeze the *measurement* next to the bytes that produced it.

    `speech_end` is when the last word stops — **not** the beat's length. The
    pause a beat gets after it speaks is added by `timing.seconds_from_last_word`
    at assembly time, and caching the sum instead would be a determinism bug
    rather than a shortcut: a warm cache would answer with the old pause and a
    cold one with the new, so tuning `TAIL_PAUSE_SECONDS` would change only the
    beats that happened to be synthesized fresh. The measured thing belongs to
    the audio; the pause is a presentation choice laid on top.

    **Stored exactly as given, and given exactly as stored.** Rounding belongs to
    whoever computes the number, not to the file that remembers it — because a
    value that is rounded on the way in and used unrounded on the way out is two
    different beat lengths depending on whether the cache was warm. That is not a
    hypothetical: it is what this function did, at three decimals, until two real
    runs of the same spec came out differing by a hundredth of a second on nine
    of thirty-six beats. `service.measure` quantizes once, and this stores it.

    `text` is here for a person, not for the code — the key already commits to
    it. Opening a directory of forty `a3f9….mp3` and knowing which one is which
    is the difference between debugging this in a minute and debugging it by
    listening to all of them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                SPEECH_END_FIELD: speech_end,
                "engine": engine,
                "voice": voice,
                "rate": rate,
                "chars": len(text),
                "text": text,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
