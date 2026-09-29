"""How long a beat holds once it has a voice.

`layout.beat_duration` answers the same question from the character count, and
it is not wrong — it is just answering about a different thing. Measured across
the 106 beats of three real films, the voice speaks at **5.87 / 6.44 / 6.41
characters a second**, which is the 6.0 the pipeline already assumes. What the
formula never saw is the silence around the speech: a clip opens with about
0.17s of lead-in and closes with about 0.82s of tail, and at a beat and a half
that is most of a beat's worth of clock spread across every beat in the film.

    a 18-character beat, 云健, +8%
    ├─ 0.10s lead
    ├─ 3.57s speech          <- the 6.0 is about this
    └─ 0.86s tail
    = 4.54s clip             <- layout writes 3.00s

**The lead-in is already accounted for and must not be added again.** Both ends
are measured from t=0 of the clip, so "when the last word ends" already contains
the head; only the tail is a number this module gets to choose.

**These constants belong to Python and only to Python.** `RenderSpeech.seconds`
is what crosses into the spec, with the pause already baked in. A JavaScript
copy would be a second pause stacked on the first — the same audio played under
a clock that waits twice as long. There is a test that greps `voice.js` for a
`TAIL_PAUSE` to keep it that way, because the temptation to add one there is
real and its symptom — everything about half a beat late — looks like a bug in
the audio rather than a bug in a constant.
"""

#: Silence left after the last word, before the player moves on.
#:
#: A quarter second, chosen rather than derived: the clips end with anywhere from
#: 0.6 to 0.9 seconds of near-silence that the encoder pads on, and holding for
#: all of it is dead air between every pair of beats — while holding for none of
#: it runs the sentences together. Anything from about 0.2 to 0.4 sounds like a
#: narrator pausing; this is the middle.
#:
#: The padding itself is never heard: the player stops the clip when the beat is
#: over, so the tail is spent as clock and not as sound.
TAIL_PAUSE_SECONDS = 0.25

#: Lead-in before the first word, as measured on real clips.
#:
#: **Documentation, not arithmetic.** It is inside `last_word_end` already; this
#: constant exists so that the sum is legible, and so that the next person to
#: read `seconds_from_last_word` can see both halves of the silence rather than
#: discovering the missing one by measuring.
HEAD_SILENCE_SECONDS = 0.17


def seconds_from_last_word(last_word_end: float) -> float:
    """How long this beat holds, given when the voice stops speaking.

    The whole of the translation from "a clip" to "a beat": the player counts
    against this, and it is the number that ends up in `RenderSpeech.seconds`.
    """
    return last_word_end + TAIL_PAUSE_SECONDS
