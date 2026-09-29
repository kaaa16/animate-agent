"""The characters a caption shows, and the characters the voice is given.

A caption and a reading are not the same string, and Chinese is where the two
come apart. 重 is *chóng* in 重铺 and *zhòng* in 重要; which one a reader picks is
a fact about the word, not about the character. A synthesizer is a reader that
cannot be told, and the first film that said 「把整条路重铺一遍」 had the voice say
*zhòng*.

**The obvious fix does not exist.** SSML has `<phoneme>` for exactly this, and
edge-tts will carry a raw fragment to the wire if you bypass the escaping its
`__init__` does — but the endpoint behind it is Edge's read-aloud service, not
Azure Speech, and it takes `<voice>` and `<prosody>` and nothing else. Both
`<phoneme alphabet='sapi' ph='chong 2'>` and `<sub alias='…'>` come back as
`NoAudioReceived` — which `edge-tts` reports as "your parameters are wrong" —
while the identical request with the tags removed returns audio. Measured
2026-09-29; the control that makes this a finding rather than a guess is worth
repeating before trusting it: send the same text *unescaped and untagged*, and
if that still returns audio, the tags are what the service refused.

So the lever is the text, and there are two ways to pull it. Rewriting the
caption would change what a viewer *reads* in order to change what they *hear*.
This module pulls the other one: it rewrites only the string handed to
`service.attach_speech`, so the caption on screen and the `duration` the
storyboard budgeted both keep the original characters.

**A substituted character has to be a homophone with no other reading.** 崇 is
*chóng* and nothing else, so 重铺 → 崇铺 leaves a reader that cannot be told
which reading is meant with nothing to choose between. Nothing here is clever:
it is a pronunciation dictionary, spelled out, one line per word.
"""

from __future__ import annotations

#: Pairs of `(as written, as said)`.
#:
#: Both halves are **whole words**, because a single character has no reading out
#: of context — which is the entire problem being solved. A pair whose left side
#: is one character would silently rewrite every occurrence of that character in
#: every film, including the ones where it was already right.
READINGS: tuple[tuple[str, str], ...] = (
    # 「加密也不是把整条路重铺一遍」 — 重 is chóng («lay it again»), not zhòng
    # («heavy»). 崇 can only be chóng.
    ("重铺", "崇铺"),
)


def pronounced(text: str) -> str:
    """`text` with every listed word replaced by the way it has to be said.

    Applied in table order rather than all at once, so a longer word listed above
    a shorter one it contains wins — the rule a term-rewriting system uses, and
    the reason `READINGS` is a tuple and not a mapping: with no duplicates to
    reject, a `dict` would drop the one property the table actually has.

    Doing nothing is the common case and costs one loop over a short tuple.
    """
    for written, said in READINGS:
        text = text.replace(written, said)
    return text
