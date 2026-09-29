"""Speech: giving a finished RenderSpec a voice track.

The last stage of the pipeline, and the only one that **leaves the machine**.
Everything upstream — parsing, the Knowledge Agent, the Storyboard Agent, the
layout pass — is a pure function of what came before it, and `rendering` is
careful to say so in as many words (`layout.py`: "Deterministic: same input,
byte-identical output"). Synthesis is not: it is a request to somebody else's
server, and the answer arrives as bytes.

That is why this is a package of its own rather than a function inside
`rendering`. The dependency runs one way — `speech` imports `rendering`, never
the reverse — which is also why the voice list lives here and `RenderStep.speech`
is keyed by a plain `str`. `rendering` must not know which voices exist any more
than it knows which palettes do.

**Off by default.** `SpeechSettings.enabled` is `False`, `attach_speech` returns
its argument untouched when it is, and nothing in `rendering`, `knowledge` or
`storyboard` ever calls in here. The test suite therefore reaches no network,
and that is a property of the wiring rather than of a mock.
"""
