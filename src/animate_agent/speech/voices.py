"""The voices a lesson may be read in, as a closed set of edge-tts voice ids.

Same shape as `registry.PALETTES`, and for the same reason: the names appear in
more than one place — here, and in `frontend/player/voice.js` where the picker
draws them — so they are a list that has to be *held* together rather than a
literal typed twice. `tests/unit/test_player_voice.py` is what holds them.

`zh-CN-YunjianNeural` leads because it is the default, the way `neon` leads
`PALETTES`: a picker whose first item is not what you are looking at reads as
broken.

**These are not a menu for the model.** Nothing in `knowledge` or `storyboard`
is told these names, and `RenderSpec.voice` is a plain string precisely so that
`rendering` never has to import this module. Which voice a film is generated in
is a decision made by whoever asked for the film — a checkbox on the upload
page, a flag on the command line — and the player offers whichever ones the spec
actually carries.
"""

#: Voice id -> the label a picker shows. Order is the order they are drawn in.
SPEECH_VOICES: tuple[tuple[str, str], ...] = (
    ("zh-CN-YunjianNeural", "云健 · 男声播报"),
    ("zh-CN-XiaoyiNeural", "晓伊 · 女声活泼"),
)

VOICE_IDS: tuple[str, ...] = tuple(voice for voice, _ in SPEECH_VOICES)

#: The voice a film is read in when nobody says otherwise.
DEFAULT_VOICE_ID = VOICE_IDS[0]


def is_voice(name: str) -> bool:
    return name in VOICE_IDS


def label_of(name: str) -> str:
    """The label for `name`, or `name` itself if it is not one we offer.

    Never raises. A spec generated on a machine that had a voice this one does
    not is still a spec worth playing, and the label is only ever shown next to
    a choice — a name with no label is a worse answer than an ugly one.
    """
    for voice, label in SPEECH_VOICES:
        if voice == name:
            return label
    return name
