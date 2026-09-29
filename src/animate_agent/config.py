"""Application configuration loaded from YAML."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from animate_agent.speech.voices import DEFAULT_VOICE_ID


class KnowledgeSettings(BaseModel):
    """Tunable parameters for the Knowledge Agent."""

    model_config = ConfigDict(extra="forbid")

    # 12, and since the scene count became *derived* this is a guard rather than
    # a target. The Knowledge Agent is told to work out how many scenes its word
    # budget buys — about 70 characters a scene, which is four beats of a screen
    # each (see `prompts.KNOWLEDGE_SYSTEM_PROMPT`) — so a 90-second film wants
    # eight and this only has to be loose enough not to bind on a long script.
    # It does still bind, and that is the one thing holding the film's length
    # down in script mode: twelve scenes of the longest narration the schema
    # allows is about four minutes.
    max_scenes: int = Field(default=12, ge=1)
    max_retries: int = Field(default=3, ge=1)
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    require_full_coverage: bool = True
    # Costs a second LLM call per generation; see knowledge/fidelity.py.
    verify_fidelity: bool = False
    #: Whether the document was written *for* this film rather than merely found
    #: and animated. Off by default, because until the chat window exists every
    #: caller is feeding in a document somebody else wrote for another purpose:
    #: a slide deck, a PDF, a paragraph about projectile motion.
    #:
    #: On, the Knowledge Agent stops treating the text as raw material — it
    #: keeps the author's words and the author's order, and the film's length
    #: follows the script's. Off, it may reorganise and compress freely to land
    #: on the target length. Either way the finished film is held to the same
    #: one-to-two minutes; this knob decides who gets to choose *what* was said,
    #: not *how much*.
    script_mode: bool = False


class AnimationSettings(BaseModel):
    """Renderer selection, consumed by the storyboard layer and the API mounts."""

    model_config = ConfigDict(extra="forbid")

    default_renderer: str = "canvas_2d"
    allowed_renderers: tuple[str, ...] = ("canvas_2d", "svg_2d", "three_3d")


class StoryboardSettings(BaseModel):
    """Tunable parameters for the Storyboard Agent and its validation pass.

    Read from **two** YAML blocks: the step limits and demo requirements were
    already reserved under `agent:`, and only the model-call knobs are new under
    `storyboard:`. Splitting them keeps the reservation from being restated.
    """

    model_config = ConfigDict(extra="forbid")

    min_storyboard_steps: int = Field(default=3, ge=1, le=7)
    # 5, not 7. The 3..7 shot rule was written when a beat was 3.8~7.0 seconds
    # with a floor nobody could edit below; at four beats a scene that put the
    # shortest possible video at 45 seconds and the longest at 196. A scene now
    # wants to be about 6 seconds — three or four beats of a second and a half —
    # and the ceiling that expresses that is 5.
    max_storyboard_steps: int = Field(default=5, ge=3, le=7)
    # The whole video, in seconds, and how far off it a storyboard may still
    # land — 90 ± 35%, so about one to two minutes.
    #
    # This is the only number in the pipeline that is about the *finished thing*
    # rather than about one layer, and that is the point: every other threshold
    # here is a local rule, and a model can satisfy all of them locally while
    # producing three and a half minutes. `validation._check_total_duration`
    # multiplies it by the band on both sides, and what it refuses is what turns
    # "make a one-to-two-minute video" into something a retry can act on.
    #
    # It was 60 ± 25% until the captions were the thing being fixed: at six
    # characters a second a 350-character script is a minute and no longer, and
    # a caption cut to fit a minute is a label rather than a sentence. The target
    # is not what the film *should* be — it is what the word budget is derived
    # from, so moving it moves how much there is to say.
    #
    # Both ends are enforced, and they are enforced at different layers on
    # purpose: the length is *written* in the Knowledge Agent's word budget and
    # only *measured* here. The ceiling is checked in both places; the floor only
    # in the one that can do something about it — see `StoryboardLimits`.
    target_seconds: float = Field(default=90.0, gt=0)
    target_band: float = Field(default=0.35, gt=0, le=1)
    require_visual_objects: bool = True
    require_interactive_demo: bool = True
    max_retries: int = Field(default=3, ge=1)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


class SpeechSettings(BaseModel):
    """Tunable parameters for the speech stage.

    The only stage that leaves the machine, and the only one that is **off by
    default**. `enabled` is read in exactly one place — the API, to decide what a
    request that says nothing about speech should do — and never inside
    `attach_speech`, whose every caller has already decided.
    """

    model_config = ConfigDict(extra="forbid")

    #: Whether an upload page that does not mention 配音 gets one anyway. Off,
    #: because the upload page is the old road: dropping in a PDF should give the
    #: picture it gave yesterday, and a silent film that suddenly has a voice is
    #: a surprise rather than an improvement.
    enabled: bool = False
    #: Which voice a film is read in when nobody says. The first id in
    #: `speech.voices.SPEECH_VOICES`, imported rather than retyped so that the
    #: picker and this cannot disagree about the default.
    default_voice: str = DEFAULT_VOICE_ID
    #: How fast the narrator reads, in edge-tts' own notation.
    #:
    #: `+8%` and not `+0%`, and it is not a preference — it is what the
    #: pipeline's own `SPEECH_CHARS_PER_SECOND = 6.0` was measured at. That
    #: constant came from the reference project's timeline, "1490 字 = 274.9 秒,
    #: 语速 5.96 字/秒", which is this voice at this rate. At +0% the same words
    #: take about nine percent longer than every length in the pipeline assumes.
    rate: str = "+8%"
    #: How many clips to synthesize at once.
    #:
    #: Measured on this machine: one at a time is 6.9 seconds a clip, because the
    #: bottleneck is the round trip and not the CPU. Eight concurrent is 0.89
    #: seconds, sixteen is 0.47 — so eight takes most of the win and leaves the
    #: endpoint a reasonable number of connections.
    concurrency: int = Field(default=8, ge=1, le=32)


DEFAULT_CONFIG_PATH = Path("config/app.example.yaml")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return {}
    return cast("dict[str, Any]", raw)


def _read_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name, {})
    if not isinstance(section, dict):
        return {}
    return cast("dict[str, Any]", section)


@lru_cache(maxsize=1)
def load_knowledge_settings(path: Path = DEFAULT_CONFIG_PATH) -> KnowledgeSettings:
    """Load the `knowledge` section from the YAML config, falling back to defaults."""
    return KnowledgeSettings.model_validate(_read_section(_read_yaml(path), "knowledge"))


@lru_cache(maxsize=1)
def load_animation_settings(path: Path = DEFAULT_CONFIG_PATH) -> AnimationSettings:
    """Load the `animation` section from the YAML config, falling back to defaults."""
    return AnimationSettings.model_validate(_read_section(_read_yaml(path), "animation"))


@lru_cache(maxsize=1)
def load_storyboard_settings(path: Path = DEFAULT_CONFIG_PATH) -> StoryboardSettings:
    """Load the `agent` + `storyboard` sections, falling back to defaults."""
    raw = _read_yaml(path)
    merged = {**_read_section(raw, "agent"), **_read_section(raw, "storyboard")}
    return StoryboardSettings.model_validate(merged)


@lru_cache(maxsize=1)
def load_speech_settings(path: Path = DEFAULT_CONFIG_PATH) -> SpeechSettings:
    """Load the `speech` section from the YAML config, falling back to defaults."""
    return SpeechSettings.model_validate(_read_section(_read_yaml(path), "speech"))
