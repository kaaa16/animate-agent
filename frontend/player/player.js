/**
 * Player: fetch a `RenderSpec`, simulate it, draw it, bind the controls.
 *
 * Everything here consumes the spec as data. There is no `eval`, no
 * `new Function`, and spec text only ever reaches `textContent` or `fillText`
 * (decision D4 — a model-authored string is data, never code).
 *
 * The simulation runs on a **fixed timestep** with an accumulator. The baseline
 * demo advances a clock by a constant per rendered frame
 * (`frontend/demo/app.js:299`: `state.t += 0.016 * speed`), which makes playback
 * speed depend on the display's refresh rate, and reads `Date.now()` in two
 * places besides. "Same spec, same frames" is the property the whole pipeline is
 * built to keep, and a wall clock is what destroys it.
 */

import { createVoiceTrack } from "./audio.js";
import { createSimulation } from "./behaviors.js";
import {
  CAPTION_FONT_SIZE,
  CAPTION_LINE_HEIGHT,
  CAPTION_PAD_X,
  captionBand,
  captionTop,
  wrapCaption,
} from "./caption.js";
import { EASE_SECONDS, blend, easeOutCubic, handsOver } from "./easing.js";
import { createRecorder, extensionFor, recordableType, saveBlob } from "./recorder.js";
import { LIVE_PROPS, PRESENCE_PROP, drawElement } from "./registry.js";
import { createStage, readTheme } from "./stage.js";
import { STORAGE_KEY, THEMES, resolveTheme } from "./themes.js";
import {
  STORAGE_KEY as VOICE_STORAGE_KEY,
  beatSeconds,
  clipUrl,
  resolveVoice,
  voiceIdsIn,
  voiceLabel,
} from "./voice.js";

const FIXED_STEP = 1 / 60;
/** Beyond this, drop the backlog rather than replaying a long stall frame by frame. */
const MAX_CATCHUP_STEPS = 5;
/**
 * Each half of a scene change: fade to the background, swap, fade back in.
 *
 * A *dip*, not a cross-fade. A cross-fade needs both scenes on screen at once,
 * which means a second canvas or an offscreen buffer, and it reads as a
 * dissolve — two pictures briefly superimposed. A dip goes through a moment of
 * empty stage instead, which is honest about what happened (the old picture is
 * gone, a new one is starting) and needs nothing but one `fillRect` over what
 * was already drawn.
 */
const TRANSITION_SECONDS = 0.45;

/**
 * The backdrop: a field of dots behind everything, in place of a ruled grid.
 *
 * The stage was covered in 40-unit graph paper — lines in `--line` at
 * `GRID_ALPHA`, which had been a literal cyan `rgba(83, 246, 255, 0.07)` before
 * it learned to ask the palette. Both are the 示意图 register: graph paper says
 * *measurement*, and the reference this project is chasing is an illustration.
 * The change is meant for the sparse scenes, where the half of the frame with
 * nothing in it read as unfinished rather than as margin.
 *
 * Same spacing and the same colour, dots instead of lines — the swap
 * anything2explainer ships as their default backdrop (`common/DotFieldBg.tsx`),
 * and their rule for it is one sentence: 背景只有幕底. Nothing else belongs back
 * here. Their other result is the one worth writing down: **a backdrop does not
 * make a small subject bigger**, so this changes how the empty parts read and is
 * not a fix for a scene whose largest object is 48px tall.
 *
 * Static. There is no clock here that outlives a beat — `state.sim` is reset by
 * every `setStep` — so a drifting field would jump at each beat, and "paused"
 * has to mean the picture stops moving.
 */
const BACKDROP_STEP = 40;
const BACKDROP_ORIGIN = 20;
const BACKDROP_RADIUS = 1.5;
const BACKDROP_ALPHA = 0.30;

/**
 * Every element this file drives, by id.
 *
 * **The ids here and the ids in `index.html` are one list cut in two**, and
 * nothing at runtime notices when they stop matching: a missing one is `null`,
 * the first write to it throws, and the page says so on the error plate (or, if
 * the write happens inside `setStep`, which runs inside the frame loop, says
 * nothing at all and draws nothing for ever). `tests/unit/test_player_transport.py`
 * is what holds the two lists together, because the upload page has had that
 * test for a while and this page never did.
 *
 * `player` is the element fullscreen is asked for — the stage and the transport
 * bar, deliberately not the whole document.
 */
const ui = {
  canvas: document.getElementById("stage"),
  title: document.getElementById("title"),
  controls: document.getElementById("controls"),
  themes: document.getElementById("themes"),
  voicePicker: document.getElementById("voice-picker"),
  voices: document.getElementById("voices"),
  voiceHint: document.getElementById("voice-hint"),
  player: document.getElementById("player"),
  error: document.getElementById("error"),
  play: document.getElementById("play"),
  clock: document.getElementById("clock"),
  seek: document.getElementById("seek"),
  gain: document.getElementById("gain"),
  mute: document.getElementById("mute"),
  fullscreen: document.getElementById("fullscreen"),
  download: document.getElementById("download"),
  recordNote: document.getElementById("record-note"),
};

const state = {
  spec: null,
  scene: null,
  //: Which scene is loaded. `advanceBeat` needs it to know whether the beat it
  //: just finished was the last beat of a scene or of the whole storyboard.
  sceneIndex: 0,
  //: `{ toScene, toStep, phase, t }` while a scene change is on screen, else
  //: `null`. Phase is `"out"` (old scene fading) or `"in"` (new scene rising).
  transition: null,
  //: `"<element>.<prop>"` -> what it read *before this beat*. Refilled on every
  //: beat change: from the previous beat when there is one (`captureDeparting`),
  //: from "everything is absent" when a scene opens (`seedDeparting`), and left
  //: empty when there is nothing to ease from at all — a beat reloaded, or any
  //: change while paused, where a ramp would have nothing to drive it.
  departing: new Map(),
  sim: null,
  viewport: null,
  theme: null,
  byId: new Map(),
  obstacleIds: [],
  obstacleRadius: new Map(),
  overrides: {},
  lookup: null,
  stepIndex: 0,
  //: Overwritten by `main`, which knows whether the link asked for a film that
  //: starts on its own. A player opens paused; the exceptions are argued there.
  playing: true,
  accumulator: 0,
  last: 0,
  stopped: null,
  //: The film as one strip of time, rebuilt whenever the voice changes.
  //: `beatSeconds` prefers each beat's *measured* clip, and the two narrators do
  //: not read at the same speed — so `total` is a different number in each, and
  //: a seek bar holding the other one's table would be a bar that lies.
  //:
  //: `sceneStart[i]` and `sceneEnd[i]` bound scene `i`'s beats, `beatStart[i][k]`
  //: is where its beat `k` begins. The fade after a scene sits in the gap
  //: between `sceneEnd[i]` and `sceneStart[i+1]`, so the gap *is* the dip and
  //: nothing has to store it twice. Nothing stores a playhead either — see
  //: `playhead()` for the argument.
  total: 0,
  sceneStart: [],
  sceneEnd: [],
  beatStart: [],
  //: What the viewer asked the volume to be, and whether they asked for silence.
  //: Held here rather than read back off the element, because the element stops
  //: knowing about either the moment the recording graph exists.
  muted: false,
  //: Whether a download is being recorded right now. Locks the two controls that
  //: would contradict it — see `startRecording`.
  recording: false,
  //: The voice tracks this spec carries, in picker order — `[]` for a spec with
  //: no speech, which is every spec written before there was any.
  voiceIds: [],
  //: Which of them is playing, or `null` when there is nothing to play.
  voice: null,
  //: Where the spec was fetched from. A beat's `speech[voice].src` is relative
  //: *to the spec*, so this is the base every clip URL is built from.
  specUrl: "",
  track: null,
};

/**
 * Whether `setStep`'s own `playBeat` is being held back by a seek.
 *
 * A module-level flag rather than a parameter on `setStep`, and that is not a
 * style choice: `test_a_beat_boundary_is_eased_and_not_stepped` pins that
 * function's signature to the literal `setStep(index)`, and `test_the_last_beat_stops`
 * pins its `duration > 0` guard by name. The seam had to go somewhere the tests
 * do not name, and "this one call is part of a seek" is a fact about the seek.
 */
let suppressingBeatAudio = false;

/* ------------------------------------------------------------------ lookup */

/**
 * Resolve a property, most specific source first.
 *
 * 1. a control the user moved — user intent outranks everything
 * 2. the current step's `object_states` — this is what makes a step a beat
 * 3. the element's own `props`, carried verbatim from the StoryboardIR
 * 4. the scene's `params`, reached by `scene.<prop>` controls
 *
 * `safe_distance` is the case that exercises tier 4: the baseline exposes it as
 * a scene-level slider, not as a property of the circle drawn for it.
 */
function makeLookup(scene) {
  const raw = (elementId, prop, fallback) => {
    const key = `${elementId}.${prop}`;
    if (key in state.overrides) return state.overrides[key];

    const step = scene.steps[state.stepIndex];
    const fromStep = step?.states?.[elementId];
    if (fromStep && prop in fromStep) return fromStep[prop];

    const element = state.byId.get(elementId);
    // A binding names the control that drives this property, and it outranks
    // the element's own props — a static prop is exactly what a binding exists
    // to override. Without this tier the safe-distance circle kept the radius
    // layout baked in while the slider moved the danger threshold, so the
    // picture changed but the circle named 安全距离 did not: the label and the
    // thing disagreed, and only a person would notice.
    const bound = element?.binds?.[prop];
    if (bound && bound in state.overrides) return state.overrides[bound];

    if (element?.props && prop in element.props) return element.props[prop];

    const sceneKey = `scene.${prop}`;
    if (sceneKey in state.overrides) return state.overrides[sceneKey];
    if (prop in (scene.params ?? {})) return scene.params[prop];

    return fallback;
  };

  // The whole of the easing: everything above still decides *what* a property
  // is worth this beat, and this decides only how it gets there. Answering from
  // `raw` and blending is what keeps the two questions apart — a beat that sets
  // nothing leaves `from` undefined and is returned untouched, so a property a
  // control is driving never passes through here at all.
  return (elementId, prop, fallback) => {
    const target = raw(elementId, prop, fallback);
    const from = state.departing.get(`${elementId}.${prop}`);
    if (from === undefined) return target;
    const progress = easeProgress();
    if (progress >= 1) return target;
    return blend(prop, from, target, progress);
  };
}

/** How far into the ease a beat is, shaped. `easing.js` owns the arithmetic. */
function easeProgress() {
  return easeOutCubic(Math.min(1, state.sim.t / EASE_SECONDS));
}

/**
 * What every live property reads as right now, taken *before* the beat moves —
 * `state.lookup` resolves against `state.stepIndex`, so the moment it changes
 * the old values are unrecoverable.
 *
 * Only the properties a beat is allowed to change are captured, because those
 * are the only ones that can differ between two beats. Read off `LIVE_PROPS`
 * rather than listed again here: it is the same table the validator and the
 * prompt are built from, and a second copy would be a second thing to drift.
 *
 * **This and `seedDeparting` are the two halves of one question.** This half is
 * a beat handed over from another beat: it eases from whatever the last beat
 * left on screen. The other half is a scene's *first* beat, which has no
 * previous beat to read — `loadScene` calls `seedDeparting` for it, and that
 * function carries the argument for why the gate is the only thing seeded.
 *
 * Until that landed, the first beat of every scene was a hard cut and the
 * vocabulary was quietly wrong about it: 「想让它讲到才出现」 worked from beat 2
 * onward, and a scene whose opening object was meant to arrive had to spend its
 * first beat on something else. That was a requirement on the storyboard that
 * no layer checked. It is recorded, with the reasoning that got it fixed, in
 * `docs/storyboard-milestone.md`.
 */
function captureDeparting() {
  const from = new Map();
  for (const element of state.scene.elements) {
    for (const prop of LIVE_PROPS[element.kind] ?? []) {
      from.set(`${element.id}.${prop}`, state.lookup(element.id, prop, undefined));
    }
  }
  return from;
}

/**
 * How strongly an element should be drawn, 0 to 1.
 *
 * Asked of the lookup rather than of `state.departing` directly, so that the
 * number here and the value the drawer's own gate reads are the same number —
 * one eased to a fraction while it is on its way in, and exactly `false` (or
 * exactly `true`) once it has landed.
 */
function presence(elementId) {
  const prop = PRESENCE_PROP[state.byId.get(elementId)?.kind];
  if (prop === undefined) return 1;
  const value = state.lookup(elementId, prop, true);
  if (typeof value === "number") return Math.max(0, Math.min(1, value));
  return value === false ? 0 : 1;
}

/* -------------------------------------------------------------------- view */

function buildView() {
  const step = state.scene.steps[state.stepIndex];
  return {
    live: state.sim.live,
    time: state.sim.t,
    theme: state.theme,
    // Handed to the drawers so one of them can ask where the caption starts —
    // `drawBody` holds a body's name out of it. It is read off the spec rather
    // than passed down, because `render` and this run at different times and the
    // only thing either wants the stage for is its size.
    stage: state.spec.stage,
    lookup: state.lookup,
    elementById: state.byId,
    obstacleIds: state.obstacleIds,
    obstacleRadius: state.obstacleRadius,
    // Glyph geometry travels in the spec (`RenderSpec.glyphs`), so the player
    // resolves a `body.glyph` name against data it already holds rather than a
    // file it would have to go and find.
    glyphs: state.spec.glyphs ?? {},
    highlighted: new Set(step?.highlights ?? []),
    isDangerous: (id) => state.sim.isDangerous(id),
    // Handed to `drawElement` rather than to the drawers: it is the one place
    // every element is drawn through, so the alpha is applied once instead of
    // in each drawer — five chances to forget it, and no way to tell from the
    // code which of the five dropped it.
    presence,
  };
}

/* ----------------------------------------------------------------- drawing */

/**
 * The dot field, drawn under everything else.
 *
 * `fillRect` per dot rather than `arc`: four hundred arcs a frame is four
 * hundred path constructions, and at a radius of 1.5 the square and the circle
 * land on the same pixels. Read the constants above for why there is a field
 * here at all, and why it does not move.
 */
function drawBackdrop(ctx, theme, width, height) {
  const size = BACKDROP_RADIUS * 2;
  ctx.save();
  ctx.fillStyle = theme.line;
  ctx.globalAlpha *= BACKDROP_ALPHA;
  for (let y = BACKDROP_ORIGIN; y < height; y += BACKDROP_STEP) {
    for (let x = BACKDROP_ORIGIN; x < width; x += BACKDROP_STEP) {
      ctx.fillRect(x - BACKDROP_RADIUS, y - BACKDROP_RADIUS, size, size);
    }
  }
  ctx.restore();
}

/**
 * The lane line: the one piece of chrome that is *semantic* rather than texture.
 *
 * It used to share this function with the graph-paper grid, in two `save`/
 * `restore` pairs because the grid ran at a fraction of the palette's opacity and
 * the line does not. The grid is gone — see `BACKDROP_STEP` — so what is left is
 * the road under a `lane` scene's car, drawn exactly as it always was.
 */
function drawChrome(ctx, scene, theme, width, height) {
  const lane = scene.elements.find((element) => element.role === "vehicle");
  const laneY = lane ? lane.y : height / 2;

  ctx.save();
  ctx.setLineDash([10, 14]);
  ctx.strokeStyle = theme.line;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, laneY);
  ctx.lineTo(width, laneY);
  ctx.stroke();
  ctx.restore();
}

/**
 * The words at the bottom: this beat's narration, in the strip layout kept clear.
 *
 * On the canvas rather than in a DOM element under it, and that is the whole
 * reason `caption.js` and `layout.py` have to agree on `CAPTION_BAND_RATIO`: the
 * reserved strip is a piece of the stage's coordinates, so what is drawn in it
 * has to be too. A DOM overlay would be laid out in CSS pixels against a canvas
 * that is *scaled to fit*, and the two would part company the first time the
 * window changed size — invisibly, and only at some window sizes.
 *
 * The plate is the theme's own panel colour, so a caption over the background
 * grid reads as a card rather than as text on graph paper. Same problem
 * `drawReadout` solves the same way.
 */
function drawCaption(ctx, step, theme, stage) {
  const text = (step?.narration ?? "").trim();
  if (text.length === 0) return;

  const top = captionTop(stage);
  const height = captionBand(stage);

  ctx.save();
  ctx.font = `${CAPTION_FONT_SIZE}px Inter, 'Microsoft YaHei', sans-serif`;
  // `measureText`, not a width table: the font is the renderer's, and a
  // half-width Latin table would break a Chinese caption in the wrong place.
  const lines = wrapCaption(
    text,
    (line) => ctx.measureText(line).width,
    stage.width - 2 * CAPTION_PAD_X,
  );

  ctx.fillStyle = theme.panel;
  ctx.fillRect(0, top, stage.width, height);

  ctx.fillStyle = theme.text;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  // Centred as a block, so a one-line beat and a two-line beat share a middle
  // instead of one sitting on the floor of the band and the other floating.
  const block = lines.length * CAPTION_LINE_HEIGHT;
  const first = top + (height - block) / 2 + CAPTION_LINE_HEIGHT / 2;
  lines.forEach((line, index) => {
    ctx.fillText(line, stage.width / 2, first + index * CAPTION_LINE_HEIGHT);
  });
  ctx.restore();
}

/**
 * How much of the picture is covered by the scene change, 0 (none) to 1 (all).
 *
 * Read every frame rather than stored, because it is a pure function of the
 * transition clock: one clock, one answer, no chance of the two disagreeing.
 */
function fadeAmount() {
  const transition = state.transition;
  if (transition === null) return 0;
  const progress = Math.min(1, transition.t / TRANSITION_SECONDS);
  return transition.phase === "out" ? progress : 1 - progress;
}

function render() {
  const stage = state.spec.stage;
  const ctx = state.viewport.ctx;
  state.viewport.clear();
  // The backdrop goes under everything, the lane line over it: the line is a
  // road a car is on, and a car is not behind a texture.
  drawBackdrop(ctx, state.theme, stage.width, stage.height);
  drawChrome(ctx, state.scene, state.theme, stage.width, stage.height);

  const view = buildView();
  for (const element of state.scene.elements) {
    try {
      drawElement(ctx, element, view);
    } catch (error) {
      // A blank canvas and a half-drawn one look the same to a person. Say what
      // broke, on the page, and stop rather than throwing once per frame.
      fail(error instanceof Error ? error.message : String(error));
      return;
    }
  }

  // After the elements and before the fade: the words are part of *this*
  // scene, so a scene change has to take them with it. Drawn last, a caption
  // would sit on top of the fade and survive into the next scene, reading as
  // narration for a picture that is already gone.
  drawCaption(ctx, state.scene.steps[state.stepIndex], state.theme, stage);

  // Last, so it covers the chrome and the elements alike: a scene change is
  // about the whole stage, not about the pictures inside it.
  const fade = fadeAmount();
  if (fade > 0) {
    ctx.save();
    ctx.globalAlpha = fade;
    ctx.fillStyle = state.theme.background;
    ctx.fillRect(0, 0, stage.width, stage.height);
    ctx.restore();
  }
}

function fail(message) {
  state.stopped = message;
  state.playing = false;
  ui.error.textContent = message;
  ui.error.hidden = false;
}

/* --------------------------------------------------------------------- loop */

function tick(now) {
  requestAnimationFrame(tick);
  if (state.stopped !== null) return;

  const elapsed = state.last === 0 ? 0 : Math.min((now - state.last) / 1000, 0.25);
  state.last = now;

  if (state.playing) {
    state.accumulator += elapsed;
    let steps = 0;
    while (state.accumulator >= FIXED_STEP && steps < MAX_CATCHUP_STEPS) {
      stepOnce();
      state.accumulator -= FIXED_STEP;
      steps += 1;
    }
    if (steps === MAX_CATCHUP_STEPS) state.accumulator = 0;
    // The beat's clock and its animation are the same clock: `update` advances
    // `sim.t` and `setStep` puts it back to zero, so pausing pauses both, and a
    // beat that is on screen is a beat that is running.
    //
    // At most one beat advances here. The loop above moves `sim.t` by at most
    // `MAX_CATCHUP_STEPS` fixed steps, and the shortest beat a layout can write
    // is several seconds, so the two are nowhere near touching.
    advanceBeat();
  }

  // After `advanceBeat`, so that a beat boundary and the bar agree within the
  // frame it happened on, and before `render`, so what is drawn is what the bar
  // is measuring.
  syncTransport();

  render();
}

/**
 * One fixed step of everything that moves: the simulation, and the scene change
 * if one is running.
 *
 * The simulation keeps stepping through a fade rather than freezing. A scene
 * caught mid-stride — a car three quarters of the way down its lane — should
 * carry on out of the picture, not stop dead and then dissolve; a freeze frame
 * is what a person reads as a stall.
 */
function stepOnce() {
  state.sim.update(FIXED_STEP, state.lookup);

  const transition = state.transition;
  if (transition === null) return;
  transition.t += FIXED_STEP;
  if (transition.t < TRANSITION_SECONDS) return;

  if (transition.phase === "out") {
    // The stage is fully covered. Swap underneath it and start uncovering.
    loadScene(transition.toScene);
    if (transition.toStep > 0) setStep(transition.toStep);
    state.transition = {
      toScene: transition.toScene,
      toStep: transition.toStep,
      phase: "in",
      t: 0,
    };
    return;
  }
  state.transition = null;
}

/* --------------------------------------------------------------------- step */

/**
 * Move on once the current beat has held for as long as it was given.
 *
 * Until this existed nothing computed a beat's length at all. The layout pass
 * bakes a duration onto a *body* (`element.duration`, for a thrown one) but
 * wrote none onto a beat, so the player had nothing to count against and did
 * the only thing left: it sat on whatever beat was loaded. Beats 2..N were
 * reachable only by clicking. A scene could be written, validated, rendered and
 * reviewed end to end without anyone seeing its second frame.
 *
 * Three rules, and the middle one is the change:
 *
 * - A `duration` of `0` means the spec predates the field. Then this returns
 *   immediately and the player behaves exactly as it did when that spec was
 *   written — an old file keeps its old behaviour instead of silently
 *   acquiring a rhythm nobody chose for it.
 * - A scene's last beat hands over to the next **scene**, through the
 *   transition, rather than stopping there. Stopping was right when a scene was
 *   the whole story: the player took one scene and the URL said which. But a
 *   storyboard is a sequence of scenes, the upload page never passed `?scene=`,
 *   and so every run of the real chain was watched one scene deep — which is
 *   how a document whose second scene is the good one got judged on its first.
 *   The beat clock is what knows the storyboard is longer than a scene; this is
 *   the only place that knows it.
 * - The end of the **storyboard** is still the end. It does **not** wrap to beat
 *   1. A scene that quietly restarts is a scene whose last beat is never seen
 *   to finish, which is the same defect as the one above wearing the opposite
 *   coat. The last beat simply holds, still animating, and it is the transport
 *   bar that tells the viewer the film is over: it reads the same held beat
 *   against `state.total`, stops the clock there and turns the play button into
 *   重播. That is deliberately **not** done here — a beat clock that stopped
 *   itself would be this same edit wearing the first coat all over again.
 */
function advanceBeat() {
  // Still called `duration`, though what it holds is now the measured length of
  // a clip rather than a division. `test_the_last_beat_stops` greps this body for
  // `duration > 0`, and renaming the local would fail it with a message about the
  // guard rather than about the name.
  const step = state.scene.steps[state.stepIndex];
  const duration = beatSeconds(step, state.voice);
  if (!(duration > 0)) return;
  if (state.sim.t < duration) return;
  // A scene change already owns the clock. Without this the beat that was
  // playing when the fade started would hand over a second time underneath it.
  if (state.transition !== null) return;

  if (state.stepIndex >= state.scene.steps.length - 1) {
    if (state.sceneIndex >= state.spec.scenes.length - 1) return;
    beginTransition(state.sceneIndex + 1);
    return;
  }
  setStep(state.stepIndex + 1);
}

/**
 * Start a scene change. `toStep` is where the new scene opens — 0 going
 * forwards, and the previous scene's last beat going back, so that ◀ from the
 * first beat of a scene lands on the beat a person was just watching.
 */
function beginTransition(toScene, toStep = 0) {
  // The fade owns the stage from here. A beat's clock normally runs out before
  // the dip begins, so there is usually nothing left to stop — but ◀ into the
  // previous scene's last beat arrives mid-word by design, and a sentence
  // carrying on under a fade that is covering its own picture is the one thing
  // a scene change must not sound like.
  if (state.track !== null) state.track.stop();
  state.transition = { toScene, toStep, phase: "out", t: 0 };
}

/**
 * Move `delta` beats, crossing a scene boundary when that is where `delta`
 * leads.
 *
 * This was the ◀ ▶ transport's handler and is now ←/→'s. The buttons went out
 * with the rest of the inspector — the seek bar is the coarse control now — but
 * the keyboard kept it, and the reason is that this film is a sequence of
 * *narrated beats* rather than of seconds. A video player spends its arrow keys
 * on ±5 seconds because a video has no structure between them to skip; here
 * "the next sentence" is what a person reaches for, and it is the only way to
 * cross a scene boundary without dragging.
 */
function stepBy(delta) {
  // One hand-over at a time. A second press mid-fade would restart the fade-out
  // from `t = 0`, which snaps the picture back to visible for no reason anyone
  // asked for.
  if (state.transition !== null) return;
  const index = state.stepIndex + delta;
  if (index >= 0 && index < state.scene.steps.length) {
    setStep(index);
    return;
  }
  const scene = state.sceneIndex + (delta > 0 ? 1 : -1);
  if (scene < 0 || scene >= state.spec.scenes.length) return;
  beginTransition(scene, delta > 0 ? 0 : (state.spec.scenes[scene].steps.length - 1));
}

function setStep(index) {
  const total = state.scene.steps.length;
  const clamped = Math.max(0, Math.min(total - 1, index));
  // Captured while `state.stepIndex` still names the beat being left. Only a
  // change that *has* something to hand over from captures anything — the same
  // beat reloaded (重置, a `reset_scene` button), and any change at all while
  // paused, both land instead of easing. `easing.js:handsOver` argues both, and
  // the paused half is the one that was missing: a camera paused to inspect
  // beat 3 was shown beat 1, because nothing was left to advance the ease.
  state.departing = handsOver(state.playing, clamped === state.stepIndex)
    ? captureDeparting()
    : new Map();
  state.stepIndex = clamped;
  // Restart the beat from rest: a step is a moment, and a beat that inherits the
  // previous one's elapsed time can never be looked at twice. It is also what
  // starts the ease above at zero.
  state.sim.reset();
  state.accumulator = 0;
  playBeat();
}

/**
 * Sound the beat that is now on screen, from its beginning.
 *
 * Called from `setStep` and from nowhere else, which is what keeps the audio
 * tied to the one event that means "the picture moved on". Not from `tick` or
 * `render` — those run sixty times a second, and an `Audio` element rebuilt on
 * every frame is a stutter rather than a voice.
 *
 * Paused, this stops instead of starting: the viewer pressed 暂停 and the
 * silence is part of what that means. Resuming goes through `togglePlay`, which
 * knows where the picture got to.
 */
function playBeat() {
  // Set for the length of one `seekTo`, which wants the beat to be *loaded* but
  // not *spoken*. Without it a seek asks for the same clip twice — once from the
  // top by this function, once from the offset by the seek — and, when the film
  // is paused, the `stop` below runs and then the seek starts the sound again,
  // giving a moving voice over a stopped picture.
  if (suppressingBeatAudio) return;
  if (state.track === null) return;
  const url = clipUrl(state.scene.steps[state.stepIndex], state.voice, state.specUrl);
  if (url === "" || !state.playing) {
    // An unvoiced beat, or a spec with no speech at all. `stop` on an element
    // that never had a clip is a no-op, so a silent film pays nothing for this.
    state.track.stop();
    return;
  }
  state.track.play(url, 0);
}

/* ------------------------------------------------------------------ timeline */

/**
 * Lay every beat of every scene end to end, and remember where each one starts.
 *
 * The film has always been one continuous thing — `advanceBeat` hands a scene
 * over to the next one on its own — but nothing ever *measured* it. A seek bar
 * is that measurement, and it is the only reason this exists.
 *
 * Two decisions in here:
 *
 * - **A scene change costs `TRANSITION_SECONDS`, not twice that.** The dip has
 *   two halves and only the first one is extra: `stepOnce` loads the next scene
 *   and starts its first beat *inside* the dip, so the beat's own clock is
 *   already running while the picture fades back in. Charging for both halves
 *   would put every scene boundary 0.45 seconds further along than it is, and by
 *   nine scenes the bar would be four seconds out. (`storyboard/validation.py`
 *   charges for both; that number was told to the model rather than to the
 *   viewer, and this is the one the film actually keeps.)
 * - **The last scene has no dip after it.** The film ends on its final beat, not
 *   on a fade to nothing, so adding one would leave a tail of dead time that can
 *   never be played.
 */
function rebuildTimeline() {
  const sceneStart = [];
  const sceneEnd = [];
  const beatStart = [];
  let at = 0;

  state.spec.scenes.forEach((scene, index) => {
    sceneStart.push(at);
    const starts = [];
    for (const step of scene.steps) {
      starts.push(at);
      at += beatSeconds(step, state.voice);
    }
    beatStart.push(starts);
    sceneEnd.push(at);
    if (index < state.spec.scenes.length - 1) at += TRANSITION_SECONDS;
  });

  state.sceneStart = sceneStart;
  state.sceneEnd = sceneEnd;
  state.beatStart = beatStart;
  state.total = at;
}

/**
 * Where the film is, in seconds from its start.
 *
 * Derived on every read rather than kept in a variable, and that is the whole
 * design. The console's own buttons can move the film — `reset_scene` reloads a
 * beat, `advance_timeline` steps one, `toggle_play` stops the clock — and a
 * stored playhead would have to be kept in step with all three of them and with
 * ←/→ and with the seek bar itself. There is one clock, `sim.t`, and this is a
 * function of it.
 *
 * **A fade reports the fade**, whichever way it was started. The obvious version
 * of this reads `sim.t`, which during a dip belongs to the scene being covered
 * and is running out its last beat — so the bar would sail past the boundary
 * while the picture was still fading. `stepBy` crosses a boundary the same way,
 * so the test is `state.transition`, not "did we arrive here on the beat".
 */
function playhead() {
  if (state.transition !== null) {
    const destination = positionOf(state.transition.toScene, state.transition.toStep);
    // Forwards: the fade sits after the scene on screen, so it reads as the last
    // 0.45 seconds before the destination. Backwards (`stepBy(-1)` off the first
    // beat): the destination is *behind* the playhead, and holding there for the
    // length of the fade is closer to what is happening than leaping a whole
    // scene forward and back would be.
    return Math.min(destination, state.sceneEnd[state.sceneIndex]) + state.transition.t;
  }
  return positionOf(state.sceneIndex, state.stepIndex) + state.sim.t;
}

/** The second a given beat begins at. Safe for a beat index that does not exist. */
function positionOf(sceneIndex, stepIndex) {
  const starts = state.beatStart[sceneIndex] ?? [];
  return starts[stepIndex] ?? state.sceneStart[sceneIndex] ?? 0;
}

/** `1:33`. Minutes do not roll over into hours; nothing here is that long. */
function formatClock(seconds) {
  const whole = Math.max(0, Math.floor(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

/**
 * Put the film at `seconds`, and make the picture and the voice both agree.
 *
 * Everything the transport bar does funnels through here, which is why the
 * awkward parts are all in one place:
 *
 * - **The fade is cancelled first.** One already running owns the swap: it is
 *   counting down in `stepOnce` and will call `loadScene` on a scene nobody asked
 *   for, 0.45 seconds later, which arrives looking like a second and unrelated
 *   bug.
 * - **A landing inside a dip is rounded forward** to the next scene's first beat.
 *   A fade is a way into something, so the thing being asked for is what comes
 *   out of it; the alternative — replaying half a fade — is both harder and a
 *   worse guess about what a person meant. The cost is at most 0.45 seconds.
 * - **The beat's audio is suppressed and then asked for from the offset**, so
 *   the clip is heard once, from the right place, and below a paused picture
 *   nothing is heard at all. See `suppressingBeatAudio`.
 * - **The simulation is replayed, not fast-forwarded by assignment.** Most of
 *   what a drawer reads is a pure function of the clock, but `behaviors.js`
 *   keeps a `dodge` map across updates — a body steering around an obstacle
 *   arrives at time `t` with a history, and setting the clock and drawing puts it
 *   back on its start line with four seconds on the clock. The loop is bounded
 *   by one beat, because `sim.reset()` zeroes the clock at every beat: the
 *   longest beat in `data/generated` is ten seconds, which is 600 steps over a
 *   handful of elements, and it is over before the next frame.
 */
function seekTo(seconds) {
  if (!(state.total > 0)) return;
  const wanted = Math.max(0, Math.min(state.total, seconds));

  state.transition = null;

  let scene = state.spec.scenes.length - 1;
  for (let index = 0; index < state.spec.scenes.length; index += 1) {
    if (wanted < state.sceneEnd[index]) {
      scene = index;
      break;
    }
  }

  const starts = state.beatStart[scene] ?? [];
  let step = 0;
  for (let index = 0; index < starts.length; index += 1) {
    if (wanted >= starts[index]) step = index;
  }
  const offset = Math.max(0, wanted - (starts[step] ?? 0));

  // Set around both of them, not just around the `setStep` at the end.
  // `loadScene` reaches `setStep(0)` on its own way through, and a cross-scene
  // seek would otherwise open the new scene's *first* beat out loud for the
  // instant before the line below points the element at the beat actually asked
  // for — a syllable of the wrong sentence, which is the kind of thing a person
  // hears and cannot place.
  suppressingBeatAudio = true;
  if (scene !== state.sceneIndex) loadScene(scene);
  setStep(step);
  suppressingBeatAudio = false;

  // Cleared *before* the replay. `state.lookup` blends from `departing` for the
  // first `EASE_SECONDS` of a beat, and after a seek those are the values of a
  // picture the viewer has just left — two unrelated numbers to interpolate
  // between. Clearing it after the loop would leave the replayed frames wrong
  // and only the ones drawn afterwards right.
  state.departing = new Map();
  state.accumulator = 0;
  state.last = 0;

  const steps = Math.round(offset / FIXED_STEP);
  for (let index = 0; index < steps; index += 1) state.sim.update(FIXED_STEP, state.lookup);

  if (state.playing && state.track !== null) {
    state.track.play(
      clipUrl(state.scene.steps[state.stepIndex], state.voice, state.specUrl),
      offset,
    );
  }
  syncTransport();
}

/* ----------------------------------------------------------------- controls */

function renderControls() {
  const scene = state.scene;
  ui.controls.replaceChildren();

  for (const control of scene.controls) {
    if (control.type === "button") {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "control-button";
      button.textContent = control.label || control.action || control.id;
      button.addEventListener("click", () => {
        if (control.action === "reset_scene") setStep(state.stepIndex);
        else if (control.action === "toggle_play") togglePlay();
        else if (control.action === "advance_timeline") setStep(state.stepIndex + 1);
      });
      ui.controls.append(button);
      continue;
    }

    if (control.type === "toggle") {
      const label = document.createElement("label");
      label.className = "control";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = Boolean(control.default);
      box.addEventListener("change", () => {
        state.overrides[control.target_property] = box.checked ? 1 : 0;
      });
      state.overrides[control.target_property] = box.checked ? 1 : 0;
      const caption = document.createElement("span");
      caption.textContent = control.label;
      label.append(box, caption);
      ui.controls.append(label);
      continue;
    }

    const label = document.createElement("label");
    label.className = "control";
    const caption = document.createElement("span");
    const value = document.createElement("output");
    const input = document.createElement("input");
    input.type = "range";
    input.min = String(control.min ?? 0);
    input.max = String(control.max ?? 1);
    input.step = String(control.step ?? 0.1);
    input.value = String(control.default ?? control.min ?? 0);
    state.overrides[control.target_property] = Number(input.value);

    const show = () => {
      value.textContent = `${input.value}${control.unit ? ` ${control.unit}` : ""}`;
    };
    show();
    input.addEventListener("input", () => {
      state.overrides[control.target_property] = Number(input.value);
      show();
    });

    caption.textContent = control.label;
    label.append(caption, input, value);
    ui.controls.append(label);
  }
}

/* ------------------------------------------------------------------- theme */

/**
 * Point the page at a palette and hand the canvas the colours it now resolves
 * to.
 *
 * `state.theme` is the only place a colour comes from — every drawer takes it
 * as an argument and `render` passes this one — so re-reading it here *is* the
 * whole switch. There is no cache to invalidate and nothing to repaint by hand;
 * the next frame is already in the new palette.
 *
 * The attribute goes on `<html>` rather than on `<body>` so that the palette
 * also covers whatever the stylesheet hangs off `:root`, and so that a swatch's
 * own `data-theme` — which does the same job for one chip — nests inside it
 * without either one having to know about the other.
 */
function applyTheme(id) {
  document.documentElement.dataset.theme = id;
  state.theme = readTheme(document.documentElement);
  renderSwatches(id);
}

/**
 * What a click on a swatch does: switch, remember, and keep the URL honest.
 *
 * The URL is rewritten because the acceptance run hands these links to a person,
 * and a link that says `?theme=paper` while the page shows薄荷 is worse than a
 * link with no theme in it at all. `replaceState`, not `pushState`: switching
 * the colours twice is not two places to go back to.
 */
function chooseTheme(id) {
  applyTheme(id);
  try {
    localStorage.setItem(STORAGE_KEY, id);
  } catch (error) {
    // Private mode can refuse `localStorage`. The palette still changes; it
    // just will not survive a reload, and that is not worth failing over.
  }
  try {
    const url = new URL(window.location.href);
    url.searchParams.set("theme", id);
    history.replaceState(null, "", url);
  } catch (error) {
    // A document that cannot rewrite its own URL — opened from a `file:` path,
    // say. Same answer: the colours are what was asked for, the address is not.
  }
}

/**
 * Build the picker. Called on every switch, which is what moves the ring.
 *
 * Rebuilt rather than updated in place: six buttons is nothing, and a
 * `replaceChildren` cannot leave a stale `.is-active` on a button that no longer
 * is one.
 */
function renderSwatches(current) {
  ui.themes.replaceChildren(
    ...THEMES.map((theme) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = theme.id === current ? "swatch is-active" : "swatch";
      button.title = `换成「${theme.label}」配色`;
      // The chip carries the palette id and the button does not. Everything
      // inside a `[data-theme]` subtree resolves against *that* palette, and
      // the label is text on the page — under `neon` it would come out
      // near-white and vanish on a light background.
      const chip = document.createElement("span");
      chip.className = "swatch-chip";
      chip.dataset.theme = theme.id;
      const name = document.createElement("span");
      // `textContent`, never `innerHTML`. These strings are ours rather than the
      // model's, but a second way of writing text into the document is a second
      // thing to have to audit for the one that is not.
      name.textContent = theme.label;
      button.append(chip, name);
      button.addEventListener("click", () => chooseTheme(theme.id));
      return button;
    }),
  );
}

/** What was chosen last time, or null. Never throws on a locked-down browser. */
function stored(key) {
  try {
    return localStorage.getItem(key);
  } catch (error) {
    return null;
  }
}

function storedTheme() {
  return stored(STORAGE_KEY);
}

function storedVoice() {
  return stored(VOICE_STORAGE_KEY);
}

/* ------------------------------------------------------------------- voice */

/**
 * Say that the browser is holding the sound until it is asked.
 *
 * Only ever about sound. The hint is not a gate on the picture: the player draws
 * whatever frame it is on whether or not anyone has clicked, which is what the
 * `?scene=N&step=N` links are for — a screenshot cannot click, and a link that
 * opens paused still has to show the beat it names.
 *
 * Since the player opens paused, a viewer who presses play has already provided
 * the gesture and this never appears for them. What is left for it is the links
 * that ask to start on their own, where playback begins before anyone has
 * touched the page.
 */
function showVoiceHint(blocked) {
  ui.voiceHint.hidden = !blocked;
}

/**
 * What a click on a voice does: switch, remember, keep the URL honest — and
 * reload the beat that is playing, because the point of the switch is to hear
 * the rest of it in the other voice.
 *
 * `speech/voices.py` and `voice.js` are held to each other by a test, the way
 * the palettes are: a voice one lists and the other does not is a button that
 * does nothing.
 */
function chooseVoice(id) {
  state.voice = id;
  try {
    localStorage.setItem(VOICE_STORAGE_KEY, id);
  } catch (error) {
    // Private mode can refuse `localStorage`. The voice still changes; it just
    // will not survive a reload, and that is not worth failing over.
  }
  try {
    const url = new URL(window.location.href);
    url.searchParams.set("voice", id);
    history.replaceState(null, "", url);
  } catch (error) {
    // A document that cannot rewrite its own URL — opened from a `file:` path,
    // say. Same answer: the voice is what was asked for, the address is not.
  }
  renderVoices(id);

  // The film is a different length in the other voice — see `rebuildTimeline` —
  // so the bar's `max` is rebuilt with it. The playhead is not stored, so there
  // is nothing to move: whatever beat and offset were on screen, `playhead()`
  // works out where they now sit on the new table.
  rebuildTimeline();
  syncTransport();

  if (state.track === null || !state.playing || state.transition !== null) return;
  // From where the picture has got to, not from the top. The two reads of one
  // line are within a few tenths of a second of each other, so `sim.t` is a fair
  // offset in either — and seeking rather than restarting is what makes a switch
  // cost the viewer nothing but the voice. A `sim.t` already past the new clip's
  // end simply finishes it, and the next beat arrives on the beat.
  state.track.play(
    clipUrl(state.scene.steps[state.stepIndex], state.voice, state.specUrl),
    state.sim.t,
  );
}

/**
 * Build the voice picker — or hide it, when there is nothing to choose between.
 *
 * A film generated with one voice carries one, and a picker with a single button
 * in it reads as a control that is broken rather than as a film with one
 * narrator. A spec with no speech at all hides it too, which is what leaves
 * every film made before this unchanged.
 *
 * Rebuilt on every switch for the same reason `renderSwatches` is: it is what
 * moves the ring, and a `replaceChildren` cannot leave a stale `.is-active` on a
 * button that no longer is one.
 */
function renderVoices(current) {
  ui.voicePicker.hidden = state.voiceIds.length < 2;
  ui.voices.replaceChildren(
    ...state.voiceIds.map((id) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = id === current ? "voice is-active" : "voice";
      button.title = `换成「${voiceLabel(id)}」`;
      const name = document.createElement("span");
      // `textContent`, never `innerHTML`. These strings are ours rather than the
      // model's, but a second way of writing text into the document is a second
      // thing to have to audit for the one that is not.
      name.textContent = voiceLabel(id);
      button.append(name);
      button.addEventListener("click", () => chooseVoice(id));
      return button;
    }),
  );
}

/* -------------------------------------------------------------------- boot */

//: How close to the end counts as "the end". Long enough that a press a frame
//: before the last beat finishes means 重播 rather than resume; short enough that
//: it never swallows a beat.
const END_OF_FILM = 0.05;

//: The recorder, built in `main` — it needs the canvas and the voice track, and
//: neither exists until then.
let recorder = null;

//: True between a pointer going down on the seek bar and coming back up.
//: `syncTransport` writes the bar's value sixty times a second, and while a
//: hand is on the handle that is the one writer too many.
let draggingSeek = false;

function togglePlay() {
  // Recording is a straight run from beat one to the end — see `startRecording`
  // for why a pause in the middle of one leaves a hole in the file. The button
  // is disabled while it runs, so this is the second line of that guard rather
  // than the first.
  if (state.recording) return;

  // The end of the film is a place the play button can be pressed from, and what
  // it means there is 重播. Resuming instead would carry on through a beat that
  // has been standing still since the button changed, which reads as the film
  // having forgotten where it was.
  if (state.total > 0 && playhead() >= state.total - END_OF_FILM) {
    state.playing = true;
    seekTo(0);
    syncTransport();
    return;
  }

  state.playing = !state.playing;
  state.last = 0;
  if (state.track === null) return;

  if (!state.playing) {
    state.track.pause();
    return;
  }
  // Resumed only when the beat on screen is the one that was being spoken to. A
  // fade paused part way through has a scene on its way out under it, and
  // `stepOnce` is frozen mid-dip — speaking there would narrate the picture that
  // is being covered up.
  if (state.transition !== null) return;
  state.track.play(
    clipUrl(state.scene.steps[state.stepIndex], state.voice, state.specUrl),
    state.sim.t,
  );
}

/**
 * Bring the transport bar up to date with the film. Once a frame, from `tick`.
 *
 * The bar has no state of its own — every number in it is read from the film,
 * which is what keeps it honest when the console's own buttons move the film
 * behind its back. The one exception is the drag: `draggingSeek` is the page's
 * way of saying "the value on screen is the viewer's, leave it alone".
 */
function syncTransport() {
  const total = state.total;
  const now = total > 0 ? Math.min(playhead(), total) : 0;

  // Played out. `advanceBeat` holds on the last beat and keeps animating it,
  // which is its own documented rule and is not changed here — but a player that
  // has reached the end of the file stops, and the button becomes 重播.
  if (state.playing && total > 0 && playhead() >= total) {
    state.playing = false;
    if (state.track !== null) state.track.pause();
    if (state.recording) stopRecording();
  }

  const atEnd = total > 0 && now >= total - END_OF_FILM;
  ui.play.textContent = state.playing ? "暂停" : atEnd ? "重播" : "播放";
  ui.clock.textContent =
    total > 0 ? `${formatClock(now)} / ${formatClock(total)}` : "--:-- / --:--";

  // A spec whose beats all declare zero seconds — `data/generated/render-ros_pub_sub.json`
  // is one — has no timeline at all: `advanceBeat` sits on the first beat for
  // ever, so there is nothing for a bar to measure. It goes dead rather than
  // showing a length invented for it.
  ui.seek.max = String(total > 0 ? total : 0);
  ui.seek.disabled = state.recording || !(total > 0);
  if (!draggingSeek) ui.seek.value = String(now);

  if (state.recording) {
    const percent = total > 0 ? Math.min(100, Math.round((now / total) * 100)) : 0;
    setRecordNote(`正在录制 ${percent}% —— 让这个标签页留在前台，切走画面会冻住。`);
  }
}

/**
 * Say something under the transport bar, or clear it.
 *
 * Called sixty times a second while a recording runs, hence the comparison: the
 * text is the same string for most of those frames, and writing identical text
 * into a live region is both wasted work and, in a screen reader, a stutter.
 */
function setRecordNote(message) {
  if (ui.recordNote.textContent === message) return;
  ui.recordNote.textContent = message;
  ui.recordNote.hidden = message === "";
}

/* ---------------------------------------------------------------- recording */

/** Undo what `startRecording` locked, from whichever way a recording ended. */
function releaseRecordingLock() {
  state.recording = false;
  ui.download.textContent = "下载视频";
  ui.play.disabled = false;
}

/** The saved file's name: the film's title, minus what a filesystem refuses. */
function filmFileName() {
  const raw = String(state.spec?.title ?? "").trim();
  const safe = raw.replace(/[\\/:*?"<>|]/g, "").slice(0, 60).trim();
  return safe || "动画";
}

/**
 * Record the film and save it as a file.
 *
 * **The film is played to be recorded.** There is no faster path: the picture is
 * drawn in real time and `MediaRecorder` writes in real time, so a ninety-second
 * film is ninety seconds of the page showing that film. Everything awkward below
 * follows from that one fact.
 *
 * - **From the top, and running.** A recording is a file, and a file that starts
 *   in the middle of a film is a file nobody can use. So the playhead goes back
 *   to zero and the clock starts, whatever the viewer was doing.
 * - **Pause is locked for the duration.** `track.stop()` at a scene change is
 *   meant and gives the dip its silence; a pause is not, and would put a stretch
 *   of frozen picture and silence into the middle of the file.
 * - **The graph has to exist before the recorder opens.** `audio.js:captureTrack`
 *   builds it and resumes its context; starting the recorder first would give a
 *   file whose first fraction of a second has no sound at all.
 * - **Nothing here goes through `fail()`.** A browser that cannot record, or a
 *   recorder that dies, says so under the bar. `fail()` stops the tick loop, so
 *   routing this through it would answer "you cannot record" with a black screen
 *   — the same trade `audio.js` makes for a clip that will not load.
 */
async function startRecording() {
  if (state.recording) {
    stopRecording();
    return;
  }
  if (!(state.total > 0)) {
    setRecordNote("这条片子的每一拍都是 0 秒，量不出长度，录不了。");
    return;
  }
  if (recordableType() === "") {
    setRecordNote("这个浏览器不能录制。换 Chrome 或 Edge 打开这一页再试。");
    return;
  }

  const wasPlaying = state.playing;
  // Land on the first frame *before* the recorder opens, so the file begins on
  // beat one rather than on whatever happened to be on screen when the button
  // was pressed.
  state.playing = false;
  seekTo(0);

  if (!(await recorder.start())) {
    // Nothing was recorded, so the film goes back to what it was doing. The
    // playhead stays at the top — that is the one part of the move not worth
    // undoing, and the note already said what went wrong.
    state.playing = wasPlaying;
    if (wasPlaying) playBeat();
    return;
  }

  state.recording = true;
  state.playing = true;
  state.last = 0;
  playBeat();
  ui.download.textContent = "停止录制";
  ui.play.disabled = true;
  syncTransport();
}

/** End the recording. The file is saved by the recorder's own `stop` handler. */
function stopRecording() {
  if (!state.recording) return;
  releaseRecordingLock();
  setRecordNote("");
  if (recorder !== null) recorder.stop();
}

/**
 * Build the recorder. Called from `main`, once the canvas and the track exist.
 *
 * `audioTrack` is a function rather than a track because `audio.js` builds its
 * WebAudio graph lazily, the first time something is recorded — so there is no
 * track to hand over until the recording has already been asked for.
 */
function buildRecorder() {
  return createRecorder({
    canvas: ui.canvas,
    audioTrack: () => state.track.captureTrack(),
    onProblem: (message) => {
      releaseRecordingLock();
      setRecordNote(message);
    },
    onDone: (blob, mimeType) => {
      saveBlob(blob, `${filmFileName()}${extensionFor(mimeType)}`);
      const megabytes = Math.round(blob.size / 1048576);
      setRecordNote(`录好了，已经存成文件（约 ${megabytes} MB）。`);
    },
  });
}

/* ------------------------------------------------------------------ chrome */

/**
 * What the viewer hears, and nothing else.
 *
 * "Nothing else" is the load-bearing half: the download is recorded off the same
 * element, and `audio.js` taps it *before* this gain, so a film downloaded with
 * the slider at zero is still a film with a voice.
 */
function applyVolume() {
  if (state.track === null) return;
  state.track.setVolume(Number(ui.gain.value));
  state.track.setMuted(state.muted);
}

function setMuted(value) {
  state.muted = Boolean(value);
  ui.mute.textContent = state.muted ? "取消静音" : "静音";
  if (state.track !== null) state.track.setMuted(state.muted);
}

/**
 * Fullscreen the stage and the bar, not the page — the palette and voice pickers
 * are the one pair of controls a real player has no equivalent of, so fullscreen
 * is exactly where they should not be.
 *
 * A browser without the API simply stays windowed. That is not worth a message:
 * every browser this page is meant for has it.
 */
function toggleFullscreen() {
  if (document.fullscreenElement === null) {
    const request = ui.player.requestFullscreen;
    if (typeof request === "function") request.call(ui.player).catch(() => {});
    return;
  }
  document.exitFullscreen().catch(() => {});
}

function bind() {
  ui.play.addEventListener("click", togglePlay);
  ui.download.addEventListener("click", () => {
    startRecording().catch((error) => {
      releaseRecordingLock();
      setRecordNote(`录制没起来：${error instanceof Error ? error.message : error}`);
    });
  });

  // `input` and not `change`: the picture should follow the handle, not wait for
  // it to be let go. A seek replays at most one beat of a handful of elements,
  // so there is nothing to throttle.
  ui.seek.addEventListener("pointerdown", () => {
    draggingSeek = true;
  });
  window.addEventListener("pointerup", () => {
    draggingSeek = false;
  });
  ui.seek.addEventListener("input", () => seekTo(Number(ui.seek.value)));

  ui.gain.addEventListener("input", () => {
    // Sliding the volume up is how everyone expects to leave mute, so it does.
    if (state.muted && Number(ui.gain.value) > 0) setMuted(false);
    applyVolume();
  });
  ui.mute.addEventListener("click", () => setMuted(!state.muted));
  ui.fullscreen.addEventListener("click", toggleFullscreen);
  document.addEventListener("fullscreenchange", () => {
    ui.fullscreen.textContent = document.fullscreenElement === null ? "全屏" : "退出全屏";
  });

  /*
   * ←/→ still step a beat, and the reason is that the *seek bar* is the coarse
   * control now. A video player spends its arrow keys on ±5 seconds because a
   * video has no structure to skip between; this film is a sequence of narrated
   * beats, and "the next sentence" is what a person reaches for. Crossing a
   * scene boundary is what `stepBy` is for, and this is its only caller left.
   *
   * The guard is not decoration. A focused range input consumes the arrow keys
   * itself, so without it nudging the volume would also walk the film; a focused
   * button consumes Space, so pressing it after clicking 全屏 would both press
   * that button again and toggle playback.
   */
  window.addEventListener("keydown", (event) => {
    // Nothing moves the film while it is being recorded, for the same reason the
    // pause button is dead: the file is meant to be a straight run, and a beat
    // stepped through here would be a jump in the middle of it. `startRecording`
    // is the way out.
    if (state.recording) return;
    const target = event.target;
    if (target && typeof target.closest === "function" && target.closest("input, button")) {
      return;
    }
    if (event.code === "Space") {
      event.preventDefault();
      togglePlay();
    } else if (event.code === "ArrowRight") {
      stepBy(1);
    } else if (event.code === "ArrowLeft") {
      stepBy(-1);
    }
  });
  window.addEventListener("resize", () => state.viewport.resize());

  // A browser will not start audio before the page has been interacted with, and
  // the only way to find out is to ask and be refused — which `audio.js` reports
  // through `onBlocked`. Any gesture counts and all of them do the same thing, so
  // they are the same handler.
  //
  // It starts wherever the picture has got to, not from the beginning. A viewer
  // who clicks four seconds in has missed four seconds and has not been shown
  // them out of order; rewinding the film to resync with the voice would be the
  // worse trade, and a screenshot with no gesture behind it must still draw.
  const armAudio = () => {
    if (state.track === null || !state.track.blocked) return;
    if (!state.playing || state.transition !== null) return;
    state.track.play(
      clipUrl(state.scene.steps[state.stepIndex], state.voice, state.specUrl),
      state.sim.t,
    );
  };
  window.addEventListener("pointerdown", armAudio);
  window.addEventListener("keydown", armAudio);
}

function loadScene(index) {
  state.sceneIndex = index;
  state.scene = state.spec.scenes[index];
  state.byId = new Map(state.scene.elements.map((element) => [element.id, element]));
  state.obstacleIds = state.scene.elements
    .filter((element) => element.role === "obstacle")
    .map((element) => element.id);
  state.obstacleRadius = new Map(
    state.obstacleIds.map((id) => [id, Number(state.byId.get(id)?.props?.radius ?? 0)]),
  );
  state.lookup = makeLookup(state.scene);
  state.sim = createSimulation(state.scene, state.spec.stage);
  state.overrides = {};
  state.accumulator = 0;
  state.last = 0;

  // The header used to be written from here — the film's title, then a scene
  // counter, the storyboard's eyebrow and this scene's teaching goal. All four
  // changed under the viewer while the film ran, and that is what made the page
  // read as an inspector. What is left of the header is the film's own name, and
  // `main` writes it once.
  //
  // A scene opens at rest. `setStep(0)` below would otherwise compare against
  // the beat index the *previous* scene was sitting on, read this scene's beats
  // at that index, and slide the first frame in from values nobody wrote.
  state.stepIndex = 0;
  state.departing = new Map();
  renderControls();
  setStep(0);
  // And then the first beat is given something to ease from, which is what
  // makes a scene *open* rather than switch on. Set after `setStep` rather than
  // passed into it because `setStep`'s own decision — hand over, or land — is
  // about a beat boundary and has no business knowing about scene openings; the
  // assignment here is the one line that says "this was not a beat boundary".
  //
  // **Skipped while paused, and that guard is not decoration.** Nothing
  // advances `sim.t` when the clock is stopped, so `easeProgress()` stays at 0,
  // `blend` keeps returning the seeded `false`, presence stays at 0 and
  // `drawElement` returns early for every element: the stage would be blank,
  // for ever, and the only way out would be to press play on a picture that
  // gave no sign there was one. `loadScene` is not reachable while paused today
  // — it is called from `stepOnce` behind `if (state.playing)`, and once at
  // startup before the first tick — but that is a coincidence of the current
  // call graph, and the guard is what makes it a fact.
  if (state.playing) state.departing = seedDeparting();
}

/**
 * Every element starts out absent, so a scene's first beat arrives rather than
 * appearing.
 *
 * This is the closing of `captureDeparting`'s known gap, which was real and
 * visible: a beat handed over from another beat eases from what it read then,
 * and a scene's first beat had nothing to ease from at all — `loadScene`
 * emptied the map and called `setStep(0)`, so whatever the first beat turned on
 * was simply *there*, and whatever it turned off was simply missing. A scene
 * opening is the one moment the audience is not expecting continuity, which
 * made it the worst place in the film for the only hard cut.
 *
 * **The value is the boolean `false`, not `0`.** `blend` ramps a presence gate
 * only when *both* ends are booleans; handed a `0` it falls through to
 * `return target` and this whole function becomes a no-op that looks
 * implemented — the failure mode where the arithmetic is right and nothing
 * moves.
 *
 * **Presence props only.** There is no "previous" to seed a number from — what
 * did `heading` read before the scene began? — and inventing one is exactly
 * what `captureDeparting` refuses. Seeding only the gates also keeps the
 * opening from animating every value it declares, which would be a second
 * gesture stacked on the scene change's own fade.
 */
function seedDeparting() {
  const from = new Map();
  for (const element of state.scene.elements) {
    const prop = PRESENCE_PROP[element.kind];
    if (prop === undefined) continue;
    from.set(`${element.id}.${prop}`, false);
  }
  return from;
}

async function main() {
  const params = new URLSearchParams(window.location.search);
  const specUrl = params.get("spec") || "/data/generated/render-robot_obstacle_avoidance.json";

  let spec;
  try {
    const response = await fetch(specUrl);
    if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`);
    spec = await response.json();
  } catch (error) {
    fail(`取不到 spec：${specUrl}\n${error instanceof Error ? error.message : error}`);
    return;
  }

  if (!spec.scenes || spec.scenes.length === 0) {
    fail("spec 里一个场景都没有");
    return;
  }

  state.spec = spec;
  // Kept because a beat's clip is named relative to the spec, and this is the
  // only place the spec's own address is known.
  state.specUrl = specUrl;

  // Whether the film starts on its own.
  //
  // A player opens **paused**, showing its first frame and waiting. That is what
  // a video does, and it is also why the page no longer has to explain that a
  // browser will not make a sound before it has been clicked — the press that
  // starts the film *is* the click.
  //
  // It also costs the first scene nothing. `loadScene`'s `seedDeparting` gives a
  // beat something to ease in from, which is right for a scene arriving mid-film
  // and pointless for the one frame the viewer is already looking at; skipped,
  // the opening frame is simply already there, which is what a paused frame is.
  //
  // Two kinds of link are the exception, and both are for machines rather than
  // for people:
  //
  // - `?scene=N` / `?step=N` is the acceptance link. It exists so a headless
  //   capture can reach a beat other than the first, and a capture cannot press
  //   play. It also has to *run*: a ballistic body at `t = 0` is a ball that has
  //   not been thrown and a trace has covered no ground, so a still frame on
  //   those is a picture of nothing.
  // - `?autoplay=1` is what the upload page's preview frame asks for. A result
  //   pane that sits on a still until you find the play button inside the frame
  //   reads as broken, next to a 生成动画 button that has just said it worked.
  state.playing =
    params.get("autoplay") === "1" || params.has("scene") || params.has("step");
  // Before the first frame, and before `bind` — the picker's listeners are
  // attached in here, and a palette that arrived after the first draw would be
  // one painted frame in the wrong colours.
  //
  // Which palette the page opens in, most specific first: a `?theme=` in the
  // link, then the one the lesson was *generated* with, then whatever the viewer
  // picked last time, then the default.
  //
  // `?theme=` first because a link someone was handed is a statement about the
  // picture they were handed, and the whole point of putting the palette in the
  // URL is that opening it shows what it says. The spec's palette comes second
  // and *ahead of the stored choice* deliberately: variety between lessons is the
  // entire reason the field exists, and a stored preference that outranked it
  // would mean anyone who had ever clicked a swatch never saw another palette
  // again. A click still wins for the rest of the page — `chooseTheme` writes the
  // URL, and the URL is first.
  //
  // `||` and not `??`, on all three. An empty `?theme=` and a spec written before
  // this field existed both say `""`, and neither is a choice.
  applyTheme(resolveTheme(params.get("theme") || spec.theme || storedTheme()));
  state.viewport = createStage(ui.canvas, spec.stage);
  state.viewport.resize();

  // Which voices this spec carries, and which one opens. The chain is the
  // theme's — `?voice=` first, then what the film was generated in, then the
  // stored choice — and `resolveVoice` narrows all of it to what is actually on
  // the film, so a link to a voice this spec does not have opens on one it does
  // rather than on silence.
  //
  // Before `loadScene`, because that reaches `setStep`, and `setStep` is what
  // starts the first clip. A track created after the first beat is a first beat
  // with no sound.
  state.voiceIds = voiceIdsIn(spec);
  state.voice = resolveVoice(
    params.get("voice") || spec.voice || storedVoice(),
    state.voiceIds,
  );
  state.track = createVoiceTrack({ onBlocked: showVoiceHint });
  renderVoices(state.voice);

  // The film's name, written **once**. The header used to be rewritten from
  // `loadScene` on every scene change — see the note there.
  ui.title.textContent = spec.title || "";
  // The strip of time the transport bar measures. Built after `state.voice`
  // because it is a different length in each voice, and before the first frame
  // because the bar reads it on the way past.
  rebuildTimeline();
  recorder = buildRecorder();

  // `?scene=N&step=N` opens the player at a given beat. Added for acceptance:
  // a headless screenshot can only capture whatever is on screen when it fires,
  // so without this the only frame anyone could look at was the first beat of
  // the first scene — which is exactly how "four identical rounded rectangles"
  // got recorded for one document and never checked in the other two.
  //
  // It still earns its place now that the player walks scene to scene on its
  // own: it is how you *start* somewhere other than the beginning, and how you
  // hand someone a link to one scene rather than to the whole run.
  //
  // Out of range is clamped rather than rejected, here and in `setStep`: a bad
  // index in a URL should still show you a picture, not a blank page.
  const sceneIndex = Number(params.get("scene") ?? 0);
  const stepIndex = Number(params.get("step") ?? 0);
  loadScene(Number.isInteger(sceneIndex) ? Math.max(0, Math.min(spec.scenes.length - 1, sceneIndex)) : 0);
  bind();
  if (Number.isInteger(stepIndex) && stepIndex > 0) {
    // An explicit `?step=` means "show me this frame", so the scene's arrival is
    // dropped rather than played: otherwise the frame the link names is a
    // half-arrived one, which is the opposite of what a link to one frame is
    // for. The URL's own comment above says its purpose is acceptance — a
    // screenshot of a stage that is still fading in is a screenshot of nothing.
    state.departing = new Map();
    setStep(stepIndex);
  }
  // Before the first frame rather than on it: a booted-paused player has no
  // `tick` work to do at all, and the bar would otherwise show the placeholder
  // `--:--` and a disabled-looking play button until something moved.
  syncTransport();

  // And load the beat that is now on screen, without sounding it. A player opens
  // paused, so this is the clip the very next press will want — and the first
  // press is the only one that pays for a cold element. `audio.js:prime` carries
  // the measurement and the reason it lands on the *end* of the sentence.
  if (state.track !== null) {
    state.track.prime(clipUrl(state.scene.steps[state.stepIndex], state.voice, state.specUrl));
  }

  requestAnimationFrame(tick);
}

main().catch((error) => fail(error instanceof Error ? error.message : String(error)));
