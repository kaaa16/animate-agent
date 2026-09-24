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
import { LIVE_PROPS, drawElement } from "./registry.js";
import { createStage, readTheme } from "./stage.js";
import { STORAGE_KEY, THEMES, resolveTheme } from "./themes.js";

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
 * Which prop decides whether an element is on screen at all.
 *
 * The drawers ask this same question — `drawBody` and `drawTrace` test
 * `visible`, `drawEmitter` and `drawZone` test `enabled` — so the two have to
 * name the same prop, or an element would fade against a gate that never
 * opened. `test_the_presence_table_names_the_props_the_drawers_test` holds them
 * together.
 */
const PRESENCE_PROP = {
  body: "visible",
  emitter: "enabled",
  zone: "enabled",
  trace: "visible",
};

/**
 * How faint the background grid is, as a fraction of the palette's own
 * structural colour.
 *
 * The grid was a literal `rgba(83, 246, 255, 0.07)` — `neon`'s `--line` at 0.22
 * — so under a light palette the stage was covered in cyan graph paper. Read as
 * a fraction of `--line` instead, `0.32 × 0.22` is the same 0.07 it has always
 * been on the default theme, and the grid belongs to whatever palette is up.
 */
const GRID_ALPHA = 0.22;

const ui = {
  canvas: document.getElementById("stage"),
  title: document.getElementById("title"),
  eyebrow: document.getElementById("eyebrow"),
  sceneCount: document.getElementById("scene-count"),
  goal: document.getElementById("goal"),
  steps: document.getElementById("steps"),
  controls: document.getElementById("controls"),
  themes: document.getElementById("themes"),
  play: document.getElementById("play"),
  reset: document.getElementById("reset"),
  prev: document.getElementById("prev"),
  next: document.getElementById("next"),
  error: document.getElementById("error"),
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
  //: `"<element>.<prop>"` -> what it read under the *previous* beat. Emptied
  //: whenever a beat is loaded rather than handed over from one, so the first
  //: beat of a scene eases from nothing and simply appears.
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
  playing: true,
  accumulator: 0,
  last: 0,
  stopped: null,
};

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

function drawChrome(ctx, scene, theme, width, height) {
  const lane = scene.elements.find((element) => element.role === "vehicle");
  const laneY = lane ? lane.y : height / 2;

  // Two `save`/`restore` pairs rather than one, because the grid runs at a
  // fraction of the palette's opacity and the lane line does not. Sharing one
  // block would mean handing the lane line the grid's alpha and drawing it a
  // fifth as strongly as it has always been drawn.
  ctx.save();
  ctx.strokeStyle = theme.line;
  ctx.globalAlpha *= GRID_ALPHA;
  ctx.lineWidth = 1;
  for (let x = 0; x <= width; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y <= height; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  ctx.restore();

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
 *   coat. The last beat simply holds, still animating; ◀ ▶ and the beat list
 *   still go back.
 */
function advanceBeat() {
  const duration = Number(state.scene.steps[state.stepIndex]?.duration ?? 0);
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
  state.transition = { toScene, toStep, phase: "out", t: 0 };
}

/**
 * Move `delta` beats, crossing a scene boundary when that is where `delta`
 * leads.
 *
 * ◀ ▶ used to clamp at the edges of the loaded scene, which made the transport
 * a control for looking at one scene rather than for looking at the animation.
 * The upload page hands over the whole storyboard, so this is the button a
 * person will reach for to check the scene after this one.
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
  renderSteps();
}

function renderSteps() {
  ui.steps.replaceChildren(
    ...state.scene.steps.map((step, index) => {
      const item = document.createElement("li");
      item.className = index === state.stepIndex ? "step is-current" : "step";
      const title = document.createElement("span");
      title.className = "step-title";
      title.textContent = `${String(index + 1).padStart(2, "0")} ${step.title}`;
      const body = document.createElement("p");
      body.className = "step-body";
      // `textContent`, never `innerHTML` — this string may be model-authored.
      body.textContent = step.narration;
      item.append(title, body);
      item.addEventListener("click", () => setStep(index));
      return item;
    }),
  );
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
 * Rebuilt rather than updated in place for the same reason `renderSteps` is:
 * six buttons is nothing, and a `replaceChildren` cannot leave a stale
 * `.is-active` on a button that no longer is one.
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
function storedTheme() {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch (error) {
    return null;
  }
}

/* -------------------------------------------------------------------- boot */

function togglePlay() {
  state.playing = !state.playing;
  state.last = 0;
  ui.play.textContent = state.playing ? "暂停" : "播放";
}

function bind() {
  ui.play.addEventListener("click", togglePlay);
  // 重置 means "put it back where it started", and a scene change in flight is
  // part of what has to be put back: left running, it would finish the fade and
  // load the scene you just tried to leave.
  ui.reset.addEventListener("click", () => {
    state.transition = null;
    setStep(0);
  });
  ui.prev.addEventListener("click", () => stepBy(-1));
  ui.next.addEventListener("click", () => stepBy(1));
  window.addEventListener("keydown", (event) => {
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

  ui.title.textContent = state.spec.title || state.scene.title || "";
  ui.eyebrow.textContent = state.spec.eyebrow || state.spec.subject || "";
  // Which scene of how many, in words. The scene's own title and goal change
  // with it, but neither says "and there are two more after this" — and until
  // the player could leave a scene at all, nothing needed to.
  ui.sceneCount.textContent = `第 ${index + 1} / ${state.spec.scenes.length} 幕`;
  ui.goal.textContent = state.scene.teaching_goal || "";

  // A scene opens at rest. `setStep(0)` below would otherwise compare against
  // the beat index the *previous* scene was sitting on, read this scene's beats
  // at that index, and slide the first frame in from values nobody wrote.
  state.stepIndex = 0;
  state.departing = new Map();
  renderControls();
  setStep(0);
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
  // Before the first frame, and before `bind` — the picker's listeners are
  // attached in here, and a palette that arrived after the first draw would be
  // one painted frame in the wrong colours.
  //
  // A `?theme=` outranks the stored choice: a link someone was handed is a
  // statement about the picture they were handed, and the whole point of
  // putting the palette in the URL is that opening it shows what it says.
  applyTheme(resolveTheme(params.get("theme") ?? storedTheme()));
  state.viewport = createStage(ui.canvas, spec.stage);
  state.viewport.resize();

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
  if (Number.isInteger(stepIndex) && stepIndex > 0) setStep(stepIndex);
  requestAnimationFrame(tick);
}

main().catch((error) => fail(error instanceof Error ? error.message : String(error)));
