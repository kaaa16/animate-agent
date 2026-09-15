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
import { drawElement } from "./registry.js";
import { createStage, readTheme } from "./stage.js";

const FIXED_STEP = 1 / 60;
/** Beyond this, drop the backlog rather than replaying a long stall frame by frame. */
const MAX_CATCHUP_STEPS = 5;

const ui = {
  canvas: document.getElementById("stage"),
  title: document.getElementById("title"),
  eyebrow: document.getElementById("eyebrow"),
  goal: document.getElementById("goal"),
  steps: document.getElementById("steps"),
  controls: document.getElementById("controls"),
  play: document.getElementById("play"),
  reset: document.getElementById("reset"),
  prev: document.getElementById("prev"),
  next: document.getElementById("next"),
  error: document.getElementById("error"),
};

const state = {
  spec: null,
  scene: null,
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
  return (elementId, prop, fallback) => {
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
    highlighted: new Set(step?.highlights ?? []),
    isDangerous: (id) => state.sim.isDangerous(id),
  };
}

/* ----------------------------------------------------------------- drawing */

function drawChrome(ctx, scene, theme, width, height) {
  const lane = scene.elements.find((element) => element.role === "vehicle");
  const laneY = lane ? lane.y : height / 2;

  ctx.save();
  ctx.strokeStyle = "rgba(83, 246, 255, 0.07)";
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
  ctx.setLineDash([10, 14]);
  ctx.strokeStyle = theme.line;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, laneY);
  ctx.lineTo(width, laneY);
  ctx.stroke();
  ctx.restore();
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
      state.sim.update(FIXED_STEP, state.lookup);
      state.accumulator -= FIXED_STEP;
      steps += 1;
    }
    if (steps === MAX_CATCHUP_STEPS) state.accumulator = 0;
  }

  render();
}

/* --------------------------------------------------------------------- step */

function setStep(index) {
  const total = state.scene.steps.length;
  state.stepIndex = Math.max(0, Math.min(total - 1, index));
  // Restart the beat from rest: a step is a moment, and a beat that inherits the
  // previous one's elapsed time can never be looked at twice.
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

/* -------------------------------------------------------------------- boot */

function togglePlay() {
  state.playing = !state.playing;
  state.last = 0;
  ui.play.textContent = state.playing ? "暂停" : "播放";
}

function bind() {
  ui.play.addEventListener("click", togglePlay);
  ui.reset.addEventListener("click", () => setStep(0));
  ui.prev.addEventListener("click", () => setStep(state.stepIndex - 1));
  ui.next.addEventListener("click", () => setStep(state.stepIndex + 1));
  window.addEventListener("keydown", (event) => {
    if (event.code === "Space") {
      event.preventDefault();
      togglePlay();
    } else if (event.code === "ArrowRight") {
      setStep(state.stepIndex + 1);
    } else if (event.code === "ArrowLeft") {
      setStep(state.stepIndex - 1);
    }
  });
  window.addEventListener("resize", () => state.viewport.resize());
}

function loadScene(index) {
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
  ui.goal.textContent = state.scene.teaching_goal || "";

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
  state.theme = readTheme(document.documentElement);
  state.viewport = createStage(ui.canvas, spec.stage);
  state.viewport.resize();

  // `?scene=N&step=N` opens the player at a given beat. Added for acceptance:
  // a headless screenshot can only capture whatever is on screen when it fires,
  // so without this the only frame anyone could look at was the first beat of
  // the first scene — which is exactly how "four identical rounded rectangles"
  // got recorded for one document and never checked in the other two.
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
