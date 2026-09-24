/**
 * Headless smoke test for the player's drawing and simulation modules.
 *
 * Why this exists
 * ---------------
 *
 * The browser check and the automated check are separate acceptance steps, by
 * standing correction — but "separate" must not mean "the automated half covers
 * nothing". A canvas drawer is exactly the kind of code that fails in ways only
 * a person notices: a misspelled `ctx.qadraticCurveTo` throws, but a wrong sign
 * in an angle just draws something slightly odd.
 *
 * So this harness pins down the part a machine *can* judge: every element of a
 * real spec, across every step, drawing without throwing, through a context that
 * records what it was asked to do. It catches typos, missing methods and
 * elements that silently draw nothing. It cannot tell you the car looks right —
 * that stays a human check, and the two are deliberately not conflated.
 *
 * Usage
 * -----
 *
 *     node tools/player_smoke.mjs data/generated/render-robot_obstacle_avoidance.json
 *
 * Exits non-zero on the first failure, with the offending element named.
 */

import { readFileSync } from "node:fs";

import { createSimulation } from "../frontend/player/behaviors.js";
import { CAPTION_MAX_LINES, wrapCaption } from "../frontend/player/caption.js";
import {
  EASE_SECONDS,
  EASINGS,
  blend,
  curve,
  easeOutCubic,
  handsOver,
} from "../frontend/player/easing.js";
import { EMPHASIS_NAMES, EMPHASIS_SECONDS, emphasisAt } from "../frontend/player/emphasis.js";
import { drawElement } from "../frontend/player/registry.js";
import { fitTransform, readTheme } from "../frontend/player/stage.js";

/**
 * The context properties `fakeContext` records — and the ones `serializeCalls`
 * can subtract again.
 *
 * Recorded because a highlight *is* a style change. `applyHighlight` sets
 * `shadowColor`/`shadowBlur` and nothing else, so a plain field would swallow
 * the only thing most beats do, and the "no two beats alike" check below would
 * call every highlight-only beat a still. It did, on the first run: the
 * avoidance document's 系统四大组成 and 前方180度扫描 beats differ in which
 * elements glow and in nothing else.
 *
 * Named as a list because the stricter reading further down exists to subtract
 * exactly these. `textAlign` is the one a drawer also reads *back*
 * (`drawReadout` picks its text origin from it) — which the recording supports,
 * because each property is defined with a getter as well as a setter.
 */
const STYLE_PROPERTIES = [
  "fillStyle",
  "strokeStyle",
  "lineWidth",
  "globalAlpha",
  "shadowColor",
  "shadowBlur",
  "font",
  "textAlign",
  "textBaseline",
];

/**
 * `Path2D` is a browser global, and `glyphs.js` builds one per glyph part from
 * the `d` string the spec carries. Node has none, so a car body would fail the
 * smoke run with `Path2D is not defined` — a failure that says nothing about
 * whether the picture is right. Same reasoning as `measureText` below: emulate
 * the platform the player actually runs on.
 *
 * The `d` string is kept so a recorded `stroke` call can still be told apart by
 * what it draws rather than only by how many times it was called.
 */
class RecordingPath2D {
  constructor(d) {
    this.d = d;
  }

  /** Without this a recorded `stroke(...)` reads `stroke([object Object])`. */
  toString() {
    return this.d;
  }
}
globalThis.Path2D = RecordingPath2D;

/** A 2D context that records calls instead of rasterising. */
function fakeContext() {
  const calls = [];
  const record =
    (name) =>
    (...args) => {
      calls.push({ name, args });
    };
  const ctx = {
    calls,
    save: record("save"),
    restore: record("restore"),
    beginPath: record("beginPath"),
    closePath: record("closePath"),
    moveTo: record("moveTo"),
    lineTo: record("lineTo"),
    arc: record("arc"),
    // A body carries a live `heading`, so the drawer works in a translated and
    // rotated frame rather than adding an angle at every corner.
    translate: record("translate"),
    rotate: record("rotate"),
    // A glyph is drawn in its own view-box units, so the glyph drawer scales
    // into them rather than converting every coordinate.
    scale: record("scale"),
    quadraticCurveTo: record("quadraticCurveTo"),
    // A `branch` or `mind` tree draws each edge as one bump curve, which is
    // `bezierCurveTo` and nothing else. Same reason `quadraticCurveTo` is here:
    // a context missing a method the drawer calls reports a real spec as a
    // broken one, and the failure names the primitive rather than the harness.
    bezierCurveTo: record("bezierCurveTo"),
    fill: record("fill"),
    stroke: record("stroke"),
    fillText: record("fillText"),
    fillRect: record("fillRect"),
    setLineDash: record("setLineDash"),
    setTransform: record("setTransform"),
    clearRect: record("clearRect"),
    // Annotations measure their own text to size the plate behind it. The number
    // is arbitrary — nothing here judges layout — but it has to be a number, or
    // every label draws a NaN-wide box and the smoke run would be measuring a
    // context that no browser provides.
    measureText: (text) => ({ width: String(text).length * 7 }),
  };
  for (const name of STYLE_PROPERTIES) {
    let value = null;
    Object.defineProperty(ctx, name, {
      get: () => value,
      set: (next) => {
        value = next;
        calls.push({ name, args: [next] });
      },
    });
  }
  return ctx;
}

/**
 * A view for one step, optionally with controls overridden.
 *
 * `overrides` mirrors what the player does when a slider moves, and it has to go
 * through the same tiers in the same order — a smoke harness that resolved
 * properties differently from `player.js` would be exercising a lookup nobody
 * runs. `binds` in particular: without that tier a bound property keeps the
 * value layout baked in, which is exactly the bug the tier exists to prevent.
 */
function viewFor(scene, simulation, stepIndex, theme, overrides = {}, glyphs = {}) {
  const byId = new Map(scene.elements.map((element) => [element.id, element]));
  const obstacleIds = scene.elements
    .filter((element) => element.role === "obstacle")
    .map((element) => element.id);
  const step = scene.steps[stepIndex];
  return {
    live: simulation.live,
    time: simulation.t,
    theme,
    elementById: byId,
    obstacleIds,
    obstacleRadius: new Map(
      obstacleIds.map((id) => [id, Number(byId.get(id)?.props?.radius ?? 0)]),
    ),
    // `RenderSpec.glyphs`, handed through like the theme: the player resolves a
    // `body.glyph` name against geometry the spec carries, never against a file
    // it goes and finds.
    glyphs,
    highlighted: new Set(step?.highlights ?? []),
    isDangerous: (id) => simulation.isDangerous(id),
    lookup: (elementId, prop, fallback) => {
      const key = `${elementId}.${prop}`;
      if (key in overrides) return overrides[key];

      const fromStep = step?.states?.[elementId];
      if (fromStep && prop in fromStep) return fromStep[prop];

      const element = byId.get(elementId);
      const bound = element?.binds?.[prop];
      if (bound && bound in overrides) return overrides[bound];
      if (element?.props && prop in element.props) return element.props[prop];

      const sceneKey = `scene.${prop}`;
      if (sceneKey in overrides) return overrides[sceneKey];
      if (prop in (scene.params ?? {})) return scene.params[prop];
      return fallback;
    },
    // Always 1 here. The ramp this feeds is a *player* behaviour — it needs a
    // beat boundary and a clock counting into it, and this harness draws each
    // beat as a single instant. What it does need from this view is that the
    // element is drawn at all, which is what a missing `presence` would break:
    // `drawElement` calls it unconditionally.
    presence: () => 1,
  };
}

/**
 * The closest the vehicle gets to each obstacle over a full run, in stage units.
 *
 * Reported, never asserted — and reported at the slider's **maximum** as well as
 * its default, because the two disagree. The dodge is a fixed 46px
 * (`behaviors.js`'s `MAX_DODGE`) while the danger threshold is a slider, so once
 * the threshold is raised past `offset + MAX_DODGE` the envelope becomes
 * inescapable and the car ends up holding its swerve right where an obstacle is.
 * That is a defect in the behaviour layer — not in the layout, which only fixed
 * the geometry — and it is recorded in `docs/storyboard-milestone.md`.
 *
 * Printing the number on every run is what keeps it *measured*: prose in a doc
 * rots, and this one has no assertion that can fail yet, because the honest fix
 * (derive the dodge from geometry instead of a constant) has not been done.
 */
function closestApproach(scene, stage, overrides) {
  const vehicle = scene.elements.find((element) => element.role === "vehicle");
  const obstacles = scene.elements.filter((element) => element.role === "obstacle");
  if (!vehicle || obstacles.length === 0) return null;

  const simulation = createSimulation(scene, stage);
  const nearest = new Map(obstacles.map((obstacle) => [obstacle.id, Number.POSITIVE_INFINITY]));
  for (let frame = 0; frame < 60 * 20; frame += 1) {
    const { lookup } = viewFor(scene, simulation, 0, THEME, overrides);
    simulation.update(1 / 60, lookup);
    const node = simulation.live.get(vehicle.id);
    for (const obstacle of obstacles) {
      const other = simulation.live.get(obstacle.id);
      if (!other) continue;
      const distance = Math.hypot(other.x - node.x, other.y - node.y);
      nearest.set(obstacle.id, Math.min(nearest.get(obstacle.id), distance));
    }
  }
  return nearest;
}

/** The widest `scene.safe_distance` the scene's own slider allows, if it has one. */
function maxSafeDistance(scene) {
  const slider = scene.controls.find(
    (control) => control.type === "slider" && control.target_property === "scene.safe_distance",
  );
  return typeof slider?.max === "number" ? slider.max : null;
}

function describeGaps(gaps) {
  return [...gaps].map(([id, distance]) => `${id} ${distance.toFixed(0)}px`).join("、");
}

/**
 * The centre distance below which the vehicle and an obstacle visibly overlap:
 * the sum of their half-extents. An obstacle's `width` is its diameter, so this
 * is the same number the drawer works from rather than a second opinion about
 * how big the shapes are.
 */
function touchDistance(scene, obstacleId) {
  const vehicle = scene.elements.find((element) => element.role === "vehicle");
  const obstacle = scene.elements.find((element) => element.id === obstacleId);
  return ((vehicle?.width ?? 0) + (obstacle?.width ?? 0)) / 2;
}

/**
 * Every mounted element must sit exactly on the thing it is mounted on.
 *
 * This invariant has already been broken once. The dodge pass used to run over
 * *every* element and reset each one's `y` from its own home position, which
 * undid the attachment pass that ran before it — so the lidar fan and the dashed
 * safe circle stayed pinned to the lane line while the car swerved out from
 * under them. Nothing threw. The picture still moved. The only symptom was
 * "the radar doesn't follow the car", which is a person noticing a thing no
 * assertion was watching.
 *
 * Equality is exact on purpose: the two values are a direct assignment apart, so
 * any drift at all is a real defect rather than floating-point noise.
 */
/**
 * The recorded calls as one comparable string.
 *
 * `record` stores `{name, args}` objects, so joining the array directly yields
 * `[object Object]|[object Object]` — every call identical to every other, which
 * makes the "no two beats alike" comparison below pass on nothing. It did: the
 * first version of that check reported the avoidance scene's step-5 as a still,
 * and every element in it as "same", for exactly this reason.
 */
function serializeCalls(calls, { geometryOnly = false } = {}) {
  const kept = geometryOnly
    ? calls.filter(({ name }) => !STYLE_PROPERTIES.includes(name))
    : calls;
  return kept.map(({ name, args }) => `${name}(${args.map(String).join(",")})`).join("|");
}

/**
 * Whether this element is *supposed* to draw nothing at this moment.
 *
 * `enabled: false` on an emitter or a zone is a beat doing its job — the
 * avoidance document's 扫描周期 beat turns the scanning beam off. Drawing
 * nothing is then the correct answer, and the "invisible element" check below
 * would otherwise call a working beat a defect.
 */
/**
 * The first `undefined` or `NaN` in what a drawer asked the context to do.
 *
 * Every colour a drawer uses comes from the theme and every coordinate from the
 * spec, and both arrive by name. A misspelled theme slot or a prop that is not
 * there is not an exception — the canvas accepts `undefined` for `fillStyle`
 * and keeps whatever style was set last, accepts `NaN` for a coordinate and
 * skips the path. The picture comes out subtly wrong and nothing says so.
 *
 * `serializeCalls` already renders both as text, so the recording is where they
 * were always visible; this is what looks. It is here rather than only in the
 * theme check below because the same hole opens on the *prop* path, which no
 * list of theme slots can cover.
 */
function firstHole(calls) {
  for (const { name, args } of calls) {
    for (const arg of args) {
      if (arg === undefined) return `${name}(undefined)`;
      if (typeof arg === "number" && Number.isNaN(arg)) return `${name}(NaN)`;
    }
  }
  return null;
}

/**
 * `readTheme` must hand back a filled slot for every name, not a hole.
 *
 * The failure this is built against is a typo in a *fallback* call — the second
 * argument of `read`, which nothing else in the project reads. `read("--accent",
 * "")` would look perfectly reasonable in the source and would put `""` into
 * every highlight on the page.
 */
function checkThemeSlots() {
  const empty = Object.entries(THEME).filter(
    ([, value]) => typeof value !== "string" || value === "",
  );
  if (empty.length > 0) {
    return (
      `readTheme 有 ${empty.length} 个槽位是空的：${empty.map(([key]) => key).join("、")}——` +
      "画布会照常画完，只是那几个颜色是空的"
    );
  }
  console.log(
    `  ✓ 主题槽位：readTheme 交出 ${Object.keys(THEME).length} 个非空值（默认调色板）`,
  );
  return null;
}

/**
 * The caption's line-breaking, against a ruler this test owns.
 *
 * `wrapCaption` takes its width measurement as a parameter precisely so that
 * this can exist: a ruler of "every character is 10 wide" makes the breaks
 * arithmetic, instead of depending on which fonts the machine happens to have.
 *
 * What is checked is the two things the band depends on — no line wider than
 * the band, and never more lines than `layout.py` reserved room for. The second
 * is the one that would go unnoticed: a caption that outgrows the strip draws
 * over the picture, and nothing on screen would say the strip was the wrong
 * size, so it has to be said here.
 */
function checkCaptionWrapping() {
  const ruler = (text) => text.length * 10;
  const maxWidth = 400;
  const texts = [
    "",
    "抛体被抛出后只受重力",
    "十".repeat(40),
    "十".repeat(41),
    "十".repeat(200),
    "mixed 中英文 together in one caption line",
  ];

  for (const text of texts) {
    const lines = wrapCaption(text, ruler, maxWidth);
    if (lines.length > CAPTION_MAX_LINES) {
      return `「${text.slice(0, 12)}…」折出 ${lines.length} 行，超过字幕条留的 ${CAPTION_MAX_LINES} 行`;
    }
    for (const line of lines) {
      if (ruler(line) > maxWidth) {
        return `「${line.slice(0, 12)}…」宽 ${ruler(line)}，超过 ${maxWidth}——会画出字幕条`;
      }
    }
  }

  // A caption that fits keeps every character: folding is not supposed to be
  // lossy except at the very end of something too long for the band.
  const fits = texts[1];
  const oneLine = wrapCaption(fits, ruler, maxWidth);
  if (oneLine.length !== 1 || oneLine[0] !== fits) {
    return `「${fits}」一行放得下，却被折成 ${JSON.stringify(oneLine)}`;
  }

  // And the too-long case is cut *visibly*. Silently spilling and silently
  // dropping look the same to whoever reads the code; only one of them shows up
  // on the page.
  const spilled = wrapCaption("十".repeat(200), ruler, maxWidth);
  if (!spilled[spilled.length - 1].endsWith("…")) {
    return `200 字折出来末尾没有省略号：${spilled.map((line) => line.length).join("/")}`;
  }

  console.log(
    `  ✓ 字幕折行：${texts.length} 个样例都没有超宽，最长的一例折成 ${CAPTION_MAX_LINES} 行并截断`,
  );
  return null;
}

function switchedOff(element, view) {
  if (element.kind === "emitter" || element.kind === "zone") {
    return view.lookup(element.id, "enabled", true) === false;
  }
  if (element.kind === "body" || element.kind === "trace") {
    return view.lookup(element.id, "visible", true) === false;
  }
  return false;
}

function brokenAttachment(scene, simulation) {
  for (const [childId, parentId] of Object.entries(scene.attachment ?? {})) {
    const child = simulation.live.get(childId);
    const parent = simulation.live.get(parentId);
    if (!child || !parent) continue;
    if (child.x !== parent.x || child.y !== parent.y) {
      return (
        `挂载脱离：\`${childId}\` 应贴合 \`${parentId}\`，实际在 ` +
        `(${child.x.toFixed(1)}, ${child.y.toFixed(1)})，母体在 ` +
        `(${parent.x.toFixed(1)}, ${parent.y.toFixed(1)})`
      );
    }
  }
  return null;
}

/**
 * The theme every drawer is handed, built by `readTheme` itself.
 *
 * It used to be a literal copied out of `stage.js`, which is a second list to
 * keep in step, and the failure is silent in exactly the wrong direction: a
 * colour added to `readTheme` and forgotten here reaches the drawer as
 * `undefined`, and `serializeCalls` writes `shadowColor(undefined)` — which
 * compares equal to itself, so the "no two beats differ" check below is
 * perfectly happy with a highlight that draws nothing.
 *
 * `readTheme` touches the DOM exactly once, in `getComputedStyle`, so stubbing
 * that one global makes it hand back its own fallbacks — the `neon` palette.
 * No element is needed and `null` is passed to say so. `checkThemeSlots` below
 * then insists none of them came back empty.
 */
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "" });
const THEME = readTheme(null);

/**
 * A per-frame step larger than this is a wrap, not travel.
 *
 * Both things that move in this simulator wrap: a `lane` body round the right
 * edge (`behaviors.js`: `wrap(base.x + t * speed * PIXELS_PER_SPEED, width)`),
 * and a traveller's `progress` back to zero. Those frames are discontinuous,
 * and counting them added a whole stage width each time — 7602px for a car that
 * actually drove about 1440. Nothing legitimate comes near it: the fastest the
 * contract allows is 5× at `PIXELS_PER_SPEED = 90`, which is 7.5px in a 1/60s
 * frame, so this has three orders of magnitude of room.
 */
const MAX_STEP_PER_FRAME = 60;

/**
 * How long the link each traveller rides actually is, in stage units.
 *
 * A traveller has no position of its own: `behaviors.js` advances `progress`
 * and the *drawer* places the token along the link from it, never writing the
 * node's `x`/`y`. So how far a message token went is `Δprogress × link length`,
 * and without the link length there is no number at all — which is how this
 * harness first printed `累计路程=0.0px` for a `chain` scene whose token
 * crosses the whole frame, the same line it printed for a `hub` scene where
 * genuinely nothing moves.
 */
function linkLengths(scene) {
  const byId = new Map(scene.elements.map((element) => [element.id, element]));
  const lengths = new Map();
  for (const element of scene.elements) {
    if (element.kind !== "traveler") continue;
    const link = byId.get(element.props?.along);
    const from = byId.get(link?.props?.from);
    const to = byId.get(link?.props?.to);
    lengths.set(
      element.id,
      from && to ? Math.hypot(to.x - from.x, to.y - from.y) : 0,
    );
  }
  return lengths;
}

function main() {
  const path = process.argv[2];
  if (!path) {
    console.error("用法：node tools/player_smoke.mjs <render-spec.json>");
    return 2;
  }
  const spec = JSON.parse(readFileSync(path, "utf8"));
  const transform = fitTransform(spec.stage, { width: 960, height: 560 }, 2);
  console.log(
    `stage ${spec.stage.width}x${spec.stage.height} -> scale ${transform.scale.toFixed(3)} ` +
      `@dpr${transform.dpr} (${transform.deviceWidth}x${transform.deviceHeight})`,
  );

  let drawn = 0;
  // The stricter reading, tallied across every scene and printed at the end.
  let transitions = 0;
  let shapeStill = 0;
  let totalSeconds = 0;
  for (const scene of spec.scenes) {
    const simulation = createSimulation(scene, spec.stage);
    const everDanger = new Set();
    /** What each beat drew, for the "no two consecutive beats alike" check below. */
    const pictures = [];
    // Tracked so the attachment check below cannot pass vacuously. If nothing
    // ever moves sideways, "mounted things match their parent" is true for the
    // same reason 0 === 0 is, and the check would prove nothing at all.
    const homeY = new Map(scene.elements.map((element) => [element.id, element.y]));
    let maxLateral = 0;
    // How far anything went, summed frame to frame over every element — a
    // different question from `maxLateral`, and the one that answers "is this
    // scene dead".
    //
    // Not end-to-end displacement: a `lane` body wraps round to the left edge,
    // so `|x - x0|` saws back and forth and measures the wrap rather than the
    // journey. Not lateral-only either: a `chain` traveller walks along a link,
    // which a y-only reading cannot see at all. Both of those printed
    // `最大侧移=0.0px` — the ROS sample because its message token slides
    // sideways without ever moving in y, the generated `hub` scene because
    // nothing moves whatsoever — and one number meaning two opposite things is
    // how a dead scene stayed invisible.
    //
    // Read as a liveness signal, not a distance: it is a sum over elements, so
    // a sensor mounted on a driving car is counted as well as the car. Zero is
    // the only value with a meaning, and it is unambiguous.
    let travelled = 0;
    // And how far anything *turned*, which is a different question from how far
    // it went and was invisible here until now.
    //
    // A body's `heading` is read by the drawer through `lookup`, never written
    // back into `live`, so a scene that only swings reads as perfectly dead to a
    // translation-only measure — and an `arm` scene is exactly that: the arm is
    // hinged at its base and the base does not move. Printed separately rather
    // than folded into `travelled`, because they are different units and adding
    // them would invent a number with no meaning.
    let turned = 0;
    // And how far an accent moved the pen. A third unit, kept apart for the
    // same reason: it is drawn, not simulated, so it is in neither of the two
    // above. See the loop below.
    let accented = 0;
    let previous = new Map();
    let previousHeading = new Map();
    const ridden = linkLengths(scene);
    const seconds = scene.steps.reduce((sum, step) => sum + Number(step.duration ?? 0), 0);
    totalSeconds += seconds;
    for (let stepIndex = 0; stepIndex < scene.steps.length; stepIndex += 1) {
      // Advance far enough for the simulation to have moved things, so a drawer
      // that only works at t=0 does not pass by luck. Danger is sampled *during*
      // the run rather than read at the end: the car passes its obstacles and is
      // safe again by then, so the final frame alone would report nothing.
      // Hoisted out of the frame loop: `lookup` reads through the closure and
      // never reads `view.time`, which is the one field that would go stale.
      const beatView = viewFor(scene, simulation, stepIndex, THEME);
      for (let frame = 0; frame < 240; frame += 1) {
        simulation.update(1 / 60, beatView.lookup);
        // Seconds since this beat began, which is what the player's `sim.t` is
        // and what an accent is defined against. See the counter below.
        const beatTime = (frame + 1) / 60;
        for (const id of simulation.danger.keys()) everDanger.add(id);
        // Checked every frame, not just at the end: the detachment only exists
        // while the car is actually dodging, and it is back in its lane by the
        // last frame of every step.
        const broken = brokenAttachment(scene, simulation);
        if (broken) {
          console.error(`\n❌ ${scene.id} / 第 ${frame} 帧：${broken}`);
          return 1;
        }
        const positions = new Map();
        for (const [id, node] of simulation.live) {
          const before = previous.get(id);
          if (before) {
            const length = ridden.get(id);
            // A traveller's distance is its progress along the link; everything
            // else moves by its own coordinates.
            const step =
              length > 0 && typeof node.progress === "number"
                ? Math.abs(node.progress - before.progress) * length
                : Math.hypot(node.x - before.x, node.y - before.y);
            if (step < MAX_STEP_PER_FRAME) travelled += step;
          }
          positions.set(id, { x: node.x, y: node.y, progress: node.progress });
          const startY = homeY.get(id);
          if (startY !== undefined) {
            maxLateral = Math.max(maxLateral, Math.abs(node.y - startY));
          }
          const heading = Number(beatView.lookup(id, "heading", node.heading ?? 0));
          const was = previousHeading.get(id);
          // Degrees, and only the size of the change: a body that swings back
          // and forth has turned twice as far as one that swung once, which is
          // the honest reading of "how much turning happened".
          if (Number.isFinite(was) && Number.isFinite(heading)) {
            turned += Math.abs(heading - was);
          }
          previousHeading.set(id, heading);

          // And how far an *accent* moved the pen, which is the third question
          // and the one that is invisible to both of the above.
          //
          // An accent is deliberately never written back into `live` — that is
          // the whole of `emphasis.js` — so a scene whose only motion is
          // `emphasis: "shake"` reads as a still frame to a position-and-heading
          // measure. The first `emphasis` demo printed "没有任何东西移动过"
          // over six beats of visible movement, which is the same false warning
          // the rotation counter above was added to stop, one layer further out.
          // On the *beat's* clock, not the simulation's, because that is the
          // clock the player uses: `setStep` restarts `sim.t` at zero, so every
          // beat gets its own accent. This harness runs one continuous clock
          // across all the beats (it never resets), so reading `simulation.t`
          // here would fire an accent once at the top of the scene and never
          // again — which is what the first run of this counter reported, as
          // `累计强调=0`. The frame count since the beat began is the same
          // quantity the player would have.
          const accent = emphasisAt(beatView.lookup(id, "emphasis", "none"), beatTime);
          // Three units added together — pixels, degrees and a scale fraction
          // (as a percentage). That is loose on purpose and it is worth saying
          // so: nothing reads the number, it exists only to answer "did
          // anything at all happen", and the threshold below is 1. A metric
          // that mixed units and was then *interpreted* would be worse than no
          // metric; one that only has to be non-zero is not.
          accented +=
            Math.abs(accent.dx) +
            Math.abs(accent.dy) +
            Math.abs(accent.rotate) +
            Math.abs(accent.scale) * 100;
        }
        previous = positions;
      }
      const view = viewFor(scene, simulation, stepIndex, THEME, {}, spec.glyphs ?? {});
      const beat = [];
      const shape = [];
      for (const element of scene.elements) {
        const ctx = fakeContext();
        try {
          drawElement(ctx, element, view);
        } catch (error) {
          console.error(
            `\n❌ ${scene.id} / ${scene.steps[stepIndex].id} / 元素 \`${element.id}\` ` +
              `(${element.kind}) 绘制失败：\n${error.message}`,
          );
          return 1;
        }
        if (ctx.calls.length === 0 && !switchedOff(element, view)) {
          console.error(
            `\n❌ ${scene.id} / 元素 \`${element.id}\` (${element.kind}) ` +
              `没发出任何绘制调用——它在画面上是隐形的，但不会报错`,
          );
          return 1;
        }
        const hole = firstHole(ctx.calls);
        if (hole) {
          console.error(
            `\n❌ ${scene.id} / ${scene.steps[stepIndex].id} / 元素 \`${element.id}\` ` +
              `(${element.kind}) 有一个参数是空的：${hole}\n` +
              "   画布不会为此报错。它照常画完，只是画错——多数是主题槽位或属性名写错了",
          );
          return 1;
        }
        beat.push(`${element.id}: ${serializeCalls(ctx.calls)}`);
        shape.push(`${element.id}: ${serializeCalls(ctx.calls, { geometryOnly: true })}`);
        drawn += 1;
      }
      pictures.push({
        id: scene.steps[stepIndex].id,
        calls: beat.join("\n"),
        geometry: shape.join("\n"),
      });
    }

    // Every beat must draw something different from the one before it.
    //
    // This is `step_state_inert`'s twin, one layer out: that rule checks the
    // contract, this checks what actually came out of the renderer. A beat that
    // repeats the previous picture is a slide, and the failure is invisible — no
    // exception, no missing element, just a step the viewer clicks through
    // without the screen doing anything.
    //
    // The number that motivated it, measured over the three sample documents:
    // 20 of 29 `object_states` targeted a prop no drawing code read, and 6 of the
    // 7 scenes came out as stills. Both halves are needed because they fail
    // differently — a rule can be right while the drawer ignores it, and a
    // drawer can read a prop while the value it was handed changes nothing.
    for (let index = 1; index < pictures.length; index += 1) {
      if (pictures[index].calls === pictures[index - 1].calls) {
        console.error(
          `\n❌ ${scene.id} / 节拍 \`${pictures[index].id}\` 画出来的画面与上一拍` +
            `（\`${pictures[index - 1].id}\`）逐调用完全相同——这一拍在屏幕上是静止的`,
        );
        return 1;
      }
    }

    // The same comparison again with the style assignments subtracted — a
    // second, stricter reading of "did this beat change the picture".
    //
    // **Reported, not asserted**, like `maxLateral` and `closestApproach` above,
    // and for a reason that is not squeamishness: the contract permits these
    // beats. `StoryboardStep` lets a beat consist of a highlight alone, and the
    // avoidance document's 系统组成 scene is exactly that on purpose — step 1
    // lights the whole system, step 2 lights the lidar, and nothing is supposed
    // to move. A harness that failed here would be stricter than the schema and
    // would reject legal output.
    //
    // It still has to be *measured*, because the failure this round was about
    // lives in exactly this blind spot. The projectile document's five beats
    // rotated the highlight between 球 / 坐标轴 / v₀ and never moved anything;
    // the weak reading called all five fine, and the only reason anyone noticed
    // was a person watching the screen. Printed with names here so the number
    // comes from this tool rather than from a one-off script — measured over
    // every spec under `data/`, 31 of 164 adjacent pairs are highlight-only, and
    // over the three current `r2-*` documents it is 1 of 38: the `hub` second
    // beat, which `docs/storyboard-milestone.md` already records as a hole.
    //
    // Counted in *adjacent pairs*, not beats. A scene's first beat has nothing
    // before it and can never be a repeat, so counting beats puts a floor under
    // the denominator that is not a measurement — which is exactly how the table
    // in the milestone doc first read `0/22` where the honest figure is `0/17`.
    const shapeStills = [];
    for (let index = 1; index < pictures.length; index += 1) {
      transitions += 1;
      if (pictures[index].geometry === pictures[index - 1].geometry) {
        shapeStill += 1;
        shapeStills.push(`${pictures[index - 1].id}→${pictures[index].id}`);
      }
    }
    if (shapeStills.length > 0) {
      console.log(
        `    ⚠️ 本幕 ${shapeStills.length}/${pictures.length - 1} 处相邻节拍几何与上一拍` +
          `完全相同，画面只换了高亮：${shapeStills.join("、")}` +
          `（契约允许，但整幕都是它，就成了一叠幻灯片）`,
      );
    }
    console.log(
      `  ${scene.id}（${scene.preset}）: ${scene.elements.length} 个元素 × ` +
        `${scene.steps.length} 拍全部绘制成功；` +
        // Absent on a spec written before beats carried one — the same `0` the
        // player itself reads as "no rhythm was ever chosen for this".
        (seconds > 0 ? `${seconds.toFixed(1)} 秒；` : "") +
        `t=${simulation.t.toFixed(1)}s ` +
        `运行中曾进入危险态的对象=${everDanger.size ? [...everDanger].join(",") : "无"}；` +
        `累计路程=${travelled.toFixed(1)}px；最大侧移=${maxLateral.toFixed(1)}px；` +
        `累计转角=${turned.toFixed(0)}°；累计强调=${accented.toFixed(0)}`,
    );
    // Reported, not asserted, on the same grounds as the still-beat count below:
    // a `hub` or `generic` scene has no role that walks, so a motionless one is
    // legal output rather than a bug. What it must not be is *invisible* — the
    // one scene of the avoidance document that nobody could stand to watch was
    // the one where every number this tool printed looked the same as the
    // reference sample's.
    // Turning counts as moving, and so does an accent. An `arm` scene is hinged
    // at its base and nothing in it ever translates, so a translation-only
    // reading called it dead while the arm was swinging through 105 degrees;
    // an `emphasis` scene translates and turns nothing at all, because an
    // accent is drawn rather than simulated. Both were warnings about scenes
    // that work, which is worse than no warning.
    if (travelled === 0 && turned < 1 && accented < 1) {
      console.log(
        "    ⚠️ 这一幕从头到尾没有任何东西移动过——对象只在原地亮灭。" +
          "契约允许，但分镜连着几幕都这样，观众看到的就是一叠会发光的幻灯片",
      );
    }

    // Clearance, at the two ends of the safe-distance slider. Reported, not
    // asserted — see `closestApproach`. A vehicle and an obstacle "touch" when
    // the gap drops below the sum of their half-extents.
    const defaultGap = closestApproach(scene, spec.stage, {});
    if (defaultGap) {
      const widest = maxSafeDistance(scene);
      let line = `    最近接近（默认安全距离）：${describeGaps(defaultGap)}`;
      if (widest !== null) {
        const wideGap = closestApproach(scene, spec.stage, { "scene.safe_distance": widest });
        line += `\n    最近接近（安全距离拉到滑杆上限 ${widest}）：${describeGaps(wideGap)}`;
        const penetrated = [...wideGap].filter(
          ([id, distance]) => distance < touchDistance(scene, id),
        );
        if (penetrated.length > 0) {
          line +=
            `\n    ⚠️ ${penetrated.map(([id]) => id).join("、")} 被车身穿过——` +
            "避让幅度是常数（`MAX_DODGE`），安全距离一拉大就再也出不了危险圈。" +
            "已知缺陷，见 docs/storyboard-milestone.md";
        }
      }
      console.log(line);
    }
  }
  const themeFailure = checkThemeSlots();
  if (themeFailure) {
    console.error(`\n❌ ${themeFailure}`);
    return 1;
  }

  const easingFailure = checkEasing();
  if (easingFailure) {
    console.error(`\n❌ 缓动：${easingFailure}`);
    return 1;
  }

  const formsFailure = checkTreeForms(spec.glyphs ?? {});
  if (formsFailure) {
    console.error(`\n❌ ${formsFailure}`);
    return 1;
  }

  const spinFailure = checkSpinningPartsTurn();
  if (spinFailure) {
    console.error(`\n❌ ${spinFailure}`);
    return 1;
  }

  const laneFailure = checkALaneBodyHoldsItsLane();
  if (laneFailure) {
    console.error(`\n❌ ${laneFailure}`);
    return 1;
  }

  const arcFailure = checkAProjectileKeepsItsArc();
  if (arcFailure) {
    console.error(`\n❌ ${arcFailure}`);
    return 1;
  }

  const pivotFailure = checkABodyTurnsAboutItsPivot();
  if (pivotFailure) {
    console.error(`\n❌ ${pivotFailure}`);
    return 1;
  }

  const curveFailure = checkEveryEasingCurve();
  if (curveFailure) {
    console.error(`\n❌ ${curveFailure}`);
    return 1;
  }

  const emphasisFailure = checkEveryEmphasisReturnsToRest();
  if (emphasisFailure) {
    console.error(`\n❌ ${emphasisFailure}`);
    return 1;
  }

  const accentFailure = checkEmphasisMovesThePenAndNotTheBody();
  if (accentFailure) {
    console.error(`\n❌ ${accentFailure}`);
    return 1;
  }

  const captionFailure = checkCaptionWrapping();
  if (captionFailure) {
    console.error(`\n❌ ${captionFailure}`);
    return 1;
  }

  console.log(
    `\n✅ 共绘制 ${drawn} 次，无异常。` +
      `强口径静拍：${shapeStill}/${transitions} 处相邻节拍几何与上一拍完全相同。` +
      // The number the one-minute target is measured against, printed where
      // nobody has to add it up by hand.
      (totalSeconds > 0 ? `全片时长：${totalSeconds.toFixed(1)} 秒。` : ""),
  );
  return 0;
}

/**
 * A part marked `spin` has to actually turn, and no sample document can show it.
 *
 * None of the three hand-written samples gives a body the `lidar` glyph — the
 * lidar in the avoidance sample is an `emitter`, which draws a fan and never
 * calls the glyph drawer at all — so the sweep arm is data no spec in this repo
 * exercises. Without this check, `anchor` and `spin` would be two more fields
 * declared, validated, shipped, and read by nothing, which is the exact failure
 * this round exists to close.
 *
 * So the drawer is driven directly: one body carrying the lidar glyph, drawn at
 * two different times, and the two *pictures* compared rather than the code
 * inspected. Same standard as everything else here — what came out of the
 * renderer, not what the source says it does.
 *
 * The glyph is read from `assets/glyphs/` rather than from a spec, which is the
 * one place that is right: the player must resolve glyphs out of the spec it was
 * handed, but a harness checking the *asset* is exactly the thing that should
 * open the asset.
 */
/**
 * The easing arithmetic, run rather than described.
 *
 * `easing.js` is DOM-free precisely so this is possible — `player.js` reaches
 * for `document` on load and nothing in it can be executed from here. Without
 * this the rule "a beat boundary is not a step function" would be checkable
 * only by watching the screen, which is the gap the whole round came out of:
 * the player's own timing was the one rule in the pipeline nobody could test.
 *
 * Returns a failure message, or `null`.
 */
function checkEasing() {
  // Shaped at both ends and monotone between them. A curve that overshoots
  // would send a radius past its target and back, which reads as a wobble.
  if (easeOutCubic(0) !== 0) return "easeOutCubic(0) 不是 0：属性会从半路开始";
  if (easeOutCubic(1) !== 1) return "easeOutCubic(1) 不是 1：属性永远差一点到不了目标值";
  if (easeOutCubic(0.5) <= 0.5) {
    return "easeOutCubic 在走到一半时还没过半：这是 ease-in，不是 ease-out";
  }
  let previous = -1;
  for (let i = 0; i <= 10; i += 1) {
    const value = easeOutCubic(i / 10);
    if (value < previous) return "easeOutCubic 不是单调的：属性会往回走";
    previous = value;
  }

  // A number travels. At the ends it is exact, or a beat that finished easing
  // would be drawing a hair off the value the spec actually wrote.
  if (blend("speed", 1.0, 0.5, 0) !== 1.0) return "blend 在起点不是旧值";
  if (blend("speed", 1.0, 0.5, 1) !== 0.5) return "blend 在终点不是新值";
  if (Math.abs(blend("speed", 1.0, 0.5, 0.5) - 0.75) > 1e-9) return "blend 在中点不是两端的平均";

  // The two presence gates travel as 1 and 0, and hand back a *number* for the
  // whole journey — that is what makes a drawer's `=== false` let a half-arrived
  // element through so `drawElement` can fade it in. Landing on the exact
  // boolean is the other half: at rest the gate must be a gate, not 0.9999.
  if (blend("enabled", false, true, 0.5) !== 0.5) return "enabled 的过渡不是 1/0 之间的数";
  if (blend("visible", true, false, 1) !== false) return "visible 落在终点时不是精确的 false";
  if (blend("visible", false, true, 0) !== false) return "visible 落在起点时不是精确的 false";
  if (blend("visible", true, false, 0) !== true) return "visible 落在起点时不是精确的 true";

  // Everything else is a style or a word, and is left exactly as written —
  // a `danger` of 0.4 is truthy, which would draw a danger state that no beat
  // asked for.
  if (blend("danger", false, true, 0.5) !== true) return "布尔样式被过渡了：danger 会半亮";
  if (blend("text", "直行", "转向", 0.5) !== "转向") return "文本被过渡了";
  if (blend("state", "待发布", "已发布", 0.5) !== "已发布") return "令牌状态被过渡了";

  // A beat change hands over only when there is a last beat to hand over
  // *from* — so only while playing, and only to a different beat. The paused
  // half is the one that was missing, and it is not a one-frame artefact: the
  // departing values were recorded and the clock was reset, `blend(…, 0)` hands
  // back the old value by design, and with the clock stopped nothing ever moved
  // it on. Every later beat drew the first beat's values, for as long as the
  // player stayed paused. Reported from the browser as a code block that kept
  // the previous beat's emphasis band no matter which beat was selected.
  if (handsOver(false, false)) return "暂停时换拍还想做渐变：新值永远到不了，画面停在第一拍";
  if (!handsOver(true, false)) return "播放时换拍不渐变了：每一拍之间都硬切";
  if (handsOver(true, true)) return "重放同一拍还做渐变：重置变成慢慢滑回去";
  if (handsOver(false, true)) return "暂停时重放同一拍做了渐变";

  if (!(EASE_SECONDS > 0.3 && EASE_SECONDS < 1.0)) {
    return `EASE_SECONDS = ${EASE_SECONDS}：短于 0.3 秒像抽搐，长于一秒就在花这一拍自己的时间`;
  }
  return null;
}

/**
 * The five tree forms draw five *different* pictures.
 *
 * Layout already measures each form's geometry and the Python tests hold those
 * numbers. What nothing else can reach is the *wiring*: the drawer dispatches on
 * `element.form`, so a form whose branch was deleted falls through to the
 * outline and draws a perfectly plausible picture that is not the one the
 * vocabulary offered. That is the `tone` defect — a name the model was given
 * with nothing behind it — at a much larger blast radius, because a whole scene
 * silently becomes a different picture rather than a colour going wrong.
 *
 * Compared as recorded calls rather than as pixels. The harness has no
 * rasteriser, and "these five drew differently from one another" is the property
 * that survives not having one — it is exactly the property a missing branch
 * destroys, and nothing weaker than a real comparison finds it.
 */
function checkTreeForms(glyphs) {
  const icon = "flask" in glyphs ? "flask" : "";
  const base = {
    id: "t",
    kind: "tree",
    role: "structure",
    label: "结构",
    x: 480,
    y: 300,
    width: 420,
    height: 240,
    focus: "",
    tone: "normal",
    lines: [
      { depth: 0, prefix: "", text: "实验配置", value_kind: "text", icon: "", x: 0, y: 13 },
      { depth: 1, prefix: "器材: ", text: '"器材"', value_kind: "string", icon, x: 22, y: 39 },
      { depth: 2, prefix: "", text: "小球", value_kind: "text", icon: "", x: 44, y: 65 },
      { depth: 1, prefix: "", text: "参数", value_kind: "text", icon: "", x: 22, y: 91 },
      { depth: 2, prefix: "角度: ", text: "45", value_kind: "number", icon: "", x: 44, y: 117 },
    ],
  };
  // A view built by hand rather than by `viewFor`: this draws one element at one
  // instant through no simulation at all, and standing up a scene and a
  // simulation in order to hand it `presence: () => 1` is ceremony around a
  // constant.
  const view = {
    live: null,
    time: 0,
    theme: THEME,
    elementById: new Map(),
    obstacleIds: [],
    obstacleRadius: new Map(),
    glyphs,
    highlighted: new Set(),
    isDangerous: () => false,
    lookup: (id, prop, fallback) => fallback,
    presence: () => 1,
  };

  const drawn = new Map();
  for (const form of ["outline", "branch", "mind", "brace", "boxes"]) {
    const ctx = fakeContext();
    try {
      drawElement(ctx, { ...base, form }, view);
    } catch (error) {
      return `结构树的 \`${form}\` 形态画不出来：${error.message}`;
    }
    drawn.set(form, JSON.stringify(serializeCalls(ctx.calls, { geometryOnly: true })));
  }
  const forms = [...drawn.keys()];
  for (let i = 0; i < forms.length; i += 1) {
    for (let j = i + 1; j < forms.length; j += 1) {
      if (drawn.get(forms[i]) === drawn.get(forms[j])) {
        return `结构树的 \`${forms[i]}\` 和 \`${forms[j]}\` 画出来一模一样：其中一个形态根本没接线`;
      }
    }
  }
  return null;
}

function checkSpinningPartsTurn() {
  const path = new URL("../assets/glyphs/lidar.json", import.meta.url);
  let glyph;
  try {
    glyph = JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    return `读不出 ${path.pathname}：${error.message}`;
  }
  const sweep = glyph.parts?.sweep;
  if (!sweep?.spin || !sweep?.anchor) {
    return "lidar 字形的 sweep part 没有 spin/anchor——雷达就不会扫，而画面看起来和它有意的静止一样";
  }

  const element = {
    id: "sensor-unit",
    kind: "body",
    role: "device",
    label: "2D 激光雷达",
    x: 100,
    y: 100,
    tone: "normal",
    props: {},
    binds: {},
    shape: "rect",
    width: 72,
    height: 56,
    sides: 6,
    inner_ratio: 1,
    heading: 0,
    glyph: "lidar",
    path: [],
    duration: 0,
  };
  const drawAt = (time) => {
    const ctx = fakeContext();
    drawElement(ctx, element, {
      live: new Map([[element.id, { x: element.x, y: element.y }]]),
      time,
      theme: THEME,
      elementById: new Map([[element.id, element]]),
      obstacleIds: [],
      obstacleRadius: new Map(),
      glyphs: { lidar: glyph },
      highlighted: new Set(),
      isDangerous: () => false,
      lookup: (_id, _prop, fallback) => fallback,
      // `drawElement` calls this unconditionally, so a view without it throws
      // before a single drawer runs. The ramp itself is the player's, and there
      // is no player here.
      presence: () => 1,
    });
    return serializeCalls(ctx.calls);
  };

  const start = drawAt(0);
  const later = drawAt(2);
  if (start === later) {
    return "lidar 的扫描臂在 t=0 与 t=2 画出的调用序列完全相同——标了 spin 却没转起来";
  }

  // Counting `rotate(` alone would not be a check. `drawBody` always writes
  // `ctx.rotate(heading)` — at `heading: 0` that is `rotate(0)`, so the substring
  // is present whether or not the arm turns. What is asserted is the *angle*:
  // two of them, the heading's zero and the arm's, and the arm's proportional to
  // simulated time.
  //
  // Proportionality is the part that matters. A single non-zero angle would also
  // be produced by a constant, by a random number, or by an accumulator that
  // drifts — and "the same spec draws the same frames" is the property the whole
  // pipeline is built around. Doubling the time has to double the angle.
  const spinAngles = (picture) =>
    [...picture.matchAll(/rotate\(([-\d.eE+]+)\)/g)]
      .map((match) => Number(match[1]))
      .filter((angle) => angle !== 0);

  const none = spinAngles(start);
  const single = spinAngles(later);
  if (none.length !== 0) {
    return `t=0 时扫描臂已经转了（角度 ${none.join("、")}）——每一步节拍都从 t=0 重来，起点必须是它被画出来的样子`;
  }
  if (single.length !== 1) {
    return `t=2 时预期恰好一个非零的 rotate（扫描臂的），实际有 ${single.length} 个：${single.join("、")}`;
  }
  const doubled = spinAngles(drawAt(4))[0];
  if (Math.abs(doubled - 2 * single[0]) > 1e-9) {
    return `扫描臂的角度不是按模拟时间线性来的：t=2 是 ${single[0]}，t=4 是 ${doubled}，后者不是前者的两倍`;
  }
  console.log(
    `  ✓ 旋转 part：lidar 的扫描臂绕 anchor 转，t=2 时 ${single[0].toFixed(3)} rad，` +
      `t=4 时 ${doubled.toFixed(3)} rad（随模拟时间线性，可复现）`,
  );
  return null;
}

/** A body element with the fields `drawBody` and `behaviors.js` reach for. */
function bodyElement(overrides = {}) {
  return {
    id: "car",
    kind: "body",
    role: "vehicle",
    label: "小车",
    x: 120,
    y: 310,
    tone: "normal",
    props: { speed: 1 },
    binds: {},
    shape: "rect",
    width: 62,
    height: 40,
    sides: 6,
    inner_ratio: 1,
    heading: 0,
    glyph: null,
    path: [],
    duration: 0,
    pivot: [0, 0],
    ...overrides,
  };
}

/** A `lane` scene holding one body, with no obstacles and so no danger. */
function laneScene(element) {
  return {
    preset: "lane",
    attachment: {},
    thresholds: {},
    params: {},
    elements: [element],
  };
}

/** A lookup that answers from the element's own authored props. */
function authoredLookup(element) {
  return (_id, prop, fallback) => element.props?.[prop] ?? fallback;
}

/**
 * Where a body ends up after `seconds` of a lane preset, starting from rest.
 */
function travel(element, seconds) {
  const simulation = createSimulation(laneScene(element), STAGE);
  simulation.reset();
  simulation.update(seconds, authoredLookup(element));
  return simulation.live.get(element.id);
}

const STAGE = { width: 960, height: 600 };

/** Stage pixels per second at `speed === 1`. Mirrors `behaviors.js`. */
const PIXELS_PER_SPEED = 90;

/**
 * A lane body holds its lane, whatever way its nose points.
 *
 * This is the inverse of the check that used to sit here, and the inversion is
 * the whole content of it. `heading` was made to steer — `x`/`y` decomposed
 * through `cos`/`sin`, with a corridor clamp as a backstop — and the arithmetic
 * was right while the playback was worse: a constant facing is a straight line
 * *forever*, so 转向绕行 walked the car up out of its lane and along the ceiling.
 * Reverted by decision. See the `behaviors.js` docstring for why.
 *
 * So what is asserted is the property the revert restores: the travelled
 * direction is +x and the lane line survives, **for every facing**. A future
 * change that reintroduces steering fails here by name, instead of being caught
 * by somebody noticing a drift three beats into a render.
 *
 * A remembered-coordinate assertion would not do that. `(587.7, 40.0)` tells you
 * that *some* arithmetic ran; "y never left the lane, at any angle" tells you
 * which arithmetic, and it keeps holding if the stage, the speed constant or the
 * start position ever move.
 */
function checkALaneBodyHoldsItsLane() {
  // 90 is in the list on purpose. It is the crab walk in its purest form — a
  // car pointing straight down the screen, still driving right — and writing it
  // down as an *expected* result is how the trade stays visible.
  for (const heading of [0, -30, 30, 75, -80, 90]) {
    const element = bodyElement({ heading });
    const node = travel(element, 3);
    // 3s at speed 1 is 270 units on a 960-wide stage, so nothing here is near
    // the horizontal wrap and a failure cannot be blamed on it.
    const expected = element.x + 3 * PIXELS_PER_SPEED;
    if (Math.abs(node.x - expected) > 1e-9) {
      return (
        `朝向 ${heading}° 时 x 不是沿车道匀速走的：x = ${node.x.toFixed(1)}，` +
        `应为 ${expected.toFixed(1)}`
      );
    }
    if (node.y !== element.y) {
      return (
        `朝向 ${heading}° 时小车离开了车道线：y = ${node.y.toFixed(1)}，` +
        `作者写的是 ${element.y}` +
        (heading === 0 ? "（朝向 0 也偏，说明不是朝向的问题）" : "——朝向又驱动位移了")
      );
    }
  }

  console.log(
    "  ✓ 车道线不受朝向影响：0/-30/30/75/-80/90 度走 3 秒，x 都是 +270、y 都停在 310" +
      "（转向只转车头、不改路线——这是撤掉的甲，不是漏掉的）",
  );
  return null;
}

/**
 * A baked arc has to actually be flown, not only travelled along in x.
 *
 * This is a defect that was already in the player and not introduced with
 * `heading`: the clearance pass wrote `node.y = base.y + dodge` for *every*
 * body, reading from the element's authored position rather than from where
 * motion had just put it. A `field` projectile therefore crossed the frame at
 * its launch height, tracing a straight line while `layout.py` had sampled a
 * parabola for it and the `trace` element drew the parabola underneath.
 *
 * Checked by flying one and asking whether y ever left the launch height. A
 * flat line is the failure, and it is the failure that reports success.
 */
function checkAProjectileKeepsItsArc() {
  const element = bodyElement({
    id: "ball",
    role: "projectile",
    props: {},
    path: [
      { x: 100, y: 500 },
      { x: 300, y: 200 },
      { x: 500, y: 500 },
    ],
    duration: 2,
    x: 100,
    y: 500,
  });
  const scene = { preset: "field", attachment: {}, thresholds: {}, params: {}, elements: [element] };
  const simulation = createSimulation(scene, STAGE);
  simulation.reset();

  let peak = Number.POSITIVE_INFINITY;
  for (let frame = 0; frame < 60 * 2; frame += 1) {
    simulation.update(1 / 60, authoredLookup(element));
    peak = Math.min(peak, simulation.live.get("ball").y);
  }

  if (!(peak < element.y - 1)) {
    return (
      `抛体全程没有离开出手高度：最低 y = ${peak.toFixed(1)}，出手在 y = ${element.y}。` +
      "它画出来的是一条直线，而 layout 已经为它采样过抛物线了"
    );
  }
  console.log(`  ✓ 抛体真的飞了弧线：出手 y=${element.y}，最高点 y=${peak.toFixed(1)}`);
  return null;
}

/**
 * A body marked as hinged has to turn about the hinge, not about its middle.
 *
 * `glyphs.js` has had the standard pivot rotation — translate, rotate,
 * translate back — since the lidar's sweep arm, but it was reachable only from
 * a glyph *part*. A body turned about its own centre, which is right for a car
 * and wrong for an arm: pivot a stick through its middle and it spins, pivot it
 * at one end and it swings.
 *
 * Checked on the recorded calls, because that is where the difference lives.
 * The two translates bracketing the rotation *are* the pivot, and the pivot is
 * by definition the point the rotation leaves where it was — for `(0, 0)` the
 * pair cancels and nothing is said, which is why the vehicle is checked for
 * exactly that and the arm for exactly not that.
 */
function checkABodyTurnsAboutItsPivot() {
  const drawWith = (element, time) => {
    const ctx = fakeContext();
    drawElement(ctx, element, {
      live: new Map([[element.id, { x: element.x, y: element.y, heading: element.heading }]]),
      time,
      theme: THEME,
      elementById: new Map([[element.id, element]]),
      obstacleIds: [],
      obstacleRadius: new Map(),
      glyphs: {},
      highlighted: new Set(),
      isDangerous: () => false,
      lookup: (_id, prop, fallback) => (prop === "heading" ? element.heading : fallback),
      presence: () => 1,
    });
    return ctx.calls;
  };

  /** The `translate` immediately before `rotate`, and the one immediately after. */
  function pivotPair(calls) {
    const index = calls.findIndex(({ name }) => name === "rotate");
    if (index <= 0 || index + 1 >= calls.length) return null;
    const before = calls[index - 1];
    const after = calls[index + 1];
    if (before.name !== "translate" || after.name !== "translate") return null;
    return { before: before.args, after: after.args };
  }

  const car = bodyElement({ heading: 45, pivot: [0, 0] });
  const carPair = pivotPair(drawWith(car, 0));
  if (!carPair) return "车身旋转前后不是一对 translate：支点写法变了";
  if (carPair.before[0] !== 0 || carPair.before[1] !== 0) {
    return `车身的支点不是中心，而是 (${carPair.before.join(", ")})——车会绕着车身外面的一点转`;
  }

  // The arm's pivot is baked by layout as `(0, height / 2)`: the middle of its
  // bottom edge. Any non-zero point would prove the offset exists; this one
  // proves it is the shoulder.
  const arm = bodyElement({ role: "arm", heading: 45, width: 34, height: 96, pivot: [0, 48] });
  const armPair = pivotPair(drawWith(arm, 0));
  if (!armPair) return "机械臂旋转前后不是一对 translate：支点写法变了";
  if (armPair.before[0] !== 0 || armPair.before[1] !== 48) {
    return `机械臂绕着 (${armPair.before.join(", ")}) 转，而 layout 烘的是 (0, 48)`;
  }
  if (armPair.after[0] !== 0 || armPair.after[1] !== -48) {
    return `机械臂的第二段 translate 是 (${armPair.after.join(", ")})，不是支点的相反数——它不是绕支点转的`;
  }

  // And it has to be visible in the picture, not only in the offsets: two
  // headings must produce two different drawings.
  const still = serializeCalls(drawWith(arm, 0));
  const swung = serializeCalls(drawWith({ ...arm, heading: 120 }, 0));
  if (still === swung) {
    return "机械臂在 45° 与 120° 画出的调用序列完全相同——支点烘好了，但没转到画面上";
  }

  console.log(
    "  ✓ 支点旋转：车身绕中心（0, 0），机械臂绕底边中点 (0, 48)，两者都没有画成自转",
  );
  return null;
}

/**
 * Every curve has to be a curve: 0 at 0, 1 at 1, and monotone if it says so.
 *
 * The endpoints are not a formality. `easing.js` uses these two ways — as a
 * *progress* for a beat transition, and as an *envelope* (`1 - curve(u)`) for
 * an accent — and both use the endpoints as their guarantee that nothing is
 * left over. A curve that started at 0.02 would leave a beat permanently 2%
 * short of the value the spec wrote; one that ended at 0.98 would leave an
 * accent permanently 2% short of home, which is the permanent-offset failure
 * `emphasis.js` was written to be incapable of.
 *
 * Monotonicity is listed by name rather than assumed, and **both directions are
 * checked**: the ones named here must be monotone, and the ones not named must
 * not be. That second half is what stops a new curve from being added and
 * silently inheriting a property nobody checked — the failure would otherwise
 * be a `pop` that does not pop.
 */
function checkEveryEasingCurve() {
  const MONOTONE = ["easeOutQuad", "easeOutCubic"];

  for (const [name, fn] of Object.entries(EASINGS)) {
    if (fn(0) !== 0) return `${name}(0) = ${fn(0)}，不是 0：它会从半路开始`;
    if (fn(1) !== 1) return `${name}(1) = ${fn(1)}，不是 1：它永远差一点到不了`;

    let previous = -Infinity;
    let rose = false;
    let fell = false;
    for (let i = 0; i <= 20; i += 1) {
      const value = fn(i / 20);
      if (value > previous) rose = true;
      if (value < previous) fell = true;
      previous = value;
    }
    if (MONOTONE.includes(name) && fell) {
      return `${name} 被列为单调，实际会往回走——顶点会冲过目标再退回来`;
    }
    if (!MONOTONE.includes(name) && !fell) {
      return (
        `${name} 会冲出 [0, 1] 正是它存在的理由，但它现在是单调的。` +
        "要么这个曲线名不副实，要么它该被加进 MONOTONE 里"
      );
    }
    if (!rose) return `${name} 全程没上升过`;
  }

  // An unknown curve name is a broken *table*, not a bad document, and the two
  // are answered differently on purpose. See `curve()` in `easing.js`.
  let threw = false;
  try {
    curve("banana");
  } catch {
    threw = true;
  }
  if (!threw) return "curve() 对没注册的名字没有报错——emphasis.js 里写错曲线会静默变成 NaN";

  console.log(
    `  ✓ 缓动曲线：${Object.keys(EASINGS).length} 条都在 (0,0)/(1,1) 上，` +
      `${MONOTONE.join("、")} 单调，其余会过冲（那是它们的作用）；名字写错会报错`,
  );
  return null;
}

/**
 * Every accent has to end where it started, and has to have been somewhere.
 *
 * The "back to zero" half is the correction this whole area earned the hard
 * way: steering by `heading` was arithmetically correct and visually wrong
 * because a constant facing moves a body forever. An accent that did not decay
 * to exactly zero would be the same defect with a smaller amplitude — a car
 * that never quite returns to its lane — so it is asserted at every end: before
 * the accent starts, at the instant it ends, and long after.
 *
 * The "was somewhere" half is the guard on the guard. `emphasisAt` returning
 * the resting pose for everything would satisfy every assertion above it, and
 * would mean a misspelled preset, a wrong axis name or an empty `axes` object
 * shipped as "the accent is implemented". Six motions that do nothing is
 * exactly the failure mode of a table like this one.
 */
function checkEveryEmphasisReturnsToRest() {
  for (const name of EMPHASIS_NAMES) {
    for (const t of [0, -1, EMPHASIS_SECONDS, EMPHASIS_SECONDS + 5]) {
      if (!isRestPose(emphasisAt(name, t))) {
        const pose = emphasisAt(name, t);
        return (
          `\`${name}\` 在 t=${t} 时没有归位（${JSON.stringify(pose)}）——` +
          "强调会在画面里留下一处永久的偏移"
        );
      }
    }
    if (name === "none") continue;

    let moved = false;
    for (let i = 1; i < 20; i += 1) {
      if (!isRestPose(emphasisAt(name, (EMPHASIS_SECONDS * i) / 20))) moved = true;
    }
    if (!moved) return `\`${name}\` 从头到尾都是静止姿势——这个名字挂在表上，但没有动作`;
  }

  // A name the player does not know is `none`, not a throw: it came from a
  // spec, and a body without its accent is still a body.
  if (!isRestPose(emphasisAt("banana", 0.3))) {
    return "没注册的 emphasis 名没有被当成 none——播放器会画出一个表里没有的姿势";
  }

  console.log(
    `  ✓ 强调动作：${EMPHASIS_NAMES.length - 1} 个都在 t=0 与 t=${EMPHASIS_SECONDS}s 归位、` +
      "中间确实动过；没注册的名字当成 none",
  );
  return null;
}

/**
 * An accent moves the pen, not the body — checked, not promised.
 *
 * This is the distinction `emphasis.js` exists to keep, and it is one line of
 * code away from being lost: `drawBody` has `view.live` in hand, and writing
 * the shake back into `node.x` would make the body *actually* shake — which
 * would then feed `proximity_gate` and the dodge on the next frame, so a car
 * being emphasised near an obstacle would set off its own danger flag. Nothing
 * would throw. The picture would even look right.
 *
 * So: snapshot the simulation, draw, snapshot again. Drawing may not change a
 * single live number.
 */
function checkEmphasisMovesThePenAndNotTheBody() {
  const element = bodyElement({ props: { emphasis: "shake" } });
  const scene = laneScene(element);
  scene.steps = [{ id: "beat", states: {} }];
  const simulation = createSimulation(scene, STAGE);
  simulation.update(0.2, authoredLookup(element));

  const snapshot = () =>
    JSON.stringify([...simulation.live].map(([id, node]) => [id, node.x, node.y, node.heading]));

  // The resting pose, for comparison. `t = 0` is inside the accent window, so
  // this is the accent at the instant `shape` is zero — which is also what a
  // body with no `emphasis` at all draws, and that equivalence is the point of
  // decay-to-zero rather than a coincidence.
  simulation.t = 0;
  const restCalls = serializeCalls(
    (() => {
      const ctx = fakeContext();
      drawElement(ctx, element, viewFor(scene, simulation, 0, THEME));
      return ctx.calls;
    })(),
  );

  let differing = 0;
  for (let i = 1; i < 18; i += 1) {
    const t = (EMPHASIS_SECONDS * i) / 18;
    simulation.t = t;
    const before = snapshot();
    const ctx = fakeContext();
    drawElement(ctx, element, viewFor(scene, simulation, 0, THEME));
    const after = snapshot();
    if (before !== after) {
      return (
        `画这一拍（t=${t.toFixed(2)}s）的时候改动了 live 里的坐标——` +
        "强调必须只动画笔，动了身体就会被避障读到"
      );
    }
    if (serializeCalls(ctx.calls) !== restCalls) differing += 1;
  }

  if (differing === 0) {
    return "整段强调画出来的调用序列和静止姿势一模一样——算术在动，画面没动";
  }

  console.log(
    `  ✓ 强调只动画笔：17 帧里 ${differing} 帧画得和静止姿势不同，` +
      "live 里的坐标一个都没变",
  );
  return null;
}

function isRestPose(pose) {
  return pose.dx === 0 && pose.dy === 0 && pose.rotate === 0 && pose.scale === 0;
}

process.exit(main());
