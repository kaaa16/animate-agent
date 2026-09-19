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
import { EASE_SECONDS, blend, easeOutCubic } from "../frontend/player/easing.js";
import { drawElement } from "../frontend/player/registry.js";
import { fitTransform } from "../frontend/player/stage.js";

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

const THEME = {
  background: "#07080d",
  panel: "rgba(13,18,28,0.86)",
  line: "rgba(83,246,255,0.32)",
  lineHot: "#42f7ff",
  magenta: "#ff2fd6",
  yellow: "#f4f06d",
  green: "#5dff9c",
  text: "#ecfbff",
  muted: "#91a6b8",
  danger: "#ff5e7a",
};

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
    let previous = new Map();
    const ridden = linkLengths(scene);
    const seconds = scene.steps.reduce((sum, step) => sum + Number(step.duration ?? 0), 0);
    totalSeconds += seconds;
    for (let stepIndex = 0; stepIndex < scene.steps.length; stepIndex += 1) {
      // Advance far enough for the simulation to have moved things, so a drawer
      // that only works at t=0 does not pass by luck. Danger is sampled *during*
      // the run rather than read at the end: the car passes its obstacles and is
      // safe again by then, so the final frame alone would report nothing.
      for (let frame = 0; frame < 240; frame += 1) {
        simulation.update(1 / 60, viewFor(scene, simulation, stepIndex, THEME).lookup);
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
        `累计路程=${travelled.toFixed(1)}px；最大侧移=${maxLateral.toFixed(1)}px`,
    );
    // Reported, not asserted, on the same grounds as the still-beat count below:
    // a `hub` or `generic` scene has no role that walks, so a motionless one is
    // legal output rather than a bug. What it must not be is *invisible* — the
    // one scene of the avoidance document that nobody could stand to watch was
    // the one where every number this tool printed looked the same as the
    // reference sample's.
    if (travelled === 0) {
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
  const easingFailure = checkEasing();
  if (easingFailure) {
    console.error(`\n❌ 缓动：${easingFailure}`);
    return 1;
  }

  const spinFailure = checkSpinningPartsTurn();
  if (spinFailure) {
    console.error(`\n❌ ${spinFailure}`);
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

  if (!(EASE_SECONDS > 0.3 && EASE_SECONDS < 1.0)) {
    return `EASE_SECONDS = ${EASE_SECONDS}：短于 0.3 秒像抽搐，长于一秒就在花这一拍自己的时间`;
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

process.exit(main());
