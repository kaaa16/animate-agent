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
import { drawElement } from "../frontend/player/registry.js";
import { fitTransform } from "../frontend/player/stage.js";

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
  // Style properties are assigned and never read back by the drawers — but they
  // are recorded, because a highlight *is* a style change. `applyHighlight` sets
  // `shadowColor`/`shadowBlur` and nothing else, so a plain field would swallow
  // the only thing most beats do, and the "no two beats alike" check below would
  // call every highlight-only beat a still. It did, on the first run: the
  // avoidance document's 系统四大组成 and 前方180度扫描 beats differ in which
  // elements glow and in nothing else.
  for (const name of [
    "fillStyle",
    "strokeStyle",
    "lineWidth",
    "globalAlpha",
    "shadowColor",
    "shadowBlur",
    "font",
    "textAlign",
    "textBaseline",
  ]) {
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
function viewFor(scene, simulation, stepIndex, theme, overrides = {}) {
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
function serializeCalls(calls) {
  return calls.map(({ name, args }) => `${name}(${args.map(String).join(",")})`).join("|");
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
        for (const [id, startY] of homeY) {
          const node = simulation.live.get(id);
          if (!node) continue;
          maxLateral = Math.max(maxLateral, Math.abs(node.y - startY));
        }
      }
      const view = viewFor(scene, simulation, stepIndex, THEME);
      const beat = [];
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
        drawn += 1;
      }
      pictures.push({ id: scene.steps[stepIndex].id, calls: beat.join("\n") });
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
    console.log(
      `  ${scene.id}（${scene.preset}）: ${scene.elements.length} 个元素 × ` +
        `${scene.steps.length} 拍全部绘制成功；t=${simulation.t.toFixed(1)}s ` +
        `运行中曾进入危险态的对象=${everDanger.size ? [...everDanger].join(",") : "无"}；` +
        `最大侧移=${maxLateral.toFixed(1)}px`,
    );

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
  console.log(`\n✅ 共绘制 ${drawn} 次，无异常。`);
  return 0;
}

process.exit(main());
