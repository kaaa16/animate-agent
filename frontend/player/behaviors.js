/**
 * Behaviours: the deterministic local simulation that makes a picture move.
 *
 * These are the JS half of `rendering/registry.py`'s `BEHAVIORS` table. That
 * table names what each behaviour *reads*, which is what makes the project rule
 * "an interactive parameter must change the result, not just a number in the UI"
 * machine-checkable. Here the reads become arithmetic.
 *
 * What decides which behaviour applies is derived, never hand-listed:
 *
 * | element                                    | behaviour          |
 * |--------------------------------------------|--------------------|
 * | `body` with a baked `path` and a `duration` | `ballistic`        |
 * | `body` carrying a numeric `speed`           | `linear_motion`    |
 * | any body, against the obstacles around it   | `proximity_gate`   |
 * | a body that just went dangerous             | `clearance_choice` |
 * | `traveler` bound to a `path_id`             | `hop_along_path`   |
 *
 * A body with no `speed` and no `path` simply does not move — not a special
 * case, just the absence of what either reads.
 *
 * **No LLM runs here, ever.** Playback is pure arithmetic on the spec, which is
 * why the same spec always produces the same frames.
 *
 * The two motions are genuinely different and the difference is where the
 * arithmetic lives. `linear_motion` is a **lane** model — advance along +x at
 * `speed`, offset sideways to dodge — and the player computes it. `ballistic`
 * moves a body along a polyline `layout.py` already sampled from the parabola a
 * throw at that angle would follow; this file only interpolates. That is
 * decision D3 applied to motion: the curve is the code's job, and the player's
 * job is to be somewhere along it at time `t`.
 *
 * **`heading` does not steer, and that is a decision, not an oversight.** It
 * steered once: `x` and `y` decomposed through `cos`/`sin`, with a corridor
 * clamped to `[0.06, 0.94]` of the stage height as a backstop. The arithmetic
 * worked and the picture was worse. A constant facing is a *straight line
 * forever*, so the avoidance car's 转向绕行 beat sent it climbing steadily out
 * of its lane and then sliding along the ceiling — a drift nobody asked for, on
 * a beat whose point was a detour. The corridor did not cause that; it was the
 * arithmetic underneath, and clamping it only decided where the drift stopped.
 *
 * The cost of reverting is real and worth naming: 转向绕行 now turns the car's
 * nose and leaves it driving the way it was already going — a crab walk, with
 * the arrow pointing at nothing. That is the accepted trade. A lane is a lane:
 * a body holds its line, and turning is a change of appearance, not of route.
 * Steering a *route* is what `ballistic` and a baked `path` are for.
 */

import { pointAlong } from "./primitives.js";
import { SPEED_PRESETS } from "./registry.js";

/** Stage pixels travelled per second at `speed === 1`. */
const PIXELS_PER_SPEED = 90;

/** How far sideways `clearance_choice` may push a body, in stage units. */
const MAX_DODGE = 46;

/** Stage units per second the dodge moves, so a swerve is visible not instant. */
const DODGE_RATE = 70;

/**
 * Build the live simulation for one scene.
 *
 * `home` keeps the authored positions because position is recomputed from
 * elapsed time every frame rather than accumulated. Accumulating `x += v*dt`
 * would let a dropped frame or a backgrounded tab change the picture, and "same
 * spec, same frames" is the property the whole pipeline exists to keep.
 */
export function createSimulation(scene, stage) {
  const home = new Map();
  for (const element of scene.elements) {
    home.set(element.id, { x: element.x, y: element.y, heading: element.heading ?? 0 });
  }
  const obstacles = scene.elements.filter((element) => element.role === "obstacle");
  // Which bodies sense, and with what. Derived from the spec rather than listed:
  // `proximity_gate` reads `radius` and `safe_distance`, and it is an emitter
  // that ranges while a threshold decides — so the behaviour belongs to whatever
  // body carries a sensor. A body with no sensor simply never goes dangerous,
  // which is the correct answer for an obstacle.
  const sensors = new Map();
  for (const element of scene.elements) {
    if (element.kind === "emitter" && element.anchor) sensors.set(element.anchor, element.id);
  }
  // Which zone is whose threshold, resolved by layout from the relations. See
  // `RenderScene.thresholds` — the short version is that looking for a prop
  // called `safe_distance` on the body works for the baseline and silently fails
  // for every document that roled a zone instead, which is what the avoidance
  // run did.
  const thresholds = scene.thresholds ?? {};
  const byId = new Map(scene.elements.map((element) => [element.id, element]));
  // Who the gate applies to: any body that has a threshold, whether it got one
  // from a zone or from carrying a sensor. A body with neither never goes
  // dangerous — the correct answer for an obstacle.
  const gated = new Set([...Object.keys(thresholds), ...sensors.keys()]);

  return {
    t: 0,
    live: new Map(),
    danger: new Map(),
    /** Lateral offset per body, in stage units. Persisted across frames. */
    dodge: new Map(),

    reset() {
      this.t = 0;
      this.live = new Map();
      this.danger = new Map();
      this.dodge = new Map();
      this.update(0, () => null);
    },

    /** Advance by `dt` seconds and recompute every derived value. */
    update(dt, lookup) {
      this.t += dt;
      const live = new Map();
      this.live = live;

      for (const [id, base] of home) live.set(id, { ...base });

      // 1. motion. The dodge is read from the previous frame, so this frame
      //    draws the decision that was already made rather than one being made
      //    mid-move.
      for (const element of scene.elements) {
        if (element.kind !== "body") continue;
        const node = live.get(element.id);
        const base = home.get(element.id);

        // Ballistic first, because it is the more specific statement: a body
        // with a baked flight path is a thrown body, whatever its `speed` says.
        // The loop repeats, which is what makes a still frame's throw visible
        // at all — a step is a moment and `setStep` starts it from t=0, so a
        // one-shot arc would end the beat with the ball on the ground and the
        // rest of the step holding nothing.
        const duration = element.duration;
        if (element.path && element.path.length >= 2 && duration > 0) {
          const travelled = (this.t % duration) / duration;
          const point = pointAlong(element.path, travelled);
          node.x = point.x;
          node.y = point.y;
          // Published so a trace anchored to this body can be drawn only as far
          // as the body has flown, which is what makes 轨迹 appear to be written
          // by the thing tracing it.
          node.progress = travelled;
          continue;
        }

        // The preset decides whether a body can move at all, and asking only
        // "does it have a numeric speed?" is what let a `chain` link diagram
        // carry its 底盘 off the right edge: `speed` is a real prop, the beat
        // that set it was a working beat, and nothing anywhere objected. See
        // `SPEED_PRESETS` in `registry.js` for why it is only the one preset.
        if (!SPEED_PRESETS.includes(scene.preset)) continue;

        const speed = lookup(element.id, "speed", null);
        if (typeof speed !== "number") continue;
        // +x only, whatever `heading` says. See the file docstring: steering by
        // facing was tried, and a constant facing drifts a body straight out of
        // its lane.
        //
        // `heading` is still read through `lookup` by `drawBody`, so the nose
        // eases around over `EASE_SECONDS` while the lane does not. The two
        // disagreeing for half a second is the crab walk, and it is the agreed
        // behaviour rather than a bug to fix here.
        //
        // `y` is left to the dodge in pass 3, which *adds* to whatever is here.
        // Wrap horizontally only: a lane is a corridor, and wrapping y would
        // teleport a car out of its lane.
        node.x = wrap(base.x + this.t * speed * PIXELS_PER_SPEED, stage.width);
        node.y = base.y;
      }

      // 2. proximity gate, for sensing bodies only. Two distances matter and
      //    they are not the same one: the sensor's `radius` bounds what can be
      //    seen at all, and the threshold decides what counts as too close.
      //
      //    The threshold has two sources, tried in that order:
      //
      //    - the zone layout resolved as this body's `safe_distance` role one,
      //      which is the shape a real document writes (`safe_zone.radius`);
      //    - `safe_distance` on the body, which walks the lookup's scene tier
      //      and is how the hand-written baseline's `scene.safe_distance`
      //      slider reaches it.
      //
      //    Only the second used to exist, and it is why the avoidance run's car
      //    never went dangerous: the document roled a zone instead of setting a
      //    scene param, the lookup returned null, and this loop `continue`d past
      //    every frame. Nothing errored. The car simply drove through.
      this.danger = new Map();
      for (const bodyId of gated) {
        const node = live.get(bodyId);
        if (!node) continue;
        const thresholdId = thresholds[bodyId];
        const safe = thresholdId
          ? lookup(thresholdId, "radius", byId.get(thresholdId)?.radius ?? null)
          : lookup(bodyId, "safe_distance", null);
        if (typeof safe !== "number") continue;
        // No sensor in the scene means nothing bounds the range, not that
        // nothing is measured. The avoidance document's decision scene is
        // exactly that: a car, a threshold circle and an obstacle, no lidar —
        // it is about the *judgement*, and the ranging happened in the scene
        // before. Keying participation off the emitter alone meant the gate
        // never ran there, so the car drove through an obstacle that was 12px
        // away while the threshold circle around it was 24px across.
        const sensorId = sensors.get(bodyId);
        const reach = sensorId
          ? lookup(sensorId, "radius", Number.POSITIVE_INFINITY)
          : Number.POSITIVE_INFINITY;
        const nearest = nearestDistance(node, obstacles, live, bodyId);
        if (nearest !== null && nearest <= reach && nearest < safe) {
          this.danger.set(bodyId, true);
        }
      }

      // 3. clearance choice: shift toward whichever side has more room, and
      //    relax back into the lane when out of trouble.
      //
      //    **Bodies only.** This loop used to run over every element, and the
      //    line below resets `node.y` from the element's own home position —
      //    so it overwrote the attachment pass that used to follow it, pinning
      //    a mounted sensor to the lane line while its car swerved away. The
      //    fan detached from the car, and the only symptom was a picture that
      //    still moved. An element that is not a body never dodges, so it has
      //    no business in this loop at all.
      //
      //    The dodge is **added** to whatever pass 1 left, not written over it.
      //    `node.y = base.y + next` reads as equivalent — pass 1 has just run,
      //    and for a body that only dodges the two agree — but it silently
      //    discarded the y of every other motion, because it rewrote from the
      //    *authored* position rather than from the current one. A `field`
      //    projectile travelled the x of its arc with its y pinned to the
      //    launch height: a straight line where the lesson drew a parabola,
      //    correct-looking, and nothing anywhere objected. The bug survived the
      //    forward pass because it is *invisible* in the case the loop was
      //    written for — a lane body is at `base.y` after pass 1, so the two
      //    forms agree for cars and disagree only for things that fly.
      for (const element of scene.elements) {
        if (element.kind !== "body") continue;
        const node = live.get(element.id);
        if (!node) continue;
        const current = this.dodge.get(element.id) ?? 0;
        let target = 0;
        if (this.danger.has(element.id)) {
          const { above, below } = sideClearance(node, obstacles, live, element.id);
          target = above > below ? -MAX_DODGE : MAX_DODGE;
        }
        const next = approach(current, target, DODGE_RATE * dt);
        if (next === 0) this.dodge.delete(element.id);
        else this.dodge.set(element.id, next);
        node.y += next;
      }

      // 4. attachments follow their anchor — last, so a mounted thing lands on
      //    where its parent actually ended up this frame rather than lagging a
      //    frame behind. This is why `attachment` survives into the spec at all:
      //    an emitter is not "at" a coordinate, it is mounted on a body.
      for (const [childId, parentId] of Object.entries(scene.attachment ?? {})) {
        const child = live.get(childId);
        const parent = live.get(parentId);
        if (child && parent) {
          child.x = parent.x;
          child.y = parent.y;
          child.heading = parent.heading;
        }
      }

      // 5. travelers ride their path. Progress is a function of time, not an
      //    accumulator, for the same reason as step 1.
      for (const element of scene.elements) {
        if (element.kind !== "traveler") continue;
        const speed = lookup(element.id, "speed", 0.5);
        live.get(element.id).progress = ((this.t * speed) / 2) % 1;
      }
    },

    isDangerous(id) {
      return this.danger.get(id) === true;
    },
  };
}

/** Move `value` toward `target` by at most `maxStep`, snapping when it lands. */
function approach(value, target, maxStep) {
  const delta = target - value;
  if (Math.abs(delta) <= maxStep) return target;
  return value + Math.sign(delta) * maxStep;
}

function wrap(value, span) {
  if (span <= 0) return value;
  return ((value % span) + span) % span;
}

function nearestDistance(node, obstacles, live, selfId) {
  let nearest = null;
  for (const obstacle of obstacles) {
    // Skipping self is not an optimisation. An obstacle is at distance 0 from
    // itself, so including it would make every obstacle permanently "in danger"
    // and quietly pin the danger flag on. The first smoke run did exactly that.
    if (obstacle.id === selfId) continue;
    const other = live.get(obstacle.id);
    if (!other) continue;
    const distance = Math.hypot(other.x - node.x, other.y - node.y);
    if (nearest === null || distance < nearest) nearest = distance;
  }
  return nearest;
}

/**
 * How much room there is above and below a body, in stage units.
 *
 * A rough proxy on purpose: it drives a visible swerve, and the baseline's own
 * `clearanceAt` (`frontend/demo/app.js:342`) is no more principled. Making it
 * exact would model physics the lesson never claims to teach.
 */
function sideClearance(node, obstacles, live, selfId) {
  let above = Number.POSITIVE_INFINITY;
  let below = Number.POSITIVE_INFINITY;
  for (const obstacle of obstacles) {
    if (obstacle.id === selfId) continue;
    const other = live.get(obstacle.id);
    if (!other) continue;
    const vertical = Math.abs(other.y - node.y);
    const horizontal = Math.abs(other.x - node.x);
    const room = vertical + horizontal;
    if (other.y <= node.y) above = Math.min(above, room);
    else below = Math.min(below, room);
  }
  return { above, below };
}
