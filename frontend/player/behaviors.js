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
 * | element                                   | behaviour          |
 * |-------------------------------------------|--------------------|
 * | `body` carrying a numeric `speed`          | `linear_motion`    |
 * | any body, against the obstacles around it  | `proximity_gate`   |
 * | a body that just went dangerous            | `clearance_choice` |
 * | `traveler` bound to a `path_id`            | `hop_along_path`   |
 *
 * A body with no `speed` simply does not move — not a special case, just the
 * absence of the prop `linear_motion` reads.
 *
 * **No LLM runs here, ever.** Playback is pure arithmetic on the spec, which is
 * why the same spec always produces the same frames.
 *
 * `linear_motion` here is a **lane** model: advance along +x at `speed`, offset
 * sideways to dodge. It is not a ballistic model, and pretending otherwise would
 * be wrong — a projectile follows a parabola, which the plan defers to a later
 * slice along with the `field` preset. The name is the registry's; the motion is
 * what the `lane` preset means by it.
 */

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

      // 1. linear motion. The dodge is read from the previous frame, so this
      //    frame draws the decision that was already made rather than one being
      //    made mid-move.
      for (const element of scene.elements) {
        if (element.kind !== "body") continue;
        const speed = lookup(element.id, "speed", null);
        if (typeof speed !== "number") continue;
        const node = live.get(element.id);
        const base = home.get(element.id);
        // Wrap horizontally only: a lane is a corridor, and wrapping y would
        // teleport a car out of its lane.
        node.x = wrap(base.x + this.t * speed * PIXELS_PER_SPEED, stage.width);
        node.y = base.y + (this.dodge.get(element.id) ?? 0);
      }

      // 2. proximity gate, for sensing bodies only. Two distances matter and
      //    they are not the same one: the sensor's `radius` bounds what can be
      //    seen at all, and `safe_distance` decides what counts as too close.
      //    Reading `safe_distance` off the body walks the lookup's scene tier,
      //    which is how a `scene.safe_distance` slider reaches it.
      this.danger = new Map();
      for (const [bodyId, sensorId] of sensors) {
        const node = live.get(bodyId);
        if (!node) continue;
        const safe = lookup(bodyId, "safe_distance", null);
        if (typeof safe !== "number") continue;
        const reach = lookup(sensorId, "radius", Number.POSITIVE_INFINITY);
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
      for (const element of scene.elements) {
        if (element.kind !== "body") continue;
        const base = home.get(element.id);
        if (base === undefined) continue;
        const current = this.dodge.get(element.id) ?? 0;
        let target = 0;
        if (this.danger.has(element.id)) {
          const { above, below } = sideClearance(live.get(element.id), obstacles, live, element.id);
          target = above > below ? -MAX_DODGE : MAX_DODGE;
        }
        const next = approach(current, target, DODGE_RATE * dt);
        if (next === 0) this.dodge.delete(element.id);
        else this.dodge.set(element.id, next);
        live.get(element.id).y = base.y + next;
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
