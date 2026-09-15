/**
 * The registry: `kind` -> drawer, and the hard failure when there is none.
 *
 * The JS twin of `rendering/registry.py`, and it keeps that file's rule — an
 * unknown name is never silently dropped. The failure is split in two on
 * purpose, because the fixes differ:
 *
 * - `未知图元` — the name is not in the Python registry either. The spec is
 *   wrong, or came from a version of the contract this player does not know.
 * - `尚未实现` — the primitive is registered but this player has no drawer yet.
 *   The spec is fine; the picture is incomplete.
 *
 * Collapsing those into one message would send someone to the wrong file half
 * the time.
 */

import {
  drawAngle,
  drawAxis,
  drawBody,
  drawDimension,
  drawEmitter,
  drawLink,
  drawReadout,
  drawTrace,
  drawTraveler,
  drawVector,
  drawZone,
} from "./primitives.js";

/** Kinds with a real drawer. */
const DRAWERS = {
  body: drawBody,
  emitter: drawEmitter,
  zone: drawZone,
  link: drawLink,
  traveler: drawTraveler,
  readout: drawReadout,
  vector: drawVector,
  trace: drawTrace,
  axis: drawAxis,
  dimension: drawDimension,
  angle: drawAngle,
};

/**
 * Kinds `registry.py` declares that no drawer handles yet.
 *
 * Kept as an explicit list rather than inferred, so a Python test can assert
 * `DRAWABLE_KINDS + PENDING_KINDS` covers every T1 primitive and the two files
 * cannot drift apart in silence.
 *
 * `region` and `wave` are the whole remainder. Neither appears in the three
 * sample documents, so neither has been drawn against real input — and a
 * primitive implemented from imagination is how a plausible-but-wrong picture
 * gets shipped. The player says so by name rather than drawing a guess.
 */
export const PENDING_KINDS = ["region", "wave"];

/**
 * kind -> the props the drawers in `primitives.js` read at playback.
 *
 * The mirror of `Primitive.live_props` in `rendering/registry.py`, written out
 * by hand for the same reason `PENDING_KINDS` is: so that the two lists drifting
 * apart is a failing test rather than a picture that quietly does not move.
 *
 * The Python side is the one that matters — `step_state_inert` rejects a beat
 * that sets a prop not in this table, and the Storyboard Agent's prompt is
 * generated from it. This copy exists so that claim can be checked against the
 * code that actually draws: a name listed here that no drawer reads makes the
 * prompt a promise the player breaks, which is exactly how `drawVector` spent
 * a whole document drawing one arrow while the model wrote three different
 * directions.
 *
 * Adding a prop here without making a drawer read it is therefore the one thing
 * not to do. If a prop should not be settable per beat, leave it out — the
 * validator will say so, with the list of props that would work instead.
 */
export const LIVE_PROPS = {
  body: ["speed", "heading", "scale", "visible", "danger"],
  emitter: ["radius", "enabled"],
  zone: ["radius", "enabled"],
  link: ["active"],
  traveler: ["speed", "progress", "state"],
  trace: ["length", "visible"],
  vector: ["magnitude", "direction"],
  axis: ["range", "ticks"],
  dimension: ["label"],
  angle: ["degrees", "radius"],
  // `align` is absent on purpose. Layout settles it once (`_align_of` in
  // `layout.py`) and `drawReadout` reads the baked value; nothing asks
  // `view.lookup` for it. It used to be listed and was not read, which is the
  // one shape of wrong this table exists to make impossible — see the note above.
  readout: ["text", "tone"],
};

export const DRAWABLE_KINDS = Object.keys(DRAWERS);

export function drawElement(ctx, element, view) {
  const drawer = DRAWERS[element.kind];
  if (drawer) {
    drawer(ctx, element, view);
    return;
  }
  if (PENDING_KINDS.includes(element.kind)) {
    throw new Error(
      `图元 \`${element.kind}\` 已在 registry 里声明，但播放器还没有实现它` +
        `（元素 \`${element.id}\`）—— 补一个绘制函数，或先别在校验台放行它`,
    );
  }
  throw new Error(
    `未知图元 \`${element.kind}\`（元素 \`${element.id}\`）—— registry 里没有这个名字`,
  );
}
