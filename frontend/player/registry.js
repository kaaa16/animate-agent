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
