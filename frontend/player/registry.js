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

import { rad } from "./atoms.js";
import { emphasisAt } from "./emphasis.js";
import { DEFAULT_ENTRANCE, entranceAt, hingePoint } from "./entrance.js";
import {
  drawAngle,
  drawAxis,
  drawBody,
  drawBubble,
  drawCard,
  drawCode,
  drawDimension,
  drawEmitter,
  drawLink,
  drawNote,
  drawOrdinal,
  drawReadout,
  drawTrace,
  drawTraveler,
  drawTree,
  drawVector,
  drawVerdict,
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
  note: drawNote,
  vector: drawVector,
  trace: drawTrace,
  axis: drawAxis,
  dimension: drawDimension,
  angle: drawAngle,
  verdict: drawVerdict,
  bubble: drawBubble,
  ordinal: drawOrdinal,
  card: drawCard,
  tree: drawTree,
  code: drawCode,
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
 *
 * **`visible` is on nearly every row, and it is read somewhere other than a
 * drawer.** `drawBody` and `drawTrace` test it (that is the `lookup(...) ===
 * false` the presence-table test greps for), and everybody else carries it
 * without a gate: the alpha comes from `drawElement`, which asks
 * `view.presence` for the *kind*. So the entry is what makes an object able to
 * arrive at a beat rather than being there from the first frame, and the
 * reading happens in one place for all of them. `emitter` and `zone` say
 * `enabled` instead — the same rule, an older name, and one the storyboards on
 * disk already write. See `PRESENCE_PROP` in `player.js`.
 *
 * **`emphasis` is on every row for the same reason, and it is read in the same
 * place.** It used to be `body`'s alone, and the read used to be inside
 * `drawBody`. Both moved: the accent is applied in `drawElement`, beside the
 * fade and the rise, so a card, a verdict or a line of code can be jolted on
 * the beat that says 「注意看这个」. That is why this table can list it for
 * every kind without lying — see `OFFSET_KINDS` there for how a mounted
 * element gets its pivot.
 */
export const LIVE_PROPS = {
  body: ["speed", "heading", "scale", "visible", "danger", "emphasis"],
  emitter: ["radius", "enabled", "emphasis"],
  zone: ["radius", "enabled", "emphasis"],
  link: ["active", "visible", "emphasis"],
  traveler: ["speed", "progress", "state", "visible", "emphasis"],
  trace: ["length", "visible", "emphasis"],
  vector: ["magnitude", "direction", "visible", "emphasis"],
  axis: ["range", "ticks", "visible", "emphasis"],
  dimension: ["label", "visible", "emphasis"],
  angle: ["degrees", "radius", "visible", "emphasis"],
  // `align` is absent on purpose. Layout settles it once (`_align_of` in
  // `layout.py`) and `drawReadout` reads the baked value; nothing asks
  // `view.lookup` for it. It used to be listed and was not read, which is the
  // one shape of wrong this table exists to make impossible — see the note above.
  readout: ["text", "tone", "emphasis"],
  // `text` and `tone`, the same pair the retired panel had, and `text` is the one
  // that carries weight here: a note's words are its whole content and the
  // component that decides where it sits is `_beside`, which runs at layout. A
  // beat that rewrites the text draws the new words in the room the note
  // reserved — which is why `NoteElement.lines` is a table keyed by the string,
  // the arrangement `bubble` already has.
  //
  // `of` is absent because it is a relation, settled once by the layout that
  // used it to find the anchor. Absent is also what an unanchored note looks
  // like, so the two cases need no separate entry.
  note: ["text", "tone", "visible", "emphasis"],
  // `mark` and `text` are the two a beat can change, and `mark` is the one worth
  // stating: a beat that flips a ✓ to a ✗ *is* the beat. `of` is not here
  // because it is a relation, settled once by the layout that resolved it.
  verdict: ["mark", "text", "visible", "emphasis"],
  // Both are live and `text` is the one worth stating. The box is cut for every
  // string a beat can write (`BubbleElement.lines`), so a beat that rewrites it
  // draws *its* sentence in *its* shape rather than a squeezed version of the
  // declared one. `of` is absent because it is a relation, settled once by the
  // layout that used it to find the anchor.
  bubble: ["text", "tone", "visible", "emphasis"],
  // `type` alongside `text`, because 「同一个位置换一种值」 is only a lesson if
  // the tag moves with the value. Both are read through `view.lookup` in
  // `drawCard`, and the words for the new type come from the element's own
  // `tags` table rather than being derived from the declared one.
  card: ["text", "type", "visible", "emphasis"],
  // `text` is absent on purpose, and it is the one a reader would expect. Both
  // of these cut their box for the rows they hold at *layout* time, so a beat
  // that swapped in a longer structure would draw it out over its own panel —
  // the same call `readout.align` makes, for a stronger version of the same
  // reason. What a beat does here instead is `focus`: a tree is walked through
  // one node per beat and a listing is read one line at a time, and that is the
  // beat this primitive exists for.
  //
  // `form` is absent for a second reason stacked on the same one. The five forms
  // do not merely draw the box differently, they *size* it differently: `branch`
  // puts a whole chain on one row and `boxes` spends height per level, so a beat
  // that changed the form would be drawing into a panel cut for another picture.
  // `registry.TREE_FORMS` says the same thing from the layout's side.
  tree: ["focus", "tone", "visible", "emphasis"],
  code: ["focus", "tone", "visible", "emphasis"],
  // The one primitive whose *only* live prop is presence. `n` is absent on
  // purpose and the argument is in `registry.py`: a step number is a fact about
  // the object, not a thing that changes during the scene. When the dot shows
  // up is the other question, and it is the one a walk-through procedure asks.
  ordinal: ["visible", "emphasis"],
};

/**
 * The presets whose bodies travel by `speed` — the mirror of `SPEED_PRESETS`
 * in `rendering/registry.py`, hand-written for the reason `LIVE_PROPS` is.
 *
 * `behaviors.js` used to ask only "is this a body with a numeric speed?", which
 * is true of every body that was ever given one. The recorded avoidance
 * document writes `car_body.speed: 60` in a `chain` scene, so the first thing
 * that lesson put on screen was a link diagram with the chassis sliding off the
 * right edge and wrapping, forever — no error anywhere, because `speed` is a
 * real prop and the beat that set it was a working beat.
 *
 * The preset is what decides whether a body can move at all. Only `lane` has a
 * corridor; `chain` and `hub` hold their bodies still, and `field` flies them
 * along a parabola `layout.py` baked, where the flight time is the arc's
 * property rather than a `speed`.
 *
 * `traveler` is not in this table and does not need to be: a token moves along
 * its link in every preset, so its `speed` is read wherever it is written.
 */
export const SPEED_PRESETS = ["lane"];

/**
 * Which prop decides whether an element is on screen at all — every kind.
 *
 * The hand-written twin of `registry.PRESENCE_PROP`, and a test holds the two
 * equal. It lives here rather than in the player because it is the same kind of
 * table as `LIVE_PROPS` one screen up: a fact about a primitive, keyed by kind,
 * that more than one file needs. `player.js` reads it to ramp the value,
 * `drawElement` below reads it through `view.presence`, and `player_smoke.mjs`
 * reads it to decide whether an element was supposed to draw at all.
 *
 * Keyed by kind rather than asked of the element because the answer is a
 * property of the *primitive*: an `emitter` is gated on `enabled` and a `link`
 * on `visible`, and the spec carries no statement about which. A table that
 * named the wrong one would look right in every file and fade an element
 * against a gate nothing reads.
 *
 * Four of these are also read by a drawer — `drawBody` and `drawTrace` test
 * `visible`, `drawEmitter` and `drawZone` test `enabled` — and the other
 * fourteen carry it without a gate, taking their alpha from `drawElement`. Both
 * halves are needed and they are not the same half: the gate is the hard
 * on/off, this table is what `easing.js` ramps and what `drawElement` turns
 * into a fade and a rise.
 *
 * `readout` is here and is retired. It neither offers the prop nor gates on it,
 * so its entry is inert — the lookup falls through to the default and every
 * picture already on disk draws exactly as it did. Keeping it complete is worth
 * that: the alternative is a table with a hole in it that only a reader who
 * already knew the history could explain.
 */
export const PRESENCE_PROP = {
  body: "visible",
  emitter: "enabled",
  zone: "enabled",
  link: "visible",
  traveler: "visible",
  trace: "visible",
  vector: "visible",
  axis: "visible",
  dimension: "visible",
  angle: "visible",
  readout: "visible",
  note: "visible",
  verdict: "visible",
  bubble: "visible",
  card: "visible",
  tree: "visible",
  code: "visible",
  ordinal: "visible",
};

export const DRAWABLE_KINDS = Object.keys(DRAWERS);

/**
 * The accent — 「这一拍强调它」 — applied to whatever element is being drawn.
 *
 * It used to live inside `drawBody`, which is the only reason a body was the
 * only thing that could be emphasised. The move is the point: a beat that says
 * 「注意看这个」 is about *the thing on screen*, and the thing on screen is as
 * often the card holding the wrong value as it is a car.
 *
 * Three properties, all load-bearing:
 *
 * - **At rest it emits nothing.** `emphasisAt` returns a frozen zero object for
 *   `none`, for an accent that has finished, and for a name it does not know —
 *   and the guard below turns that into no drawing calls at all. A settled
 *   frame is therefore byte-identical to the one this function replaced, which
 *   is what keeps every recorded smoke snapshot and every on-disk artifact
 *   meaning what it meant.
 * - **It is an outer transform.** It runs before the drawer, in stage space, so
 *   `drawBody`'s own `translate(node.x, node.y)` plus pivot rotation compose
 *   *inside* it: the pen moves and the body does not. That is also why
 *   `drawBody` no longer adds `accent.rotate` to its heading — the rotation is
 *   here now, about the same point.
 * - **It rides the live position.** `view.live` holds a node for every element,
 *   not only the ones that move, so a car asked to `shake` mid-lane shakes
 *   where it currently *is*. Reading `element.x` would shake it at the place it
 *   started the scene, which is the crab walk `emphasis.js` exists to refuse.
 *
 * `scale` is applied about the same origin as the rotation. It used to be folded
 * into the body's own `size`, i.e. scaled about the shape's centre; for every
 * role but `arm` those are the same point, and an `arm` now pops about the
 * shoulder it is bolted to. It also now scales whatever the drawer strokes, so
 * a `pop` thickens a 2px outline by the same 12% it grows the shape — which is
 * what a CSS `transform: scale()` would do and what the eye expects.
 */
function applyAccent(ctx, element, view) {
  const accent = emphasisAt(view.lookup(element.id, "emphasis", "none"), view.time);
  if (!(accent.dx || accent.dy || accent.rotate || accent.scale)) return;

  ctx.translate(accent.dx, accent.dy);
  if (!(accent.rotate || accent.scale)) return;

  const { x: cx, y: cy } = hingePoint(element, view);
  ctx.translate(cx, cy);
  ctx.rotate(rad(accent.rotate));
  const grown = 1 + accent.scale;
  ctx.scale(grown, grown);
  ctx.translate(-cx, -cy);
}

export function drawElement(ctx, element, view) {
  const drawer = DRAWERS[element.kind];
  if (drawer) {
    // Applied here rather than inside each drawer, and multiplied rather than
    // assigned so a drawer's own alpha still means what it says. This is the
    // second half of the presence ramp: a drawer's gate reads the *eased* value
    // and so lets a half-arrived element through, and the fraction it lets
    // through is this one.
    const presence = view.presence(element.id);
    if (presence <= 0) return;
    ctx.save();
    // `ctx.globalAlpha *= presence` rather than an assignment, so a drawer's own
    // alpha still means what it says: a drawer's gate reads the *eased* value
    // and so lets a half-arrived element through, and the fraction it lets
    // through is this one.
    ctx.globalAlpha *= presence;
    // What the arrival looks like is `entrance.js`'s business, and it emits
    // nothing at all once `presence` reaches 1 — so an element that never hides
    // is drawn on exactly the path it was on before there was more than one
    // style to choose from.
    //
    // The clip goes first, and that is not a preference: a clip path is taken in
    // the space current at `ctx.clip()` time, so a rectangle in stage
    // coordinates has to be applied before the translate and the scale below put
    // anything else on the matrix.
    const entrance = entranceAt(element.enter ?? DEFAULT_ENTRANCE, presence, element, view);
    if (entrance.clip) {
      ctx.beginPath();
      ctx.rect(entrance.clip.x, entrance.clip.y, entrance.clip.width, entrance.clip.height);
      ctx.clip();
    }
    if (entrance.dy) ctx.translate(0, entrance.dy);
    if (entrance.scale) {
      const { x: cx, y: cy } = hingePoint(element, view);
      ctx.translate(cx, cy);
      ctx.scale(entrance.scale, entrance.scale);
      ctx.translate(-cx, -cy);
    }
    applyAccent(ctx, element, view);
    try {
      drawer(ctx, element, view);
    } finally {
      ctx.restore();
    }
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
