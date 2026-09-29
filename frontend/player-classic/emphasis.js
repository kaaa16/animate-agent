/**
 * Accents: the six ways a body can say "look here", plus the word for "don't".
 *
 * A separate file from `primitives.js` for the reason `easing.js` is separate
 * from `player.js`: this is arithmetic, arithmetic can be *run*, and a rule
 * that can only be checked by watching the screen is the gap this project keeps
 * closing. Nothing here touches `document`.
 *
 * What an accent is not
 * ---------------------
 *
 * An accent is **decoration drawn on top of the body, not a position**. It
 * never writes to `view.live`, never reaches `behaviors.js`, and is gone by the
 * next frame's simulation. That distinction is the whole reason this is safe to
 * add one week after steering was removed for drifting a car out of its lane:
 * `heading` moved the body, so an unbounded facing was an unbounded picture. An
 * accent moves the *pen*, and every one of them is back at exactly zero before
 * the beat is half over.
 *
 * It is also why the accent is drawn about the pivot `element.pivot` rather than
 * about the centre: on an `arm`, a `swing` hinges from the shoulder, which is
 * the same fact `heading` uses and the same reason `_BODY_PIVOTS` exists.
 *
 * Words in, numbers out
 * ---------------------
 *
 * The model writes one of these names into `emphasis` and nothing else. It may
 * not write an amplitude, a cycle count or a duration — `prompts.py` forbids
 * 时长 and 缓动 in as many words, and an accent is made of exactly those three.
 * So the name is the model's and the arithmetic is this file's, which is
 * `tone` again: a word the model chooses, a colour the code owns.
 *
 * Why every one decays to zero
 * ----------------------------
 *
 * A beat is at rest at its start and at rest at its end; the accent happens in
 * between. Two consequences, both checked in `tools/player_smoke.mjs`:
 *
 * - `shape(0) === 0` and `shape(1) === 0`, so a beat that holds still, or a
 *   still frame grabbed from a render, shows the body where layout put it.
 *   An accent that did not return to zero would bake a permanent offset in,
 *   which is the crab walk wearing a different hat.
 * - The whole accent is over in `EMPHASIS_SECONDS`, inside `BEAT_MIN_SECONDS`
 *   (1.2s, in `layout.py`), so no accent is ever cut off mid-motion by the beat
 *   moving on. The margin is thinner than it was — that floor used to be 3.8s
 *   when a beat was a paragraph, and a beat is now a line, so the accent is
 *   most of a short beat rather than a gesture at the start of a long one.
 *   Whether 0.9s still reads as an accent at that length is a judgement for
 *   somebody watching, not for this comment.
 */

import { curve } from "./easing.js";

/**
 * How long one accent lasts, in seconds.
 *
 * Shorter than a beat on purpose. This is an accent at the *start* of a beat,
 * not the beat's own animation: the object draws attention to itself and then
 * holds still while the narration catches up. Stretching it across a 6-second
 * beat would make it a wobble the audience has to sit through.
 */
export const EMPHASIS_SECONDS = 0.9;

/**
 * The seven names — six motions and `none` — and what each one is made of.
 *
 * `wave` is the oscillation and `curve` is its envelope, and every motion in
 * the table is `envelope × wave` — see `shape()` below. Keeping them as two
 * declared facts rather than as eight hand-written functions is what makes
 * "does it return to zero" a property of the *table* rather than something
 * eight functions would each have to be trusted about.
 *
 * - `hump` — `sin(πu)`: out and back once, never negative. A push.
 * - `buzz` — `sin(2π·cycles·u)`: alternating, and `cycles` is how many times.
 *
 * The axes are offsets, and each is applied where it belongs: `dx`/`dy` in
 * stage units before the rotation, `rotate` in degrees about the pivot, and
 * `scale` as a fraction **added to 1** so it composes with the `scale` prop
 * instead of overwriting it.
 *
 * The four curves are all reached by at least one motion here, and the table is
 * where that is visible: a curve nothing names is a curve nothing checks.
 * `easeOutBack` and `easeOutElastic` are non-monotone, so their envelopes dip
 * *below* zero — the accent passes through the resting pose and comes back,
 * which is what makes `pop` pop and `spring` ring rather than just fade.
 */
const EMPHASIS = {
  none: null,
  pulse: { axes: { scale: 0.09 }, wave: "hump", cycles: 1, curve: "easeOutQuad" },
  shake: { axes: { dx: 7 }, wave: "buzz", cycles: 3, curve: "easeOutQuad" },
  wobble: { axes: { rotate: 6 }, wave: "buzz", cycles: 3, curve: "easeOutQuad" },
  swing: { axes: { rotate: 16 }, wave: "hump", cycles: 1, curve: "easeOutCubic" },
  pop: { axes: { scale: 0.12, rotate: 4 }, wave: "hump", cycles: 1, curve: "easeOutBack" },
  spring: { axes: { rotate: 12 }, wave: "hump", cycles: 1, curve: "easeOutElastic" },
};

/** The names, in table order. `registry.py`'s glosses must be these, in any order. */
export const EMPHASIS_NAMES = Object.keys(EMPHASIS);

/**
 * The resting pose: an accent that has finished is indistinguishable from one
 * that was never asked for. Frozen, so two calls cannot hand out two objects
 * that a caller might mutate into a permanent offset.
 */
const AT_REST = Object.freeze({ dx: 0, dy: 0, rotate: 0, scale: 0 });

/**
 * How far a body is displaced by `name` at `t` seconds into its beat.
 *
 * `t` is the *beat's* own clock (`view.time`, which `player.js` restarts at
 * zero on every step), so the accent re-fires on each beat that asks for it and
 * needs no state of its own — the same "recompute from `t`" rule the whole
 * player runs on, and the reason a dropped frame cannot change the picture.
 *
 * An unknown name is `none`, not a throw. This is the `toneColor` arrangement
 * and not the `curve` one: the name came from a spec, the player did not write
 * it, and a body drawn without its accent is a body — the same call the player
 * makes for a tone it does not know. `validation.py`'s `unknown_emphasis` is
 * what makes the name right in the first place.
 */
export function emphasisAt(name, t) {
  const motion = EMPHASIS[name];
  if (!motion || !(t >= 0) || t >= EMPHASIS_SECONDS) return AT_REST;
  const amount = shape(motion, t / EMPHASIS_SECONDS);
  const { dx = 0, dy = 0, rotate = 0, scale = 0 } = motion.axes;
  return {
    dx: dx * amount,
    dy: dy * amount,
    rotate: rotate * amount,
    scale: scale * amount,
  };
}

/**
 * The one shape every accent is: an envelope that falls from 1 to 0, times a
 * wave. `u` is the fraction of the accent already spent, in `[0, 1)`.
 *
 * `1 - curve(u)` is the envelope, so it is exactly 1 at `u = 0` and exactly 0
 * at `u = 1` — and the waves are all 0 at both ends too, which is what puts the
 * body back where it started without either term having to be special-cased.
 * For `easeOutBack` and `easeOutElastic` the envelope goes negative in between
 * and the motion briefly reverses, which is the overshoot those curves are for.
 */
function shape(motion, u) {
  const envelope = 1 - curve(motion.curve)(u);
  const wave =
    motion.wave === "hump"
      ? Math.sin(Math.PI * u)
      : Math.sin(2 * Math.PI * motion.cycles * u);
  return envelope * wave;
}
