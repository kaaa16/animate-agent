/**
 * Moving a property from what the last beat set it to, to what this one does.
 *
 * Split out of `player.js` for the reason `behaviors.js` and `stage.js` are
 * separate files: this is arithmetic, and arithmetic can be *run*. `player.js`
 * reaches for `document` the moment it loads, so nothing in it is reachable from
 * a test; the rule "a beat boundary is not a step function" would then be
 * checkable only by a person watching the screen, which is exactly the gap this
 * whole round of work came out of.
 *
 * There is no library behind this. `tween.js` is MIT and would do the same
 * arithmetic — but it also brings a global ticker and its own
 * `requestAnimationFrame` loop, and the player already runs one. Two loops
 * racing for the same frame jitters in a way nobody would trace back to a
 * dependency choice, and the equations themselves are four characters long.
 */

/**
 * ease-out cubic — Robert Penner's equation, in the one-line form easings.net
 * publishes it.
 *
 * The slow finish is the point. A property that arrives at full speed and stops
 * dead reads as a mechanical jump, and "it moves without being interesting" was
 * the complaint this answers.
 */
export function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

/** The two props a drawer tests with `=== false`, and so the two that may ramp. */
const PRESENCE_PROPS = ["visible", "enabled"];

/**
 * Where a property sits on its way from what it was to what it is.
 *
 * `progress` is already eased — this interpolates, it does not shape.
 *
 * Numbers interpolate, and so do the two presence gates, as 1 and 0. That
 * second one is load-bearing rather than decorative: `drawBody` and `drawTrace`
 * decide whether to draw at all with `lookup(id, "visible") === false`, and
 * `drawEmitter`/`drawZone` with `enabled`. A gate caught mid-ramp is a *number*,
 * so that check lets the element through, and `drawElement` supplies the
 * matching alpha — which is how a fan that switches on at beat 2 fades up
 * instead of appearing whole between two frames.
 *
 * Everything else is returned untouched. `danger`, `active`, `state`, `text`
 * and `tone` are a style or a word, and there is no halfway point between two
 * words to show: for those the cut *is* the change.
 */
export function blend(prop, from, target, progress) {
  // The endpoints are exact, and the caller does not have to be the reason why.
  // `player.js` already returns the target once an ease has landed, so this is
  // belt-and-braces for the far end — but the near one matters on its own: the
  // first frame of every beat eases from the old value, and a gate that came
  // back as `0` instead of `false` would be a number, which a drawer's
  // `=== false` reads as "draw it".
  if (progress >= 1) return target;
  if (progress <= 0) return from;

  if (typeof from === "number" && typeof target === "number") {
    return from + (target - from) * progress;
  }
  if (PRESENCE_PROPS.includes(prop) && typeof from === "boolean" && typeof target === "boolean") {
    const start = from ? 1 : 0;
    return start + ((target ? 1 : 0) - start) * progress;
  }
  return target;
}

/**
 * How long a property takes to travel, in seconds.
 *
 * Shorter than about a third of a second reads as a glitch rather than as
 * movement; longer than about a second and the beat starts spending its own
 * duration on the transition instead of on the thing being taught.
 */
export const EASE_SECONDS = 0.55;
