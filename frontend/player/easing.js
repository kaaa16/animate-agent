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

/**
 * The rest of the family, and the reason they are here rather than in a library.
 *
 * These are Robert Penner's equations in the normalised form `easings.net`
 * publishes: `[0, 1] -> [0, 1]`, `f(0) === 0`, `f(1) === 1`. Four of them, not
 * thirty. A curve with no caller is a curve nobody has looked at, and the rule
 * this file already keeps for `tween.js` — refuse the dependency, keep the
 * arithmetic — cuts the other way too: keep the arithmetic you can *check*, and
 * `tools/player_smoke.mjs` checks every one of these at both ends and for
 * monotonicity where monotonicity is claimed.
 *
 * Which one is wanted is never a decision the model gets to make — see
 * `prompts.py` on 缓动. The name comes from `emphasis.js`'s table, which is
 * code, exactly as `tone` names come from `registry.py` and land on colours.
 */
export function easeOutQuad(t) {
  return 1 - (1 - t) * (1 - t);
}

/**
 * Overshoot, then settle — the "pop".
 *
 * The only curve here that is **not** monotone: it rises past 1 around `t≈0.7`
 * and comes back down. That is the whole effect, and it is why this one is used
 * as a *decay* rather than as a progress: `1 - easeOutBack(u)` dips below zero
 * near the end, so a `pop`-ped scale passes through its normal size on the way
 * out instead of easing into it. Values outside `[0, 1]` are the point.
 */
export function easeOutBack(t) {
  // The endpoints are pinned rather than computed, and the reason is not
  // tidiness. The three terms below sum to zero at `t = 0` on paper, and to
  // `2.22e-16` in binary floating point — which the accent envelope turns into
  // a body that is permanently a fraction of a pixel from where layout put it.
  // `tools/player_smoke.mjs` caught exactly that on this curve's first run.
  // `easeOutElastic` pins them for the same reason; `easeOutQuad`/`easeOutCubic`
  // land exactly and do not need to.
  if (t <= 0) return 0;
  if (t >= 1) return 1;
  const back = 1.70158;
  const shifted = t - 1;
  return 1 + (back + 1) * shifted * shifted * shifted + back * shifted * shifted;
}

/**
 * A damped spring, written as a curve. Ringing, then rest.
 *
 * `c4 = 2π/3` is what puts the oscillation *after* the first rise rather than
 * on top of it — the standard amplitude/period pair from `easings.net`, kept
 * because inventing a nicer-looking one here would make this file the only
 * place the numbers live.
 */
export function easeOutElastic(t) {
  if (t <= 0) return 0;
  if (t >= 1) return 1;
  const c4 = (2 * Math.PI) / 3;
  return Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
}

/** Every curve by name, for the tables that name one. */
export const EASINGS = {
  easeOutQuad,
  easeOutCubic,
  easeOutBack,
  easeOutElastic,
};

/**
 * A curve by name, or a hard failure.
 *
 * Throws rather than falling back, and the asymmetry with `emphasis.js` — which
 * answers an unknown *emphasis* with "no emphasis" — is deliberate. The two
 * names arrive from different places: an emphasis name is written by a model
 * into a spec, and a player handed a spec it did not write should still draw
 * something; a curve name is written by *this* repo, into the table below, and
 * a typo in it is a broken preset rather than a document that needs a retry.
 */
export function curve(name) {
  const found = EASINGS[name];
  if (!found) {
    throw new Error(
      `未知缓动曲线 \`${name}\`——` +
        `已注册的是 ${Object.keys(EASINGS).join("、")}。` +
        "这个名字应当由 emphasis.js 的表提供，不是从 spec 里读来的",
    );
  }
  return found;
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
