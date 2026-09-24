/**
 * The line under the picture: what this beat is saying, in words.
 *
 * A beat has carried its narration since `RenderStep` existed, and until now
 * the only place it appeared was the 教学节拍 list in the side panel — which is
 * chrome, outside the canvas, and therefore not in the picture at all. The
 * reference film this project is aiming at has the words on screen; that is
 * most of what makes it a video rather than a diagram with a legend beside it.
 *
 * Arithmetic, not drawing
 * -----------------------
 *
 * Split out of `player.js` for the reason `emphasis.js` and `easing.js` are:
 * whatever can be *run* is worth separating, and line-breaking is the part of
 * this that can be wrong in a way nobody spots. So the width of a string is a
 * function passed in, and the tests hand it a fake ruler and check where the
 * breaks land without a canvas anywhere.
 *
 * Why the band is a reservation and not an overlay
 * ------------------------------------------------
 *
 * `layout.py` gives up `CAPTION_BAND_RATIO` of the stage's height at the bottom
 * and then refuses to place anything inside it (`_check_fit` reads it as a
 * fourth edge). Drawing the words over the picture instead would have been
 * less code and would have put a subtitle on top of whatever the lesson was
 * pointing at, which is the one thing a subtitle must never do.
 *
 * That reservation is exactly `CAPTION_MAX_LINES` tall. The schema caps a
 * beat's `description` at 200 characters and the whole real corpus runs 18~57
 * (median 34, 80% within 40), so two lines is a measurement rather than a
 * hope — but a hand-written storyboard is not bound by the corpus, which is why
 * the overflow is ellipsised rather than allowed to spill out of the band and
 * over the picture.
 */

/** Stage units, and the stage is 960x600 — so these are canvas pixels. */
export const CAPTION_FONT_SIZE = 20;

/** Baseline to baseline. What the band is measured in, so it has to divide. */
export const CAPTION_LINE_HEIGHT = 26;

export const CAPTION_PAD_X = 20;

/**
 * How many lines the band holds.
 *
 * Two, and `layout.py`'s `CAPTION_BAND_RATIO` is sized for exactly this: pad,
 * this many lines, pad. A test holds the two files to that agreement, because
 * a band the layout reserved and a box the caption draws are the same number
 * written twice, and a mismatch is invisible until words sit on top of a car.
 */
export const CAPTION_MAX_LINES = 2;

/** The ellipsis that marks a caption too long for the band. */
const ELLIPSIS = "…";

/**
 * Break `text` into at most `maxLines` lines that each fit `maxWidth`.
 *
 * `measure` is `(text) => width` — `ctx.measureText(...).width` in the player,
 * a stub in the tests. Widths are asked of the *renderer* rather than computed
 * from a table here, because the font is the renderer's: `primitives.js` draws
 * Chinese with `'Microsoft YaHei'` and a half-width Latin table would break the
 * lines in the wrong place on the one string this exists to wrap.
 *
 * Greedy, longest-fitting-prefix first. A break is nudged back to the last
 * space when there is one more than half-way along, which is what keeps a
 * caption that mixes 中文 and English from splitting a word — Chinese has no
 * spaces to prefer, so the character break is already right for it.
 *
 * The last line is ellipsised rather than returned whole. Spilling would draw
 * outside the reserved band, on top of the picture; losing the tail of a long
 * sentence is worse than nothing at all, and it is visible, which the spill is
 * not.
 */
export function wrapCaption(text, measure, maxWidth, maxLines = CAPTION_MAX_LINES) {
  const lines = [];
  let rest = String(text ?? "").trim();

  while (rest.length > 0 && lines.length < maxLines) {
    const last = lines.length === maxLines - 1;
    const fits = longestPrefix(rest, measure, maxWidth);

    if (fits >= rest.length) {
      lines.push(rest);
      break;
    }
    if (last) {
      // No room left for another line: cut this one back far enough that the
      // ellipsis itself fits, and stop.
      lines.push(ellipsised(rest, measure, maxWidth));
      break;
    }
    let cut = fits;
    const space = rest.lastIndexOf(" ", fits);
    if (space > fits / 2) cut = space;
    if (cut <= 0) cut = 1; // one glyph wider than the whole line
    lines.push(rest.slice(0, cut));
    rest = rest.slice(cut).trimStart();
  }

  return lines.length > 0 ? lines : [""];
}

/** How much of `text` fits in `maxWidth`, as a character count. */
function longestPrefix(text, measure, maxWidth) {
  let fits = 0;
  for (let end = 1; end <= text.length; end += 1) {
    if (measure(text.slice(0, end)) > maxWidth) break;
    fits = end;
  }
  return fits;
}

/** The longest prefix of `text` that fits once `…` is appended. */
function ellipsised(text, measure, maxWidth) {
  const width = measure(ELLIPSIS);
  const fits = longestPrefix(text, measure, maxWidth - width);
  return fits > 0 ? text.slice(0, fits) + ELLIPSIS : ELLIPSIS;
}

/**
 * How tall the band is, in stage units — the same number `layout.py` gives up.
 *
 * Read from the height rather than from the font, so that the plate the caption
 * paints and the strip the layout kept clear are one number and not two that
 * have to be kept in step by hand.
 */
export function captionBand(stage) {
  return stage.height * CAPTION_BAND_RATIO;
}

/**
 * The fraction of the stage's height the caption owns.
 *
 * 0.13 of 600 is 78, and `CAPTION_MAX_LINES` lines of `CAPTION_LINE_HEIGHT` is
 * 52 — so the block sits with 13 of room above it and 13 below, which is why
 * the lines are centred in the band rather than stacked from its top edge.
 *
 * `layout.py` holds the same constant. The two are compared by a test rather
 * than shared, because no module system spans Python and the browser here — and
 * a duplicated constant that nothing checks is how a caption ends up drawn over
 * a wheel.
 */
export const CAPTION_BAND_RATIO = 0.13;

/** Where the plate's top edge sits, and where a line's baseline sits. */
export function captionTop(stage) {
  return stage.height - captionBand(stage);
}
