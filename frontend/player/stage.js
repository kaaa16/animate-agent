/**
 * Stage: the mapping from spec coordinates to device pixels.
 *
 * `fitTransform` is deliberately a pure function of numbers, with no canvas and
 * no DOM. Fitting arithmetic is where "everything is blurry" and "everything is
 * offset by the toolbar height" come from, and those are far easier to pin down
 * with a unit test than by squinting at a screenshot.
 */

/**
 * Work out the device transform that fits `stage` inside `viewport`.
 *
 * The result is what goes into `ctx.setTransform`: uniform scale (never
 * anisotropic — a stretched car reads as a bug, not as a wide screen), centred,
 * with the device pixel ratio folded in so 1 spec unit maps to the same visual
 * size on any display.
 */
export function fitTransform(stage, viewport, dpr = 1) {
  const scale = Math.min(viewport.width / stage.width, viewport.height / stage.height);
  const offsetX = (viewport.width - stage.width * scale) / 2;
  const offsetY = (viewport.height - stage.height * scale) / 2;
  return {
    scale,
    offsetX,
    offsetY,
    dpr,
    a: scale * dpr,
    d: scale * dpr,
    e: offsetX * dpr,
    f: offsetY * dpr,
    deviceWidth: Math.max(1, Math.round(viewport.width * dpr)),
    deviceHeight: Math.max(1, Math.round(viewport.height * dpr)),
  };
}

export function createStage(canvas, stage) {
  const ctx = canvas.getContext("2d");
  let current = fitTransform(stage, { width: stage.width, height: stage.height }, 1);

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const viewport = {
      width: rect.width || stage.width,
      height: rect.height || stage.height,
    };
    current = fitTransform(stage, viewport, dpr);
    canvas.width = current.deviceWidth;
    canvas.height = current.deviceHeight;
    // Set every frame rather than once: `resize` is called on a window resize,
    // and the transform is what carries the new fit into the next draw.
    ctx.setTransform(current.a, 0, 0, current.d, current.e, current.f);
  }

  return {
    ctx,
    get transform() {
      return current;
    },
    resize,
    clear() {
      ctx.save();
      ctx.setTransform(current.dpr, 0, 0, current.dpr, 0, 0);
      ctx.clearRect(0, 0, current.deviceWidth, current.deviceHeight);
      ctx.restore();
    },
  };
}

/**
 * Read the theme from the CSS custom properties the stylesheet owns.
 *
 * Called once at boot and again on every palette change, which is the whole of
 * what switching costs: nothing caches a colour. Every drawer takes the theme as
 * an argument and `render` passes `state.theme`, so the next frame is already
 * drawn in the new palette.
 *
 * The fallbacks are the `neon` values, repeated from `player.css` on purpose —
 * they are what a page with no stylesheet at all still draws, and they are not
 * dead code: `tools/player_smoke.mjs` stubs `getComputedStyle` and calls this to
 * build its own theme, so the harness gets the real slots rather than a second
 * hand-copied list. `tests/unit/test_player_theme.py` checks the names read here
 * are declared in `player.css`.
 */
export function readTheme(element) {
  const styles = getComputedStyle(element);
  const read = (name, fallback) => styles.getPropertyValue(name).trim() || fallback;
  return {
    background: read("--bg", "#07080d"),
    panel: read("--panel", "rgba(13, 18, 28, 0.86)"),
    line: read("--line", "rgba(83, 246, 255, 0.32)"),
    lineHot: read("--line-hot", "#42f7ff"),
    // Not the same slot as `lineHot`, though on `neon` it holds the same value.
    // `lineHot` is what an ordinary object is outlined in; this is what a
    // highlighted one glows, and what the page chrome is coloured with. A light
    // palette needs the two apart — see the top of `player.css`.
    accent: read("--accent", "#42f7ff"),
    magenta: read("--magenta", "#ff2fd6"),
    yellow: read("--yellow", "#f4f06d"),
    green: read("--green", "#5dff9c"),
    text: read("--text", "#ecfbff"),
    muted: read("--muted", "#91a6b8"),
    danger: read("--danger", "#ff5e7a"),
  };
}
