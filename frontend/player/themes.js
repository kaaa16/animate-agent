/**
 * The palettes the picker offers: which ones exist, and what to call them.
 *
 * The colours themselves are not here. They are in `player.css`, one
 * `[data-theme="…"]` block each, because a palette is a fact about how the page
 * is presented and the stylesheet is where those live — and because that is what
 * lets a swatch paint itself with the theme it is offering instead of with a
 * copy of it (see the note at the top of `player.css`).
 *
 * What is left for this file is the part CSS cannot express: an ordered list,
 * a label per id, and which one is the default. The ids are compared against the
 * stylesheet's selectors by `tests/unit/test_player_theme.py` — a theme added to
 * one and not the other is either an unreachable palette or a button that does
 * nothing, and neither shows up until somebody clicks it.
 */

/**
 * In the order the picker draws them: the two dark alternatives, then the three
 * light ones. `neon` leads because it is the default, and a picker whose first
 * item is not what you are looking at reads as broken.
 */
export const THEMES = [
  { id: "neon", label: "霓虹" },
  { id: "dusk", label: "暮紫" },
  { id: "blueprint", label: "蓝图" },
  { id: "paper", label: "纸白" },
  { id: "mint", label: "薄荷" },
  { id: "sakura", label: "樱粉" },
];

/** The palette a page with no `?theme=`, no stored choice and no `player.css` gets. */
export const DEFAULT_THEME = "neon";

/**
 * Where a click is remembered.
 *
 * Read a second time by the inline script in `index.html`, which sets the
 * attribute before the first paint and so cannot import this module. The two
 * strings have to match; `test_the_prepaint_script_reads_the_same_key` holds
 * them together, and it is a test rather than a comment because the failure is
 * silent — a mistyped key here means the choice is saved and never restored.
 */
export const STORAGE_KEY = "player-theme";

/** `id` if the picker offers it, otherwise the default. */
export function resolveTheme(id) {
  return THEMES.some((theme) => theme.id === id) ? id : DEFAULT_THEME;
}
