/**
 * The voices a spec may carry: which ones exist, and what to call them.
 *
 * The same three things `themes.js` holds — an ordered list, a label per id, and
 * which is the default — and for the same reason: colour lives in CSS, and a
 * voice lives in an mp3 the spec points at, so all that is left for a module is
 * the part neither can express.
 *
 * **This file must stay free of the DOM.** `tools/player_smoke.mjs` imports it
 * under `node` and runs `beatSeconds` against real specs, which is the only way
 * any of this gets executed outside a browser. `audio.js` is the module that
 * touches an `Audio` element; this one is arithmetic.
 *
 * The ids are held to `speech/voices.py` by `tests/unit/test_player_voice.py`,
 * the way `themes.js` is held to `registry.PALETTES`. A voice in one and not the
 * other is a button that does nothing, or a voice nobody can reach.
 */

/**
 * In the order the picker draws them. `zh-CN-YunjianNeural` leads because it is
 * the default in `SpeechSettings`, and a picker whose first item is not what you
 * are hearing reads as broken.
 */
export const VOICES = [
  { id: "zh-CN-YunjianNeural", label: "云健 · 男声播报" },
  { id: "zh-CN-XiaoyiNeural", label: "晓伊 · 女声活泼" },
];

/** The voice a page with no `?voice=`, no `spec.voice` and no stored choice gets. */
export const DEFAULT_VOICE = VOICES[0].id;

/**
 * Where a click is remembered. Deliberately not the theme's key: the two
 * preferences are independent, and sharing one would mean switching the palette
 * changed the narrator.
 */
export const STORAGE_KEY = "player-voice";

/**
 * Which voices this spec actually has, in picker order.
 *
 * Derived from the steps rather than stored on the spec, because the spec saying
 * so would be a second copy of a fact it already contains — and a film whose
 * `voices` list disagreed with its own audio would be a picker offering a
 * silence. A spec with no speech at all answers `[]`, which is what makes "is
 * there anything to choose between" a question about data.
 */
export function voiceIdsIn(spec) {
  const present = new Set();
  for (const scene of spec?.scenes ?? []) {
    for (const step of scene.steps ?? []) {
      for (const id of Object.keys(step.speech ?? {})) present.add(id);
    }
  }
  // `VOICES` order, not discovery order: the two are the same when the spec was
  // written by `attach_speech`, and differ exactly when it was not — in which
  // case the picker should look like the picker.
  return VOICES.map((voice) => voice.id).filter((id) => present.has(id));
}

/**
 * Which voice to open with, given what was asked for and what is on the film.
 *
 * `asked` is the whole chain, already collapsed by the caller the way the
 * theme's is: `?voice=` first, because a link someone was handed is a statement
 * about what they were handed; then the spec's own voice, because a viewer who
 * once clicked 晓伊 should still hear what the filmmaker chose; then what they
 * chose last time. It is one expression at the call site so that the order is
 * legible in one place.
 *
 * What is left for this function is the narrowing, and it is the part that
 * matters for a page that must not go silent: a spec carries only the voices it
 * was *generated* with, and a link asking for one it does not have would
 * otherwise resolve to a voice with no clips behind it. Falling through to the
 * first voice the film actually has is the difference between the wrong narrator
 * and no narrator.
 *
 * Returns `null` when the spec carries no voice at all. That is not an error —
 * it is an unvoiced spec, which is every spec written before this existed.
 */
export function resolveVoice(asked, offered) {
  if (!offered || offered.length === 0) return null;
  return offered.includes(asked) ? asked : offered[0];
}

/**
 * How long this beat holds, in the voice now playing.
 *
 * The speech pass measures a beat rather than deriving it — the clip opens with
 * a little silence and ends with a lot more, and the character formula cannot
 * see either — so when there is a track for this voice, its length is the
 * answer. `step.duration` is the fallback for a spec with no audio, and it still
 * carries the generated voice's measured length as well, so a player that knows
 * nothing about `speech` walks the film at the right pace.
 *
 * Zero means "no timing was declared": `advanceBeat` reads that as "sit here",
 * which is what every spec written before `duration` existed expects.
 */
export function beatSeconds(step, voice) {
  const measured = voice ? step?.speech?.[voice]?.seconds : undefined;
  if (typeof measured === "number" && measured > 0) return measured;
  const declared = step?.duration;
  return typeof declared === "number" ? declared : 0;
}

/** Where a beat's clip is, given where its spec came from. */
export function clipUrl(step, voice, specUrl) {
  const src = voice ? step?.speech?.[voice]?.src : undefined;
  if (!src) return "";
  // A string join rather than `new URL(src, spec)`: the second argument has to
  // be absolute, and this module cannot reach `location` to make it so without
  // ceasing to run under `node`.
  //
  // Correct here because `src` is not a URL anyone typed — it is
  // `audio/<voice>/<hash>.mp3`, written by `speech.cache.relative_src`, with no
  // scheme, no leading slash and no `..` in it, and the spec it is relative to
  // sits in the directory the clips are under. Anything else would need the URL
  // parser, so `test_player_voice.py` pins the flat shape rather than trusting
  // that this stays true.
  const clean = String(specUrl ?? "").split(/[?#]/)[0];
  const cut = clean.lastIndexOf("/");
  return cut >= 0 ? clean.slice(0, cut + 1) + src : src;
}

/** The label for `id`, falling back to the id itself for a voice we do not know. */
export function voiceLabel(id) {
  const found = VOICES.find((voice) => voice.id === id);
  return found ? found.label : id;
}
