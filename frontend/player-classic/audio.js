/**
 * The voice track: one `Audio` element, reused, and the five ways it can drift
 * out of step with the picture.
 *
 * `sim.t` is the beat's clock and `setStep` puts it back to zero; an `Audio`
 * element is an object that outlives a beat. The two are kept together by hand,
 * which means there is a right answer for every one of the paths below and a
 * symptom for every one that is missed — and no automated test can see any of
 * them, because there is no audio in a grep and none in the smoke harness. It is
 * the one part of this player that only a person listening can check. So:
 *
 * - **Beat change** — stop, then point at the new clip. "Stop first" and not
 *   "load and let the old one finish", or a fast ◀▶ leaves two voices talking.
 * - **Pause and resume** — the element is paused with the picture, and resumed
 *   from `sim.t`, which is where the picture is. Not from where the audio was:
 *   the two agree only if both were paused, and a run that started at beat five
 *   has them apart from the beginning.
 * - **Scene change** — stopped the moment the fade starts. The beat's own clock
 *   still runs through the dip, so without this the last sentence of a scene
 *   plays over the fade and arrives under the next scene's title.
 * - **A fade paused halfway** — nothing is resumed, because the beat on screen
 *   belongs to a scene that is on its way out. `player.js` owns that decision;
 *   this module only has to not fight it.
 * - **Voice change** — the new clip is loaded from `sim.t`, because the viewer
 *   asked to hear the rest of this beat in the other voice, not to hear it from
 *   the top.
 *
 * Two rules that are not about timing. A clip that will not load is **silence
 * for that beat and not a failure**: the beat's length comes from the spec, so
 * a 404 costs nothing but the sound, and `fail()` — which stops the whole
 * player — would turn one missing mp3 into a black screen. And `play()` returns
 * a promise that browsers reject when there has been no gesture yet, so it is
 * always caught: the rejection is the only signal that the page is muted by
 * policy, and it is what `onBlocked` reports.
 */

/**
 * @param {(blocked: boolean) => void} [onBlocked] Called when the browser starts
 *   or stops refusing to play without a gesture, so the page can show or hide
 *   its hint. Only ever called on a change.
 */
export function createVoiceTrack({ onBlocked } = {}) {
  const element = new Audio();
  // Nothing is fetched until a beat is actually played. The player opens on beat
  // one and only ever plays one clip at a time, so preloading would be a request
  // for audio the viewer may never reach — and on a page opened with
  // `?scene=5` it would be a request for the wrong scene's.
  element.preload = "none";

  let loaded = "";
  let blocked = false;

  function setBlocked(value) {
    if (blocked === value) return;
    blocked = value;
    if (onBlocked) onBlocked(blocked);
  }

  element.addEventListener("error", () => {
    // See the module docstring. `loaded` is cleared so that the next beat does
    // not compare equal to a clip that never arrived and skip the reload.
    loaded = "";
  });

  //: Bumped whenever the clip is replaced. A seek that has to wait for metadata
  //: captures the value it was issued under and does nothing if the clip has
  //: moved on — otherwise a fast ◀▶ leaves a pending seek to jump the *new* clip
  //: to the old beat's offset.
  let generation = 0;

  /**
   * Put the playhead at `from`, once there is a clip to put it in.
   *
   * Assigning `currentTime` before the metadata has arrived throws in some
   * browsers and is silently ignored in others, so the seek waits for
   * `loadedmetadata` when it has to.
   *
   * **Always sets it, including to zero.** That is what makes one code path
   * serve all four ways a clip starts: a new beat (`from` is 0), the same beat
   * replayed with 重置 (0 again, and a no-op early return here would leave it
   * playing from the middle), a resume (`from` is `sim.t`), and a voice changed
   * mid-beat (a new clip, checked in at `sim.t`). A `from` past the end clamps to
   * the end and the clip finishes at once, which is right for a beat resumed
   * long after it started.
   */
  function seek(from) {
    const target = from > 0 ? from : 0;
    const issued = generation;
    if (element.readyState >= 1) {
      element.currentTime = target;
      return;
    }
    element.addEventListener(
      "loadedmetadata",
      () => {
        if (issued === generation) element.currentTime = target;
      },
      { once: true },
    );
  }

  async function start(from) {
    seek(from);
    try {
      await element.play();
      setBlocked(false);
    } catch (error) {
      // Not a failure to report. The page is allowed to draw; the browser is
      // refusing only to make a sound, and the hint says so.
      setBlocked(true);
    }
  }

  return {
    /** Whether the browser is currently refusing to play without a gesture. */
    get blocked() {
      return blocked;
    },

    /** Which clip is loaded, so a caller can avoid restarting the same one. */
    get src() {
      return loaded;
    },

    /** Point at a beat's clip and play it from `from` seconds in. */
    async play(url, from = 0) {
      if (!url) {
        this.stop();
        return;
      }
      if (loaded !== url) {
        loaded = url;
        generation += 1;
        element.src = url;
      }
      await start(from);
    },

    /** Stop the sound without forgetting which clip it was. */
    pause() {
      element.pause();
    },

    /** Carry on from `from`, which is where the picture has got to. */
    async resume(from) {
      if (!loaded) return;
      await start(from);
    },

    /** Silence, and let go of the clip. Safe to call when there is none. */
    stop() {
      element.pause();
      if (loaded === "") return;
      loaded = "";
      generation += 1;
      element.removeAttribute("src");
      // Required after clearing `src`, or the element keeps decoding what it was
      // given. Guarded by the `loaded` check above because `load()` with no
      // source fires `error` in some browsers, which would look like a broken
      // clip on every scene change.
      element.load();
    },
  };
}
