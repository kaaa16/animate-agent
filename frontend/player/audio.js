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
 *   "load and let the old one finish", or a fast seek leaves two voices talking.
 * - **Pause and resume** — the element is paused with the picture, and resumed
 *   from `sim.t`, which is where the picture is. Not from where the audio was:
 *   the two agree only if both were paused, and a film started at beat five has
 *   them apart from the beginning.
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
 * A sixth path arrived with the seek bar, and it is the same move as the voice
 * change: `player.js:seekTo` lands on a beat and then asks for that beat's clip
 * from an offset. Nothing here had to change for it — `play(url, from)` already
 * serves it — but it is the reason `setStep`'s own `playBeat` has to be
 * suppressed while a seek is in flight. That suppression lives in `player.js`,
 * because it is a fact about the seek and not about the sound.
 *
 * Two rules that are not about timing. A clip that will not load is **silence
 * for that beat and not a failure**: the beat's length comes from the spec, so
 * a 404 costs nothing but the sound, and `fail()` — which stops the whole
 * player — would turn one missing mp3 into a black screen. And `play()` returns
 * a promise that browsers reject when there has been no gesture yet, so it is
 * always caught: the rejection is the only signal that the page is muted by
 * policy, and it is what `onBlocked` reports.
 *
 * ## Volume, and the second output the recording needs
 *
 * There are two things that can be done with this element's sound: hear it, and
 * record it. They want different volumes — a viewer who turned the film down
 * does not want the downloaded file turned down with it — so the two cannot
 * share the element's own `volume`/`muted`, which sit *upstream* of everything.
 *
 * Until `captureTrack` is called there is no graph and no recording, so the
 * element's own properties cost nothing and are what the slider drives. The
 * first recording builds a WebAudio graph around the element, and from then on
 * the element's sound reaches the speakers *only* through that graph — at which
 * point the slider becomes a gain node and the element's properties are left
 * alone for good. `captureTrack` carries the rest of that argument.
 */

/**
 * @param {(blocked: boolean) => void} [onBlocked] Called when the browser starts
 *   or stops refusing to play without a gesture, so the page can show or hide
 *   its hint. Only ever called on a change.
 */
export function createVoiceTrack({ onBlocked } = {}) {
  const element = new Audio();
  // Nothing is fetched until a beat is actually played, with one exception:
  // `prime`, which `player.js` calls once at startup for the beat already on
  // screen. The rule this line is really keeping is that a clip the viewer may
  // never reach is never fetched — the player opens on one beat and only ever
  // plays one clip at a time, so preloading a run of them would be a request for
  // audio nobody asked for, and on a page opened with `?scene=5` a request for
  // the wrong scene's.
  element.preload = "none";

  let loaded = "";
  let blocked = false;

  //: What the viewer asked for, 0 to 1, and whether they asked for silence.
  //: Held here rather than read back off the element, because after the graph
  //: exists the element no longer knows about either.
  let volume = 1;
  let muted = false;

  //: `{ context, source, speaker, tap }`, or null until the first recording.
  //: See the module docstring for why it does not exist from the start.
  let graph = null;

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
  //: moved on — otherwise a fast seek leaves a pending seek to jump the *new*
  //: clip to the old beat's offset.
  let generation = 0;

  /**
   * Put the playhead at `from`, once there is a clip to put it in.
   *
   * Assigning `currentTime` before the metadata has arrived throws in some
   * browsers and is silently ignored in others, so the seek waits for
   * `loadedmetadata` when it has to.
   *
   * **Always sets it, including to zero.** That is what makes one code path
   * serve every way a clip starts: a new beat (`from` is 0), the same beat
   * replayed (0 again, and a no-op early return here would leave it playing
   * from the middle), a resume (`from` is `sim.t`), a voice changed mid-beat (a
   * new clip, checked in at `sim.t`), and the seek bar landing anywhere inside
   * a beat. A `from` past the end clamps to the end and the clip finishes at
   * once, which is right for a beat resumed long after it started.
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

  /**
   * Push the stored volume at whichever of the two places is currently live.
   *
   * Called on every change and once when the graph is built, so a volume set
   * before there was any recording is not lost the moment there is one.
   */
  function applyVolume() {
    if (graph === null) {
      element.volume = volume;
      element.muted = muted;
      return;
    }
    graph.speaker.gain.value = muted ? 0 : volume;
  }

  /**
   * The one sound, as a track `MediaRecorder` can be handed.
   *
   * Builds the graph on first use and returns the recording branch's track. The
   * shape is two branches off one source node:
   *
   *     element -> source -> tap     (full volume, goes into the recording)
   *                      -> speaker -> destination   (what the viewer hears)
   *
   * The tap is deliberately **before** the speaker gain. A downloaded film that
   * came out at whatever the slider happened to be at would be a file whose
   * loudness is an accident of one person's session.
   *
   * Three things here are load-bearing:
   *
   * - **`resume()` before anything is connected.** A context built outside a
   *   gesture starts suspended, and from the moment `createMediaElementSource`
   *   exists the element reaches the speakers only through the graph. A
   *   suspended graph is therefore *silence*, while `element.play()` still
   *   resolves and `onBlocked` never fires — the film goes quiet with nothing on
   *   the page saying why. That is the worst failure this module can have, and
   *   it is one `await` away.
   * - **Built once.** `createMediaElementSource` may be called once per element
   *   for the lifetime of the page and throws on a second call, which is why the
   *   element is reused at all and why this caches.
   * - **The element's own `volume`/`muted` are never touched again.** They apply
   *   upstream of this tap, so using them for the slider would silence the
   *   recording along with the room — the footgun this whole arrangement exists
   *   to avoid.
   */
  async function captureTrack() {
    if (graph === null) {
      const Context = window.AudioContext || window.webkitAudioContext;
      const context = new Context();
      await context.resume();

      const source = context.createMediaElementSource(element);
      const speaker = context.createGain();
      const tap = context.createMediaStreamDestination();
      source.connect(tap);
      source.connect(speaker);
      speaker.connect(context.destination);

      graph = { context, source, speaker, tap };
      // The slider may already have been moved. The element remembers it and the
      // graph does not, so this is where the two hand over.
      applyVolume();
    }
    await graph.context.resume();
    return graph.tap.stream.getAudioTracks()[0];
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

    /** What the viewer hears, 0 to 1. Never what the recording hears. */
    setVolume(value) {
      volume = Math.max(0, Math.min(1, Number(value)));
      applyVolume();
    },

    setMuted(value) {
      muted = Boolean(value);
      applyVolume();
    },

    /** See `captureTrack`. Async because the context has to be resumed first. */
    captureTrack,

    /**
     * Fetch and decode a clip without making a sound, so the press that plays it
     * does not wait for the bytes.
     *
     * The film opens paused on a beat with the transport right there, and the
     * first press of 播放 is the one press whose cost is **audible**. `setStep`
     * starts a beat's clock at the moment it hands the clip to `play`, and the
     * sound does not start until the browser has it — so the beat runs on the
     * spec's clock while the voice is still arriving, and the difference comes
     * off the *end* of the sentence, because nothing stops a clip early. The
     * last word is the one that goes.
     *
     * Measured on a cold element, 2026-09-29: **460ms** to first sound, against
     * **15ms** for the next beat. That is what the reader hears as 「回车」 missing
     * from the opening line — 460ms is longer than the word is spoken, and the
     * caption carries it either way.
     *
     * Only the element's own first play is that expensive; every later beat
     * inherits a warm pipeline. So this is called once, for the beat already on
     * screen, and `preload` stays `"none"` for everything the viewer has not
     * been shown.
     */
    prime(url) {
      if (!url || loaded === url) return;
      loaded = url;
      generation += 1;
      element.preload = "auto";
      element.src = url;
      element.load();
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
