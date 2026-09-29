/**
 * Recording the film to a file, in the browser — the download button's engine.
 *
 * The picture is drawn from a spec, sixty times a second, and there is no file
 * anywhere behind it: right-clicking the canvas offers a still image and nothing
 * else. So "download the video" is not a fetch, it is a *recording* — the canvas
 * is captured as a live stream, the voice track is captured alongside it, and
 * `MediaRecorder` writes both into one container.
 *
 * Three consequences follow, and all three are visible to the person waiting:
 *
 * - **It takes as long as the film.** A recorder records; it cannot be asked to
 *   go faster. A 93-second film is 93 seconds of waiting, during which the page
 *   is playing that film.
 * - **Frames come from `requestAnimationFrame`.** `captureStream` emits a frame
 *   when the canvas is painted, and browsers throttle painting to nothing in a
 *   background tab. Recording and switching tabs therefore gives a file whose
 *   picture freezes at the moment of the switch, with no error anywhere. The
 *   page says so before it starts; it cannot prevent it.
 * - **A failure here is not the film's failure.** A missing codec or a refused
 *   recorder is reported next to the button, never through `fail()` — the same
 *   reasoning `audio.js` records for a 404'd clip. `fail()` stops the tick loop,
 *   which would turn "this browser cannot record" into a black screen.
 *
 * ## The codec has to name both halves
 *
 * `MediaRecorder` accepts a stream whose tracks need not share a container, and
 * a `mimeType` naming only a video codec is accepted happily — and then writes
 * a file with no audio in it. So the candidates below name *both* codecs, and
 * `isTypeSupported` is asked about the whole string. `video/mp4` on its own is
 * kept only as a last resort before WebM, because it is the one entry that
 * cannot be checked for an audio codec at all.
 *
 * Which one wins is the browser's answer, not ours: Chromium gives MP4 with
 * H.264 and AAC, Firefox gives WebM with VP9 and Opus and refuses every MP4
 * spelling. Hence the feature detection rather than a constant.
 */

/**
 * In preference order. MP4 first because it is the format that opens in
 * everything — an editor, a phone, a chat window — and WebM as the floor,
 * because a file that plays in the browser it was made in beats no file.
 */
const CANDIDATES = [
  'video/mp4;codecs="avc1.42E01E,mp4a.40.2"',
  'video/mp4;codecs="avc1.4D401E,mp4a.40.2"',
  "video/mp4",
  'video/webm;codecs="vp9,opus"',
  'video/webm;codecs="vp8,opus"',
  "video/webm",
];

/**
 * 30 rather than the display's refresh rate. The source is a canvas drawn from a
 * fixed-timestep simulation, so there is no motion between frames to lose, and a
 * 60 fps capture would double the file for a picture that is already smooth at
 * 30. It is also the one number here that a person can change if the file comes
 * out too big.
 */
const FRAMES_PER_SECOND = 30;

/**
 * Whether this browser can record, and what it would write.
 *
 * Returns the `mimeType` to hand `MediaRecorder`, or `""` when there is no
 * recorder to hand it to. Asking first is what lets the button say "this
 * browser cannot" instead of failing at the end of a ninety-second wait.
 */
export function recordableType() {
  if (typeof MediaRecorder === "undefined") return "";
  if (typeof HTMLCanvasElement === "undefined") return "";
  if (typeof HTMLCanvasElement.prototype.captureStream !== "function") return "";
  for (const type of CANDIDATES) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return "";
}

/** The extension a container wants, so the saved file opens by double-click. */
export function extensionFor(mimeType) {
  return String(mimeType).startsWith("video/mp4") ? ".mp4" : ".webm";
}

/**
 * Hand a finished blob to the browser as a download.
 *
 * The anchor is built with `createElement` and a `download` attribute rather
 * than a navigation: nothing here is a URL anyone typed, and the object URL is
 * revoked later rather than at once — a browser that has not finished taking the
 * file when the URL dies saves nothing, and its failure is silent.
 */
export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}

/**
 * @param {object} options
 * @param {HTMLCanvasElement} options.canvas What to record.
 * @param {() => Promise<MediaStreamTrack|null>} options.audioTrack The sound, as
 *   a track, fetched at `start()` rather than held: the WebAudio graph behind it
 *   is built lazily and `audio.js` owns when that happens.
 * @param {(message: string) => void} options.onProblem Say what went wrong. Not
 *   `fail()` — see the module docstring.
 * @param {(blob: Blob, mimeType: string) => void} options.onDone The finished
 *   file, once there is one.
 */
export function createRecorder({ canvas, audioTrack, onProblem, onDone }) {
  let recorder = null;
  //: The stream this recording opened. Kept so its video track can be stopped
  //: when the recording ends: a `captureStream` that is never stopped goes on
  //: capturing the canvas for the life of the page, and every download would
  //: leave another one behind.
  let stream = null;
  let chunks = [];
  let type = "";

  function finish() {
    const mimeType = type || recorder?.mimeType || "video/webm";
    const blob = new Blob(chunks, { type: mimeType });
    chunks = [];
    recorder = null;
    if (stream !== null) {
      for (const track of stream.getVideoTracks()) track.stop();
      stream = null;
    }
    if (blob.size === 0) {
      onProblem("录出来的文件是空的——录制期间页面可能被切到后台了。");
      return;
    }
    onDone(blob, mimeType);
  }

  return {
    /** Whether a recording is running. The transport bar locks pause on it. */
    get active() {
      return recorder !== null && recorder.state === "recording";
    },

    /**
     * Begin. Returns whether it did — the caller uses the answer to decide
     * between the recording overlay and staying put.
     */
    async start() {
      if (this.active) return true;
      type = recordableType();
      if (type === "") {
        onProblem("这个浏览器不能录制（缺 MediaRecorder 或 canvas.captureStream）。换 Chrome / Edge 试试。");
        return false;
      }

      const track = await audioTrack();
      if (track === null) {
        onProblem("拿不到声音轨，先随便点一下页面再试。");
        return false;
      }

      stream = canvas.captureStream(FRAMES_PER_SECOND);
      stream.addTrack(track);

      try {
        recorder = new MediaRecorder(stream, {
          mimeType: type,
          // Flat-colour motion graphics compress far better than footage, so
          // this is generous rather than thrifty; the alternative default is a
          // number nobody chose. `audio.js` has already fixed the loudness, so
          // the audio rate is the one thing here that only has to not be tiny.
          videoBitsPerSecond: 4000000,
          audioBitsPerSecond: 128000,
        });
      } catch (error) {
        for (const opened of stream.getVideoTracks()) opened.stop();
        stream = null;
        onProblem(`录制器打不开：${error instanceof Error ? error.message : error}`);
        return false;
      }

      chunks = [];
      recorder.addEventListener("dataavailable", (event) => {
        if (event.data && event.data.size > 0) chunks.push(event.data);
      });
      recorder.addEventListener("stop", finish);
      recorder.addEventListener("error", (event) => {
        const error = event.error ?? event;
        onProblem(`录制中断：${error instanceof Error ? error.message : error}`);
      });
      // A chunk a second, rather than one blob at the very end: ninety seconds
      // of video held only in the recorder's own buffer is ninety seconds that
      // a crash or a closed tab throws away.
      recorder.start(1000);
      return true;
    },

    /** End it. Idempotent — the film may also have reached its own end. */
    stop() {
      if (recorder !== null && recorder.state !== "inactive") recorder.stop();
    },
  };
}
