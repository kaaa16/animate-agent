/**
 * Talk about what to explain, then have it made.
 *
 * Two ways in and one button out. Drop a document and 「生成动画」 runs it through
 * the pipeline; say nothing about files and the same button has the model turn
 * the conversation into an article first, then runs *that* through the pipeline.
 * The page does not care which — it posts, waits, and shows what comes back.
 *
 * One request, one response, no polling, no streaming. The chain behind the
 * animation endpoints makes two or three model calls, so this can take minutes —
 * which is exactly why the progress display says it has no progress: there is one
 * HTTP response and nothing reports on it.
 *
 * Server-authored strings reach the page only through `textContent` (decision
 * D4 — model output is data, never code), and the spec URL that comes back is
 * treated as data too: checked, then encoded, before it is allowed near the
 * frame's `src`.
 */

const FILE_ENDPOINT = "/api/animations/from-file";
const CHAT_ENDPOINT = "/api/chat";
const CHAT_ANIMATION_ENDPOINT = "/api/animations/from-chat";
const PLAYER_URL = "/player/index.html";
const SPEC_PREFIX = "/specs/";

/**
 * Typing one of these and pressing 发送 runs the pipeline instead of asking a
 * question. The whole message has to be one of them, not merely contain one:
 * 「光合作用如何生成有机物」 contains 生成 and is a question about plants.
 * A near-miss costs one chat turn; a false positive costs a three-minute run.
 */
const GENERATE_COMMANDS = ["生成动画", "开始生成", "生成吧"];

/**
 * The same six extensions the server accepts (`SUPPORTED_EXTENSIONS` in
 * `documents/file_parser.py`). Duplicated deliberately, and a test asserts the
 * two lists agree: `accept=` on the input is only a hint — a picker may ignore
 * it and drag-and-drop ignores it completely, so the check has to exist here.
 */
const ALLOWED_EXTENSIONS = [".pptx", ".docx", ".pdf", ".md", ".markdown", ".txt"];

const ui = {
  dropZone: document.getElementById("dropZone"),
  fileInput: document.getElementById("fileInput"),
  pickBtn: document.getElementById("pickBtn"),
  clearFileBtn: document.getElementById("clearFileBtn"),
  goBtn: document.getElementById("goBtn"),
  voiceSelect: document.getElementById("voiceSelect"),
  fileStatus: document.getElementById("fileStatus"),
  chatLog: document.getElementById("chatLog"),
  chatInput: document.getElementById("chatInput"),
  sendBtn: document.getElementById("sendBtn"),
  progress: document.getElementById("progress"),
  progressLine: document.getElementById("progressLine"),
  alert: document.getElementById("alert"),
  result: document.getElementById("result"),
  resultTitle: document.getElementById("resultTitle"),
  resultMeta: document.getElementById("resultMeta"),
  openLink: document.getElementById("openLink"),
  player: document.getElementById("player"),
};

const state = {
  file: null,
  busy: false,
  timer: null,
  startedAt: 0,
  /**
   * The conversation as it will be sent, which is not the same thing as what is
   * on screen. The article that comes back is shown in the log but deliberately
   * *not* pushed in here: the next request asks the model to turn "the above
   * discussion" into a script, and a previous draft sitting in the transcript is
   * material to copy from rather than a conversation to summarise.
   *
   * In memory only. Reloading loses the conversation, and nothing about it
   * reaches the server except when a request is in flight — there is no session,
   * no job id, and no transcript on disk.
   */
  messages: [],
  /** A chat turn is in flight. Distinct from `busy`, which is a film rendering. */
  chatting: false,
};

/* ------------------------------------------------------------------ display */

function extensionOf(name) {
  const dot = name.lastIndexOf(".");
  return dot === -1 ? "" : name.slice(dot).toLowerCase();
}

function humanSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function showAlert(message) {
  ui.alert.textContent = message;
  ui.alert.hidden = false;
}

function clearAlert() {
  ui.alert.hidden = true;
  ui.alert.textContent = "";
}

function scrollLogToEnd() {
  ui.chatLog.scrollTop = ui.chatLog.scrollHeight;
}

/**
 * Add one line to the conversation log.
 *
 * Built with `createElement` and `textContent` rather than markup, because half
 * of what goes in here is written by the model (D4).
 */
function append(role, text) {
  const item = document.createElement("li");
  // Class names go in without a leading dot: `test_web_page.py` reads every
  // quoted `.word` string in this file as a file extension.
  item.className = role === "user" ? "bubble from-user" : "bubble from-ai";
  item.textContent = text;
  ui.chatLog.appendChild(item);
  scrollLogToEnd();
  return item;
}

/** A note from the page itself — an attachment, a failure. Not part of the chat. */
function appendNote(text) {
  const item = document.createElement("li");
  item.className = "note";
  item.textContent = text;
  ui.chatLog.appendChild(item);
  scrollLogToEnd();
  return item;
}

/**
 * What 「生成动画」 should be. A disabled button that looks live is worse than an
 * obviously dead one, and "is there anything to generate" is now two questions
 * rather than one.
 */
function refreshGenerateButton() {
  ui.goBtn.disabled = state.busy || (state.file === null && state.messages.length === 0);
  ui.sendBtn.disabled = state.chatting;
  ui.clearFileBtn.hidden = state.file === null;
}

function setBusy(busy) {
  state.busy = busy;
  ui.pickBtn.disabled = busy;
  ui.goBtn.textContent = busy ? "生成中…" : "生成动画";
  ui.progress.hidden = !busy;
  refreshGenerateButton();

  if (busy) {
    // Elapsed seconds rather than a percentage: it is a true thing this page can
    // know, and the only true thing it can show about the server's progress.
    state.startedAt = Date.now();
    ui.progressLine.textContent = "正在生成… 已等待 0 秒";
    state.timer = window.setInterval(() => {
      const seconds = Math.round((Date.now() - state.startedAt) / 1000);
      ui.progressLine.textContent = `正在生成… 已等待 ${seconds} 秒`;
    }, 1000);
    return;
  }

  if (state.timer !== null) {
    window.clearInterval(state.timer);
    state.timer = null;
  }
}

/* ------------------------------------------------------------------- choose */

function choose(file) {
  if (!file) return;
  const extension = extensionOf(file.name);
  if (!ALLOWED_EXTENSIONS.includes(extension)) {
    state.file = null;
    ui.fileStatus.textContent = `已拒绝：${file.name}`;
    refreshGenerateButton();
    showAlert(
      `不支持的文件格式：${extension || "（没有扩展名）"}。` +
        `支持：${ALLOWED_EXTENSIONS.join("、")}。`,
    );
    return;
  }
  clearAlert();
  state.file = file;
  const size = humanSize(file.size);
  ui.fileStatus.textContent = `已挂上：${file.name}（${size}）`;
  appendNote(`已挂上文件：${file.name}（${size}）。点「生成动画」就按这份文档做。`);
  refreshGenerateButton();
}

function clearFile() {
  if (state.file === null) return;
  const name = state.file.name;
  state.file = null;
  ui.fileInput.value = "";
  ui.fileStatus.textContent = "还没有挂文件。";
  appendNote(`已移除文件：${name}。现在点「生成动画」会按上面的对话来做。`);
  refreshGenerateButton();
}

/* ----------------------------------------------------------------- transports */

/**
 * Read a JSON body, or throw something a person can act on.
 *
 * An unhandled exception comes back as `text/plain` ("Internal Server Error",
 * starlette/middleware/errors.py), so the parse failure is the ordinary shape of
 * a server-side crash rather than a freak case.
 */
async function readJson(response) {
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(
      `服务端返回了非 JSON 响应（HTTP ${response.status}）。` +
        `500 的正文是 text/plain，所以这一条多半意味着服务端抛了异常——去看服务端日志。`,
    );
  }
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}：${payload.detail ?? "服务端没有给出原因"}`);
  }
  return payload;
}

function unreachable(error) {
  return `请求没有发出去，或被中断：${error instanceof Error ? error.message : error}`;
}

/**
 * The voice, as form/JSON fields, or nothing at all.
 *
 * An empty string is the 「不要配音」 option, and an absent `speak` is the
 * server's "do what the config says" — and the config says no. So an unchosen
 * voice and a page that had never heard of voices are the same request.
 */
function voiceFields() {
  const voice = ui.voiceSelect.value;
  return voice ? { speak: true, voice } : null;
}

async function postFile(file) {
  const body = new FormData();
  body.append("file", file, file.name);
  const speech = voiceFields();
  if (speech) {
    body.append("speak", "true");
    body.append("voice", speech.voice);
  }

  let response;
  try {
    response = await fetch(FILE_ENDPOINT, { method: "POST", body });
  } catch (error) {
    throw new Error(unreachable(error));
  }
  return readJson(response);
}

async function postTurn(messages) {
  let response;
  try {
    response = await fetch(CHAT_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages }),
    });
  } catch (error) {
    throw new Error(unreachable(error));
  }
  return readJson(response);
}

async function postFromChat(messages) {
  const speech = voiceFields();
  const body = speech ? { messages, speak: true, voice: speech.voice } : { messages };

  let response;
  try {
    response = await fetch(CHAT_ANIMATION_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (error) {
    throw new Error(unreachable(error));
  }
  return readJson(response);
}

/* ------------------------------------------------------------------ talking */

function isGenerateCommand(text) {
  // One trailing punctuation mark forgiven, because a sentence typed into a box
  // tends to collect one. Anything more than that is a sentence, not a command.
  const trimmed = text.trim().replace(/[。！!，,、~～]+$/, "");
  return GENERATE_COMMANDS.includes(trimmed);
}

async function send() {
  if (state.chatting) return;
  const text = ui.chatInput.value.trim();
  if (!text) return;
  clearAlert();

  if (isGenerateCommand(text)) {
    ui.chatInput.value = "";
    await generate();
    return;
  }

  ui.chatInput.value = "";
  append("user", text);
  state.messages.push({ role: "user", content: text });
  state.chatting = true;
  refreshGenerateButton();

  const pending = append("ai", "正在想…");
  pending.classList.add("is-thinking");

  try {
    const payload = await postTurn(state.messages);
    pending.textContent = String(payload.reply ?? "");
    state.messages.push({ role: "assistant", content: String(payload.reply ?? "") });
  } catch (error) {
    // The failed turn stays in the log and in the transcript. It happened, the
    // user typed it, and it is part of what they want explained.
    pending.classList.remove("is-thinking");
    pending.className = "note";
    pending.textContent = `这轮没答上来：${error instanceof Error ? error.message : error}`;
    showAlert(`AI 没有回话：${error instanceof Error ? error.message : error}`);
  } finally {
    state.chatting = false;
    refreshGenerateButton();
  }
}

/* ----------------------------------------------------------------- generate */

function fail(message) {
  clearAlert();
  showAlert(message);
  appendNote(message);
  setBusy(false);
}

function succeed(payload) {
  // The frame's `src` is built out of a server value, so it is checked first and
  // encoded second. A URL that is not this site's `/specs/` path is refused
  // rather than loaded: the page has no reason to fetch anything else.
  if (typeof payload.spec_url !== "string" || !payload.spec_url.startsWith(SPEC_PREFIX)) {
    fail(
      `服务端返回的 spec 地址不是本站的 ${SPEC_PREFIX} 路径，已拒绝加载：` +
        `${String(payload.spec_url)}`,
    );
    return;
  }

  if (typeof payload.article === "string" && payload.article) {
    append("ai", `这是我整理出来的稿子（${payload.article_path}）：\n\n${payload.article}`);
  }

  ui.resultTitle.textContent = String(payload.title ?? "（没有标题）");
  ui.resultMeta.textContent =
    `${String(payload.scene_count ?? "?")} 幕 · ` +
    `${String(payload.storyboard_id ?? "")} · ${String(payload.spec_path ?? "")}`;

  const playerUrl = `${PLAYER_URL}?spec=${encodeURIComponent(payload.spec_url)}`;
  ui.openLink.href = playerUrl;
  // The frame asks to start on its own; the link does not. The player opens
  // paused, which is right for a link someone was handed — but this frame sits
  // under a 生成动画 button that has just said the run finished, and a still
  // picture there reads as a failure rather than as a film waiting to start.
  ui.player.src = `${playerUrl}&autoplay=1`;
  ui.result.hidden = false;
  setBusy(false);
}

async function generate() {
  if (state.busy) return;
  if (state.file === null && state.messages.length === 0) return;
  const file = state.file;
  clearAlert();
  ui.result.hidden = true;
  ui.player.removeAttribute("src");
  setBusy(true);

  let payload;
  try {
    payload = file === null ? await postFromChat(state.messages) : await postFile(file);
  } catch (error) {
    fail(`生成失败：${error instanceof Error ? error.message : error}`);
    return;
  }

  succeed(payload);
}

/* ---------------------------------------------------------------------- bind */

function bind() {
  ui.pickBtn.addEventListener("click", () => ui.fileInput.click());
  ui.fileInput.addEventListener("change", () => {
    const files = ui.fileInput.files;
    if (files && files.length > 0) {
      if (files.length > 1) {
        showAlert(`一次只处理一份文档，已忽略其余 ${files.length - 1} 个。`);
      }
      choose(files[0]);
    }
  });
  ui.clearFileBtn.addEventListener("click", clearFile);
  ui.goBtn.addEventListener("click", generate);
  ui.sendBtn.addEventListener("click", send);
  ui.chatInput.addEventListener("keydown", (event) => {
    // Enter sends, Shift+Enter is a newline. Without this the box is a textarea
    // that swallows the one key everybody presses in a chat box.
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  });

  ["dragenter", "dragover"].forEach((name) => {
    ui.dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      ui.dropZone.classList.add("is-dragging");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    ui.dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      ui.dropZone.classList.remove("is-dragging");
    });
  });
  ui.dropZone.addEventListener("drop", (event) => {
    const files = event.dataTransfer && event.dataTransfer.files;
    if (files && files.length > 0) choose(files[0]);
  });

  // A drop that lands a few pixels outside the zone is otherwise handled by the
  // browser, which navigates the tab to the file and replaces this page with raw
  // bytes — a broken screen with nothing of ours in the causal chain.
  ["dragover", "drop"].forEach((name) => {
    window.addEventListener(name, (event) => event.preventDefault());
  });

  refreshGenerateButton();
}

function init() {
  const missing = Object.keys(ui).filter((name) => ui[name] === null);
  if (missing.length > 0) {
    // Markup and script have drifted apart. Say which piece is gone; binding
    // nothing at all would look like a page that simply does not react.
    document.body.textContent = `页面元素没有找齐，index.html 与 app.js 不同步。缺少：${missing.join("、")}`;
    return;
  }
  bind();
}

init();
