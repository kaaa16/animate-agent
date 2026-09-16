/**
 * Upload a document and show what the pipeline makes of it.
 *
 * One request, one response, no polling, no streaming. The chain behind
 * `/api/animations/from-file` makes two model calls, so this can take a minute
 * or two — which is exactly why the progress display says it has no progress:
 * there is one HTTP response and nothing reports on it. A bar that filled in
 * stage by stage would be a drawing of a process nobody is watching.
 *
 * Server-authored strings reach the page only through `textContent` (decision
 * D4 — model output is data, never code), and the spec URL that comes back is
 * treated as data too: checked, then encoded, before it is allowed near the
 * frame's `src`.
 */

const ENDPOINT = "/api/animations/from-file";
const PLAYER_URL = "/player/index.html";
const SPEC_PREFIX = "/specs/";

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
  goBtn: document.getElementById("goBtn"),
  fileStatus: document.getElementById("fileStatus"),
  progress: document.getElementById("progress"),
  progressLine: document.getElementById("progressLine"),
  alert: document.getElementById("alert"),
  result: document.getElementById("result"),
  resultTitle: document.getElementById("resultTitle"),
  resultMeta: document.getElementById("resultMeta"),
  openLink: document.getElementById("openLink"),
  player: document.getElementById("player"),
};

const state = { file: null, busy: false, timer: null, startedAt: 0 };

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

function setBusy(busy) {
  state.busy = busy;
  ui.goBtn.disabled = busy || state.file === null;
  ui.pickBtn.disabled = busy;
  ui.goBtn.textContent = busy ? "生成中…" : "生成动画";
  ui.progress.hidden = !busy;

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
    ui.goBtn.disabled = true;
    ui.fileStatus.textContent = `已拒绝：${file.name}`;
    showAlert(
      `不支持的文件格式：${extension || "（没有扩展名）"}。` +
        `支持：${ALLOWED_EXTENSIONS.join("、")}。`,
    );
    return;
  }
  clearAlert();
  state.file = file;
  ui.fileStatus.textContent = `已选择：${file.name}（${humanSize(file.size)}）`;
  ui.goBtn.disabled = state.busy;
}

/* ----------------------------------------------------------------- generate */

function fail(message) {
  clearAlert();
  showAlert(message);
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

  ui.resultTitle.textContent = String(payload.title ?? "（没有标题）");
  ui.resultMeta.textContent =
    `${String(payload.scene_count ?? "?")} 幕 · ` +
    `${String(payload.storyboard_id ?? "")} · ${String(payload.spec_path ?? "")}`;

  const playerUrl = `${PLAYER_URL}?spec=${encodeURIComponent(payload.spec_url)}`;
  ui.openLink.href = playerUrl;
  ui.player.src = playerUrl;
  ui.result.hidden = false;
  setBusy(false);
}

async function generate() {
  if (state.busy || state.file === null) return;
  const file = state.file;
  clearAlert();
  ui.result.hidden = true;
  ui.player.removeAttribute("src");
  setBusy(true);

  const body = new FormData();
  body.append("file", file, file.name);

  let response;
  try {
    response = await fetch(ENDPOINT, { method: "POST", body });
  } catch (error) {
    fail(`请求没有发出去，或被中断：${error instanceof Error ? error.message : error}`);
    return;
  }

  let payload;
  try {
    payload = await response.json();
  } catch {
    // An unhandled exception comes back as `text/plain` ("Internal Server
    // Error", starlette/middleware/errors.py), so this branch is the ordinary
    // shape of a server-side crash rather than a freak case.
    fail(
      `服务端返回了非 JSON 响应（HTTP ${response.status}）。` +
        `500 的正文是 text/plain，所以这一条多半意味着服务端抛了异常——去看服务端日志。`,
    );
    return;
  }

  if (!response.ok) {
    fail(`生成失败（HTTP ${response.status}）：${payload.detail ?? "服务端没有给出原因"}`);
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
  ui.goBtn.addEventListener("click", generate);

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
