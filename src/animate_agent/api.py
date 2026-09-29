"""FastAPI entrypoint: the HTTP face of the pipeline, and the site that drives it.

Serves three things from one process and one port: the API itself, the upload
page at `/`, and the player plus the specs it fetches. That is deliberate — every
URL the page touches is then same-origin, which is why the CORS block below is
still only about the Next app on :3000 and did not have to grow.

    uvicorn animate_agent.api:app
    # http://127.0.0.1:8000/       upload a document, or talk one into existence
    # http://127.0.0.1:8000/player the player, ?spec=<url>
    # http://127.0.0.1:8000/specs  where generated specs land

Two ways in, one chain out. `from-file` starts from bytes somebody dragged in;
`from-chat` starts from a conversation and has the model write the document
first. Past the `DocumentIR` they are the same function, which is the point of
`_render_animation` — a second copy of the error table is a second copy to get
out of step.
"""

import hashlib
import logging
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool

from animate_agent.chat import (
    MAX_MESSAGE_CHARS,
    MAX_MESSAGES,
    MAX_TOTAL_CHARS,
    reply,
    write_article,
)
from animate_agent.config import load_speech_settings
from animate_agent.documents.file_parser import SUPPORTED_EXTENSIONS
from animate_agent.documents.models import DocumentIR
from animate_agent.documents.service import ingest_file, ingest_url
from animate_agent.knowledge.models import LessonIR
from animate_agent.knowledge.service import generate_lesson
from animate_agent.llm import LLMBudgetExhaustedError, load_llm_config
from animate_agent.rendering.layout import LayoutError
from animate_agent.rendering.service import render_spec_path, render_storyboard, write_render_spec
from animate_agent.speech.engine import EngineUnavailable, SpeechError
from animate_agent.speech.service import attach_speech
from animate_agent.storyboard.service import generate_storyboard

_LOG = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = SUPPORTED_EXTENSIONS

#: What every JSON route starts with. Used to tell the API apart from the site
#: below, whose responses want a different cache rule.
API_PREFIX = "/api"

#: Where the repository is, resolved from this file rather than from the working
#: directory. Everything below is an absolute path built from it, and that is not
#: fastidiousness: `StaticFiles` resolves each request with
#: `realpath(join(directory, path))` plus a containment check
#: (starlette/staticfiles.py:158-168), so a relative directory that stops
#: containing the file does not raise — it silently answers 404.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
SPECS_URL = "/specs"


def _mount(app: FastAPI, path: str, directory: Path, *, html: bool = False) -> None:
    """Mount `directory` at `path`, or skip it when it is not there.

    `StaticFiles(directory=...)` raises `RuntimeError` in its **constructor**
    (starlette/staticfiles.py:55-56), so an unguarded mount turns one missing
    directory into an ImportError for the whole API. `check_dir=False` is not the
    escape hatch it looks like: Starlette then re-checks on the first request
    through the mount and raises there instead (同文件 :93-95, :189-203), which is
    a 500 rather than a 404.

    The guard is for the directories committed to the repository, where its job is
    "a future move must not be able to take the API down" — not for `data/`, which
    is handled below.
    """
    if not directory.is_dir():
        return
    name = path.strip("/") or "root"
    app.mount(path, StaticFiles(directory=str(directory), html=html), name=name)


def _repo_relative(path: Path) -> str:
    """`path` relative to the repository, or absolute when it is outside it.

    The fallback is not padding: `Path.relative_to` raises `ValueError` rather
    than admitting the path is elsewhere, and a test that points the output
    directory at a temp directory would turn that into a 500 on the success path.
    """
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


class FromUrlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: AnyHttpUrl


class AnimationResponse(BaseModel):
    """What the browser needs in order to play what it just caused.

    Deliberately not the `RenderSpec` itself: that is tens of kilobytes, the
    player fetches it by URL anyway, and shipping it here as well would give one
    picture two sources of truth. The file on disk is the artifact of record.
    """

    model_config = ConfigDict(extra="forbid")

    storyboard_id: str
    lesson_id: str
    document_id: str
    title: str
    scene_count: int
    #: Repository-relative, for a person reading the response.
    spec_path: str
    #: What the player is pointed at.
    spec_url: str


class ChatMessage(BaseModel):
    """One turn of a conversation, as a client is allowed to assert it.

    `role` is a `Literal` over the two a client may claim, and `system` is not
    one of them. The system prompt is the server's — the page is the one thing in
    this system that must never be able to write it, and an endpoint that accepts
    a caller-supplied one is an endpoint that lets the caller replace the rules
    the whole feature rests on.
    """

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class ChatRequest(BaseModel):
    """A conversation to answer, bounded at the edge rather than in the handler.

    The bounds are declarative so the 422 is FastAPI's own and says which field
    was wrong. They are *also* enforced in `chat.history`, because a limit that
    only exists at the edge stops existing the moment something calls the module
    directly — which is exactly what a test does.
    """

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_MESSAGES)

    @model_validator(mode="after")
    def _the_whole_conversation_is_bounded(self) -> "ChatRequest":
        total = sum(len(message.content) for message in self.messages)
        if total > MAX_TOTAL_CHARS:
            raise ValueError(f"对话太长：{total} 字，上限 {MAX_TOTAL_CHARS} 字")
        return self


class FromChatRequest(ChatRequest):
    """A conversation, plus the same three switches `from-file` takes.

    Inherited rather than restated, and for the reason `extra="forbid"` exists:
    a field that has to be written twice is a field that can be written two
    different ways. `script`, `speak` and `voice` mean exactly what they mean on
    the upload route — absent is "whatever the YAML says".
    """

    script: bool | None = None
    speak: bool | None = None
    voice: str | None = None


class ChatReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply: str


class FromChatResponse(AnimationResponse):
    """`AnimationResponse`, plus the稿子 that produced the film.

    The article travels back because it is the intermediate thing a person may
    want to read, correct, or reuse — and because it is on disk under a
    content-addressed name, so `article_path` is a durable pointer rather than a
    copy of something that will be gone tomorrow.
    """

    article: str
    #: Repository-relative, like `spec_path`, and for the same reason.
    article_path: str


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Turn on the per-call timing line while the server runs.

    Same reason as `cli.main`'s: the pipeline is two sequential reasoning calls,
    and the line `llm._log_call` emits is the only thing that says which one is
    costing the minutes. Without a handler, INFO goes nowhere.

    In the lifespan rather than at import: importing this module must not change
    the process's logging, for the same reason `load_llm_config` reads `.env` at
    call time. And `basicConfig` is a no-op when something else has already
    configured root — which is exactly right, since under pytest that is the
    capture handler and its output is not ours to rearrange.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    yield


app = FastAPI(title="Animate Agent API", version="0.1.0", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def revalidate_the_site(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Tell the browser to revalidate the site's files instead of guessing.

    The site is served straight off disk, and `StaticFiles` sends `ETag` and
    `Last-Modified` and nothing else. That silence is not neutral: a response with
    a `Last-Modified` and no explicit freshness is *heuristically* cacheable, and
    browsers take roughly a tenth of the time since the file changed as its
    remaining lifetime. Edit `player.js`, reload within that window, and the
    browser reuses the copy it already has without asking — so the page runs the
    previous commit's code. Nothing about it looks like caching: the page loads,
    the console is clean, and the behaviour is just... the old behaviour. Which is
    indistinguishable from "my change did nothing", and is how a working
    auto-advance gets reported as still stopping at the first scene.

    `no-cache`, not `no-store`. The copy may stay; it only has to be revalidated,
    and revalidating against the `ETag` above is a 304 with an empty body — cheap
    on a loopback server, and the difference between "I reloaded" and "I am
    looking at what is on disk".

    The JSON routes are exempt. Their responses are not the thing being edited,
    a browser will not reuse a POST without being told to, and `no-cache` on an
    API reply is a promise about semantics that nothing here needs to make.
    """
    response = await call_next(request)
    if not request.url.path.startswith(API_PREFIX):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.post("/api/documents/from-url", response_model=DocumentIR)
async def create_document_from_url(request: FromUrlRequest) -> DocumentIR:
    try:
        return await ingest_url(str(request.url))
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Could not fetch document: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/lessons/from-url", response_model=LessonIR)
async def create_lesson_from_url(request: FromUrlRequest) -> LessonIR:
    try:
        document = await ingest_url(str(request.url))
        return await generate_lesson(document)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Could not fetch document: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def _ingest_uploaded_file(file: UploadFile) -> DocumentIR:
    """Turn an upload into a DocumentIR, refusing the extensions we cannot read.

    The 400 lives here rather than in each route so the rule exists once, and it
    is checked before the bytes are read — so a `.exe` never reaches a parser.
    A `ValueError` from the parser itself is the caller's to map; that is a
    different failure with a different status.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}。支持: {supported}",
        )
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(content)
    try:
        # `ingest_file` is synchronous — a read, a parse and a write. It is small
        # for a text file and not small at all for a hundred-page PDF, and it was
        # being called straight from an async handler, which blocks the loop for
        # every other request while it runs. Off the loop rather than rewritten as
        # async, because the blocking part is somebody else's library.
        return await run_in_threadpool(ingest_file, tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _require_key() -> None:
    """503 unless there is a key to spend, checked before anything is spent.

    A function rather than a line in each handler because both animation routes
    need it and neither may skip it. Without it the run dies a minute later
    inside the model client complaining about JSON — the same trap the CLI avoids
    the same way (`cli.py:137`), and a much worse message.
    """
    if not load_llm_config().api_key:
        raise HTTPException(
            status_code=503,
            detail="未配置 DEEPSEEK_KEY，无法调用模型。请在 .env 或环境变量中设置。",
        )


async def _render_animation(
    document: DocumentIR,
    *,
    script: bool | None,
    speak: bool | None,
    voice: str | None,
) -> AnimationResponse:
    """A DocumentIR in, a playable spec out.

    The half of the chain that does not care where the document came from. Both
    animation routes go through here, which is what keeps the error table below
    written once — it is the kind of code that gets a branch added to one copy
    and not the other.

    **`GENERATED_DIR` is read in the body and must stay that way.** Spelling it
    as `output_dir: Path = GENERATED_DIR` binds at import, so a test that points
    the output directory at a temp directory would be silently ignored and the
    suite would write into the real `data/generated` — while still passing. The
    same goes for every name patched on this module: they are resolved at call
    time, which is only true as long as they keep being looked up by name.
    """
    output_dir = GENERATED_DIR
    storyboard_path: Path | None = None
    try:
        lesson = await generate_lesson(document, output_dir=output_dir, script_mode=script)
        storyboard = await generate_storyboard(
            lesson, document, output_dir=output_dir, script_mode=script
        )
        storyboard_path = output_dir / f"{storyboard.storyboard_id}.json"
        spec = render_storyboard(storyboard, output_dir=output_dir)
    except LayoutError as exc:
        # `LayoutError` *is* a `ValueError`, so it has to be caught first. The
        # storyboard is already on disk and is the only thing that says why the
        # layout refused, so the refusal names it rather than leaving a person
        # with a 422 and nowhere to look.
        detail = f"布局失败: {exc}"
        if storyboard_path is not None:
            detail += f"。StoryboardIR 已落盘: {storyboard_path}"
        raise HTTPException(status_code=422, detail=detail) from exc
    except LLMBudgetExhaustedError as exc:
        # A `RuntimeError`, so without this it would escape as a bare 500 and the
        # page would have nothing to show but "Internal Server Error".
        raise HTTPException(status_code=502, detail=f"调用模型失败: {exc}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"调用模型失败: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    destination = render_spec_path(storyboard.storyboard_id, output_dir=output_dir)

    # Absent means "ask the config", and the config ships off. The page therefore
    # gets an unvoiced film unless it says otherwise, which is the same deal
    # `script` makes and for the same reason.
    wants_speech = load_speech_settings().enabled if speak is None else speak
    if wants_speech:
        voices = [part.strip() for part in (voice or "").split(",") if part.strip()] or None
        try:
            report = await attach_speech(spec, output_dir=output_dir, voices=voices)
        except EngineUnavailable as exc:
            # A server that has not installed the extra, rather than a bad
            # request. 503 with the install line, because the fix is on this side.
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except SpeechError as exc:
            raise HTTPException(status_code=502, detail=f"配音失败: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        write_render_spec(spec, destination)
        _LOG.info(
            "配音: %d 拍 / %s / 新合成 %d 复用 %d / 成片 %.1f 秒",
            report.beats,
            "、".join(report.voices),
            report.synthesised,
            report.reused,
            report.seconds,
        )
        if report.warning is not None:
            _LOG.warning("%s", report.warning)

    return AnimationResponse(
        storyboard_id=storyboard.storyboard_id,
        lesson_id=lesson.lesson_id,
        document_id=document.document_id,
        title=spec.title,
        scene_count=len(spec.scenes),
        spec_path=_repo_relative(destination),
        spec_url=f"{SPECS_URL}/{destination.name}",
    )


@app.post("/api/lessons/from-file", response_model=LessonIR)
async def create_lesson_from_file(file: Annotated[UploadFile, File()]) -> LessonIR:
    try:
        document = await _ingest_uploaded_file(file)
        return await generate_lesson(document)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/animations/from-file", response_model=AnimationResponse)
async def create_animation_from_file(
    file: Annotated[UploadFile, File()],
    script: Annotated[bool | None, Form()] = None,
    speak: Annotated[bool | None, Form()] = None,
    voice: Annotated[str | None, Form()] = None,
) -> AnimationResponse:
    """A document in, a playable animation out — the whole chain in one request.

    The lesson and the storyboard are written to the *same* directory this app
    serves, and that directory is absolute. Both matter: relative to the working
    directory they would land somewhere the `/specs` mount is not looking, and the
    picture would 404 while every step reported success.

    `script` is the one thing a caller may need to say about the *nature* of what
    it is uploading rather than about what to do with it: this text was written to
    be read aloud, so keep the author's words and let the film follow their
    length. Left out (which is what the upload page does), it falls back to the
    YAML, which is off — a page where somebody drags in a slide deck must not
    acquire script behaviour by default.

    `speak` and `voice` follow the same rule and for the same reason: absent means
    "whatever the YAML says", and the YAML says no. `voice` is one id, or two
    separated by a comma for a film that carries both — the same string a
    `<select>` gives, rather than a repeated form field, so that the page can post
    one value and not have to know how multipart repeats work.

    The spec is rewritten **in place** once it has a voice track, so `spec_url`
    keeps pointing at one file: the player fetches it once and finds out from the
    spec itself whether there is anything to play.
    """
    _require_key()
    document = await _ingest_uploaded_file(file)
    return await _render_animation(document, script=script, speak=speak, voice=voice)


#: What a generated article's filename starts with. Named so a person looking at
#: `data/generated/` can tell one from a spec without opening it.
ARTICLE_PREFIX = "article-"


def _write_article(text: str) -> Path:
    """Write `text` where the pipeline can read it back, and return the path.

    Content-addressed, so the same conversation asked for twice lands on the same
    file rather than accumulating a new one per click. It is a real file in a real
    directory and not a `NamedTemporaryFile`, for three reasons that all come down
    to the same one — this is an artifact, not scratch space: nothing leaks into
    `%TEMP%`, the `document_id` downstream is stable for the same conversation
    instead of new every time, and there is something left afterwards for a person
    to read.

    `.md` is load-bearing: `file_parser.parse_file` dispatches on the suffix, and
    an article named `.txt` would take the other parser.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    path = GENERATED_DIR / f"{ARTICLE_PREFIX}{digest}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@app.post("/api/chat", response_model=ChatReply)
async def chat_turn(request: ChatRequest) -> ChatReply:
    """One conversational turn. No pipeline, no artifacts — just an answer.

    Kept separate from the animation route on purpose: this one is expected to
    come back in seconds and may be called many times before anything is
    generated. Folding it in would make every question cost a film.
    """
    _require_key()
    try:
        text = await reply([message.model_dump() for message in request.messages])
    except LLMBudgetExhaustedError as exc:
        raise HTTPException(status_code=502, detail=f"调用模型失败: {exc}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"调用模型失败: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ChatReply(reply=text)


@app.post("/api/animations/from-chat", response_model=FromChatResponse)
async def create_animation_from_chat(request: FromChatRequest) -> FromChatResponse:
    """A conversation in, a playable animation out.

    Two extra steps in front of the upload route's chain: the model writes the
    document, and the document is read back off disk. Everything after that is
    `_render_animation`, which is the same function `from-file` calls.

    The article is generated here rather than in the browser, which is what keeps
    the page a dumb poster: it sends a conversation and renders text, and it never
    has to know the file-extension rules or synthesize a `File` out of a string to
    post back. It also means the intermediate document is a thing on disk that can
    be read afterwards rather than a string that existed for one round trip.
    """
    _require_key()
    messages = [message.model_dump() for message in request.messages]

    try:
        article = await write_article(messages)
    except LLMBudgetExhaustedError as exc:
        raise HTTPException(status_code=502, detail=f"写稿失败: {exc}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"写稿失败: {exc}") from exc
    except ValueError as exc:
        # The model answered, but not with anything the pipeline can use. The
        # caller's mistake only in the sense that the caller asked; there is
        # nothing for them to fix, so the message has to say what came back.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    path = _write_article(article)
    try:
        document = await run_in_threadpool(ingest_file, path)
    except ValueError as exc:
        # Parsing our *own* article failed, which means the model produced
        # something `parse_markdown` found no text in. Named as ours, because
        # "文件中没有可解析的文本内容" reads like the user's fault otherwise.
        raise HTTPException(
            status_code=422,
            detail=f"生成的稿子没法当成文档读：{exc}（稿子在 {_repo_relative(path)}）",
        ) from exc

    result = await _render_animation(
        document, script=request.script, speak=request.speak, voice=request.voice
    )
    return FromChatResponse(
        **result.model_dump(), article=article, article_path=_repo_relative(path)
    )


# ------------------------------------------------------------------ the site
#
# Everything below mounts a directory. `data/` is in `.gitignore`, so a fresh
# clone does not have it — and a mount that is skipped stays skipped, so `/specs`
# would go on 404ing after the first generation created the directory, until the
# next restart. Creating it here is idempotent, anchored to the repository rather
# than to the working directory, and is what lets that mount be unconditional.
# The `is_dir` guard in `_mount` is for the committed `frontend/` directories,
# where its job is "a future move must not take the API down".
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

_mount(app, "/player", REPO_ROOT / "frontend" / "player", html=True)
# A second name for the same directory, because `docs/storyboard-milestone.md:85`
# and `tools/layout_storyboard.py:22` record the manual-acceptance URL as
# `/frontend/player/index.html?spec=...`, and a recorded URL that stops working
# is a record that stops being checkable. Mounting `/frontend` wholesale would
# also publish `frontend/demo/app.js`, which has to stay unreachable.
_mount(app, "/frontend/player", REPO_ROOT / "frontend" / "player", html=True)
_mount(app, SPECS_URL, GENERATED_DIR)
# Keeps every URL under `data/generated/...` that earlier runs recorded resolvable.
_mount(app, "/data", DATA_DIR)
# MUST BE LAST. Starlette matches in registration order and `mount` appends, so
# the API routes above always win. The one cost: a path no route matched lands
# here, so a mistyped API URL gets StaticFiles' 404/405 instead of FastAPI's JSON
# one. That is the price of serving the site from the same prefix.
_mount(app, "/", REPO_ROOT / "frontend" / "web", html=True)
