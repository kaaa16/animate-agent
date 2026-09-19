"""FastAPI entrypoint: the HTTP face of the pipeline, and the site that drives it.

Serves three things from one process and one port: the API itself, the upload
page at `/`, and the player plus the specs it fetches. That is deliberate — every
URL the page touches is then same-origin, which is why the CORS block below is
still only about the Next app on :3000 and did not have to grow.

    uvicorn animate_agent.api:app
    # http://127.0.0.1:8000/       upload a document
    # http://127.0.0.1:8000/player the player, ?spec=<url>
    # http://127.0.0.1:8000/specs  where generated specs land
"""

import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import AnyHttpUrl, BaseModel, ConfigDict

from animate_agent.documents.file_parser import SUPPORTED_EXTENSIONS
from animate_agent.documents.models import DocumentIR
from animate_agent.documents.service import ingest_file, ingest_url
from animate_agent.knowledge.models import LessonIR
from animate_agent.knowledge.service import generate_lesson
from animate_agent.llm import LLMBudgetExhaustedError, load_llm_config
from animate_agent.rendering.layout import LayoutError
from animate_agent.rendering.service import render_spec_path, render_storyboard
from animate_agent.storyboard.service import generate_storyboard

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


app = FastAPI(title="Animate Agent API", version="0.1.0")
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
        return ingest_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/lessons/from-file", response_model=LessonIR)
async def create_lesson_from_file(file: Annotated[UploadFile, File()]) -> LessonIR:
    try:
        document = await _ingest_uploaded_file(file)
        return await generate_lesson(document)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/animations/from-file", response_model=AnimationResponse)
async def create_animation_from_file(file: Annotated[UploadFile, File()]) -> AnimationResponse:
    """A document in, a playable animation out — the whole chain in one request.

    The lesson and the storyboard are written to the *same* directory this app
    serves, and that directory is absolute. Both matter: relative to the working
    directory they would land somewhere the `/specs` mount is not looking, and the
    picture would 404 while every step reported success.
    """
    # Checked before the upload is read and before anything is spent. Without a
    # key the run dies a minute later inside the model client, complaining about
    # JSON — the same trap the CLI avoids the same way (cli.py:137).
    if not load_llm_config().api_key:
        raise HTTPException(
            status_code=503,
            detail="未配置 DEEPSEEK_KEY，无法调用模型。请在 .env 或环境变量中设置。",
        )

    storyboard_path: Path | None = None
    try:
        document = await _ingest_uploaded_file(file)
        lesson = await generate_lesson(document, output_dir=GENERATED_DIR)
        storyboard = await generate_storyboard(lesson, document, output_dir=GENERATED_DIR)
        storyboard_path = GENERATED_DIR / f"{storyboard.storyboard_id}.json"
        spec = render_storyboard(storyboard, output_dir=GENERATED_DIR)
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

    destination = render_spec_path(storyboard.storyboard_id, output_dir=GENERATED_DIR)
    return AnimationResponse(
        storyboard_id=storyboard.storyboard_id,
        lesson_id=lesson.lesson_id,
        document_id=document.document_id,
        title=spec.title,
        scene_count=len(spec.scenes),
        spec_path=_repo_relative(destination),
        spec_url=f"{SPECS_URL}/{destination.name}",
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
