"""FastAPI entrypoint for the first Animate Agent vertical slice."""

import tempfile
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AnyHttpUrl, BaseModel, ConfigDict

from animate_agent.documents.models import DocumentIR
from animate_agent.documents.service import ingest_file, ingest_url
from animate_agent.knowledge.models import LessonIR
from animate_agent.knowledge.service import generate_lesson

ALLOWED_EXTENSIONS = {".pptx", ".docx", ".pdf"}


class FromUrlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: AnyHttpUrl


app = FastAPI(title="Animate Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


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


@app.post("/api/lessons/from-file", response_model=LessonIR)
async def create_lesson_from_file(file: Annotated[UploadFile, File()]) -> LessonIR:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}。支持: .pptx, .docx, .pdf",
        )
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(content)
    try:
        document = ingest_file(tmp_path)
        return await generate_lesson(document)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
