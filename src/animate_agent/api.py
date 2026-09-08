"""FastAPI entrypoint for the first Animate Agent vertical slice."""

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AnyHttpUrl, BaseModel, ConfigDict

from animate_agent.documents.models import DocumentIR
from animate_agent.documents.service import ingest_url


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
