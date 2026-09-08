# Document ingestion milestone

This milestone implements the boundary:

```text
URL -> HTTP adapter -> HTML parser -> DocumentIR -> JSON file -> viewer
```

No LLM is involved. Future file adapters should produce the same `DocumentIR` model before any
AI processing begins.

## Run locally

Install the Python and frontend dependencies, then start both development servers:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m uvicorn animate_agent.api:app --reload
```

```powershell
npm --prefix frontend install
npm --prefix frontend run dev
```

Open `http://localhost:3000`. The frontend calls `http://localhost:8000` by default. Override it
with `NEXT_PUBLIC_API_BASE_URL` when needed.

Each successful request to `POST /api/documents/from-url` writes the validated representation to
`data/documents/{document_id}.json`.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy src
npm --prefix frontend run typecheck
npm --prefix frontend run build
```
