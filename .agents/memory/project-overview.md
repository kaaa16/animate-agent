# Project Overview

date: 2026-09-08
status: active

## Summary

Animate Agent / Intuition Engine Agent is a prototype for turning text, documents, PPT content, or open-source project introductions into 2D/3D teaching animations. The intended product experience is an Interactive Knowledge Movie, not a normal document reader or summary page.

## Current Artifacts

- `frontend/demo/index.html`: current static homepage/input prototype; can be opened directly in a browser.
- `frontend/demo/intuition-home.js` and `frontend/demo/intuition-home.css`: focused text/file input workflow with animated Canvas visual system.
- `frontend/demo/app.js` and `frontend/demo/styles.css`: legacy interactive lesson prototype with ROS and robot teaching scenes.
- `frontend/app/`: Next.js DocumentIR parser/viewer for the first URL ingestion milestone.
- `src/animate_agent/api.py`: FastAPI endpoint for URL ingestion.
- `src/animate_agent/documents/`: validated DocumentIR models plus HTML fetch, parse, clean, and persistence pipeline.
- `src/animate_agent/`: Python package for reusable animation elements, templates, rendering/storyboard/document modules, and interaction controls.
- `docs/`: product expectations, tech stack, todo, interaction flow notes, and draw.io source/exported diagrams.
- `tests/unit/`: current unit tests for animation elements and templates.

## Entrypoints

- Static frontend demo: open `frontend/demo/index.html` in a browser.
- Document parser API: `.\.venv\Scripts\python.exe -m uvicorn animate_agent.api:app --reload`.
- Next.js parser UI: `npm --prefix frontend run dev`.
- Python package: `src/animate_agent`.
- Unit tests: `.\.venv\Scripts\python.exe -m pytest` when the local virtual environment exists; otherwise create/install from `requirements-dev.txt`.

## Standing Corrections

- Preserve repository data and avoid broad cleanup unless explicitly requested.
- Do not let the frontend execute arbitrary model-generated code; the frontend should consume controlled JSON animation specs.
- Treat browser rendering checks separately from syntax/build checks, especially for visual or interactive frontend work.

## Validation Notes

- Python validation: `.\.venv\Scripts\python.exe -m pytest`.
- Static JS syntax checks may use Node where applicable, but they do not prove browser behavior.
- Browser behavior should be verified with a local HTTP server or direct file open plus Playwright/screenshots when frontend behavior is changed.
- URL ingestion milestone validation also includes `ruff check src tests`, strict `mypy src`, frontend `typecheck`, and `build`.
