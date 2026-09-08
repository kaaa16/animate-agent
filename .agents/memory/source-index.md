# Source Index

date: 2026-09-08
status: active

## Project Sources

- `AGENTS.md`: AgentGo v1.14.0 Chinese protocol installed at repository root on 2026-07-29.
- `README.md`: high-level project summary, current demo artifacts, quick run commands, and engineering principles.
- `pyproject.toml`: Python package metadata, dependencies, pytest configuration, Ruff, and strict mypy settings.
- `requirements.txt` and `requirements-dev.txt`: runtime and development dependency pins/ranges.
- `docs/PROJECT_EXPECTATIONS.md`: product vision and quality bar for Interactive Knowledge Movie output.
- `docs/TECH_STACK.md`: proposed backend/frontend/rendering stack and JSON animation spec direction.
- `docs/TODO.md`: MVP scope, animation quality standards, and current prototype boundaries.
- `docs/INTERACTION_FLOW.md`: interaction pipeline from input through controlled JSON spec to rendering and feedback.
- `docs/interaction-flow.drawio`: editable flowchart source; PNG exports live in `docs/`.
- `data/samples/`: sample ROS and robot-obstacle-avoidance source material.
- `frontend/demo/`: static and legacy frontend prototypes.
- `frontend/app/` and `frontend/package.json`: Next.js DocumentIR parser/viewer and its build scripts.
- `src/animate_agent/api.py`: `POST /api/documents/from-url` HTTP boundary.
- `src/animate_agent/documents/`: shared DocumentIR schema and URL HTML ingestion adapter.
- `tests/fixtures/manim_quickstart.html`: deterministic Manim-like HTML fixture.
- `docs/document-ingestion-milestone.md`: milestone boundary, run commands, and verification commands.
- `src/animate_agent/`: Python implementation area.
- `tests/`: unit and integration test area.

## Current Workflow Evidence

- Python project uses `pytest` with `testpaths = ["tests"]` and `pythonpath = ["src"]`.
- Dev dependencies include `pytest`, `ruff`, and `mypy`.
- The DocumentIR frontend uses Next.js and validates with `npm --prefix frontend run typecheck` and `build`.
- The URL ingestion path was browser-verified against the live Manim Quickstart on 2026-09-08.

## Quality/Risk Notes

- Some terminal output may display Chinese as mojibake in PowerShell, while git diff can still show correct UTF-8 content.
- Existing `output/`, `.venv/`, `.playwright-cli/`, and pytest cache directories are generated/local artifacts and should not be included in normal source commits.
- Browser rendering remains the acceptance check for frontend visual claims.
