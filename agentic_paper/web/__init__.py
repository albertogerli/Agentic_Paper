"""FastAPI + HTMX web UI for live demos.

Optional component — install with ``pip install agentic-paper[web]``.

Endpoints (see ``agentic_paper.web.server``):
    GET  /                       — drop-zone page
    POST /review                 — accepts a PDF upload, returns {run_id, redirect}
    GET  /runs/{run_id}          — HTMX SSE activity panel
    GET  /runs/{run_id}/status   — SSE stream of agent events (tail of audit.jsonl)
    GET  /runs/{run_id}/report   — markdown report rendered as HTML
"""

from __future__ import annotations

__all__: list[str] = []
