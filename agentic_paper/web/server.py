"""FastAPI + HTMX shell. Heavy lifting stays in the orchestrator."""

from __future__ import annotations

import argparse
import asyncio
import html as _html
import json
import logging
import tempfile
from pathlib import Path
from typing import AsyncGenerator

try:
    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    from fastapi.templating import Jinja2Templates
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Web extras not installed. Run `pip install agentic-paper[web]`."
    ) from e

try:
    import markdown as md_renderer
except ImportError as e:  # pragma: no cover
    raise ImportError("Missing `markdown`. Run `pip install agentic-paper[web]`.") from e

from .. import __version__
from ..audit import COST_RATES
from ..config import Config, ProviderConfig, RoutingConfig, TierConfig
from ..models import PROVIDER_MODELS, lookup as model_lookup
from ..routing import PROFILE_NAMES, PROFILES, apply_auto_mode, get_profile
from .runner import RunRegistry

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _THIS_DIR / "templates"

# Plain-vendor names accepted from the BYOK form.
_BYOK_PROVIDERS = ("openai", "anthropic", "google")


def _lookup_rates(provider: str, model: str) -> tuple[float, float] | tuple[None, None]:
    """Return (input_per_1M, output_per_1M) using longest-prefix matching."""
    if (provider, model) in COST_RATES:
        return COST_RATES[(provider, model)]
    candidates = [k for k in COST_RATES if k[0] == provider and model.startswith(k[1])]
    if not candidates:
        return (None, None)
    return COST_RATES[max(candidates, key=lambda k: len(k[1]))]


def _resolve_primary_model(primary_provider: str, primary_model: str) -> str | None:
    """Validate the user's (provider, model) pair against the catalog.

    Returns the resolved model string, or None if the pair is invalid /
    incompatible. If ``primary_model`` is empty, fall back to the catalog's
    first model for the provider.
    """
    if not primary_provider:
        return None
    catalog = PROVIDER_MODELS.get(primary_provider, [])
    if not catalog:
        return None  # unknown provider
    if not primary_model.strip():
        return catalog[0].model
    # Accept exact catalog match or heuristic-known IDs (model_lookup will
    # classify by prefix even for off-catalog model strings).
    if any(m.model == primary_model for m in catalog):
        return primary_model
    spec = model_lookup(primary_model)
    if spec and spec.provider == primary_provider:
        return primary_model
    return None


def _build_user_config(
    base_config_path: str,
    user_keys: dict[str, str],
    *,
    primary_provider: str = "",
    primary_model: str = "",
    profile: str = "",
) -> tuple[Config, list[str]]:
    """Load the on-disk config and overlay user-supplied keys and routing.

    Keys go into ``Config.providers[name].api_key`` (inline); ``api_key_env``
    is cleared so the user value wins over server env vars. The legacy
    top-level ``Config.api_key`` is also set to the OpenAI key when given.

    When ``primary_provider`` (and optionally ``primary_model``) is supplied,
    *all three* routing tiers are rewritten to send every agent through that
    pair. tier_high keeps ``thinking_budget="auto"`` so reasoning-capable
    models still think. Unknown / mismatched (provider, model) pairs are
    silently ignored.

    Auto-Mode: if the caller supplied at least one BYOK key but the active
    routing points to a provider they did not supply (e.g. profile=std needs
    Anthropic but only google_api_key was given), each unreachable tier is
    remapped to an equivalent on a provider the user did supply. The returned
    warning list is empty otherwise. Auto-Mode runs *after* the profile is
    applied but *before* the primary-model override — an explicit primary
    pair wins because the user picked it on purpose.

    The returned Config object lives only in the worker thread — never on
    disk, never in logs.
    """
    cfg = Config.from_yaml(base_config_path)

    # ----- Profile preset (overrides routing). Primary model below still wins.
    profile_routing = get_profile(profile)
    if profile_routing is not None:
        cfg.routing = profile_routing

    # ----- Keys
    providers = dict(cfg.providers)
    for name in _BYOK_PROVIDERS:
        key = (user_keys.get(name) or "").strip()
        if not key:
            continue
        existing = providers.get(name) or ProviderConfig()
        providers[name] = ProviderConfig(
            api_key=key,
            api_key_env=None,
            base_url=existing.base_url,
        )
    cfg.providers = providers
    if user_keys.get("openai", "").strip():
        cfg.api_key = user_keys["openai"].strip()

    # ----- Auto-Mode: remap unreachable tiers to providers the user actually
    # supplied. Only kicks in for BYOK runs (i.e. at least one user key); when
    # no keys are submitted the server falls back to whatever it has wired
    # locally and we don't second-guess it.
    auto_warnings: list[str] = []
    user_supplied = {name for name in _BYOK_PROVIDERS if (user_keys.get(name) or "").strip()}
    if user_supplied and cfg.routing is not None:
        cfg.routing, auto_warnings = apply_auto_mode(cfg.routing, user_supplied)
        if auto_warnings:
            for w in auto_warnings:
                logger.info("Auto-Mode: %s", w)

    # ----- Primary model override (rewrites the whole RoutingConfig).
    resolved_model = _resolve_primary_model(primary_provider, primary_model)
    if resolved_model:
        cfg.routing = RoutingConfig(
            tier_high=TierConfig(
                provider=primary_provider, model=resolved_model, thinking_budget="auto"
            ),
            tier_standard=TierConfig(
                provider=primary_provider, model=resolved_model, thinking_budget=None
            ),
            tier_basic=TierConfig(
                provider=primary_provider, model=resolved_model, thinking_budget=None
            ),
        )
        # User explicitly named a (provider, model) — any auto-mode remap is
        # irrelevant now, so don't surface its warnings.
        auto_warnings = []

    return cfg, auto_warnings


def create_app(*, config_path: str = "config.yaml", registry: RunRegistry | None = None) -> FastAPI:
    """Build the FastAPI app. Pass a custom registry to override the default runner (tests)."""
    app = FastAPI(title="Agentic_Paper Web", version=__version__)
    app.state.registry = registry or RunRegistry(config_path=config_path)
    app.state.config_path = config_path
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    # ---------------------------------------------------------------- routes

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            "index.html", {"request": request, "version": __version__}
        )

    @app.get("/api/profiles")
    async def api_profiles() -> JSONResponse:
        """Pre-baked routing bundles (max / std / quick) with per-run cost estimate.

        Estimate uses a representative 12-agent split (7 high / 4 standard / 1 basic
        at paper_complexity=0.5) and 50k input / 5k output tokens per call.
        """
        # Representative agent distribution at paper_complexity=0.5.
        AGENTS_PER_TIER = {"tier_high": 7, "tier_standard": 4, "tier_basic": 1}
        PER_CALL_IN = 50_000
        PER_CALL_OUT = 5_000

        out: dict[str, dict] = {}
        for name, cfg in PROFILES.items():
            tiers: dict[str, dict] = {}
            estimated_total = 0.0
            for tier_name, n_agents in AGENTS_PER_TIER.items():
                tier: TierConfig = getattr(cfg, tier_name)
                in_rate, out_rate = _lookup_rates(tier.provider, tier.model)
                tier_blob: dict = {
                    "provider": tier.provider,
                    "model": tier.model,
                    "thinking_budget": tier.thinking_budget,
                    "input_cost_per_million": in_rate,
                    "output_cost_per_million": out_rate,
                    "agents_in_tier": n_agents,
                }
                if in_rate is not None and out_rate is not None:
                    per_call = (PER_CALL_IN * in_rate + PER_CALL_OUT * out_rate) / 1_000_000
                    tier_blob["estimated_per_call_usd"] = round(per_call, 4)
                    estimated_total += per_call * n_agents
                tiers[tier_name] = tier_blob
            out[name] = {
                "tiers": tiers,
                "estimated_cost_usd": round(estimated_total, 4),
                "label": {
                    "max":   "max — flagship models, max thinking, slowest + most expensive",
                    "std":   "std — balanced, recommended default",
                    "quick": "quick — cheapest + fastest, minimal thinking",
                }.get(name, name),
            }
        return JSONResponse(out)

    @app.get("/api/models")
    async def api_models() -> JSONResponse:
        """Catalog of known (provider, model) pairs for the form dropdown."""
        out: dict[str, list[dict]] = {}
        for provider, models in PROVIDER_MODELS.items():
            entries = []
            for m in models:
                in_rate, out_rate = _lookup_rates(provider, m.model)
                entries.append({
                    "model": m.model,
                    "context_window": m.context_window,
                    "max_output_tokens": m.max_output_tokens,
                    "reasoning_style": m.reasoning_style,
                    "notes": m.notes,
                    "input_cost_per_million": in_rate,
                    "output_cost_per_million": out_rate,
                })
            out[provider] = entries
        return JSONResponse(out)

    @app.post("/review")
    async def review(
        file: UploadFile = File(...),
        file_v1: UploadFile | None = File(None),
        openai_api_key: str = Form(""),
        anthropic_api_key: str = Form(""),
        google_api_key: str = Form(""),
        primary_provider: str = Form(""),
        primary_model: str = Form(""),
        profile: str = Form(""),
    ) -> JSONResponse:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename")
        suffix = Path(file.filename).suffix or ".pdf"
        # NamedTemporaryFile with delete=False so the background thread can read it.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            payload = await file.read()
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        if tmp_path.stat().st_size == 0:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Empty upload")

        # Optional v1 — when present, the Revision Assessor has real diff to chew on.
        tmp_v1_path: Path | None = None
        if file_v1 is not None and file_v1.filename:
            v1_payload = await file_v1.read()
            if v1_payload:
                suffix_v1 = Path(file_v1.filename).suffix or ".pdf"
                with tempfile.NamedTemporaryFile(suffix=suffix_v1, delete=False) as tmp_v1:
                    tmp_v1.write(v1_payload)
                    tmp_v1_path = Path(tmp_v1.name)

        registry: RunRegistry = app.state.registry

        # BYOK: if any user-supplied key arrived in the form, build an
        # ephemeral Config that overlays them on top of the on-disk config.
        # The Config (and therefore the keys) only lives in the worker
        # thread for the duration of this run.
        user_keys = {
            "openai": openai_api_key,
            "anthropic": anthropic_api_key,
            "google": google_api_key,
        }
        used_byok = any(v.strip() for v in user_keys.values())
        primary_provider = primary_provider.strip()
        primary_model = primary_model.strip()
        used_primary = bool(primary_provider)
        profile = profile.strip().lower()
        used_profile = profile in PROFILE_NAMES

        auto_warnings: list[str] = []
        if used_byok or used_primary or used_profile:
            cfg, auto_warnings = _build_user_config(
                app.state.config_path,
                user_keys,
                primary_provider=primary_provider,
                primary_model=primary_model,
                profile=profile if used_profile else "",
            )
            run_id = registry.start(
                tmp_path, config=cfg, cleanup_pdf=True, pdf_v1_path=tmp_v1_path,
                auto_mode_warnings=auto_warnings,
            )
        else:
            run_id = registry.start(
                tmp_path, cleanup_pdf=True, pdf_v1_path=tmp_v1_path,
            )

        return JSONResponse({
            "run_id": run_id,
            "redirect": f"/runs/{run_id}",
            "byok": used_byok,
            "profile": profile if used_profile else None,
            "compare_versions": tmp_v1_path is not None,
            "auto_mode_warnings": auto_warnings,
            "primary": {
                "provider": primary_provider or None,
                "model": (
                    _resolve_primary_model(primary_provider, primary_model)
                    if used_primary
                    else None
                ),
            },
        })

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_page(run_id: str, request: Request) -> HTMLResponse:
        registry: RunRegistry = app.state.registry
        status = registry.get(run_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Unknown run_id")
        return templates.TemplateResponse(
            "run.html",
            {
                "request": request,
                "run_id": run_id,
                "state": status.state,
                "version": __version__,
                "auto_mode_warnings": status.auto_mode_warnings,
            },
        )

    @app.get("/runs/{run_id}/status")
    async def status_stream(run_id: str) -> StreamingResponse:
        registry: RunRegistry = app.state.registry
        status = registry.get(run_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Unknown run_id")

        async def _merged() -> AsyncGenerator[str, None]:
            """Merge two async generators (audit tail + thinking bus drain)."""
            q: asyncio.Queue[str | None] = asyncio.Queue()

            async def _forward(gen):
                try:
                    async for chunk in gen:
                        await q.put(chunk)
                finally:
                    await q.put(None)  # sentinel

            audit_task = asyncio.create_task(_forward(_audit_event_stream(registry, run_id)))
            thinking_task = asyncio.create_task(_forward(_drain_event_bus(registry, run_id)))
            done_count = 0
            try:
                while done_count < 2:
                    item = await q.get()
                    if item is None:
                        done_count += 1
                        continue
                    yield item
            finally:
                for t in (audit_task, thinking_task):
                    if not t.done():
                        t.cancel()

        return StreamingResponse(
            _merged(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get("/runs/{run_id}/report", response_class=HTMLResponse)
    async def report(run_id: str, request: Request) -> HTMLResponse:
        registry: RunRegistry = app.state.registry
        status = registry.get(run_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Unknown run_id")
        run_dir = status.output_dir or (Path(Config.from_yaml(app.state.config_path).output_dir) / run_id)
        md_files = sorted(Path(run_dir).glob("review_report_*.md"))
        if not md_files:
            html_body = (
                "<p class='text-gray-600'>Report not ready yet. "
                "Refresh in a moment, or check back when the run completes.</p>"
            )
        else:
            md_text = md_files[-1].read_text(encoding="utf-8")
            html_body = md_renderer.markdown(
                md_text,
                extensions=["tables", "fenced_code", "toc", "attr_list"],
            )
        return templates.TemplateResponse(
            "report.html",
            {"request": request, "run_id": run_id, "report_html": html_body, "version": __version__},
        )

    @app.post("/runs/{run_id}/retry")
    async def retry_failed(run_id: str) -> JSONResponse:
        """Re-run the agents whose review_*.txt is missing from the run dir.

        Uses whatever provider config the server has TODAY (BYOK keys from the
        original upload were not persisted). If the original failed because of
        a missing key, set the relevant key server-side first.
        """
        registry: RunRegistry = app.state.registry
        status = registry.get(run_id)
        if status is None or status.output_dir is None:
            raise HTTPException(status_code=404, detail="Unknown run_id")
        # Local import to keep the heavy bits out of FastAPI startup.
        from ..orchestrator import ReviewOrchestrator

        cfg = Config.from_yaml(app.state.config_path)
        orch = ReviewOrchestrator(cfg, run_id=run_id)
        try:
            result = orch.retry_failed_agents()
        except FileNotFoundError as e:
            raise HTTPException(status_code=410, detail=str(e))
        return JSONResponse(result)

    @app.get("/healthz")
    async def healthz() -> dict:
        registry: RunRegistry = app.state.registry
        return {
            "status": "ok",
            "version": __version__,
            "active_runs": sum(1 for s in registry.runs.values() if s.state == "running"),
            "total_runs": len(registry.runs),
        }

    return app


# ---------------------------------------------------------------- SSE helpers


def _sse_event(event: str, data_obj) -> str:
    """Format a Server-Sent-Events frame. ``data_obj`` may be a dict or str."""
    if isinstance(data_obj, str):
        # Each newline in `data:` requires another `data:` prefix per SSE spec.
        lines = data_obj.split("\n")
        data_payload = "\n".join(f"data: {line}" for line in lines)
    else:
        data_payload = "data: " + json.dumps(data_obj, ensure_ascii=False)
    return f"event: {event}\n{data_payload}\n\n"


def _render_agent_li(record: dict) -> str:
    """Inline HTML snippet for one agent-completed event (HTMX swaps this directly)."""
    agent = _html.escape(record.get("agent", "?"))
    provider = _html.escape(record.get("provider", "?"))
    model = _html.escape(record.get("model", "?"))
    latency = record.get("latency_ms", 0)
    tokens_in = record.get("input_token_count", 0)
    tokens_out = record.get("output_token_count", 0)
    cost = record.get("cost_estimate_usd", 0.0)
    thinking = "🧠" if record.get("thinking_mode_enabled") else ""
    return (
        '<li class="border-l-4 border-blue-400 pl-3 py-2 bg-white rounded shadow-sm">'
        f'<div class="flex items-baseline justify-between">'
        f'<span class="font-semibold">{agent}</span>'
        f'<span class="text-xs text-gray-500">{latency} ms</span>'
        f'</div>'
        f'<div class="text-xs text-gray-600">{provider} / <code>{model}</code> {thinking}</div>'
        f'<div class="text-xs text-gray-500">tokens: {tokens_in:,} in / {tokens_out:,} out · est ${cost:.4f}</div>'
        '</li>'
    )


def _render_done_div(status_state: str, count: int, error: str | None = None) -> str:
    if status_state == "completed":
        return (
            '<div class="mt-4 bg-green-100 border border-green-300 rounded p-3 text-green-900">'
            f'✅ Completed. {count} agent calls recorded.'
            '</div>'
        )
    if status_state == "failed":
        safe_error = _html.escape(error or "unknown error")
        return (
            '<div class="mt-4 bg-red-100 border border-red-300 rounded p-3 text-red-900">'
            f'❌ Failed after {count} call(s). {safe_error}'
            '</div>'
        )
    return ""


def _render_thinking_event(agent: str, kind: str, text: str = "") -> str:
    """One inline HTML snippet per live-thinking chunk, swap-friendly for HTMX."""
    safe_agent = _html.escape(agent)
    if kind == "thinking_start":
        return (
            f'<div class="thinking-block border-l-2 border-amber-300 pl-2 py-1 my-1 '
            f'text-xs text-amber-900 bg-amber-50 rounded" data-agent="{safe_agent}">'
            f'🧠 <strong>{safe_agent}</strong> is thinking…</div>'
        )
    safe_text = _html.escape(text)
    return (
        f'<div class="thinking-chunk text-xs text-amber-800 italic '
        f'pl-4 ml-2 my-0.5 leading-snug" data-agent="{safe_agent}">'
        f'{safe_text}</div>'
    )


async def _drain_event_bus(registry: RunRegistry, run_id: str) -> AsyncGenerator[str, None]:
    """Pull thinking events from the in-memory bus and emit SSE frames."""
    bus = getattr(registry, "event_bus", None)
    if bus is None:
        return
    channel = bus.channel(run_id)
    # Run drain in a worker thread to avoid blocking the event loop.
    loop = asyncio.get_running_loop()

    def _pull_batch():
        return list(channel.drain(timeout=0.4))

    status = registry.get(run_id)
    while True:
        batch = await loop.run_in_executor(None, _pull_batch)
        for evt in batch:
            yield _sse_event(
                "thinking",
                _render_thinking_event(evt.agent, evt.kind, evt.text),
            )
        status = registry.get(run_id)
        if status is None:
            return
        if status.state in ("completed", "failed") and not batch:
            return


_AUDIT_WAIT_TICKS = 60       # 60 × 0.5s = 30s cap on the initial file-appears wait
_AUDIT_TICK_SECONDS = 0.5    # also the cadence of the steady-state poll loop


async def _wait_for_audit_file(
    registry: RunRegistry, run_id: str, audit_path: Path,
) -> bool:
    """Block until ``audit_path`` exists or the run has clearly given up.

    Polls at :data:`_AUDIT_TICK_SECONDS` cadence for at most
    :data:`_AUDIT_WAIT_TICKS` ticks (30 seconds total). The wait short-circuits
    if the run has already transitioned to ``failed`` — there will be no audit
    file to wait for. ``completed`` is NOT a short-circuit on purpose: a fast
    successful run may produce all audit rows before the SSE client connects.

    Returns ``True`` if the file is present when we leave the loop, ``False``
    if the run failed before producing it. The main loop in
    :func:`_audit_event_stream` ignores the return value and still emits the
    initial heartbeat + the final ``done`` frame; the bool is exposed for
    tests and future callers.
    """
    for _ in range(_AUDIT_WAIT_TICKS):
        if audit_path.exists():
            return True
        await asyncio.sleep(_AUDIT_TICK_SECONDS)
        status = registry.get(run_id)
        if status is not None and status.state == "failed":
            return False
    return audit_path.exists()


def _read_new_audit_lines(audit_path: Path, last_pos: int) -> tuple[list[str], int]:
    """Tail-read the audit log from ``last_pos`` to EOF.

    Returns the new lines plus the updated cursor. On ``OSError`` (file
    rotated/locked/deleted between checks) we silently return the existing
    cursor — the next poll will pick the file up again if it reappears.
    """
    try:
        with open(audit_path, "r", encoding="utf-8") as f:
            f.seek(last_pos)
            chunk = f.read()
            new_pos = f.tell()
    except OSError:
        return [], last_pos
    return chunk.splitlines(), new_pos


def _parse_audit_record(line: str) -> dict | None:
    """Validate one JSONL line. Returns the parsed dict, or ``None`` for
    blank / malformed lines. Malformed lines are swallowed silently — a
    partial write would be retried on the next tail-read pass."""
    if not line.strip():
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


async def _audit_event_stream(registry: RunRegistry, run_id: str) -> AsyncGenerator[str, None]:
    """Tail ``output/<run_id>/audit.jsonl`` and emit one SSE event per row."""
    status = registry.get(run_id)
    if status is None or status.output_dir is None:
        yield _sse_event("error", {"error": "unknown run"})
        return

    audit_path = status.output_dir / "audit.jsonl"
    last_pos = 0
    seen = 0

    # Wait up to 30s for the audit file to appear (extracted; see helper).
    await _wait_for_audit_file(registry, run_id, audit_path)

    # Initial heartbeat so the client sees the connection.
    yield _sse_event("status", {"state": status.state})

    while True:
        status = registry.get(run_id)
        if status is None:
            yield _sse_event("error", {"error": "run vanished"})
            return

        if audit_path.exists():
            lines, last_pos = _read_new_audit_lines(audit_path, last_pos)
            for line in lines:
                record = _parse_audit_record(line)
                if record is None:
                    continue
                seen += 1
                yield _sse_event("agent", _render_agent_li(record))

        if status.state in ("completed", "failed"):
            yield _sse_event("done", _render_done_div(status.state, seen, status.error))
            
            # Prevent HTMX from reconnecting (and duplicating history) by holding 
            # the connection open. If the user clicks "Retry", the state changes 
            # back to "running", and we can resume tailing.
            while True:
                await asyncio.sleep(_AUDIT_TICK_SECONDS)
                status = registry.get(run_id)
                if status is None or status.state not in ("completed", "failed"):
                    break
            
            if status is None:
                yield _sse_event("error", {"error": "run vanished"})
                return
                
            # Clear the done message now that we're running again
            yield _sse_event("done", "")
            continue

        await asyncio.sleep(_AUDIT_TICK_SECONDS)


# ----------------------------------------------------------------- entrypoint


def run_main(argv: list[str] | None = None) -> int:
    """Console script: ``agentic-paper-web``."""
    import uvicorn

    parser = argparse.ArgumentParser(description="Launch the Agentic_Paper web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev only).")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    app = create_app(config_path=args.config)
    print(f"\n  Agentic_Paper web UI on http://{args.host}:{args.port}/\n", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_main())
