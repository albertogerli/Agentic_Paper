"""FastAPI web layer — endpoints + SSE wiring with a stubbed runner.

Real LLM calls are stubbed via ``RunRegistry(runner=fake_runner)`` so the test
never touches a vendor API.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from agentic_paper.audit import AuditLogger  # noqa: E402
from agentic_paper.web.runner import RunRegistry  # noqa: E402
from agentic_paper.web.server import create_app  # noqa: E402


def _make_fake_runner(out_root: Path):
    """Build a runner that synthesises a tiny audit.jsonl + markdown report."""

    def fake_runner(pdf_path: Path, run_id: str, config) -> None:
        run_dir = out_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        audit = AuditLogger(run_dir, run_id=run_id)
        audit.record(
            agent="Methodology_Expert", provider="stub", model="stub-model",
            prompt="sys\nuser", response_text='{"summary":"ok"}',
            usage={"input_tokens": 10, "output_tokens": 5},
            latency_ms=12, thinking_enabled=False, seed=None,
        )
        audit.record(
            agent="Results_Analyst", provider="stub", model="stub-model",
            prompt="sys\nuser", response_text='{"summary":"ok"}',
            usage={"input_tokens": 8, "output_tokens": 4},
            latency_ms=9, thinking_enabled=True, seed=42,
        )
        (run_dir / "review_report_20260521_120000.md").write_text(
            "# Peer Review Report\n\n## Editorial Decision\n\nAccept as is.\n", encoding="utf-8"
        )

    return fake_runner


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Force the registry's Config.from_yaml() to use a config that points at tmp_path.
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(f"output_dir: \"{tmp_path / 'out'}\"\napi_key: \"dummy\"\n", encoding="utf-8")
    out_root = tmp_path / "out"

    registry = RunRegistry(
        config_path=str(cfg_path),
        max_workers=1,
        runner=_make_fake_runner(out_root),
    )
    app = create_app(config_path=str(cfg_path), registry=registry)
    client = TestClient(app)
    yield client, registry, out_root
    registry.shutdown()


def _wait_for_state(registry, run_id, target_states, timeout_s=5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = registry.get(run_id)
        if status and status.state in target_states:
            return status
        time.sleep(0.05)
    raise AssertionError(f"Run {run_id} did not reach {target_states} within {timeout_s}s")


def test_index_renders(app_client) -> None:
    client, _, _ = app_client
    r = client.get("/")
    assert r.status_code == 200
    assert "Agentic Paper" in r.text
    assert "Drop a PDF" in r.text
    assert "hx-post=\"/review\"" in r.text
    # Auto-simulation disclaimer banner is mandatory on the drop-zone.
    assert "Auto-simulation mode" in r.text
    assert "substitute for human peer review" in r.text


def test_healthz_reports_version(app_client) -> None:
    client, _, _ = app_client
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_review_rejects_empty_upload(app_client) -> None:
    client, _, _ = app_client
    r = client.post(
        "/review",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert r.status_code == 400


def test_review_returns_run_id_and_redirect(app_client) -> None:
    client, registry, _ = app_client
    r = client.post(
        "/review",
        files={"file": ("hello.pdf", b"%PDF-1.4\n%fake content\n", "application/pdf")},
    )
    assert r.status_code == 200
    body = r.json()
    assert "run_id" in body
    assert body["redirect"] == f"/runs/{body['run_id']}"
    # The registry should know about it.
    _wait_for_state(registry, body["run_id"], {"completed", "failed"})


def test_run_page_404_for_unknown(app_client) -> None:
    client, _, _ = app_client
    r = client.get("/runs/does-not-exist")
    assert r.status_code == 404


def test_run_page_and_status_sse_for_real_run(app_client) -> None:
    client, registry, _ = app_client
    r = client.post(
        "/review",
        files={"file": ("hello.pdf", b"%PDF-1.4\nfake\n", "application/pdf")},
    )
    run_id = r.json()["run_id"]
    _wait_for_state(registry, run_id, {"completed", "failed"})

    page = client.get(f"/runs/{run_id}")
    assert page.status_code == 200
    assert run_id in page.text
    assert 'sse-connect="/runs/' in page.text  # HTMX SSE wiring rendered

    # SSE endpoint returns text/event-stream and contains our agent events + done.
    sse_resp = client.get(f"/runs/{run_id}/status")
    assert sse_resp.status_code == 200
    assert "text/event-stream" in sse_resp.headers["content-type"]
    body = sse_resp.text
    assert "event: status" in body
    assert "event: agent" in body
    assert "Methodology_Expert" in body
    assert "Results_Analyst" in body
    assert "event: done" in body
    assert "Completed" in body or "Failed" in body


def test_report_endpoint_renders_markdown_as_html(app_client) -> None:
    client, registry, _ = app_client
    r = client.post(
        "/review",
        files={"file": ("hello.pdf", b"%PDF-1.4\nfake\n", "application/pdf")},
    )
    run_id = r.json()["run_id"]
    _wait_for_state(registry, run_id, {"completed", "failed"})

    rep = client.get(f"/runs/{run_id}/report")
    assert rep.status_code == 200
    # Markdown → HTML (toc extension adds an id attribute on headings).
    assert "Peer Review Report</h1>" in rep.text
    assert "Accept as is" in rep.text
    assert run_id in rep.text


def test_report_endpoint_when_no_report_yet(app_client, tmp_path) -> None:
    """If the runner has not produced a markdown report, the report page should
    return a friendly placeholder, not 500."""
    client, registry, out_root = app_client

    # Bypass the runner: register a status with no markdown report on disk.
    from agentic_paper.web.runner import RunStatus
    rid = "manual-run"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / rid).mkdir()
    registry.runs[rid] = RunStatus(
        run_id=rid, pdf_path=tmp_path / "x.pdf", state="running",
        output_dir=out_root / rid,
    )

    rep = client.get(f"/runs/{rid}/report")
    assert rep.status_code == 200
    assert "Report not ready yet" in rep.text


def test_sse_renders_audit_record_html(app_client) -> None:
    """Each agent event must be a self-contained <li> snippet (HTMX swaps inline)."""
    client, registry, _ = app_client
    r = client.post(
        "/review",
        files={"file": ("hello.pdf", b"%PDF-1.4\nfake\n", "application/pdf")},
    )
    run_id = r.json()["run_id"]
    _wait_for_state(registry, run_id, {"completed", "failed"})

    sse_resp = client.get(f"/runs/{run_id}/status")
    body = sse_resp.text
    # The SSE 'data:' frames should contain the rendered <li> we emit.
    assert '<li class="border-l-4' in body
    assert "Methodology_Expert" in body
    # Cost surface is present (from the audit usage we recorded).
    assert "tokens:" in body


# --------------------------------------------------------------------- BYOK


def _byok_app(tmp_path: Path, captured: list):
    """App where the runner captures the Config it received without making API calls."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(f"output_dir: \"{tmp_path / 'out'}\"\napi_key: \"server-default-key\"\n", encoding="utf-8")
    out_root = tmp_path / "out"

    def capturing_runner(pdf_path: Path, run_id: str, config) -> None:
        # Snapshot the config that arrived in this worker.
        captured.append({
            "run_id": run_id,
            "api_key": config.api_key,
            "providers": {
                name: {"api_key": p.api_key, "api_key_env": p.api_key_env, "base_url": p.base_url}
                for name, p in (config.providers or {}).items()
            },
        })
        # Write a minimal audit + report so the SSE path still works.
        run_dir = out_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        AuditLogger(run_dir, run_id=run_id).record(
            agent="StubAgent", provider="stub", model="stub-model",
            prompt="hello", response_text="ok",
            usage={"input_tokens": 1, "output_tokens": 1},
            latency_ms=1, thinking_enabled=False, seed=None,
        )
        (run_dir / "review_report_byok.md").write_text("# OK\n", encoding="utf-8")

    registry = RunRegistry(config_path=str(cfg_path), max_workers=1, runner=capturing_runner)
    app = create_app(config_path=str(cfg_path), registry=registry)
    return TestClient(app), registry, out_root


def test_byok_keys_reach_runner_config(tmp_path: Path) -> None:
    captured: list = []
    client, registry, _ = _byok_app(tmp_path, captured)

    r = client.post(
        "/review",
        files={"file": ("p.pdf", b"%PDF-1.4\nfake\n", "application/pdf")},
        data={
            "openai_api_key": "sk-user-openai",
            "anthropic_api_key": "sk-ant-user",
            "google_api_key": "AIza-user-google",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["byok"] is True
    run_id = body["run_id"]
    _wait_for_state(registry, run_id, {"completed", "failed"})

    assert len(captured) == 1
    cfg = captured[0]
    assert cfg["api_key"] == "sk-user-openai"  # legacy field is updated for OpenAI fallback
    assert cfg["providers"]["openai"]["api_key"] == "sk-user-openai"
    assert cfg["providers"]["openai"]["api_key_env"] is None
    assert cfg["providers"]["anthropic"]["api_key"] == "sk-ant-user"
    assert cfg["providers"]["google"]["api_key"] == "AIza-user-google"
    registry.shutdown()


def test_byok_empty_strings_fall_back_to_server_config(tmp_path: Path) -> None:
    captured: list = []
    client, registry, _ = _byok_app(tmp_path, captured)

    r = client.post(
        "/review",
        files={"file": ("p.pdf", b"%PDF-1.4\nfake\n", "application/pdf")},
        data={"openai_api_key": "  ", "anthropic_api_key": "", "google_api_key": ""},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["byok"] is False  # no key was meaningfully supplied
    run_id = body["run_id"]
    _wait_for_state(registry, run_id, {"completed", "failed"})

    assert captured[0]["api_key"] == "server-default-key"
    registry.shutdown()


def test_byok_partial_keys_only_override_their_provider(tmp_path: Path) -> None:
    captured: list = []
    client, registry, _ = _byok_app(tmp_path, captured)

    r = client.post(
        "/review",
        files={"file": ("p.pdf", b"%PDF-1.4\nfake\n", "application/pdf")},
        data={"anthropic_api_key": "sk-ant-only"},  # only Anthropic
    )
    run_id = r.json()["run_id"]
    _wait_for_state(registry, run_id, {"completed", "failed"})

    cfg = captured[0]
    assert cfg["api_key"] == "server-default-key"   # OpenAI key untouched
    # Anthropic now in providers block
    assert cfg["providers"]["anthropic"]["api_key"] == "sk-ant-only"
    # OpenAI may or may not be in providers block depending on the base config;
    # what matters is its key was NOT replaced.
    if "openai" in cfg["providers"]:
        assert cfg["providers"]["openai"]["api_key"] != "sk-ant-only"
    registry.shutdown()


def test_byok_keys_do_not_leak_into_audit_jsonl(tmp_path: Path) -> None:
    captured: list = []
    client, registry, out_root = _byok_app(tmp_path, captured)

    SECRET = "sk-very-secret-test-key-12345"
    r = client.post(
        "/review",
        files={"file": ("p.pdf", b"%PDF-1.4\nfake\n", "application/pdf")},
        data={"openai_api_key": SECRET},
    )
    run_id = r.json()["run_id"]
    _wait_for_state(registry, run_id, {"completed", "failed"})

    # Walk every artefact under the run dir; assert the key string is nowhere.
    run_dir = out_root / run_id
    leaked_in: list[Path] = []
    for path in run_dir.rglob("*"):
        if path.is_file():
            try:
                if SECRET in path.read_text(encoding="utf-8", errors="ignore"):
                    leaked_in.append(path)
            except OSError:
                pass
    assert not leaked_in, f"User API key leaked into: {leaked_in}"
    registry.shutdown()


def test_byok_cleanup_pdf_removes_temp_upload(tmp_path: Path) -> None:
    """The uploaded PDF should be deleted from $TMPDIR after the run."""
    import os
    captured: list = []
    client, registry, _ = _byok_app(tmp_path, captured)

    # Snapshot $TMPDIR contents before the upload.
    tmpdir = Path(os.environ.get("TMPDIR", "/tmp"))
    before = {p.name for p in tmpdir.iterdir() if p.suffix == ".pdf"}

    r = client.post(
        "/review",
        files={"file": ("p.pdf", b"%PDF-1.4\nfake\n", "application/pdf")},
    )
    run_id = r.json()["run_id"]
    _wait_for_state(registry, run_id, {"completed", "failed"})

    after = {p.name for p in tmpdir.iterdir() if p.suffix == ".pdf"}
    # No new orphan PDF should be left behind.
    assert after - before == set(), f"Temp PDFs leaked: {after - before}"
    registry.shutdown()


# ----------------------------------------------------------------- Model picker


def test_api_models_returns_catalog(app_client) -> None:
    client, _, _ = app_client
    r = client.get("/api/models")
    assert r.status_code == 200
    catalog = r.json()
    assert set(catalog.keys()) == {"openai", "anthropic", "google"}
    # Spot check
    openai_models = {m["model"] for m in catalog["openai"]}
    assert "gpt-5.5" in openai_models and "gpt-5.4-mini" in openai_models
    anth_models = {m["model"] for m in catalog["anthropic"]}
    assert "claude-opus-4-7" in anth_models and "claude-sonnet-4-6" in anth_models
    google_models = {m["model"] for m in catalog["google"]}
    assert "gemini-3.1-flash-lite" in google_models
    # Per-entry metadata
    sample = catalog["openai"][0]
    assert {"model", "context_window", "reasoning_style",
            "input_cost_per_million", "output_cost_per_million"} <= set(sample.keys())


def test_api_models_includes_cost_fields(app_client) -> None:
    """Costs are present for catalog entries we have COST_RATES for."""
    client, _, _ = app_client
    catalog = client.get("/api/models").json()
    by_id = {m["model"]: m for prov in catalog.values() for m in prov}

    # Spot check known rates from agentic_paper.audit.COST_RATES
    assert by_id["gpt-5.4-mini"]["input_cost_per_million"] == 0.5
    assert by_id["gpt-5.4-mini"]["output_cost_per_million"] == 2.0
    assert by_id["gpt-5.4-nano"]["input_cost_per_million"] == 0.2
    assert by_id["gpt-5.4-nano"]["output_cost_per_million"] == 1.25
    assert by_id["claude-opus-4-7"]["input_cost_per_million"] == 15.0
    assert by_id["claude-opus-4-7"]["output_cost_per_million"] == 75.0
    assert by_id["gemini-3.5-flash"]["input_cost_per_million"] == 1.5
    assert by_id["gemini-3.5-flash"]["output_cost_per_million"] == 9.0


def test_index_renders_cost_hint_wiring(app_client) -> None:
    """The cost hint container + onchange handler must be in the rendered HTML."""
    client, _, _ = app_client
    body = client.get("/").text
    assert 'id="model-cost-hint"' in body
    assert 'onchange="onModelChange()"' in body
    assert "function onModelChange" in body
    assert "input_cost_per_million" in body


def _primary_model_app(tmp_path: Path, captured: list):
    """Like _byok_app but captures the full routing config the runner received."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"output_dir: \"{tmp_path / 'out'}\"\napi_key: \"server-default-key\"\n"
        "routing:\n"
        "  tier_high:     {provider: anthropic, model: claude-opus-4-7, thinking_budget: auto}\n"
        "  tier_standard: {provider: openai, model: gpt-5.4-mini}\n"
        "  tier_basic:    {provider: google, model: gemini-3.1-flash-lite}\n",
        encoding="utf-8",
    )
    out_root = tmp_path / "out"

    def capturing_runner(pdf_path: Path, run_id: str, config) -> None:
        snap = {
            "run_id": run_id,
            "tier_high": (config.routing.tier_high.provider, config.routing.tier_high.model,
                          config.routing.tier_high.thinking_budget),
            "tier_standard": (config.routing.tier_standard.provider, config.routing.tier_standard.model,
                              config.routing.tier_standard.thinking_budget),
            "tier_basic": (config.routing.tier_basic.provider, config.routing.tier_basic.model,
                           config.routing.tier_basic.thinking_budget),
        }
        captured.append(snap)
        run_dir = out_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        AuditLogger(run_dir, run_id=run_id).record(
            agent="StubAgent", provider="stub", model="stub-model",
            prompt="x", response_text="y",
            usage={"input_tokens": 1, "output_tokens": 1},
            latency_ms=1, thinking_enabled=False, seed=None,
        )
        (run_dir / "review_report_test.md").write_text("# OK\n", encoding="utf-8")

    registry = RunRegistry(config_path=str(cfg_path), max_workers=1, runner=capturing_runner)
    app = create_app(config_path=str(cfg_path), registry=registry)
    return TestClient(app), registry, out_root


def test_primary_model_rewrites_all_three_tiers(tmp_path: Path) -> None:
    captured: list = []
    client, registry, _ = _primary_model_app(tmp_path, captured)

    r = client.post(
        "/review",
        files={"file": ("p.pdf", b"%PDF-1.4\nfake\n", "application/pdf")},
        data={"primary_provider": "anthropic", "primary_model": "claude-haiku-4-5"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["primary"] == {"provider": "anthropic", "model": "claude-haiku-4-5"}
    run_id = body["run_id"]
    _wait_for_state(registry, run_id, {"completed", "failed"})

    snap = captured[0]
    assert snap["tier_high"]    == ("anthropic", "claude-haiku-4-5", "auto")
    assert snap["tier_standard"] == ("anthropic", "claude-haiku-4-5", None)
    assert snap["tier_basic"]    == ("anthropic", "claude-haiku-4-5", None)
    registry.shutdown()


def test_primary_provider_without_model_uses_catalog_default(tmp_path: Path) -> None:
    captured: list = []
    client, registry, _ = _primary_model_app(tmp_path, captured)

    r = client.post(
        "/review",
        files={"file": ("p.pdf", b"%PDF-1.4\nfake\n", "application/pdf")},
        data={"primary_provider": "openai"},  # no model picked
    )
    body = r.json()
    assert body["primary"]["provider"] == "openai"
    # Catalog default for openai is the first entry: gpt-5.5
    assert body["primary"]["model"] == "gpt-5.5"
    run_id = body["run_id"]
    _wait_for_state(registry, run_id, {"completed", "failed"})

    snap = captured[0]
    assert snap["tier_high"][0] == "openai"
    assert snap["tier_high"][1] == "gpt-5.5"
    registry.shutdown()


def test_primary_provider_mismatched_model_is_ignored(tmp_path: Path) -> None:
    """If the user pairs anthropic+gpt-5.5, the override is silently dropped."""
    captured: list = []
    client, registry, _ = _primary_model_app(tmp_path, captured)

    r = client.post(
        "/review",
        files={"file": ("p.pdf", b"%PDF-1.4\nfake\n", "application/pdf")},
        data={"primary_provider": "anthropic", "primary_model": "gpt-5.5"},
    )
    body = r.json()
    # _resolve_primary_model returns None → response reports model=None
    assert body["primary"]["provider"] == "anthropic"
    assert body["primary"]["model"] is None
    run_id = body["run_id"]
    _wait_for_state(registry, run_id, {"completed", "failed"})

    # Original routing from config.yaml is preserved.
    snap = captured[0]
    assert snap["tier_high"][0] == "anthropic" and snap["tier_high"][1] == "claude-opus-4-7"
    assert snap["tier_standard"][0] == "openai" and snap["tier_standard"][1] == "gpt-5.4-mini"
    registry.shutdown()


def test_api_profiles_returns_three_bundles(app_client) -> None:
    client, _, _ = app_client
    r = client.get("/api/profiles")
    assert r.status_code == 200
    profiles = r.json()
    assert set(profiles.keys()) == {"max", "std", "quick"}
    for name, info in profiles.items():
        assert "tiers" in info
        assert set(info["tiers"]) == {"tier_high", "tier_standard", "tier_basic"}
        assert "estimated_cost_usd" in info
        assert info["estimated_cost_usd"] >= 0
        for tier_name, tier in info["tiers"].items():
            assert "provider" in tier and "model" in tier
            assert "agents_in_tier" in tier
    # Sanity: max should cost more than std, std more than quick.
    assert profiles["max"]["estimated_cost_usd"] > profiles["std"]["estimated_cost_usd"]
    assert profiles["std"]["estimated_cost_usd"] > profiles["quick"]["estimated_cost_usd"]


def test_profile_overrides_all_three_tiers(tmp_path: Path) -> None:
    captured: list = []
    client, registry, _ = _primary_model_app(tmp_path, captured)

    r = client.post(
        "/review",
        files={"file": ("p.pdf", b"%PDF-1.4\nfake\n", "application/pdf")},
        data={"profile": "quick"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["profile"] == "quick"
    run_id = body["run_id"]
    _wait_for_state(registry, run_id, {"completed", "failed"})

    snap = captured[0]
    # quick profile: haiku 4.5 + gpt-5.4-nano + flash-lite
    assert snap["tier_high"]    == ("anthropic", "claude-haiku-4-5", None)
    assert snap["tier_standard"] == ("openai", "gpt-5.4-nano", "low")
    assert snap["tier_basic"]    == ("google", "gemini-3.1-flash-lite", "low")
    registry.shutdown()


def test_profile_unknown_is_silently_ignored(tmp_path: Path) -> None:
    captured: list = []
    client, registry, _ = _primary_model_app(tmp_path, captured)

    r = client.post(
        "/review",
        files={"file": ("p.pdf", b"%PDF-1.4\nfake\n", "application/pdf")},
        data={"profile": "super-deluxe-bogus"},
    )
    body = r.json()
    assert body["profile"] is None
    run_id = body["run_id"]
    _wait_for_state(registry, run_id, {"completed", "failed"})

    # Original config.yaml routing preserved.
    snap = captured[0]
    assert snap["tier_high"][0] == "anthropic" and snap["tier_high"][1] == "claude-opus-4-7"
    registry.shutdown()


def test_primary_model_wins_over_profile(tmp_path: Path) -> None:
    """Primary (specific) takes precedence over profile (preset)."""
    captured: list = []
    client, registry, _ = _primary_model_app(tmp_path, captured)

    r = client.post(
        "/review",
        files={"file": ("p.pdf", b"%PDF-1.4\nfake\n", "application/pdf")},
        data={
            "profile": "max",                  # would route to opus + gpt-5.5 + gemini-3-pro
            "primary_provider": "openai",
            "primary_model": "gpt-5.4-mini",   # but primary overrides
        },
    )
    body = r.json()
    assert body["profile"] == "max"
    assert body["primary"]["model"] == "gpt-5.4-mini"
    run_id = body["run_id"]
    _wait_for_state(registry, run_id, {"completed", "failed"})

    # All 3 tiers should be openai/gpt-5.4-mini (primary won).
    snap = captured[0]
    for tier in ("tier_high", "tier_standard", "tier_basic"):
        assert snap[tier][0] == "openai"
        assert snap[tier][1] == "gpt-5.4-mini"
    registry.shutdown()


def test_index_renders_profile_radios(app_client) -> None:
    """Drop-zone must include the profile radio block + JS loader."""
    client, _, _ = app_client
    body = client.get("/").text
    assert "Quality vs cost preset" in body
    assert 'id="profile-radios"' in body
    assert 'name="profile"' in body
    assert "async function loadProfiles()" in body


def test_primary_model_combined_with_byok_keys(tmp_path: Path) -> None:
    """Both BYOK keys AND primary model override apply in the same request."""
    captured: list = []
    client, registry, _ = _primary_model_app(tmp_path, captured)

    r = client.post(
        "/review",
        files={"file": ("p.pdf", b"%PDF-1.4\nfake\n", "application/pdf")},
        data={
            "anthropic_api_key": "sk-ant-byok",
            "primary_provider": "anthropic",
            "primary_model": "claude-sonnet-4-6",
        },
    )
    body = r.json()
    assert body["byok"] is True
    assert body["primary"]["model"] == "claude-sonnet-4-6"
    run_id = body["run_id"]
    _wait_for_state(registry, run_id, {"completed", "failed"})

    snap = captured[0]
    assert snap["tier_high"] == ("anthropic", "claude-sonnet-4-6", "auto")
    registry.shutdown()
