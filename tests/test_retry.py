"""Retry-failed-agents path — orchestrator + web endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_paper.config import (
    Config,
    ProviderConfig,
    RoutingConfig,
    TierConfig,
)
from agentic_paper.orchestrator import ReviewOrchestrator
from agentic_paper.providers import ProviderRegistry
from agentic_paper.providers.stub_provider import StubProvider
from agentic_paper.schemas import (
    AuthorEditorSummary,
    CoordinatorAssessment,
    EditorDecision,
    Review,
)


def _stub_registry(canned_review, canned_coord, canned_editor, canned_summary):
    stub = StubProvider()
    stub.set_response(Review, canned_review)
    stub.set_response(CoordinatorAssessment, canned_coord)
    stub.set_response(EditorDecision, canned_editor)
    stub.set_response(AuthorEditorSummary, canned_summary)
    reg = ProviderRegistry()
    for name in ("openai", "anthropic", "google"):
        reg.register(name, stub)
    return reg, stub


def test_retry_replays_only_missing_agents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canned_review,
    canned_coordinator,
    canned_editor,
    canned_summary,
    sample_paper_text: str,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    cfg = Config(
        api_key="dummy",
        output_dir=str(tmp_path / "out"),
        enrich_with_citations=False,
        enrich_with_statcheck=False,
        routing=RoutingConfig(
            tier_high=TierConfig(provider="openai", model="gpt-5.4-mini"),
            tier_standard=TierConfig(provider="openai", model="gpt-5.4-mini"),
            tier_basic=TierConfig(provider="openai", model="gpt-5.4-mini"),
        ),
        providers={"openai": ProviderConfig(api_key_env="OPENAI_API_KEY")},
    )
    orch = ReviewOrchestrator(cfg)
    reg, stub = _stub_registry(canned_review, canned_coordinator, canned_editor, canned_summary)
    orch.registry = reg
    orch.execute_review_process(sample_paper_text)

    run_dir = Path(cfg.output_dir) / orch.run_id
    assert (run_dir / "paper.txt").exists()
    assert (run_dir / "_run_state.json").exists()

    # All review_*.txt files present after the initial run.
    review_files_before = sorted(p.name for p in run_dir.glob("review_*.txt"))
    assert len(review_files_before) > 0

    # Simulate a failed methodology agent by deleting its saved review.
    (run_dir / "review_methodology.txt").unlink()

    # Build a fresh orchestrator on the same run_id (web layer does this).
    orch2 = ReviewOrchestrator(cfg, run_id=orch.run_id)
    orch2.registry = reg
    result = orch2.retry_failed_agents()

    assert result["retried"] == ["methodology"]
    assert "methodology" in result["succeeded"]
    assert result["failed"] == {}
    assert (run_dir / "review_methodology.txt").exists()


def test_retry_with_explicit_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    canned_review, canned_coordinator, canned_editor, canned_summary,
    sample_paper_text: str,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    cfg = Config(
        api_key="dummy", output_dir=str(tmp_path / "out"),
        enrich_with_citations=False, enrich_with_statcheck=False,
        providers={"openai": ProviderConfig(api_key_env="OPENAI_API_KEY")},
    )
    orch = ReviewOrchestrator(cfg)
    reg, _ = _stub_registry(canned_review, canned_coordinator, canned_editor, canned_summary)
    orch.registry = reg
    orch.execute_review_process(sample_paper_text)

    orch2 = ReviewOrchestrator(cfg, run_id=orch.run_id)
    orch2.registry = reg
    result = orch2.retry_failed_agents(only_keys=["literature", "ethics"])

    assert set(result["retried"]) == {"literature", "ethics"}
    assert set(result["succeeded"]) == {"literature", "ethics"}


def test_retry_404_when_run_does_not_exist(tmp_path: Path) -> None:
    cfg = Config(api_key="dummy", output_dir=str(tmp_path / "nope"))
    orch = ReviewOrchestrator(cfg, run_id="nonexistent")
    with pytest.raises(FileNotFoundError):
        orch.retry_failed_agents()


def test_retry_endpoint_returns_410_for_unknown_or_clobbered_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    canned_review, canned_coordinator, canned_editor, canned_summary,
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from agentic_paper.web.runner import RunRegistry, RunStatus
    from agentic_paper.web.server import create_app

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(f"output_dir: \"{tmp_path / 'out'}\"\napi_key: dummy\n")
    out_root = tmp_path / "out"

    def _runner(pdf_path, run_id, config) -> None:
        # Do nothing; the retry endpoint needs paper.txt which doesn't exist → 410.
        (out_root / run_id).mkdir(parents=True, exist_ok=True)

    registry = RunRegistry(config_path=str(cfg_path), runner=_runner)
    # Register a dummy run.
    rid = "manual-x"
    (out_root / rid).mkdir(parents=True, exist_ok=True)
    registry.runs[rid] = RunStatus(
        run_id=rid, pdf_path=tmp_path / "x.pdf",
        state="completed", output_dir=out_root / rid,
    )
    app = create_app(config_path=str(cfg_path), registry=registry)
    client = TestClient(app)

    r = client.post(f"/runs/{rid}/retry")
    assert r.status_code == 410
    assert "paper.txt" in r.json()["detail"]
    registry.shutdown()
