"""End-to-end orchestrator run on the sample PDF text using only stub providers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentic_paper.config import Config, ProviderConfig, RoutingConfig, TierConfig
from agentic_paper.orchestrator import ReviewOrchestrator
from agentic_paper.providers import ProviderRegistry
from agentic_paper.providers.stub_provider import StubProvider
from agentic_paper.schemas import (
    AuthorEditorSummary,
    CoordinatorAssessment,
    EditorDecision,
    Review,
)


def _stub_registry_with_canned(
    canned_review, canned_coordinator, canned_editor, canned_summary
) -> tuple[ProviderRegistry, StubProvider]:
    stub = StubProvider()
    stub.set_response(Review, canned_review)
    stub.set_response(CoordinatorAssessment, canned_coordinator)
    stub.set_response(EditorDecision, canned_editor)
    stub.set_response(AuthorEditorSummary, canned_summary)
    reg = ProviderRegistry()
    for name in ("openai", "anthropic", "google"):
        reg.register(name, stub)
    return reg, stub


@pytest.fixture
def stub_orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canned_review,
    canned_coordinator,
    canned_editor,
    canned_summary,
) -> tuple[ReviewOrchestrator, StubProvider]:
    # Isolate output to tmp_path; pretend a key exists so Config.validate() would pass.
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    cfg = Config(
        api_key="dummy",
        output_dir=str(tmp_path / "out"),
        routing=RoutingConfig(
            tier_high=TierConfig(provider="anthropic", model="claude-opus-4-7"),
            tier_standard=TierConfig(provider="openai", model="gpt-5.4-mini"),
            tier_basic=TierConfig(provider="google", model="gemini-3.1-flash-lite"),
        ),
        providers={
            "openai": ProviderConfig(api_key_env="OPENAI_API_KEY"),
        },
        # Skip slow / network-dependent enrichments in unit tests.
        enrich_with_citations=False,
        enrich_with_statcheck=False,
    )
    orch = ReviewOrchestrator(cfg)
    reg, stub = _stub_registry_with_canned(
        canned_review, canned_coordinator, canned_editor, canned_summary
    )
    # Replace the registry built by the orchestrator with our stub-only one.
    orch.registry = reg
    return orch, stub


def test_full_pipeline_with_stub_returns_typed_results(
    stub_orchestrator: tuple[ReviewOrchestrator, StubProvider],
    sample_paper_text: str,
) -> None:
    orch, stub = stub_orchestrator
    results = orch.execute_review_process(sample_paper_text)

    assert "paper_info" in results
    assert results["paper_info"]["length"] > 0

    # 12 reviewers run in the fan-out. Plus coordinator + editor + summary downstream.
    reviews: dict[str, Any] = results["reviews"]
    assert len(reviews) == 12
    for key in (
        "methodology", "results", "literature", "structure", "impact",
        "contradiction", "ethics", "ai_origin", "hallucination",
        "citation_validator", "statcheck_validator", "revision_assessor",
    ):
        assert key in reviews
        review = reviews[key]
        assert review["recommendation"] in {"accept", "minor_rev", "major_rev", "reject"}
        assert 0.0 <= review["confidence"] <= 1.0
        assert review["agent"]
        assert review["model_used"]

    assert results["coordinator"] is not None
    assert results["coordinator"]["final_recommendation"] in {
        "accept", "minor_rev", "major_rev", "reject",
    }
    assert results["editor_decision"] is not None
    assert results["editor_decision"]["decision"] in {
        "accept", "minor_rev", "major_rev", "reject",
    }
    assert results["author_editor_summary"] is not None

    # No agent should have errored.
    assert results["errors"] == {}

    # Routing metadata was recorded.
    routing = results["config"]["routing"]
    assert len(routing) == 15   # 12 reviewers + coordinator + editor + summary
    assert routing["methodology"]["schema"] == "Review"
    assert routing["coordinator"]["schema"] == "CoordinatorAssessment"
    assert routing["editor"]["schema"] == "EditorDecision"
    assert routing["author_editor_summary"]["schema"] == "AuthorEditorSummary"


def test_orchestrator_writes_report_artefacts(
    stub_orchestrator: tuple[ReviewOrchestrator, StubProvider],
    sample_paper_text: str,
    tmp_path: Path,
) -> None:
    orch, _ = stub_orchestrator
    orch.execute_review_process(sample_paper_text)

    out_dir = Path(orch.config.output_dir) / orch.run_id
    assert (out_dir / "paper_info.json").exists()
    # At least one of each report kind should land in the output dir.
    md_files = list(out_dir.glob("review_report_*.md"))
    json_files = list(out_dir.glob("review_results_*.json"))
    html_files = list(out_dir.glob("dashboard_*.html"))
    summary_files = list(out_dir.glob("executive_summary_*.md"))
    assert md_files and json_files and html_files and summary_files

    # Per-agent reviews are JSON-serialized AnnotatedReview dumps.
    methodology_blob = (out_dir / "review_methodology.txt").read_text()
    assert '"recommendation"' in methodology_blob
    assert '"summary"' in methodology_blob


def test_orchestrator_writes_audit_artefacts(
    stub_orchestrator: tuple[ReviewOrchestrator, StubProvider],
    sample_paper_text: str,
) -> None:
    orch, _ = stub_orchestrator
    results = orch.execute_review_process(sample_paper_text)

    out_dir = Path(orch.config.output_dir) / orch.run_id

    # audit.jsonl: 12 rows (9 reviewers + coordinator + summary + editor).
    audit_path = out_dir / "audit.jsonl"
    assert audit_path.exists()
    import json
    rows = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
    # 12 reviewers + coordinator + summary + editor = 15 calls.
    assert len(rows) == 15, f"expected 15 audit rows, got {len(rows)}"
    expected_fields = {
        "timestamp", "run_id", "agent", "provider", "model", "prompt_hash",
        "input_token_count", "output_token_count", "latency_ms",
        "cost_estimate_usd", "thinking_mode_enabled", "seed",
    }
    for row in rows:
        assert expected_fields.issubset(row.keys()), set(expected_fields) - set(row.keys())

    # prompts/ and responses/ both have at least 15 files.
    assert len(list((out_dir / "prompts").glob("*.txt"))) >= 15
    assert len(list((out_dir / "responses").glob("*.json"))) >= 15

    # audit_summary present in results.
    summary = results.get("audit_summary", {})
    assert summary and summary["total_calls"] == 15
    assert summary["run_id"] == orch.run_id

    # Markdown report includes the cost section.
    md_files = list(out_dir.glob("review_report_*.md"))
    assert md_files
    md_text = md_files[0].read_text()
    assert "## Cost & Token Usage" in md_text
    assert orch.run_id in md_text


def test_orchestrator_scopes_under_run_id(
    stub_orchestrator: tuple[ReviewOrchestrator, StubProvider],
    sample_paper_text: str,
) -> None:
    orch, _ = stub_orchestrator
    orch.execute_review_process(sample_paper_text)
    # All artefacts must land under output/<run_id>/, not the bare output dir.
    base = Path(orch.config.output_dir)
    assert base.exists()
    assert (base / orch.run_id).exists()
    # The bare output dir should contain *only* run-id subdirs.
    leaked = [p for p in base.iterdir() if p.is_file()]
    assert not leaked, f"expected no files at base output dir, found {leaked}"


def test_seed_is_propagated_to_agents(
    stub_orchestrator: tuple[ReviewOrchestrator, StubProvider],
    sample_paper_text: str,
) -> None:
    orch, stub = stub_orchestrator
    orch.config.seed = 12345
    # Rebuild agents to pick up the new seed.
    orch.execute_review_process(sample_paper_text)
    # Every recorded stub call should carry the seed.
    seen_seeds = {call.get("seed") for call in stub.calls}
    assert seen_seeds == {12345}, seen_seeds


def test_retry_error_is_unwrapped_to_inner_message() -> None:
    """When tenacity wraps the real failure in RetryError, the report should
    still show the underlying exception's class + message (regression for the
    live ``RetryError[]`` empty-brackets output seen on 2026-05-21)."""
    from concurrent.futures import Future
    from tenacity import RetryError

    from agentic_paper.orchestrator import _format_exception

    class _Boom(RuntimeError):
        pass

    # Build a Future that already raised so RetryError.last_attempt.result()
    # re-raises the inner exception.
    fut: Future = Future()
    fut.set_exception(_Boom("400 INVALID_ARGUMENT — additional_properties"))
    err = RetryError(last_attempt=fut)

    msg = _format_exception(err)
    assert "_Boom" in msg
    assert "INVALID_ARGUMENT" in msg
    assert "RetryError" not in msg, "the wrapper class should not bleed through"


def test_format_exception_for_plain_exception() -> None:
    from agentic_paper.orchestrator import _format_exception
    out = _format_exception(ValueError("boom"))
    assert out == "ValueError: boom"


def test_run_context_preamble_pins_today_date() -> None:
    """LLM prompts must carry today's date so old training cutoffs don't
    misread past events in the paper as future predictions."""
    from datetime import datetime

    from agentic_paper.orchestrator import _run_context_preamble

    today = datetime.now().strftime("%Y-%m-%d")
    pre = _run_context_preamble()
    assert "=== RUN CONTEXT ===" in pre
    assert today in pre
    assert "Today's date is" in pre
    # And the override works for tests that need a fixed date.
    pre_fixed = _run_context_preamble(today="2099-01-01")
    assert "2099-01-01" in pre_fixed


def test_initial_message_contains_date_preamble(
    stub_orchestrator: tuple[ReviewOrchestrator, StubProvider],
    sample_paper_text: str,
) -> None:
    """The full reviewer message body must start with the date preamble."""
    from datetime import datetime

    from agentic_paper.paper import PaperInfo

    orch, _ = stub_orchestrator
    today = datetime.now().strftime("%Y-%m-%d")
    info = PaperInfo(
        title="X", authors="Y", abstract="z", length=10, sections=[], file_path=None,
    )
    msg = orch._prepare_initial_message(info, sample_paper_text)
    assert msg.startswith("=== RUN CONTEXT ===")
    assert today in msg
    assert "Paper to be analyzed:" in msg
