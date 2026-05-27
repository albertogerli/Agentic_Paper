"""Audit logger + cost estimation + run_id generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_paper.audit import (
    COST_RATES,
    AuditLogger,
    estimate_cost,
    generate_run_id,
    hash_prompt,
)


def test_generate_run_id_is_unique_and_well_formed() -> None:
    ids = {generate_run_id() for _ in range(20)}
    assert len(ids) == 20
    for rid in ids:
        # YYYYMMDD-HHMMSS-XXXXXX
        parts = rid.split("-")
        assert len(parts) == 3
        assert parts[0].isdigit() and len(parts[0]) == 8
        assert parts[1].isdigit() and len(parts[1]) == 6
        assert len(parts[2]) == 6


def test_hash_prompt_is_deterministic_and_short() -> None:
    assert hash_prompt("hello world") == hash_prompt("hello world")
    assert hash_prompt("a") != hash_prompt("b")
    assert hash_prompt("") == hash_prompt(None)  # type: ignore[arg-type]
    assert len(hash_prompt("hello world")) == 16


@pytest.mark.parametrize(
    "provider, model, in_tok, out_tok",
    [
        ("openai", "gpt-5.5", 1_000_000, 0),         # 5 USD
        ("openai", "gpt-5.5", 0, 1_000_000),         # 25 USD
        ("anthropic", "claude-opus-4-7", 1_000_000, 1_000_000),  # 90 USD
        ("google", "gemini-3.1-flash-lite", 1_000_000, 1_000_000),  # 1.75 USD
    ],
)
def test_estimate_cost_uses_rate_table(provider, model, in_tok, out_tok) -> None:
    cost = estimate_cost(provider, model, in_tok, out_tok)
    in_rate, out_rate = COST_RATES[(provider, model)]
    expected = (in_tok / 1_000_000) * in_rate + (out_tok / 1_000_000) * out_rate
    assert abs(cost - expected) < 1e-9


def test_estimate_cost_handles_unknown_model_via_prefix() -> None:
    cost = estimate_cost("openai", "gpt-5.5-2026-04-23", 1_000_000, 0)
    # Should match the "gpt-5.5" prefix rate (5 USD per 1M input).
    assert cost == pytest.approx(5.0)


def test_estimate_cost_returns_zero_for_unknown_provider() -> None:
    assert estimate_cost("totally-unknown", "x", 1_000, 1_000) == 0.0
    assert estimate_cost("openai", "gpt-5.5", 0, 0) == 0.0


def test_audit_logger_writes_records_and_artefacts(tmp_path: Path) -> None:
    al = AuditLogger(tmp_path / "run-x", run_id="run-x")

    rec = al.record(
        agent="Methodology_Expert",
        provider="anthropic",
        model="claude-opus-4-7",
        prompt="SYS\n\nuser",
        response_text='{"summary":"ok"}',
        usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        latency_ms=1234,
        thinking_enabled=True,
        seed=42,
    )

    assert rec["agent"] == "Methodology_Expert"
    assert rec["provider"] == "anthropic"
    assert rec["seed"] == 42
    assert rec["thinking_mode_enabled"] is True
    assert rec["input_token_count"] == 100 and rec["output_token_count"] == 50
    # claude-opus-4-7: 15/75 per 1M → 100*15/1e6 + 50*75/1e6 = 0.001500 + 0.003750 = 0.00525
    assert rec["cost_estimate_usd"] == pytest.approx(0.00525)
    assert len(rec["prompt_hash"]) == 16

    # JSONL file has one row matching the in-memory record.
    audit_path = tmp_path / "run-x" / "audit.jsonl"
    assert audit_path.exists()
    rows = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["agent"] == "Methodology_Expert"

    # Prompt + response files written.
    assert (tmp_path / "run-x" / "prompts" / "Methodology_Expert.txt").read_text() == "SYS\n\nuser"
    resp = json.loads((tmp_path / "run-x" / "responses" / "Methodology_Expert.json").read_text())
    assert resp["usage"]["input_tokens"] == 100
    assert resp["text"] == '{"summary":"ok"}'


def test_audit_logger_summary_aggregates_per_provider(tmp_path: Path) -> None:
    al = AuditLogger(tmp_path / "run-y", run_id="run-y")
    al.record(
        agent="A", provider="openai", model="gpt-5.4-mini",
        prompt="p1", response_text="r1",
        usage={"input_tokens": 1000, "output_tokens": 500},
        latency_ms=10, thinking_enabled=False, seed=None,
    )
    al.record(
        agent="B", provider="openai", model="gpt-5.4-mini",
        prompt="p2", response_text="r2",
        usage={"input_tokens": 2000, "output_tokens": 700},
        latency_ms=20, thinking_enabled=True, seed=42,
    )
    al.record(
        agent="C", provider="anthropic", model="claude-haiku-4-5",
        prompt="p3", response_text="r3",
        usage={"input_tokens": 500, "output_tokens": 100},
        latency_ms=30, thinking_enabled=True, seed=42,
    )

    s = al.summary()
    assert s["run_id"] == "run-y"
    assert s["total_calls"] == 3
    assert s["total_input_tokens"] == 3500
    assert s["total_output_tokens"] == 1300
    assert "openai" in s["per_provider"] and "anthropic" in s["per_provider"]
    assert s["per_provider"]["openai"]["calls"] == 2
    assert s["per_provider"]["openai"]["input_tokens"] == 3000
    assert s["per_provider"]["anthropic"]["calls"] == 1
    assert set(s["per_agent"]) == {"A", "B", "C"}
    assert s["per_agent"]["B"]["thinking_mode_enabled"] is True


def test_audit_logger_safe_filename_for_weird_agent_names(tmp_path: Path) -> None:
    al = AuditLogger(tmp_path / "run-z", run_id="run-z")
    al.record(
        agent="weird name / with slashes",
        provider="openai", model="gpt-4o-mini",
        prompt="p", response_text="r",
        usage={"input_tokens": 1, "output_tokens": 1},
        latency_ms=1, thinking_enabled=False, seed=None,
    )
    # File should be created with sanitised name; no traversal.
    files = list((tmp_path / "run-z" / "prompts").iterdir())
    assert len(files) == 1
    assert "/" not in files[0].name
