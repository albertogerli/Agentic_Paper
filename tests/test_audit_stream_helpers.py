"""Direct unit tests for the helpers extracted from ``_audit_event_stream``.

Before the refactor these code paths were only reachable by spinning up a
FastAPI TestClient + tailing SSE. Now they're three small pure-ish functions
testable in isolation.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from agentic_paper.web.runner import RunRegistry, RunStatus  # noqa: E402
from agentic_paper.web.server import (  # noqa: E402
    _AUDIT_TICK_SECONDS,
    _AUDIT_WAIT_TICKS,
    _parse_audit_record,
    _read_new_audit_lines,
    _wait_for_audit_file,
)


# ---------------------------------------------------------------- _parse_audit_record


def test_parse_audit_record_returns_dict_for_valid_jsonl() -> None:
    out = _parse_audit_record('{"agent": "x", "model": "m", "input_token_count": 7}')
    assert out == {"agent": "x", "model": "m", "input_token_count": 7}


def test_parse_audit_record_returns_none_for_blank_or_malformed() -> None:
    assert _parse_audit_record("") is None
    assert _parse_audit_record("   \t  ") is None
    assert _parse_audit_record("{not json") is None
    assert _parse_audit_record("12345") == 12345  # valid JSON, even if not a dict
    # NOTE: a bare int still parses; the caller treats every truthy parse as
    # an event. We document the behaviour rather than constrain it here.


# ---------------------------------------------------------------- _read_new_audit_lines


def test_read_new_audit_lines_returns_cursor_to_eof(tmp_path: Path) -> None:
    p = tmp_path / "audit.jsonl"
    p.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
    lines, new_pos = _read_new_audit_lines(p, 0)
    assert lines == ['{"a":1}', '{"a":2}']
    assert new_pos == p.stat().st_size


def test_read_new_audit_lines_picks_only_new_content_since_last_pos(tmp_path: Path) -> None:
    p = tmp_path / "audit.jsonl"
    p.write_text('{"a":1}\n', encoding="utf-8")
    _, after_first = _read_new_audit_lines(p, 0)
    # Append more — the tail-read should return only the new rows.
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"a":2}\n{"a":3}\n')
    lines, new_pos = _read_new_audit_lines(p, after_first)
    assert lines == ['{"a":2}', '{"a":3}']
    assert new_pos == p.stat().st_size


def test_read_new_audit_lines_swallows_oserror(tmp_path: Path) -> None:
    """If the file is missing/locked, the helper returns ([], same_cursor)."""
    missing = tmp_path / "nope.jsonl"
    lines, new_pos = _read_new_audit_lines(missing, 42)
    assert lines == []
    assert new_pos == 42


# ---------------------------------------------------------------- _wait_for_audit_file


def _registry_with_status(state: str, run_dir: Path) -> RunRegistry:
    reg = RunRegistry(config_path=str(run_dir / "no-config.yaml"))
    reg.runs["r"] = RunStatus(
        run_id="r", pdf_path=run_dir / "x.pdf",
        state=state, output_dir=run_dir,
    )
    return reg


def test_wait_for_audit_file_returns_true_when_file_already_present(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "audit.jsonl"
    audit.write_text("{}\n", encoding="utf-8")
    reg = _registry_with_status("running", tmp_path)
    ok = asyncio.run(_wait_for_audit_file(reg, "r", audit))
    assert ok is True


def test_wait_for_audit_file_short_circuits_on_failed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the run is already failed and no audit file ever appears, we
    bail out of the polling loop early — we don't wait the full 30s."""
    audit = tmp_path / "audit.jsonl"   # deliberately not created
    reg = _registry_with_status("failed", tmp_path)

    # Patch sleep to a no-op so the 60 ticks don't waste 30s of test time.
    async def _no_sleep(_t: float) -> None:
        return None
    monkeypatch.setattr("agentic_paper.web.server.asyncio.sleep", _no_sleep)

    ok = asyncio.run(_wait_for_audit_file(reg, "r", audit))
    assert ok is False


# ---------------------------------------------------------------- constants


def test_wait_constants_match_refactor_intent() -> None:
    """30-second total cap is the spec; verify the ticks * cadence still hold."""
    assert _AUDIT_WAIT_TICKS * _AUDIT_TICK_SECONDS == pytest.approx(30.0)
