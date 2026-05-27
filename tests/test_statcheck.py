"""statcheck R subprocess wrapper — tests mock the subprocess so they run
without R installed in CI."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from agentic_paper.external import statcheck


def _fake_proc(returncode=0, stdout="{}", stderr=""):
    cp = subprocess.CompletedProcess(args=["Rscript"], returncode=returncode)
    cp.stdout, cp.stderr = stdout, stderr
    return cp


def test_no_rscript_returns_friendly_reason() -> None:
    with patch("agentic_paper.external.statcheck.shutil.which", return_value=None):
        r = statcheck.run_statcheck("any text")
    assert r.available is False
    assert "Rscript not found" in r.reason
    assert r.n_stats == 0


def test_rscript_timeout_is_caught() -> None:
    with patch("agentic_paper.external.statcheck.shutil.which", return_value="/usr/bin/Rscript"):
        with patch(
            "agentic_paper.external.statcheck.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="Rscript", timeout=5),
        ):
            r = statcheck.run_statcheck("any text", timeout=5)
    assert r.available is False
    assert "timeout" in r.reason.lower()


def test_rscript_nonzero_exit_is_reported() -> None:
    with patch("agentic_paper.external.statcheck.shutil.which", return_value="/usr/bin/Rscript"):
        with patch(
            "agentic_paper.external.statcheck.subprocess.run",
            return_value=_fake_proc(returncode=2, stderr="boom"),
        ):
            r = statcheck.run_statcheck("any text")
    assert r.available is False
    assert "Rscript returned 2" in r.reason


def test_rscript_garbage_output_reported() -> None:
    with patch("agentic_paper.external.statcheck.shutil.which", return_value="/usr/bin/Rscript"):
        with patch(
            "agentic_paper.external.statcheck.subprocess.run",
            return_value=_fake_proc(returncode=0, stdout="this is not JSON"),
        ):
            r = statcheck.run_statcheck("any text")
    assert r.available is False
    assert "not valid JSON" in r.reason


def test_rscript_package_missing_reason_surfaces() -> None:
    payload = {"available": False, "reason": "R package statcheck not installed"}
    with patch("agentic_paper.external.statcheck.shutil.which", return_value="/usr/bin/Rscript"):
        with patch(
            "agentic_paper.external.statcheck.subprocess.run",
            return_value=_fake_proc(stdout=json.dumps(payload)),
        ):
            r = statcheck.run_statcheck("any text")
    assert r.available is False
    assert "statcheck not installed" in r.reason


def test_parses_rows_into_report() -> None:
    payload = {
        "available": True,
        "n_stats": 2,
        "n_errors": 1,
        "n_decision_errors": 0,
        "rows": [
            {
                "statistic": "t", "df1": None, "df2": 28, "value": 2.3,
                "reported_comparison": "=", "reported_p_value": 0.04,
                "computed_p_value": 0.0291,
                "error": True, "decision_error": False,
                "raw": "t(28) = 2.3, p = .04",
            },
            {
                "statistic": "F", "df1": 2, "df2": 100, "value": 5.5,
                "reported_comparison": "=", "reported_p_value": 0.005,
                "computed_p_value": 0.0054,
                "error": False, "decision_error": False,
                "raw": "F(2, 100) = 5.5, p = .005",
            },
        ],
    }
    with patch("agentic_paper.external.statcheck.shutil.which", return_value="/usr/bin/Rscript"):
        with patch(
            "agentic_paper.external.statcheck.subprocess.run",
            return_value=_fake_proc(stdout=json.dumps(payload)),
        ):
            r = statcheck.run_statcheck("…")
    assert r.available is True
    assert r.n_stats == 2 and r.n_errors == 1 and r.n_decision_errors == 0
    assert len(r.rows) == 2
    assert r.rows[0].error is True and r.rows[1].error is False
    assert r.rows[0].statistic == "t"
    assert r.rows[1].df1 == 2 and r.rows[1].df2 == 100


def test_format_report_emits_error_marker() -> None:
    from agentic_paper.external.statcheck import StatcheckReport, StatRow
    r = StatcheckReport(
        available=True, n_stats=1, n_errors=1, n_decision_errors=0,
        rows=[StatRow(
            statistic="t", value=2.3, reported_p_value=0.04,
            computed_p_value=0.0291, error=True, raw="t(28) = 2.3, p = .04",
            df1=None, df2=28, reported_comparison="=",
        )],
    )
    out = statcheck.format_report_for_agent(r)
    assert "NUMERICAL ERROR" in out
    assert "0.04" in out and "0.0291" in out
    assert "t(28) = 2.3" in out


def test_format_report_when_unavailable() -> None:
    out = statcheck.format_report_for_agent(
        statcheck.StatcheckReport(available=False, reason="R not installed")
    )
    assert "not available" in out
    assert "R not installed" in out


def test_statcheck_validator_agent_metadata() -> None:
    from agentic_paper.agents import statcheck_validator as sv
    assert sv.KEY == "statcheck_validator"
    assert sv.NAME == "Statistical_Sanity_Validator"
    assert "statcheck" in sv.INSTRUCTIONS


def test_statcheck_validator_runs_via_stub(stub_provider) -> None:
    from agentic_paper.agents import statcheck_validator as sv
    from agentic_paper.agents.base import Agent
    from agentic_paper.schemas import Review
    agent = Agent(
        name=sv.NAME, instructions=sv.INSTRUCTIONS, model="stub-model",
        provider=stub_provider, schema=sv.SCHEMA, max_output_tokens=512,
    )
    result = agent.run("paper with STATCHECK REPORT attached")
    assert isinstance(result, Review)
