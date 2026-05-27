"""Python wrapper around the bundled ``statcheck.R``.

``statcheck`` (Nuijten et al., 2016) is an R package that detects inferential
statistics in scientific text and recomputes the p-values from the test
statistic + degrees of freedom. It finds numerical reporting errors that
LLMs are bad at catching (because they're not running the arithmetic).

This module:
    * Detects if ``Rscript`` is on PATH.
    * Calls the bundled ``statcheck.R`` script via subprocess, piping the
      paper text on stdin.
    * Parses the JSON the R script emits.
    * Degrades gracefully (no exception) when R, the package, or the script
      itself are unavailable — the orchestrator continues with a "not
      available" report so the agent can say so transparently.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_STATCHECK_R_SCRIPT = Path(__file__).resolve().parent / "statcheck.R"


@dataclass
class StatRow:
    """One row of statcheck output — a single detected inferential statistic."""

    statistic: str
    value: float | None
    reported_p_value: float | None
    computed_p_value: float | None
    error: bool = False
    decision_error: bool = False
    raw: str = ""
    reported_comparison: str = "="
    df1: float | None = None
    df2: float | None = None


@dataclass
class StatcheckReport:
    available: bool = False
    reason: str = ""
    n_stats: int = 0
    n_errors: int = 0
    n_decision_errors: int = 0
    rows: list[StatRow] = field(default_factory=list)


def run_statcheck(paper_text: str, *, timeout: float = 30.0) -> StatcheckReport:
    """Run the bundled R script against the paper text.

    All failure modes (no Rscript, package missing, timeout, parse error)
    return a ``StatcheckReport(available=False)`` with the reason filled in —
    they never raise. The agent layer will surface ``reason`` to the user
    instead of bombing the run.
    """
    if not shutil.which("Rscript"):
        return StatcheckReport(
            available=False,
            reason=(
                "Rscript not found on PATH. Install R from https://www.r-project.org/ "
                "and then `install.packages(c('statcheck', 'jsonlite'))` from an R REPL."
            ),
        )
    if not _STATCHECK_R_SCRIPT.exists():
        return StatcheckReport(
            available=False, reason=f"bundled statcheck.R missing at {_STATCHECK_R_SCRIPT}",
        )
    try:
        proc = subprocess.run(
            ["Rscript", "--vanilla", str(_STATCHECK_R_SCRIPT)],
            input=paper_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return StatcheckReport(available=False, reason=f"Rscript exceeded {timeout}s timeout")
    except OSError as e:
        return StatcheckReport(available=False, reason=f"subprocess failed: {e}")

    if proc.returncode != 0:
        msg = (proc.stderr or "").strip()[:200] or "non-zero exit"
        return StatcheckReport(
            available=False, reason=f"Rscript returned {proc.returncode}: {msg}",
        )

    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as e:
        return StatcheckReport(
            available=False,
            reason=f"R script output was not valid JSON ({e}); stdout head: {proc.stdout[:200]!r}",
        )

    if not data.get("available"):
        return StatcheckReport(
            available=False, reason=str(data.get("reason", "unknown statcheck failure")),
        )

    rows: list[StatRow] = []
    for r in data.get("rows", []) or []:
        rows.append(
            StatRow(
                statistic=str(r.get("statistic", "")),
                value=r.get("value"),
                reported_p_value=r.get("reported_p_value"),
                computed_p_value=r.get("computed_p_value"),
                error=bool(r.get("error", False)),
                decision_error=bool(r.get("decision_error", False)),
                raw=str(r.get("raw", "")),
                reported_comparison=str(r.get("reported_comparison", "=")),
                df1=r.get("df1"),
                df2=r.get("df2"),
            )
        )

    return StatcheckReport(
        available=True,
        n_stats=int(data.get("n_stats", 0)),
        n_errors=int(data.get("n_errors", 0)),
        n_decision_errors=int(data.get("n_decision_errors", 0)),
        rows=rows,
    )


def _stat_marker(r: StatRow) -> str:
    """One of ``❌ NUMERICAL ERROR`` / ``⚠️ DECISION ERROR`` / ``✅``.

    Numerical errors take priority over decision errors (a row can flag both —
    a recomputed p that disagrees and crosses the .05 threshold) because the
    numerical mismatch is the more actionable issue.
    """
    if r.error:
        return "❌ NUMERICAL ERROR"
    if r.decision_error:
        return "⚠️ DECISION ERROR"
    return "✅"


def _format_df_part(r: StatRow) -> str:
    """Render degrees-of-freedom suffix: ``(df1,df2)``, ``(df1)``, or empty."""
    if r.df1 is not None and r.df2 is not None:
        return f"({r.df1},{r.df2})"
    if r.df1 is not None:
        return f"({r.df1})"
    return ""


def _format_optional_number(v: float | None) -> str:
    """``"?"`` when missing, ``%g`` formatted otherwise. Same convention used
    for the three p-value / test-statistic columns of the detail rows."""
    if v is None:
        return "?"
    return f"{v:g}"


def _format_single_stat(i: int, r: StatRow) -> list[str]:
    """Return the 1-2 formatted lines for one StatRow: the head plus an
    optional ``raw:`` follow-up line. Caller does ``lines.extend(...)``."""
    head = (
        f"  [{i}] {_stat_marker(r)} {r.statistic}{_format_df_part(r)} = "
        f"{_format_optional_number(r.value)}, "
        f"reported p {r.reported_comparison} {_format_optional_number(r.reported_p_value)}, "
        f"recomputed p ≈ {_format_optional_number(r.computed_p_value)}"
    )
    if not r.raw:
        return [head]
    return [head, f"        raw: {r.raw[:120]}"]


def _format_overall_verdict(report: StatcheckReport) -> str:
    """Single-line summary right under the totals block."""
    if report.n_errors == 0 and report.n_decision_errors == 0:
        return "  ✅ No numerical or decision errors found."
    return "  ❌ At least one reported p-value does not match the recomputed one."


def format_report_for_agent(report: StatcheckReport) -> str:
    """Render the StatcheckReport as plain-text suitable for an LLM prompt."""
    if not report.available:
        return f"statcheck is not available for this run — {report.reason}"
    if report.n_stats == 0:
        return (
            "statcheck (R) ran successfully but did not detect any parseable "
            "inferential statistics in the paper. Either the paper makes no "
            "inferential claims or the statistical reporting is in a format "
            "the package does not currently parse."
        )

    lines: list[str] = [
        f"statcheck (R) analysed {report.n_stats} reported inferential statistics.",
        f"  Numerical reporting errors (reported p ≠ recomputed p): {report.n_errors}",
        f"  Decision errors (significance flips around the threshold): {report.n_decision_errors}",
        "",
        _format_overall_verdict(report),
        "",
        "Per-statistic detail:",
    ]
    for i, r in enumerate(report.rows, 1):
        lines.extend(_format_single_stat(i, r))
    return "\n".join(lines)


__all__ = ["StatRow", "StatcheckReport", "run_statcheck", "format_report_for_agent"]
