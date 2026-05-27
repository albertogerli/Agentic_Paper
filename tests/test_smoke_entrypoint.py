"""Smoke entry-point + CLI surface tests."""

from __future__ import annotations

import pytest

from agentic_paper.providers import smoke as smoke_module


def test_smoke_main_reports_no_providers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    rc = smoke_module.main(["--config", "/definitely/does/not/exist.yaml"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "No providers configured." in captured.out


def test_cli_help_does_not_raise() -> None:
    from agentic_paper.cli import _build_parser
    parser = _build_parser()
    # argparse SystemExits with 0 on --help; capture it.
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0


def test_system_health_check_handles_no_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    from agentic_paper.cli import system_health_check
    from agentic_paper.config import Config
    report = system_health_check(Config(api_key=""))
    assert "providers" in report and report["providers_ok"] is False
