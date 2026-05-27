"""Report renderer tests — markdown, JSON export, HTML dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agentic_paper.reports import (
    export_json,
    render_executive_summary,
    render_html,
    render_markdown,
)
from agentic_paper.schemas import (
    AnnotatedAuthorEditorSummary,
    AnnotatedCoordinatorAssessment,
    AnnotatedEditorDecision,
    AnnotatedReview,
    AuthorEditorSummary,
    CoordinatorAssessment,
    EditorDecision,
    Review,
)


def _build_results(
    canned_review: Review,
    canned_coordinator: CoordinatorAssessment,
    canned_editor: EditorDecision,
    canned_summary: AuthorEditorSummary,
) -> dict[str, Any]:
    ar = AnnotatedReview(
        agent="Methodology_Expert", model_used="claude-opus-4-7", **canned_review.model_dump()
    )
    ac = AnnotatedCoordinatorAssessment(
        agent="Review_Coordinator", model_used="claude-opus-4-7", **canned_coordinator.model_dump()
    )
    ae = AnnotatedEditorDecision(
        agent="Journal_Editor", model_used="gpt-5.5", **canned_editor.model_dump()
    )
    asum = AnnotatedAuthorEditorSummary(
        agent="Author_Editor_Summary_Agent", model_used="gpt-5.4-mini", **canned_summary.model_dump()
    )
    return {
        "paper_info": {
            "title": "A Tabular Transformer",
            "authors": "Doe, Smith",
            "abstract": "We propose…",
            "length": 24_000,
            "sections": ["Introduction", "Methods", "Results"],
        },
        "reviews": {"methodology": ar.model_dump()},
        "errors": {},
        "coordinator": ac.model_dump(),
        "author_editor_summary": asum.model_dump(),
        "editor_decision": ae.model_dump(),
        "timestamp": "2026-05-19T12:30:00",
        "config": {
            "models_used": {
                "powerful": "claude-opus-4-7",
                "standard": "gpt-5.4-mini",
                "basic": "gemini-3.1-flash-lite",
            },
            "num_reviewers": 1,
            "providers": ["anthropic", "openai", "google"],
            "routing": {
                "methodology": {
                    "provider": "anthropic", "model": "claude-opus-4-7",
                    "thinking_budget": "auto", "schema": "Review",
                }
            },
        },
    }


@pytest.fixture
def results(
    canned_review, canned_coordinator, canned_editor, canned_summary
) -> dict[str, Any]:
    return _build_results(canned_review, canned_coordinator, canned_editor, canned_summary)


def test_render_markdown_contains_structured_blocks(results: dict[str, Any]) -> None:
    md = render_markdown(results)
    assert "# Peer Review Report" in md
    assert "Auto-simulation report" in md          # disclaimer banner
    assert "## Editorial Decision" in md
    assert "## Coordinator Assessment" in md
    assert "Author & Editor Recommendations" in md  # new heading
    assert "Comments to the Author" in md
    assert "Confidential Comments to the Editor" in md
    assert "Methodology Expert Review" in md
    assert "MAJOR" in md  # severity badge for the major concern
    assert "Accept with minor revisions" in md


def test_render_markdown_renders_routing_table(results: dict[str, Any]) -> None:
    md = render_markdown(results)
    assert "## Routing per Agent" in md
    assert "| methodology" in md.lower() or "| `methodology`" in md
    assert "claude-opus-4-7" in md


def test_render_html_includes_severity_widgets(results: dict[str, Any]) -> None:
    html = render_html(results)
    assert "<!DOCTYPE html>" in html
    assert "Fatal concerns" in html and "Major concerns" in html and "Minor concerns" in html
    # severity-coloured badge for our major concern
    assert "bg-yellow-100" in html
    # editor decision shows the human label
    assert "Accept with minor revisions" in html


def test_render_executive_summary_includes_decision(results: dict[str, Any]) -> None:
    es = render_executive_summary(results)
    assert "Executive Summary" in es
    assert "Accept with minor revisions" in es


def test_export_json_writes_valid_round_trip(tmp_path: Path, results: dict[str, Any]) -> None:
    out = tmp_path / "results.json"
    assert export_json(results, out) is True
    loaded = json.loads(out.read_text())
    assert loaded["timestamp"] == "2026-05-19T12:30:00"
    assert loaded["config"]["num_reviewers"] == 1
    assert loaded["editor_decision"]["decision"] == "minor_rev"


def test_render_markdown_handles_empty_reviews() -> None:
    minimal = {
        "paper_info": {"title": "X", "authors": "Y", "abstract": "z", "length": 1, "sections": []},
        "reviews": {},
        "errors": {"methodology": "RuntimeError: stub failed"},
        "coordinator": None,
        "author_editor_summary": None,
        "editor_decision": None,
        "timestamp": "now",
        "config": {"models_used": {"powerful": "a", "standard": "b", "basic": "c"},
                   "num_reviewers": 0, "providers": [], "routing": {}},
    }
    md = render_markdown(minimal)
    assert "# Peer Review Report" in md
    assert "## Errors" in md and "stub failed" in md
