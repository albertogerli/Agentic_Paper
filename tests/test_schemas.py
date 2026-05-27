"""Validate that the fixture JSON parses cleanly against the structured-output schemas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_paper.schemas import (
    AnnotatedReview,
    ConcernItem,
    Recommendation,
    Review,
    Severity,
)


FIXTURE = Path(__file__).parent / "fixtures" / "canned_review.json"


def _load_fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def test_review_validates_canned_fixture() -> None:
    review = Review.model_validate(_load_fixture())
    assert review.recommendation == "major_rev"
    assert 0.0 <= review.confidence <= 1.0
    assert len(review.concerns) == 3
    assert {c.severity for c in review.concerns} == {"fatal", "major", "minor"}


def test_concern_severity_is_constrained() -> None:
    data = _load_fixture()
    data["concerns"][0]["severity"] = "catastrophic"  # not a valid Severity literal
    with pytest.raises(ValidationError):
        Review.model_validate(data)


def test_recommendation_is_constrained() -> None:
    data = _load_fixture()
    data["recommendation"] = "publish"  # not a valid Recommendation literal
    with pytest.raises(ValidationError):
        Review.model_validate(data)


def test_confidence_in_range() -> None:
    data = _load_fixture()
    data["confidence"] = 1.4
    with pytest.raises(ValidationError):
        Review.model_validate(data)


def test_page_and_suggested_fix_may_be_null() -> None:
    data = _load_fixture()
    # The 'minor' concern already has page: null. Check it round-trips.
    review = Review.model_validate(data)
    minor = [c for c in review.concerns if c.severity == "minor"][0]
    assert minor.page is None


def test_extra_fields_forbidden() -> None:
    data = _load_fixture()
    data["mystery_field"] = "should be rejected"
    with pytest.raises(ValidationError):
        Review.model_validate(data)


def test_author_editor_summary_uses_recommendation_field_names() -> None:
    """Regression: the summary agent's output must carry recommendation_for_author
    and recommendation_for_editor_only (renamed from review_for_*)."""
    from agentic_paper.schemas import AuthorEditorSummary

    s = AuthorEditorSummary(
        recommendation_for_author="public-facing comments",
        recommendation_for_editor_only="confidential comments",
    )
    blob = s.model_dump()
    assert set(blob) == {"recommendation_for_author", "recommendation_for_editor_only"}

    # Old field names must no longer be accepted (extra='forbid' enforces this).
    with pytest.raises(ValidationError):
        AuthorEditorSummary.model_validate({
            "review_for_author_and_editor": "x",
            "review_for_editor_only": "y",
        })


def test_annotated_review_wraps_metadata() -> None:
    review = Review.model_validate(_load_fixture())
    annotated = AnnotatedReview(
        agent="Methodology_Expert",
        model_used="claude-opus-4-7",
        **review.model_dump(),
    )
    dumped = annotated.model_dump()
    assert dumped["agent"] == "Methodology_Expert"
    assert dumped["model_used"] == "claude-opus-4-7"
    assert dumped["recommendation"] == "major_rev"
    assert len(dumped["concerns"]) == 3
