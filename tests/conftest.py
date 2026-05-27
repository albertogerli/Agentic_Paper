"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agentic_paper.providers.stub_provider import StubProvider
from agentic_paper.schemas import (
    AuthorEditorSummary,
    ConcernItem,
    CoordinatorAssessment,
    EditorDecision,
    Review,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


_PDF_PATH = _PROJECT_ROOT / "data" / "fixtures" / "sample_paper.pdf"


# --------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def project_root() -> Path:
    return _PROJECT_ROOT


@pytest.fixture(scope="session")
def sample_paper_pdf() -> Path:
    """Return the path to the committed 3-page test PDF; regenerate if missing."""
    if not _PDF_PATH.exists():
        from tests._make_pdf import build_pdf
        build_pdf(_PDF_PATH)
    return _PDF_PATH


@pytest.fixture(scope="session")
def sample_paper_text(sample_paper_pdf: Path) -> str:
    """Plain-text extraction of the test PDF, cached for the session."""
    from agentic_paper.paper import FileManager
    return FileManager(str(_PROJECT_ROOT / "data" / "_tmp_output")).extract_text_from_pdf(
        str(sample_paper_pdf)
    )


@pytest.fixture
def canned_review() -> Review:
    return Review(
        summary="The fixture paper is well-structured and lightly written.",
        strengths=["Clear section headings", "Reproducible (it is a fixture)"],
        concerns=[
            ConcernItem(
                severity="major",
                section="3. Results",
                page=2,
                issue="p=0.04 is reported without correction for multiple comparisons.",
                suggested_fix="Apply Bonferroni or report q-values.",
            ),
            ConcernItem(
                severity="minor",
                section="2. Methods",
                page=2,
                issue="Sample size of N=3 is small.",
                suggested_fix=None,
            ),
        ],
        recommendation="minor_rev",
        confidence=0.7,
    )


@pytest.fixture
def canned_coordinator(canned_review: Review) -> CoordinatorAssessment:
    return CoordinatorAssessment(
        executive_summary="Reviewers agree the paper is a valid fixture with minor methodological concerns.",
        consensus_strengths=["Clear structure", "Reproducibility"],
        consensus_concerns=[c.model_copy() for c in canned_review.concerns[:1]],
        disagreements=["Severity of the small-N issue"],
        methodology_score=0.55,
        novelty_score=0.30,
        overall_score=0.50,
        revision_priorities=["Address multiple-comparisons correction", "Discuss N=3 limitation"],
        final_recommendation="minor_rev",
    )


@pytest.fixture
def canned_editor() -> EditorDecision:
    return EditorDecision(
        decision="minor_rev",
        decision_label="Accept with minor revisions",
        rationale="Solid as a fixture; small statistical issues to clean up before publication.",
        author_guidance=["Add multiple-comparisons correction", "Expand discussion of N=3"],
        fits_audience=True,
        confidence=0.85,
    )


@pytest.fixture
def canned_summary() -> AuthorEditorSummary:
    return AuthorEditorSummary(
        recommendation_for_author="Recommend minor revisions; tighten the statistical reporting.",
        recommendation_for_editor_only="No ethical concerns; conditional accept after the listed fixes.",
    )


@pytest.fixture
def stub_provider(
    canned_review: Review,
    canned_coordinator: CoordinatorAssessment,
    canned_editor: EditorDecision,
    canned_summary: AuthorEditorSummary,
) -> StubProvider:
    """Stub provider pre-loaded with canned responses for every schema."""
    p = StubProvider()
    p.set_response(Review, canned_review)
    p.set_response(CoordinatorAssessment, canned_coordinator)
    p.set_response(EditorDecision, canned_editor)
    p.set_response(AuthorEditorSummary, canned_summary)
    return p
