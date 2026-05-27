"""Structured-output schemas shared by every reviewer agent.

Two layers per role:
    * ``X``          — the shape the LLM is asked to produce.
    * ``AnnotatedX`` — ``X`` plus ``agent`` + ``model_used`` metadata, set
      by the orchestrator (the model is not asked to guess its own name).

The ``Annotated*`` classes are what gets persisted under ``results["reviews"]``
and what report renderers consume.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["minor", "major", "fatal"]
Recommendation = Literal["accept", "minor_rev", "major_rev", "reject"]


# ---------------------------------------------------------------------------
# Reviewer schema
# ---------------------------------------------------------------------------
class ConcernItem(BaseModel):
    """One specific issue raised by a reviewer."""

    model_config = ConfigDict(extra="forbid")

    severity: Severity
    section: str
    page: Optional[int]
    issue: str
    suggested_fix: Optional[str]


class Review(BaseModel):
    """Reviewer agent LLM output."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="2-3 sentences capturing the key takeaway.")
    strengths: list[str]
    concerns: list[ConcernItem]
    recommendation: Recommendation
    confidence: float = Field(ge=0.0, le=1.0)


class AnnotatedReview(Review):
    """:class:`Review` + provenance for storage / rendering."""

    agent: str
    model_used: str


# ---------------------------------------------------------------------------
# Coordinator schema
# ---------------------------------------------------------------------------
class CoordinatorAssessment(BaseModel):
    """Coordinator LLM output."""

    model_config = ConfigDict(extra="forbid")

    executive_summary: str
    consensus_strengths: list[str]
    consensus_concerns: list[ConcernItem]
    disagreements: list[str]
    methodology_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    novelty_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    revision_priorities: list[str]
    final_recommendation: Recommendation


class AnnotatedCoordinatorAssessment(CoordinatorAssessment):
    agent: str
    model_used: str


# ---------------------------------------------------------------------------
# Editor schema
# ---------------------------------------------------------------------------
class EditorDecision(BaseModel):
    """Editor LLM output."""

    model_config = ConfigDict(extra="forbid")

    decision: Recommendation
    decision_label: str = Field(
        description="Human-readable label: 'Accept as is' | 'Accept with minor revisions' | "
        "'Revise and resubmit (major revisions)' | 'Reject'."
    )
    rationale: str
    author_guidance: list[str]
    fits_audience: bool
    confidence: float = Field(ge=0.0, le=1.0)


class AnnotatedEditorDecision(EditorDecision):
    agent: str
    model_used: str


# ---------------------------------------------------------------------------
# Author/Editor summary schema
# ---------------------------------------------------------------------------
class AuthorEditorSummary(BaseModel):
    """Two-section peer-review recommendation produced by the summary agent.

    The two fields mirror the standard journal review form:
        * ``recommendation_for_author`` — visible to the author. Concrete,
          actionable feedback the author should read first.
        * ``recommendation_for_editor_only`` — confidential. Editorial
          guidance the editor sees but the author does not.
    """

    model_config = ConfigDict(extra="forbid")

    recommendation_for_author: str
    recommendation_for_editor_only: str


class AnnotatedAuthorEditorSummary(AuthorEditorSummary):
    agent: str
    model_used: str


# ---------------------------------------------------------------------------
# Per-role appended output instructions
# ---------------------------------------------------------------------------
REVIEWER_OUTPUT_INSTRUCTION = """

YOUR RESPONSE MUST BE A SINGLE STRUCTURED OBJECT matching the provided schema:
  - summary (str): 2-3 sentences with the key takeaway.
  - strengths (list[str]): what the paper does well.
  - concerns (list[object]): each item must have
      severity ('minor' | 'major' | 'fatal'),
      section (str),
      page (int or null),
      issue (str),
      suggested_fix (str or null).
  - recommendation ('accept' | 'minor_rev' | 'major_rev' | 'reject').
  - confidence (float between 0.0 and 1.0).

Do not emit any prose outside the structured object. Do not wrap it in
markdown code fences. Do not echo the prior instructions.
"""

COORDINATOR_OUTPUT_INSTRUCTION = """

YOUR RESPONSE MUST BE A SINGLE STRUCTURED OBJECT matching the provided schema:
  - executive_summary (str): 3-5 sentences synthesising all reviews.
  - consensus_strengths (list[str]): strengths multiple reviewers agreed on.
  - consensus_concerns (list[object]): same shape as reviewer concerns
      (severity / section / page / issue / suggested_fix).
  - disagreements (list[str]): where reviewers explicitly disagreed.
  - methodology_score (float 0..1 or null).
  - novelty_score (float 0..1 or null).
  - overall_score (float 0..1): your synthesised judgement.
  - revision_priorities (list[str]): ordered fix-this-first list.
  - final_recommendation ('accept' | 'minor_rev' | 'major_rev' | 'reject').
"""

EDITOR_OUTPUT_INSTRUCTION = """

YOUR RESPONSE MUST BE A SINGLE STRUCTURED OBJECT matching the provided schema:
  - decision ('accept' | 'minor_rev' | 'major_rev' | 'reject').
  - decision_label (str): 'Accept as is' | 'Accept with minor revisions' |
    'Revise and resubmit (major revisions)' | 'Reject'.
  - rationale (str): 2-4 sentences justifying the decision.
  - author_guidance (list[str]): concrete steps the authors should take.
  - fits_audience (bool): whether the paper fits the journal's audience.
  - confidence (float 0..1).
"""

SUMMARY_OUTPUT_INSTRUCTION = """

YOUR RESPONSE MUST BE A SINGLE STRUCTURED OBJECT matching the provided schema.
Frame each field exactly as it would appear on a journal review form.

  - recommendation_for_author (str): "Comments to the Author".
    The author will see this verbatim. Write actionable, constructive,
    professional feedback. Lead with the high-level decision direction
    (e.g. "We recommend major revisions before this manuscript can be
    considered for publication"), then enumerate specific revision asks
    (cite section/page where possible). Cover: framing, methods,
    statistics, claims vs evidence, clarity, missing literature.
    Do NOT include anything you wouldn't say to the author's face.

  - recommendation_for_editor_only (str): "Confidential Comments to the Editor".
    The author does NOT see this. Be candid. Cover items the editor
    should weigh privately: doubts about novelty, fit with the journal,
    ethical or COI flags, suspected plagiarism / AI-origin signals,
    whether to invite revisions vs reject outright, suggested second
    reviewers if relevant, your confidence in the recommendation, and
    anything off-the-record that would shape the editorial decision.
"""


__all__ = [
    "Severity",
    "Recommendation",
    "ConcernItem",
    "Review",
    "AnnotatedReview",
    "CoordinatorAssessment",
    "AnnotatedCoordinatorAssessment",
    "EditorDecision",
    "AnnotatedEditorDecision",
    "AuthorEditorSummary",
    "AnnotatedAuthorEditorSummary",
    "REVIEWER_OUTPUT_INSTRUCTION",
    "COORDINATOR_OUTPUT_INSTRUCTION",
    "EDITOR_OUTPUT_INSTRUCTION",
    "SUMMARY_OUTPUT_INSTRUCTION",
]
