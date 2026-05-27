"""Paper-version diff utilities."""

from __future__ import annotations

from agentic_paper.diff_utils import format_diff_for_agent, paper_diff


V1 = """
1. Introduction
This paper studies stub-driven development. We argue X.

2. Methods
We used N = 12 students.

3. Results
p = 0.04 indicates significance.

4. Discussion
Stub-driven development matters.

5. References
[1] Old, A. (2020). Old paper.
"""

V2 = """
1. Introduction
This paper studies stub-driven development. We argue X and Y, and add a
discussion of competing approaches.

2. Methods
We used N = 30 students (up from 12 in our earlier draft to address
reviewer concerns).

3. Results
p = 0.04 indicates significance.

4. Discussion
Stub-driven development matters in this expanded analysis.

5. References
[1] Old, A. (2020). Old paper.
[2] New, B. (2023). A relevant new study.
"""


def test_diff_identifies_modified_methods_section() -> None:
    d = paper_diff(V1, V2)
    assert "Methods" in d.modified_sections
    # Results section is bit-identical → unchanged
    assert "Results" in d.unchanged_sections


def test_diff_overall_similarity_is_high_but_not_one() -> None:
    d = paper_diff(V1, V2)
    assert 0.5 < d.overall_similarity < 1.0


def test_diff_handles_added_section() -> None:
    v1 = "Introduction\nA\n\nMethods\nB"
    v2 = "Introduction\nA\n\nMethods\nB\n\nLimitations\nWe acknowledge X."
    d = paper_diff(v1, v2)
    assert "Limitations" in d.added_sections


def test_diff_handles_removed_section() -> None:
    v1 = "Introduction\nA\n\nMethods\nB\n\nLimitations\nWe acknowledge X."
    v2 = "Introduction\nA\n\nMethods\nB"
    d = paper_diff(v1, v2)
    assert "Limitations" in d.removed_sections


def test_diff_handles_identical_papers() -> None:
    d = paper_diff(V1, V1)
    assert d.overall_similarity == 1.0
    assert d.modified_sections == []
    assert d.added_sections == [] and d.removed_sections == []


def test_diff_handles_empty_inputs() -> None:
    d = paper_diff("", "")
    assert d.sections == []
    out = format_diff_for_agent(d)
    assert "No content" in out


def test_format_diff_for_agent_shows_unified() -> None:
    d = paper_diff(V1, V2)
    out = format_diff_for_agent(d)
    assert "Comparing v1" in out
    assert "Methods" in out
    # Unified diff hunk markers
    assert "+++" in out or "+ " in out


def test_revision_assessor_module_has_expected_metadata() -> None:
    from agentic_paper.agents import revision_assessor as ra
    assert ra.KEY == "revision_assessor"
    assert ra.NAME == "Revision_Assessor"
    assert "v1" in ra.INSTRUCTIONS and "v2" in ra.INSTRUCTIONS


def test_revision_assessor_runs_via_stub(stub_provider) -> None:
    """Even without a diff in the message, the agent must still return a valid Review
    (its instructions tell it to gracefully degrade)."""
    from agentic_paper.agents import revision_assessor as ra
    from agentic_paper.agents.base import Agent
    from agentic_paper.schemas import Review
    agent = Agent(
        name=ra.NAME, instructions=ra.INSTRUCTIONS, model="stub-model",
        provider=stub_provider, schema=ra.SCHEMA, max_output_tokens=512,
    )
    result = agent.run("paper text only, no diff attached")
    assert isinstance(result, Review)
