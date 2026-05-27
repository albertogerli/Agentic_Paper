"""Citation extractor + OpenAlex client + Citation_Validator agent."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agentic_paper.external.citations import (
    CitationCheck,
    CitationRef,
    CitationReport,
    extract_citations,
    extract_references_section,
    format_report_for_agent,
)
from agentic_paper.external.openalex import OpenAlexWork


# --------------------------------------------------------------------- extract


SAMPLE_REFS = """
1. Introduction

Some text about the paper. We cite the seminal work of Smith et al. (2020) [1]
and a related study [2] (doi: 10.1234/foo.bar.5678). Other papers use the
methodology of Jones (2021) [3].

References
[1] Smith, J. and Doe, A. (2020). "A landmark study on stub-driven development."
    Journal of Tests, 12(3), 100-115.
[2] Brown, B. (2022). Generative paper review with LLMs.
    Proceedings of CI/CD 2022, 42-47. doi:10.1234/foo.bar.5678
[3] Jones, C. (2021). "Reviewers as composable agents." Nature AI, 5, 88-99.
    https://doi.org/10.5555/jones.2021.001
"""


def test_extract_references_section_finds_header() -> None:
    refs = extract_references_section(SAMPLE_REFS)
    assert refs, "should find a References section"
    assert "Smith, J." in refs


def test_extract_citations_finds_dois_in_body_and_refs() -> None:
    cites = extract_citations(SAMPLE_REFS)
    dois = {c.doi for c in cites if c.doi}
    # Body DOI (10.1234/foo.bar.5678) + refs DOI (10.5555/jones.2021.001) — both should appear
    assert "10.1234/foo.bar.5678" in dois
    assert "10.5555/jones.2021.001" in dois


def test_extract_citations_pulls_title_from_quoted_entries() -> None:
    cites = extract_citations(SAMPLE_REFS)
    titles = {c.inferred_title for c in cites if c.inferred_title}
    # The Smith entry has a quoted title in the references list (no DOI for it).
    assert any("stub-driven development" in (t or "").lower() for t in titles)


def test_extract_citations_handles_text_with_no_references() -> None:
    cites = extract_citations("Just some prose. No references anywhere.")
    assert cites == []


def test_extract_citations_caps_at_max_refs() -> None:
    body = "\n".join(f"[{i}] Author (2020). doi:10.1/{i}" for i in range(1, 100))
    cites = extract_citations("References\n" + body, max_refs=20)
    assert len(cites) == 20


# --------------------------------------------------------------------- format


def _mk_check(*, doi=None, title=None, match=None, fab=False, sim=1.0, note="") -> CitationCheck:
    return CitationCheck(
        raw=doi or title or "?", doi=doi, inferred_title=title,
        match=match, similarity=sim, looks_fabricated=fab, note=note,
    )


def test_format_report_empty_is_explicit() -> None:
    out = format_report_for_agent(CitationReport(0, 0, 0, 0, 0))
    assert "No citations" in out


def test_format_report_renders_per_citation_markers() -> None:
    work = OpenAlexWork(
        id="https://openalex.org/W1", doi="10.1/a", title="A Real Paper",
        year=2024, authors=["Real Author"], cited_by_count=42, venue="J. Tests",
    )
    report = CitationReport(
        total=2, by_doi=1, by_title=0, not_found=1, likely_fabricated=1,
        checks=[
            _mk_check(doi="10.1/a", match=work, note="DOI resolved"),
            _mk_check(doi="10.99/nope", match=None, fab=True,
                      note="DOI does not resolve in OpenAlex"),
        ],
    )
    out = format_report_for_agent(report)
    assert "Total citations parsed: 2" in out
    assert "✅" in out and "❌" in out
    assert "A Real Paper" in out
    assert "10.99/nope" in out


# --------------------------------------------------------------------- agent


def test_citation_validator_module_has_expected_metadata() -> None:
    from agentic_paper.agents import citation_validator as cv
    assert cv.KEY == "citation_validator"
    assert cv.NAME == "Citation_Validator"
    assert 0.5 <= cv.BASE_COMPLEXITY <= 1.0
    assert "OpenAlex" in cv.INSTRUCTIONS


def test_citation_validator_runs_via_stub_provider(stub_provider) -> None:
    """Confirm the agent integrates with the stock Agent/StubProvider plumbing."""
    from agentic_paper.agents import citation_validator as cv
    from agentic_paper.agents.base import Agent
    from agentic_paper.schemas import Review

    agent = Agent(
        name=cv.NAME, instructions=cv.INSTRUCTIONS, model="stub-model",
        provider=stub_provider, schema=cv.SCHEMA, max_output_tokens=512,
    )
    result = agent.run("dummy paper with =====CITATION VALIDATION REPORT===== attached")
    assert isinstance(result, Review)


# --------------------------------------------------------------------- client


@pytest.mark.asyncio
async def test_openalex_client_get_by_doi_uses_http_mock(monkeypatch) -> None:
    """OpenAlexClient should hit /works/doi:<X> and parse the response."""
    import httpx

    from agentic_paper.external.openalex import OpenAlexClient

    captured: dict = {}

    class _MockResp:
        def __init__(self, status, payload): self.status_code, self._p = status, payload
        def json(self): return self._p

    async def _mock_get(self, url, **kw):
        captured["url"] = url
        captured["kw"] = kw
        return _MockResp(200, {
            "id": "https://openalex.org/W42",
            "doi": "https://doi.org/10.1/test",
            "title": "Mocked paper title",
            "publication_year": 2024,
            "authorships": [{"author": {"display_name": "X. Y."}}],
            "cited_by_count": 7,
            "primary_location": {"source": {"display_name": "J. Mock"}},
            "open_access": {"is_oa": True},
        })

    monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)

    async with OpenAlexClient() as client:
        work = await client.get_by_doi("10.1/test")

    assert work is not None
    assert work.title == "Mocked paper title"
    assert work.year == 2024
    assert work.authors == ["X. Y."]
    assert captured["url"].endswith("/works/doi:10.1/test")
