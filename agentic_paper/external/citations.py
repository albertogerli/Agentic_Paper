"""Extract citations from a paper and validate them against OpenAlex.

The extraction is deliberately heuristic — peer-review papers vary too much
to parse rigorously. The goal is to surface enough verifiable signal that the
``Citation_Validator`` agent can flag the obvious cases (fabricated DOIs,
non-existent titles, missing canonical works).
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable

from .openalex import OpenAlexClient, OpenAlexWork

logger = logging.getLogger(__name__)

# DOI regex per Crossref's recommendation, slightly tightened.
DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\)\]\"<>,;]+", re.IGNORECASE)

# Heuristic markers for the "References" section.
_REFS_HEADER_RE = re.compile(
    r"^\s*(References|Bibliography|Works Cited|Citations)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Reference-entry heuristics: either numbered ([1], 1., (1)) or hanging indent.
_REF_ENTRY_RE = re.compile(
    r"(?:^|\n)\s*(?:\[\d+\]|\(\d+\)|\d+[\.\)])\s+([^\n].+?)(?=(?:\n\s*(?:\[\d+\]|\(\d+\)|\d+[\.\)]))|\Z)",
    re.DOTALL,
)


@dataclass
class CitationRef:
    """A raw citation pulled out of the paper, before validation."""

    raw: str
    doi: str | None = None
    inferred_title: str | None = None


@dataclass
class CitationCheck:
    """Result of validating one CitationRef against OpenAlex."""

    raw: str
    doi: str | None
    inferred_title: str | None
    match: OpenAlexWork | None
    similarity: float = 0.0
    looks_fabricated: bool = False
    note: str = ""


@dataclass
class CitationReport:
    """Aggregate result of validating all extracted citations."""

    total: int
    by_doi: int
    by_title: int
    not_found: int
    likely_fabricated: int
    checks: list[CitationCheck] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- extraction


def extract_references_section(text: str) -> str:
    """Return the substring starting at the References header, or '' if missing."""
    m = _REFS_HEADER_RE.search(text)
    if not m:
        return ""
    return text[m.end():]


def _likely_title(entry: str) -> str | None:
    """Heuristic: in a typical reference entry the title is the longest run of
    letters between author list and venue/year. Pull the longest quoted phrase
    if present, else the longest mid-string capitalised run.
    """
    # Quoted titles
    q = re.search(r"[\"\u201c]([^\"\u201d]{8,200})[\"\u201d]", entry)
    if q:
        return q.group(1).strip().rstrip(".")
    # Strip leading authors (up to a year), strip leading initials & comma noise
    # then take the rest up to a period or "In:".
    after_year = re.split(r"\b(19|20)\d{2}\b", entry, maxsplit=1)
    body = after_year[-1] if len(after_year) > 1 else entry
    body = body.lstrip(" .,)").strip()
    # Title runs until period before next capital cluster (venue) or "In:"
    m = re.split(r"\.\s+(?:[A-Z][^\.]+\b(?:Journal|Proceedings|Conference|Press|Review|Communications|Nature|Science|IEEE|ACM|Lancet|JAMA|BMJ)\b)|\.\s+In:", body, maxsplit=1)
    candidate = m[0].strip().rstrip(".")
    candidate = re.sub(r"\s+", " ", candidate)
    return candidate if 8 <= len(candidate) <= 300 else None


def _normalise_doi(raw_doi: str) -> str:
    """Strip trailing punctuation + lowercase, matching the original behaviour."""
    return raw_doi.rstrip(".,;:)]\"'").lower()


def _extract_inline_dois(
    paper_text: str, refs: dict[str, CitationRef], max_refs: int,
) -> None:
    """Add DOI-only refs found anywhere in the paper body.

    Mutates ``refs`` in place. Stops once ``max_refs`` has been reached.
    """
    for doi in DOI_RE.findall(paper_text):
        norm = _normalise_doi(doi)
        if norm in refs:
            continue
        refs[norm] = CitationRef(raw=doi, doi=norm)
        if len(refs) >= max_refs:
            return  # equivalent to the original `break`


def _add_entry_ref(entry: str, refs: dict[str, CitationRef]) -> None:
    """Insert a single reference-section entry into ``refs``.

    DOI-bearing entries are keyed by the normalised DOI; title-only entries
    go in under a ``title::<lowercased title>`` synthetic key. ``setdefault``
    semantics — already-known keys are NOT overwritten.
    """
    entry_doi_match = DOI_RE.search(entry)
    if entry_doi_match:
        doi_norm = _normalise_doi(entry_doi_match.group(0))
        refs.setdefault(
            doi_norm,
            CitationRef(raw=entry, doi=doi_norm, inferred_title=_likely_title(entry)),
        )
        return
    title = _likely_title(entry)
    if not title:
        return
    key = "title::" + title.lower()
    refs.setdefault(key, CitationRef(raw=entry, inferred_title=title))


def _extract_section_entries(
    paper_text: str, refs: dict[str, CitationRef], max_refs: int,
) -> None:
    """Parse the References section into per-entry CitationRefs.

    No-op if the paper has no parseable References section. Mutates ``refs``
    in place; stops once ``max_refs`` has been reached.
    """
    refs_text = extract_references_section(paper_text)
    if not refs_text:
        return
    for entry_match in _REF_ENTRY_RE.finditer(refs_text):
        entry = entry_match.group(1).strip()
        if not entry:
            continue
        _add_entry_ref(entry, refs)
        if len(refs) >= max_refs:
            return  # equivalent to the original `break`


def extract_citations(paper_text: str, *, max_refs: int = 50) -> list[CitationRef]:
    """Return up to ``max_refs`` citations parsed from the paper text.

    Strategy:
      1. DOIs anywhere in the body get their own CitationRef.
      2. The References section is segmented into entries; for each entry
         without a DOI, we extract an inferred title.

    The list is deduplicated by DOI (case-insensitive) and capped at ``max_refs``.
    Order is preserved: inline DOIs come first (in body order), then
    reference-section entries (in document order).
    """
    refs: dict[str, CitationRef] = {}
    _extract_inline_dois(paper_text, refs, max_refs)
    _extract_section_entries(paper_text, refs, max_refs)
    return list(refs.values())[:max_refs]


# ---------------------------------------------------------------- validation


def _title_similarity(a: str, b: str) -> float:
    a_n = re.sub(r"\W+", " ", (a or "")).strip().lower()
    b_n = re.sub(r"\W+", " ", (b or "")).strip().lower()
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()


async def _check_one(client: OpenAlexClient, ref: CitationRef) -> CitationCheck:
    if ref.doi:
        work = await client.get_by_doi(ref.doi)
        if work:
            sim = _title_similarity(ref.inferred_title or "", work.title) if ref.inferred_title else 1.0
            return CitationCheck(
                raw=ref.raw, doi=ref.doi, inferred_title=ref.inferred_title,
                match=work, similarity=sim, looks_fabricated=False,
                note="DOI resolved",
            )
        return CitationCheck(
            raw=ref.raw, doi=ref.doi, inferred_title=ref.inferred_title,
            match=None, similarity=0.0, looks_fabricated=True,
            note="DOI does not resolve in OpenAlex (possible fabrication)",
        )
    # Title-only search
    title = ref.inferred_title or ""
    candidates = await client.search_title(title, per_page=3) if title else []
    if not candidates:
        return CitationCheck(
            raw=ref.raw, doi=None, inferred_title=title,
            match=None, similarity=0.0, looks_fabricated=bool(title),
            note="No OpenAlex hit for inferred title",
        )
    best = max(candidates, key=lambda w: _title_similarity(title, w.title))
    sim = _title_similarity(title, best.title)
    fab = sim < 0.55
    return CitationCheck(
        raw=ref.raw, doi=None, inferred_title=title,
        match=best, similarity=sim, looks_fabricated=fab,
        note=("Low-similarity match, possible fabrication" if fab
              else "Title-similarity match (no DOI in source)"),
    )


def _aggregate_stats(checks: list[CitationCheck]) -> dict[str, int]:
    """Walk ``checks`` once and accumulate the four CitationReport counters.

    Replaces four inline ``sum(1 for c in checks if ...)`` generators that
    each added a branch to ``validate_citations``' cyclomatic complexity.
    """
    stats = {"by_doi": 0, "by_title": 0, "not_found": 0, "likely_fabricated": 0}
    for c in checks:
        if c.match is None:
            stats["not_found"] += 1
        elif c.doi:
            stats["by_doi"] += 1
        else:
            stats["by_title"] += 1
        if c.looks_fabricated:
            stats["likely_fabricated"] += 1
    return stats


async def validate_citations(
    refs: Iterable[CitationRef],
    *,
    concurrency: int = 6,
    timeout: float = 8.0,
) -> CitationReport:
    """Run OpenAlex lookups for each ref. Concurrency capped at 6 to stay in the polite pool."""
    refs_list = list(refs)
    if not refs_list:
        return CitationReport(total=0, by_doi=0, by_title=0, not_found=0, likely_fabricated=0)

    sem = asyncio.Semaphore(concurrency)
    errors: list[str] = []

    async def _bounded(client: OpenAlexClient, ref: CitationRef) -> CitationCheck | None:
        async with sem:
            try:
                return await _check_one(client, ref)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{ref.raw[:60]}…: {type(e).__name__}: {e}")
                return None

    async with OpenAlexClient(timeout=timeout) as client:
        results = await asyncio.gather(*[_bounded(client, r) for r in refs_list])

    checks = [c for c in results if c is not None]
    stats = _aggregate_stats(checks)
    return CitationReport(
        total=len(refs_list),
        by_doi=stats["by_doi"],
        by_title=stats["by_title"],
        not_found=stats["not_found"],
        likely_fabricated=stats["likely_fabricated"],
        checks=checks,
        errors=errors,
    )


# ---------------------------------------------------------------- formatting


def _check_marker(c: CitationCheck) -> str:
    """One of ❌ / ✅ / ❓ depending on the validation outcome.

    Extracted so the nested ternary in the previous body becomes an
    early-return cascade with a single responsibility.
    """
    if c.looks_fabricated:
        return "❌"
    if c.match is not None:
        return "✅"
    return "❓"


def _check_identifier(c: CitationCheck) -> str:
    """Short identifier line used as the per-citation 'head': DOI > title > raw."""
    if c.doi:
        return f"doi:{c.doi}"
    if c.inferred_title:
        return f'title: "{c.inferred_title[:90]}"'
    return f"raw: {c.raw[:80]}"


def _format_match_line(c: CitationCheck) -> str:
    """The "→ OpenAlex: ..." line under a citation that resolved.

    Preconditions: ``c.match is not None``. Authors / year / venue all have
    sensible fallbacks for partial OpenAlex records.
    """
    m = c.match
    assert m is not None  # noqa: S101 — invariant from the caller
    yr = m.year or "?"
    venue = m.venue or "unknown venue"
    authors_first = m.authors[0] if m.authors else "?"
    n_more = max(0, len(m.authors) - 1)
    authors_str = authors_first + (f" et al. (+{n_more})" if n_more else "")
    return (
        f"        → OpenAlex: \"{m.title[:120]}\""
        f" — {authors_str}, {yr}, {venue}"
        f" (sim={c.similarity:.2f})"
    )


def _format_single_check(i: int, c: CitationCheck) -> list[str]:
    """Return the formatted lines for a single CitationCheck (head + detail).

    Always 2 lines: the head (`[N] marker doi:... | title:... | raw:...`) plus
    either the OpenAlex match details (if resolved) or the note from the
    validation step (if not). Caller appends to the running ``lines`` list
    with ``lines.extend(...)``.
    """
    head = f"  [{i}] {_check_marker(c)} {_check_identifier(c)}"
    if c.match is None:
        return [head, f"        → {c.note}"]
    return [head, _format_match_line(c)]


def _format_report_header(report: CitationReport) -> list[str]:
    """The top block of the agent-facing report: totals + lookup-error count."""
    lines = [
        "Extracted and queried against OpenAlex (https://openalex.org, free, no API key).",
        f"Total citations parsed: {report.total}",
        f"  DOI resolved:           {report.by_doi}",
        f"  Title-similarity match: {report.by_title}",
        f"  Not found:              {report.not_found}",
        f"  Likely fabricated:      {report.likely_fabricated}",
    ]
    if report.errors:
        lines.append(f"  Lookup errors:          {len(report.errors)} (transient HTTP)")
    lines.append("")
    return lines


def _format_report_errors(errors: list[str]) -> list[str]:
    """The bottom block: at most 5 transient HTTP errors that prevented lookup."""
    if not errors:
        return []
    lines = ["", "Lookup errors (citations the validator could not check):"]
    for e in errors[:5]:
        lines.append(f"  - {e}")
    return lines


def format_report_for_agent(report: CitationReport) -> str:
    """Format a CitationReport as plain text suitable for an LLM prompt."""
    if report.total == 0:
        return "No citations could be extracted from the paper (no DOIs, no parseable References section)."

    lines = _format_report_header(report)
    lines.append("Per-citation results:")
    for i, c in enumerate(report.checks, 1):
        lines.extend(_format_single_check(i, c))
    lines.extend(_format_report_errors(report.errors))
    return "\n".join(lines)
