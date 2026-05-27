"""Lightweight paper-version diff for revision-aware review.

Given two extracted-text bodies (v1 = older, v2 = newer), this module emits:
    * Per-section diffs (paragraph-level) for sections that changed.
    * Sections that appear only in v1 (deleted) or only in v2 (added).
    * A short summary of how much actually moved.

No external deps — just stdlib's ``difflib``. Heuristic-grade by design; the
LLM agent uses the output as context, not as ground truth.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher, unified_diff
from typing import Iterable, Literal

logger = logging.getLogger(__name__)

# Type alias: the four states a section can be in after diffing.
SectionState = Literal["added", "removed", "modified", "unchanged"]

# Same standard-sections list used by paper.py — kept inline to avoid an import
# cycle and to keep this module standalone.
_STANDARD_SECTIONS = (
    "Abstract", "Introduction", "Background", "Related Work", "Literature Review",
    "Methods", "Methodology", "Materials and Methods", "Experimental Setup",
    "Results", "Experiments", "Evaluation", "Findings",
    "Discussion", "Analysis", "Implications",
    "Conclusion", "Conclusions", "Future Work", "Limitations",
    "References", "Bibliography", "Acknowledgments", "Appendix",
)
_SECTION_HEADER_RE = re.compile(
    r"^\s*(?:\d+\.?\d*\.?\s+)?(" + "|".join(_STANDARD_SECTIONS) + r")\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class SectionDiff:
    name: str
    state: SectionState  # "added" | "removed" | "modified" | "unchanged"
    similarity: float    # 0..1, by SequenceMatcher.ratio() on the section text
    v1_len: int = 0
    v2_len: int = 0
    unified: str = ""    # truncated unified diff for the agent's eyes


@dataclass
class PaperDiff:
    v1_chars: int
    v2_chars: int
    overall_similarity: float
    sections: list[SectionDiff] = field(default_factory=list)
    added_sections: list[str] = field(default_factory=list)
    removed_sections: list[str] = field(default_factory=list)
    modified_sections: list[str] = field(default_factory=list)
    unchanged_sections: list[str] = field(default_factory=list)


# --------------------------------------------------------- helpers


def _split_sections(text: str) -> dict[str, str]:
    """Heuristically split text into a {section_name: body} map.

    Section names are normalised to title case; sections we cannot
    identify go under the synthetic key ``__preamble__``.
    """
    if not text:
        return {}
    sections: dict[str, str] = {}
    last_name = "__preamble__"
    last_start = 0
    matches = list(_SECTION_HEADER_RE.finditer(text))
    if not matches:
        return {last_name: text}

    for m in matches:
        sections.setdefault(last_name, "")
        sections[last_name] += text[last_start:m.start()]
        last_name = m.group(1).title()
        last_start = m.end()
    sections.setdefault(last_name, "")
    sections[last_name] += text[last_start:]
    # Trim whitespace
    return {k: v.strip() for k, v in sections.items() if v.strip()}


def _short_unified(a: str, b: str, *, name: str, max_lines: int = 60) -> str:
    """Compact unified diff suitable for an LLM prompt."""
    diff = list(unified_diff(
        a.splitlines(), b.splitlines(),
        fromfile=f"v1/{name}", tofile=f"v2/{name}",
        n=2, lineterm="",
    ))
    if len(diff) > max_lines:
        head = diff[:max_lines // 2]
        tail = diff[-max_lines // 2:]
        diff = head + [f"... ({len(diff) - max_lines} lines elided) ..."] + tail
    return "\n".join(diff)


# --------------------------------------------------------- section classifiers
#
# Each helper builds one SectionDiff for a specific "shape" of section pair.
# Keeping them tiny + named lets ``_classify_section`` read as a top-down
# dispatch cascade instead of a deep nested-if hairball.


def _section_removed(name: str, a: str) -> SectionDiff:
    """Section is present in v1 but absent in v2."""
    return SectionDiff(
        name=name, state="removed",
        similarity=0.0, v1_len=len(a), v2_len=0,
    )


def _section_added(name: str, b: str) -> SectionDiff:
    """Section is present in v2 but absent in v1."""
    return SectionDiff(
        name=name, state="added",
        similarity=0.0, v1_len=0, v2_len=len(b),
    )


def _section_identical(name: str, a: str) -> SectionDiff:
    """Bit-identical content in v1 and v2 — no diff needed."""
    return SectionDiff(
        name=name, state="unchanged",
        similarity=1.0, v1_len=len(a), v2_len=len(a),
    )


def _section_modified_or_close(
    name: str, a: str, b: str, modified_threshold: float
) -> SectionDiff:
    """Both v1 and v2 contain the section but text differs.

    Above ``modified_threshold`` similarity we still call it 'unchanged' (the
    diff is cosmetic — typos, whitespace, citation reformatting). Below the
    threshold it's a true 'modified' and we attach a compact unified diff.
    """
    sim = SequenceMatcher(None, a, b).ratio()
    if sim >= modified_threshold:
        return SectionDiff(
            name=name, state="unchanged",
            similarity=sim, v1_len=len(a), v2_len=len(b),
        )
    return SectionDiff(
        name=name, state="modified", similarity=sim,
        v1_len=len(a), v2_len=len(b),
        unified=_short_unified(a, b, name=name),
    )


def _classify_section(
    name: str, a: str, b: str, *, modified_threshold: float,
) -> SectionDiff:
    """Pick the right SectionDiff builder for a single (a, b) section pair.

    Early-return cascade — at most one of the four branches matches; the rest
    are skipped. This is what brought ``paper_diff``'s cyclomatic complexity
    down from 23 to 3.
    """
    if a and not b:
        return _section_removed(name, a)
    if b and not a:
        return _section_added(name, b)
    if a == b:
        return _section_identical(name, a)
    return _section_modified_or_close(name, a, b, modified_threshold)


def _section_names_in_state(
    sections: Iterable[SectionDiff], state: SectionState,
) -> list[str]:
    """Return the names of every section currently in ``state``, in order.

    Pulled out as a named helper so the four list-comprehensions in
    ``paper_diff`` collapse into a single typed call site.
    """
    return [s.name for s in sections if s.state == state]


# --------------------------------------------------------- public


def paper_diff(
    v1_text: str, v2_text: str, *, modified_threshold: float = 0.97,
) -> PaperDiff:
    """Compare two papers section-by-section. ``modified_threshold`` is the
    minimum similarity below which a section is treated as 'modified'."""
    v1 = _split_sections(v1_text or "")
    v2 = _split_sections(v2_text or "")
    sections: list[SectionDiff] = [
        _classify_section(
            name, v1.get(name, ""), v2.get(name, ""),
            modified_threshold=modified_threshold,
        )
        for name in sorted(set(v1) | set(v2))
    ]

    overall = SequenceMatcher(None, v1_text or "", v2_text or "").ratio()
    return PaperDiff(
        v1_chars=len(v1_text or ""),
        v2_chars=len(v2_text or ""),
        overall_similarity=overall,
        sections=sections,
        added_sections=_section_names_in_state(sections, "added"),
        removed_sections=_section_names_in_state(sections, "removed"),
        modified_sections=_section_names_in_state(sections, "modified"),
        unchanged_sections=_section_names_in_state(sections, "unchanged"),
    )


def format_diff_for_agent(d: PaperDiff, *, max_modified: int = 6) -> str:
    """Render a PaperDiff as plain text for prepending to a reviewer prompt."""
    if not d.sections:
        return "No content available to diff (one or both versions empty)."

    pct = int(round(d.overall_similarity * 100))
    lines = [
        f"Comparing v1 ({d.v1_chars:,} chars) vs v2 ({d.v2_chars:,} chars).",
        f"Overall text similarity: {pct}% (1.00 = identical).",
        f"  Added sections:     {', '.join(d.added_sections) or '(none)'}",
        f"  Removed sections:   {', '.join(d.removed_sections) or '(none)'}",
        f"  Modified sections:  {', '.join(d.modified_sections) or '(none)'}",
        f"  Unchanged sections: {len(d.unchanged_sections)}",
        "",
    ]

    modified = [s for s in d.sections if s.state == "modified"]
    modified.sort(key=lambda s: s.similarity)  # most-changed first
    if not modified:
        lines.append("No sections were modified within the similarity threshold.")
        return "\n".join(lines)

    lines.append(f"Per-section diffs (showing the {min(len(modified), max_modified)} "
                 "most-changed sections):")
    for s in modified[:max_modified]:
        lines.append("")
        lines.append(f"=== Section: {s.name} (similarity {s.similarity:.2f}; "
                     f"v1={s.v1_len:,} chars, v2={s.v2_len:,} chars) ===")
        lines.append(s.unified or "(diff body elided)")

    if len(modified) > max_modified:
        lines.append("")
        lines.append(f"… {len(modified) - max_modified} additional modified section(s) elided "
                     "for token budget; full diff is available in the run artefacts.")
    return "\n".join(lines)


__all__ = [
    "PaperDiff",
    "SectionDiff",
    "SectionState",
    "paper_diff",
    "format_diff_for_agent",
]
