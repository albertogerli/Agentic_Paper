"""Markdown report + executive summary rendering from structured results."""

from __future__ import annotations

from typing import Any, Callable

SEVERITY_BADGE = {"fatal": "🔴 FATAL", "major": "🟡 MAJOR", "minor": "🔵 minor"}

RECOMMENDATION_LABEL = {
    "accept": "Accept",
    "minor_rev": "Minor revisions",
    "major_rev": "Major revisions",
    "reject": "Reject",
}

REVIEWER_TITLES = {
    "methodology": "Methodology Expert",
    "results": "Results Analyst",
    "literature": "Literature Expert",
    "structure": "Structure & Clarity Reviewer",
    "impact": "Impact & Innovation Analyst",
    "contradiction": "Contradiction Checker",
    "ethics": "Ethics & Integrity Reviewer",
    "ai_origin": "AI Origin Detector",
    "hallucination": "Hallucination Detector",
    "citation_validator": "Citation Validator (OpenAlex)",
    "statcheck_validator": "Statistical Sanity (statcheck R)",
    "revision_assessor": "Revision Assessor (v1 → v2 diff)",
}


def _render_concern(c: dict[str, Any]) -> str:
    severity = c.get("severity", "minor")
    badge = SEVERITY_BADGE.get(severity, severity)
    page = f"p.{c['page']}" if c.get("page") is not None else "—"
    line = f"- **{badge}** · {c.get('section', '?')} ({page}) — {c.get('issue', '')}"
    fix = c.get("suggested_fix")
    if fix:
        line += f"\n    - _Fix:_ {fix}"
    return line


def _render_review(key: str, r: dict[str, Any]) -> str:
    title = REVIEWER_TITLES.get(key, key.replace("_", " ").title())
    rec_label = RECOMMENDATION_LABEL.get(r.get("recommendation", ""), r.get("recommendation", "?"))
    confidence = r.get("confidence", 0.0)
    out = [
        f"### {title} Review",
        "",
        f"**Recommendation:** {rec_label} · **Confidence:** {confidence:.2f}",
        f"**Reviewer:** {r.get('agent', '?')} (`{r.get('model_used', '?')}`)",
        "",
        f"**Summary.** {r.get('summary', '')}",
    ]
    strengths = r.get("strengths", [])
    if strengths:
        out.append("\n**Strengths**")
        out.extend(f"- {s}" for s in strengths)
    concerns = r.get("concerns", [])
    if concerns:
        out.append("\n**Concerns**")
        out.extend(_render_concern(c) for c in concerns)
    else:
        out.append("\n_No specific concerns flagged._")
    return "\n".join(out)


def _render_coordinator(c: dict[str, Any]) -> str:
    out = [
        "## Coordinator Assessment",
        "",
        f"**Final recommendation:** {RECOMMENDATION_LABEL.get(c.get('final_recommendation', ''), c.get('final_recommendation', '?'))}",
        f"**Overall score:** {c.get('overall_score', 0.0):.2f}",
    ]
    if c.get("methodology_score") is not None:
        out.append(f"**Methodology score:** {c['methodology_score']:.2f}")
    if c.get("novelty_score") is not None:
        out.append(f"**Novelty score:** {c['novelty_score']:.2f}")
    out.append("")
    out.append(f"**Executive summary.** {c.get('executive_summary', '')}")
    strengths = c.get("consensus_strengths", [])
    if strengths:
        out.append("\n**Consensus strengths**")
        out.extend(f"- {s}" for s in strengths)
    concerns = c.get("consensus_concerns", [])
    if concerns:
        out.append("\n**Consensus concerns**")
        out.extend(_render_concern(cc) for cc in concerns)
    disagreements = c.get("disagreements", [])
    if disagreements:
        out.append("\n**Disagreements**")
        out.extend(f"- {d}" for d in disagreements)
    priorities = c.get("revision_priorities", [])
    if priorities:
        out.append("\n**Revision priorities (ordered)**")
        out.extend(f"{i}. {p}" for i, p in enumerate(priorities, 1))
    return "\n".join(out)


def _render_editor(e: dict[str, Any]) -> str:
    out = [
        "## Editorial Decision",
        "",
        f"**Decision:** {e.get('decision_label', e.get('decision', '?'))}",
        f"**Confidence:** {e.get('confidence', 0.0):.2f}",
        f"**Fits journal audience:** {'yes' if e.get('fits_audience') else 'no'}",
        "",
        f"**Rationale.** {e.get('rationale', '')}",
    ]
    guidance = e.get("author_guidance", [])
    if guidance:
        out.append("\n**Guidance for authors**")
        out.extend(f"- {g}" for g in guidance)
    return "\n".join(out)


def _render_summary(s: dict[str, Any]) -> str:
    return (
        "## Author & Editor Recommendations\n\n"
        "### 📝 Comments to the Author\n"
        "_What the author would see on a real journal review form._\n\n"
        f"{s.get('recommendation_for_author', '')}\n\n"
        "### 🔒 Confidential Comments to the Editor\n"
        "_Editor-only. The author does **not** see this section in a real submission._\n\n"
        f"{s.get('recommendation_for_editor_only', '')}"
    )


def _render_cost_summary(audit: dict[str, Any]) -> str:
    if not audit:
        return ""
    lines = [
        "## Cost & Token Usage",
        "",
        f"**Run id:** `{audit.get('run_id', '?')}`  ",
        f"**Total calls:** {audit.get('total_calls', 0)}  ",
        f"**Total input tokens:** {audit.get('total_input_tokens', 0):,}  ",
        f"**Total output tokens:** {audit.get('total_output_tokens', 0):,}  ",
        f"**Estimated cost (USD):** ${audit.get('total_cost_usd', 0.0):.4f}",
        "",
    ]
    per_provider = audit.get("per_provider", {}) or {}
    if per_provider:
        lines.append("### Per provider")
        lines.append("")
        lines.append("| Provider | Calls | Input tokens | Output tokens | Cost (USD) |")
        lines.append("|---|---:|---:|---:|---:|")
        for p, v in sorted(per_provider.items()):
            lines.append(
                f"| {p} | {v.get('calls', 0)} | {v.get('input_tokens', 0):,} | "
                f"{v.get('output_tokens', 0):,} | ${v.get('cost_usd', 0.0):.4f} |"
            )
        lines.append("")
    per_agent = audit.get("per_agent", {}) or {}
    if per_agent:
        lines.append("### Per agent")
        lines.append("")
        lines.append("| Agent | Provider | Model | Input | Output | Latency (ms) | Cost (USD) | Thinking |")
        lines.append("|---|---|---|---:|---:|---:|---:|:---:|")
        for a in sorted(per_agent):
            v = per_agent[a]
            tk = "✓" if v.get("thinking_mode_enabled") else "—"
            lines.append(
                f"| {a} | {v.get('provider', '?')} | `{v.get('model', '?')}` | "
                f"{v.get('input_tokens', 0):,} | {v.get('output_tokens', 0):,} | "
                f"{v.get('latency_ms', 0)} | ${v.get('cost_usd', 0.0):.4f} | {tk} |"
            )
        lines.append("")
    lines.extend([
        "_Rates are estimates from `agentic_paper/audit.py` (May 2026). "
        "Cross-check against the vendor's current pricing for accounting purposes._",
        "",
    ])
    return "\n".join(lines)


def _render_header(results: dict[str, Any]) -> list[str]:
    """Title + auto-simulation disclaimer + run metadata. Always rendered."""
    return [
        "# Peer Review Report",
        "",
        "> ⚠️ **Auto-simulation report.** Produced by LLM agents simulating a "
        "peer-review committee. Intended as a self-review aid for the author's "
        "own manuscript. **Not** a substitute for human peer review; **do not** "
        "use to assess another author's paper without their consent.",
        "",
        f"**Generated:** {results.get('timestamp', '')}",
        f"**Run id:** `{results.get('run_id', '?')}`",
        "",
    ]


def _render_paper_info(paper_info: dict[str, Any]) -> list[str]:
    """Title / authors / abstract / length / detected sections block."""
    return [
        "## Paper Information",
        "",
        f"**Title:** {paper_info.get('title', 'Unknown')}",
        "",
        f"**Authors:** {paper_info.get('authors', 'Unknown')}",
        "",
        "**Abstract:**",
        paper_info.get("abstract", ""),
        "",
        f"**Document Length:** {paper_info.get('length', 0):,} characters",
        "",
        f"**Identified Sections:** {', '.join(paper_info.get('sections', [])[:10])}",
        "",
    ]


def _render_routing_table(routing: dict[str, Any]) -> list[str]:
    """Per-agent (provider, model, thinking) table; empty list when no routing info."""
    if not routing:
        return []
    lines = [
        "## Routing per Agent",
        "",
        "| Agent | Provider | Model | Thinking |",
        "|---|---|---|---|",
    ]
    for k, info in routing.items():
        lines.append(
            f"| `{k}` | {info.get('provider', '?')} | `{info.get('model', '?')}` | "
            f"{info.get('thinking_budget', '—')} |"
        )
    lines.append("")
    return lines


def _render_reviews_section(reviews: dict[str, Any]) -> list[str]:
    """Detailed reviewer outputs in the canonical REVIEWER_TITLES order."""
    if not reviews:
        return []
    lines = ["## Detailed Reviews", ""]
    for key in REVIEWER_TITLES:  # iteration order is the rendering order
        if key not in reviews:
            continue
        lines.append(_render_review(key, reviews[key]))
        lines.append("\n---\n")
    return lines


def _render_errors_section(errors: dict[str, str]) -> list[str]:
    """Per-agent error rows; empty list when no agent failed."""
    if not errors:
        return []
    lines = ["## Errors", ""]
    for k, msg in errors.items():
        lines.append(f"- **{k}** — {msg}")
    lines.append("")
    return lines


def _render_seed_footer(config: dict[str, Any] | None) -> list[str]:
    """One-liner about deterministic mode; empty when no seed was used."""
    if not config:
        return []
    seed = config.get("seed")
    if seed is None:
        return []
    return [
        f"_Reproducibility: this run used seed `{seed}` "
        f"(forwarded to OpenAI + Google; ignored by Anthropic)._"
    ]


# Order matters: editor first (the "verdict"), then coordinator (synthesis),
# then summary (author/editor recommendations). Each entry pairs the data
# look-up key with the section renderer.
_OPTIONAL_RENDERERS: tuple[tuple[str, Callable[[dict[str, Any]], str]], ...] = (
    ("editor_decision", _render_editor),
    ("coordinator", _render_coordinator),
    ("author_editor_summary", _render_summary),
)


def render_markdown(results: dict[str, Any]) -> str:
    """Render the detailed peer-review report as Markdown."""
    paper_info = results.get("paper_info", {})
    reviews = results.get("reviews", {})
    errors: dict[str, str] = results.get("errors", {}) or {}
    audit_summary = results.get("audit_summary", {}) or {}
    routing = (results.get("config") or {}).get("routing", {})

    parts: list[str] = []
    parts.extend(_render_header(results))
    parts.extend(_render_paper_info(paper_info))
    parts.extend(_render_routing_table(routing))

    # Optional decision/synthesis/summary blocks — dispatched via the table
    # above so render_markdown no longer carries one branch per block.
    for key, renderer in _OPTIONAL_RENDERERS:
        data = results.get(key)
        if not data:
            continue
        parts.append(renderer(data))
        parts.append("")

    parts.extend(_render_reviews_section(reviews))
    parts.extend(_render_errors_section(errors))

    cost_md = _render_cost_summary(audit_summary)
    if cost_md:
        parts.append(cost_md)

    parts.extend(_render_seed_footer(results.get("config")))
    return "\n".join(parts)


def render_executive_summary(results: dict[str, Any]) -> str:
    paper_info = results.get("paper_info", {})
    editor = results.get("editor_decision") or {}
    coordinator = results.get("coordinator") or {}
    summary = results.get("author_editor_summary") or {}

    out = [
        "# Executive Summary",
        "",
        f"**Paper:** {paper_info.get('title', '')}",
        f"**Authors:** {paper_info.get('authors', '')}",
        f"**Review date:** {results.get('timestamp', '')}",
        "",
        "## Editorial Decision",
        "",
        f"**Decision:** {editor.get('decision_label', editor.get('decision', '—'))}  ",
        f"**Confidence:** {editor.get('confidence', 0.0):.2f}",
        "",
        editor.get("rationale", ""),
        "",
        "## Coordinator Overall Assessment",
        "",
        f"**Final recommendation:** {coordinator.get('final_recommendation', '—')}  ",
        f"**Overall score:** {coordinator.get('overall_score', 0.0):.2f}",
        "",
        coordinator.get("executive_summary", ""),
        "",
        "## Recommendation for the Author",
        "",
        summary.get("recommendation_for_author", ""),
        "",
        "---",
        "",
        "This paper was reviewed by 12 specialised AI reviewers (methodology, results, literature,",
        "structure, impact, contradiction, ethics, AI-origin, hallucination, citation-validator,",
        "statcheck-validator, revision-assessor), synthesised by a",
        "coordinator agent and adjudicated by an editor agent. See the full report for per-reviewer",
        "structured output (strengths, concerns by severity, suggested fixes).",
    ]
    return "\n".join(out)
