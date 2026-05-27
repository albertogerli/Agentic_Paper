"""HTML dashboard renderer using Tailwind via CDN, with severity color-coding."""

from __future__ import annotations

import html as _html
from typing import Any


SEVERITY_STYLE = {
    "fatal": ("bg-red-100 border-red-300 text-red-900", "🔴", "Fatal"),
    "major": ("bg-yellow-100 border-yellow-300 text-yellow-900", "🟡", "Major"),
    "minor": ("bg-blue-100 border-blue-300 text-blue-900", "🔵", "Minor"),
}

DECISION_STYLE = {
    "accept": ("bg-green-100 border-green-300 text-green-900", "✅"),
    "minor_rev": ("bg-blue-100 border-blue-300 text-blue-900", "🔧"),
    "major_rev": ("bg-yellow-100 border-yellow-300 text-yellow-900", "⚠️"),
    "reject": ("bg-red-100 border-red-300 text-red-900", "❌"),
}

REVIEWER_DISPLAY = {
    "methodology":         ("🔬", "Methodology Expert"),
    "results":             ("📊", "Results Analyst"),
    "literature":          ("📚", "Literature Expert"),
    "structure":           ("🏗️", "Structure & Clarity Reviewer"),
    "impact":              ("💡", "Impact & Innovation Analyst"),
    "contradiction":       ("🔍", "Contradiction Checker"),
    "ethics":              ("⚖️", "Ethics & Integrity Reviewer"),
    "ai_origin":           ("🤖", "AI Origin Detector"),
    "hallucination":       ("🚨", "Hallucination Detector"),
    "citation_validator":  ("🌐", "Citation Validator (OpenAlex)"),
    "statcheck_validator": ("📐", "Statistical Sanity (statcheck R)"),
    "revision_assessor":   ("🔁", "Revision Assessor (v1 → v2)"),
}


def _esc(text: Any) -> str:
    return _html.escape(str(text))


def _render_concern_card(c: dict[str, Any]) -> str:
    severity = c.get("severity", "minor")
    style, icon, label = SEVERITY_STYLE.get(severity, SEVERITY_STYLE["minor"])
    page = f"p.{_esc(c['page'])}" if c.get("page") is not None else "—"
    fix_html = ""
    if c.get("suggested_fix"):
        fix_html = (
            f'<div class="mt-2 text-xs text-gray-700">'
            f'<span class="font-semibold uppercase tracking-wide">Suggested fix:</span> '
            f'{_esc(c["suggested_fix"])}</div>'
        )
    return (
        f'<div class="border-2 {style} rounded-md p-3 mb-2">'
        f'<div class="flex items-baseline justify-between">'
        f'<span class="font-semibold">{icon} {label}</span>'
        f'<span class="text-xs">{_esc(c.get("section", "?"))} · {page}</span>'
        f'</div>'
        f'<div class="mt-1 text-sm">{_esc(c.get("issue", ""))}</div>'
        f'{fix_html}'
        f'</div>'
    )


def _render_review_block(key: str, r: dict[str, Any]) -> str:
    icon, title = REVIEWER_DISPLAY.get(key, ("📝", key.replace("_", " ").title()))
    confidence = r.get("confidence", 0.0)
    rec = r.get("recommendation", "?")
    rec_style, rec_icon = DECISION_STYLE.get(rec, ("bg-gray-100", "📋"))
    strengths_html = ""
    if r.get("strengths"):
        items = "".join(f"<li>{_esc(s)}</li>" for s in r["strengths"])
        strengths_html = (
            f'<div class="mt-3"><div class="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-1">'
            f'Strengths</div><ul class="list-disc list-inside text-sm text-gray-800">{items}</ul></div>'
        )
    concerns_html = ""
    if r.get("concerns"):
        cards = "".join(_render_concern_card(c) for c in r["concerns"])
        concerns_html = (
            f'<div class="mt-3"><div class="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-1">'
            f'Concerns ({len(r["concerns"])})</div>{cards}</div>'
        )
    else:
        concerns_html = '<div class="mt-3 text-xs italic text-gray-500">No concerns flagged.</div>'

    return f"""
<div class="review-card border-2 bg-white border-gray-200 rounded-lg p-5">
  <div class="flex items-start justify-between mb-3">
    <h3 class="text-lg font-semibold flex items-center gap-2">
      <span class="text-2xl">{icon}</span> {_esc(title)}
    </h3>
    <div class="text-xs text-gray-500 text-right">
      <div>{_esc(r.get("agent", ""))}</div>
      <div><code>{_esc(r.get("model_used", ""))}</code></div>
    </div>
  </div>
  <div class="flex items-center gap-3 mb-3">
    <span class="px-2 py-1 rounded {rec_style} text-xs font-semibold">{rec_icon} {_esc(rec)}</span>
    <span class="text-xs text-gray-500">confidence {confidence:.2f}</span>
  </div>
  <div class="text-sm text-gray-800">{_esc(r.get("summary", ""))}</div>
  {strengths_html}
  {concerns_html}
</div>
"""


def _render_coordinator(c: dict[str, Any] | None) -> str:
    if not c:
        return ""
    final = c.get("final_recommendation", "?")
    style, icon = DECISION_STYLE.get(final, ("bg-gray-100", "📋"))
    score = c.get("overall_score", 0.0)
    extras: list[str] = []
    if c.get("methodology_score") is not None:
        extras.append(f"methodology {c['methodology_score']:.2f}")
    if c.get("novelty_score") is not None:
        extras.append(f"novelty {c['novelty_score']:.2f}")
    extra_html = f' · {" · ".join(extras)}' if extras else ""

    concerns = c.get("consensus_concerns", [])
    concerns_html = "".join(_render_concern_card(cc) for cc in concerns)
    strengths_html = ""
    if c.get("consensus_strengths"):
        items = "".join(f"<li>{_esc(s)}</li>" for s in c["consensus_strengths"])
        strengths_html = (
            f'<div class="mt-3"><div class="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-1">'
            f'Consensus strengths</div><ul class="list-disc list-inside text-sm">{items}</ul></div>'
        )
    disagreements_html = ""
    if c.get("disagreements"):
        items = "".join(f"<li>{_esc(d)}</li>" for d in c["disagreements"])
        disagreements_html = (
            f'<div class="mt-3"><div class="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-1">'
            f'Disagreements</div><ul class="list-disc list-inside text-sm">{items}</ul></div>'
        )
    priorities_html = ""
    if c.get("revision_priorities"):
        items = "".join(f"<li>{_esc(p)}</li>" for p in c["revision_priorities"])
        priorities_html = (
            f'<div class="mt-3"><div class="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-1">'
            f'Revision priorities</div><ol class="list-decimal list-inside text-sm">{items}</ol></div>'
        )
    return f"""
<div class="bg-white rounded-lg shadow-lg p-6 mb-8">
  <h2 class="text-2xl font-semibold mb-4 flex items-center gap-2">
    🎯 Coordinator Assessment
  </h2>
  <div class="flex items-center gap-3 mb-3">
    <span class="px-2 py-1 rounded {style} text-sm font-semibold">{icon} {_esc(final)}</span>
    <span class="text-sm text-gray-600">overall {score:.2f}{extra_html}</span>
  </div>
  <div class="text-sm text-gray-800">{_esc(c.get("executive_summary", ""))}</div>
  {strengths_html}
  <div class="mt-3"><div class="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-1">Consensus concerns ({len(concerns)})</div>{concerns_html}</div>
  {disagreements_html}
  {priorities_html}
</div>
"""


def _render_editor(e: dict[str, Any] | None) -> str:
    if not e:
        return ""
    decision = e.get("decision", "?")
    style, icon = DECISION_STYLE.get(decision, ("bg-gray-100", "📋"))
    label = e.get("decision_label", decision)
    confidence = e.get("confidence", 0.0)
    fits = "yes" if e.get("fits_audience") else "no"
    guidance_html = ""
    if e.get("author_guidance"):
        items = "".join(f"<li>{_esc(g)}</li>" for g in e["author_guidance"])
        guidance_html = (
            f'<div class="mt-3"><div class="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-1">'
            f'Guidance for authors</div><ul class="list-disc list-inside text-sm">{items}</ul></div>'
        )
    return f"""
<div class="bg-white rounded-lg shadow-lg p-6 mb-8">
  <h2 class="text-2xl font-semibold mb-4 flex items-center gap-2">
    <span>{icon}</span> Editorial Decision
  </h2>
  <div class="border-2 {style} rounded-lg p-4 mb-3">
    <div class="text-lg font-semibold">{_esc(label)}</div>
    <div class="text-xs mt-1">confidence {confidence:.2f} · fits audience: {fits}</div>
  </div>
  <div class="text-sm text-gray-800">{_esc(e.get("rationale", ""))}</div>
  {guidance_html}
</div>
"""


def _render_summary(s: dict[str, Any] | None) -> str:
    if not s:
        return ""
    return f"""
<div class="bg-white rounded-lg shadow-lg p-6 mb-8">
  <h2 class="text-2xl font-semibold mb-2">📝 Author & Editor Recommendations</h2>
  <p class="text-xs text-gray-500 mb-4">
    Mirrors the two-box layout of a standard journal review form: the author
    sees the left panel; the right panel is editor-only and would be stripped
    before sending feedback to the author.
  </p>
  <div class="grid md:grid-cols-2 gap-4">
    <div class="bg-purple-50 border-2 border-purple-200 rounded-lg p-4">
      <div class="text-xs font-semibold uppercase tracking-wide text-purple-700 mb-2">
        Comments to the Author
      </div>
      <div class="text-sm whitespace-pre-wrap text-gray-800">{_esc(s.get("recommendation_for_author", ""))}</div>
    </div>
    <div class="bg-amber-50 border-2 border-amber-300 rounded-lg p-4">
      <div class="text-xs font-semibold uppercase tracking-wide text-amber-700 mb-2">
        🔒 Confidential Comments to the Editor
      </div>
      <div class="text-sm whitespace-pre-wrap text-gray-800">{_esc(s.get("recommendation_for_editor_only", ""))}</div>
    </div>
  </div>
</div>
"""


def render_html(results: dict[str, Any]) -> str:
    paper = results.get("paper_info", {})
    reviews = results.get("reviews", {})
    editor = results.get("editor_decision")
    coordinator = results.get("coordinator")
    summary = results.get("author_editor_summary")
    timestamp = results.get("timestamp", "")

    # Severity counts across all reviews
    severity_counts = {"fatal": 0, "major": 0, "minor": 0}
    for r in reviews.values():
        for c in r.get("concerns", []) or []:
            sev = c.get("severity", "minor")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

    review_blocks = "".join(
        _render_review_block(key, reviews[key])
        for key in REVIEWER_DISPLAY
        if key in reviews
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Paper Review Dashboard - {_esc(paper.get('title', 'Untitled'))}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    body {{ font-family: 'Inter', sans-serif; }}
    .gradient-bg {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
    .review-card {{ transition: all 0.3s ease; }}
    .review-card:hover {{ transform: translateY(-2px); box-shadow: 0 12px 24px rgba(0,0,0,0.1); }}
  </style>
</head>
<body class="bg-gray-50">
  <div class="gradient-bg text-white">
    <div class="container mx-auto px-6 py-12">
      <h1 class="text-4xl font-bold mb-2">📚 Peer Review Dashboard</h1>
      <p class="text-purple-100">Advanced Multi-Agent Review System</p>
    </div>
  </div>

  <div class="container mx-auto px-6 py-8 max-w-7xl">
    <div class="bg-white rounded-lg shadow-lg p-8 mb-8">
      <h2 class="text-2xl font-semibold mb-6">📄 Paper Information</h2>
      <div class="grid md:grid-cols-2 gap-6">
        <div>
          <h3 class="text-sm font-medium text-gray-500 uppercase tracking-wide mb-2">Title</h3>
          <p class="text-lg font-medium text-gray-900">{_esc(paper.get('title', 'Not specified'))}</p>
        </div>
        <div>
          <h3 class="text-sm font-medium text-gray-500 uppercase tracking-wide mb-2">Authors</h3>
          <p class="text-lg text-gray-700">{_esc(paper.get('authors', 'Not specified'))}</p>
        </div>
        <div>
          <h3 class="text-sm font-medium text-gray-500 uppercase tracking-wide mb-2">Document Length</h3>
          <p class="text-lg text-gray-700">{paper.get('length', 0):,} characters</p>
        </div>
        <div>
          <h3 class="text-sm font-medium text-gray-500 uppercase tracking-wide mb-2">Review Date</h3>
          <p class="text-lg text-gray-700">{_esc(timestamp)}</p>
        </div>
      </div>
    </div>

    <div class="grid md:grid-cols-3 gap-4 mb-8">
      <div class="bg-red-50 border-2 border-red-300 rounded-lg p-4 text-center">
        <div class="text-3xl font-bold text-red-700">{severity_counts["fatal"]}</div>
        <div class="text-sm text-red-700 mt-1">🔴 Fatal concerns</div>
      </div>
      <div class="bg-yellow-50 border-2 border-yellow-300 rounded-lg p-4 text-center">
        <div class="text-3xl font-bold text-yellow-700">{severity_counts["major"]}</div>
        <div class="text-sm text-yellow-700 mt-1">🟡 Major concerns</div>
      </div>
      <div class="bg-blue-50 border-2 border-blue-300 rounded-lg p-4 text-center">
        <div class="text-3xl font-bold text-blue-700">{severity_counts["minor"]}</div>
        <div class="text-sm text-blue-700 mt-1">🔵 Minor concerns</div>
      </div>
    </div>

    {_render_editor(editor)}
    {_render_coordinator(coordinator)}
    {_render_summary(summary)}

    <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
      <h2 class="text-2xl font-semibold mb-6">📋 Detailed Expert Reviews</h2>
      <div class="space-y-6">
        {review_blocks}
      </div>
    </div>
  </div>
</body>
</html>"""
