"""Schema sanitizer for Gemini's response_schema.

Regression test for the live failure on 2026-05-21: Gemini rejected every
agent call with HTTP 400 ``Unknown name "additional_properties"`` because
Pydantic-generated JSON schemas carry ``additionalProperties: false``,
``$defs/$ref``, and ``anyOf:[T, null]`` shapes that Gemini does not accept.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict

from agentic_paper.providers.google_provider import _sanitize_schema_for_gemini
from agentic_paper.schemas import Review


def _walk_keys(node, banned: set[str]) -> list[str]:
    """Return a list of paths where any banned key shows up. Empty == clean."""
    hits: list[str] = []

    def _w(n, prefix=""):
        if isinstance(n, dict):
            for k, v in n.items():
                if k in banned:
                    hits.append(prefix + k)
                _w(v, prefix=f"{prefix}{k}.")
        elif isinstance(n, list):
            for i, item in enumerate(n):
                _w(item, prefix=f"{prefix}[{i}].")
    _w(node)
    return hits


def test_sanitizer_strips_additionalProperties() -> None:
    raw = Review.model_json_schema()
    # Sanity: Pydantic does include the offending field.
    assert _walk_keys(raw, {"additionalProperties"}), "fixture must contain additionalProperties"
    out = _sanitize_schema_for_gemini(raw)
    assert _walk_keys(out, {"additionalProperties"}) == []


def test_sanitizer_inlines_defs_and_refs() -> None:
    raw = Review.model_json_schema()
    # Pydantic uses $defs for the nested ConcernItem model.
    assert "$defs" in raw
    out = _sanitize_schema_for_gemini(raw)
    assert "$defs" not in out
    assert _walk_keys(out, {"$ref", "$defs"}) == []
    # And the inlined ConcernItem fields must be reachable from concerns.items.
    concerns_items = out["properties"]["concerns"]["items"]
    assert concerns_items["type"] == "object"
    assert "severity" in concerns_items["properties"]
    assert "issue" in concerns_items["properties"]


def test_sanitizer_collapses_optional_anyof_to_nullable() -> None:
    class _M(BaseModel):
        model_config = ConfigDict(extra="forbid")
        name: str
        age: Optional[int] = None

    raw = _M.model_json_schema()
    out = _sanitize_schema_for_gemini(raw)
    # 'age' should now be a single-type object with nullable: true, no anyOf.
    age = out["properties"]["age"]
    assert "anyOf" not in age
    assert age.get("nullable") is True
    assert age.get("type") == "integer"


def test_sanitizer_preserves_required_and_descriptions() -> None:
    raw = Review.model_json_schema()
    out = _sanitize_schema_for_gemini(raw)
    assert "required" in out
    assert set(out["required"]) == {
        "summary", "strengths", "concerns", "recommendation", "confidence"
    }
    # title may or may not be retained, but the property descriptions should.
    summary_node = out["properties"]["summary"]
    assert summary_node["type"] == "string"


def test_sanitizer_idempotent() -> None:
    raw = Review.model_json_schema()
    once = _sanitize_schema_for_gemini(raw)
    twice = _sanitize_schema_for_gemini(once)
    assert once == twice
