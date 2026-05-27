"""Model catalog + lookup heuristics."""

from __future__ import annotations

import pytest

from agentic_paper.models import (
    CATALOG,
    PROVIDER_MODELS,
    ModelSpec,
    lookup,
    models_for,
)


def test_every_catalog_model_has_a_provider() -> None:
    assert set(PROVIDER_MODELS) == {"openai", "anthropic", "google"}
    for provider, models in PROVIDER_MODELS.items():
        assert models, f"{provider} should have at least one curated model"
        for m in models:
            assert m.provider == provider


def test_catalog_is_indexed_by_model_id() -> None:
    for model_id, spec in CATALOG.items():
        assert spec.model == model_id


@pytest.mark.parametrize(
    "model, provider, style",
    [
        ("gpt-5.5", "openai", "effort"),
        ("gpt-5.5-pro", "openai", "effort"),
        ("gpt-5.4-mini", "openai", "effort"),
        ("gpt-4o", "openai", "none"),
        ("claude-opus-4-7", "anthropic", "adaptive"),
        ("claude-sonnet-4-6", "anthropic", "adaptive"),
        ("claude-haiku-4-5", "anthropic", "adaptive"),
        ("claude-sonnet-4-5", "anthropic", "budget"),
        ("gemini-3-pro", "google", "level"),
        ("gemini-3.1-pro-preview", "google", "level"),
        ("gemini-3.1-flash-lite", "google", "level"),
        ("gemini-2.5-flash", "google", "budget"),
    ],
)
def test_lookup_returns_known_styles(model: str, provider: str, style: str) -> None:
    spec = lookup(model)
    assert spec is not None
    assert spec.provider == provider
    assert spec.reasoning_style == style


@pytest.mark.parametrize(
    "model, provider, style",
    [
        ("claude-opus-5-0", "anthropic", "adaptive"),  # future 5.x
        ("claude-opus-3-0", "anthropic", "budget"),    # older
        ("gemini-3.2-flash", "google", "level"),        # future 3.x
        ("gemini-1.5-pro", "google", "budget"),         # older
        ("gpt-5.9", "openai", "effort"),                # future 5.x
        ("o5-mini", "openai", "effort"),                # future o-series
        ("gpt-3.5-turbo", "openai", "none"),            # legacy non-reasoning
    ],
)
def test_lookup_heuristic_fallback(model: str, provider: str, style: str) -> None:
    spec = lookup(model)
    assert spec is not None
    assert spec.provider == provider
    assert spec.reasoning_style == style


def test_lookup_returns_none_for_empty_or_unknown() -> None:
    assert lookup("") is None
    assert lookup("totally-unknown-vendor-model") is None


def test_models_for_returns_list_or_empty() -> None:
    assert len(models_for("openai")) > 0
    assert models_for("unknown") == []
    assert models_for("openai_compat") == []
