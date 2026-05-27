"""Catalog of vendor models per provider with call-modality metadata.

Each :class:`ModelSpec` carries the info that providers need to pick the
right API surface (OpenAI Responses vs Chat Completions; Anthropic
``adaptive`` vs ``enabled`` thinking; Gemini ``thinking_level`` vs
``thinking_budget``). Use :func:`lookup` to resolve a model name to a
spec — unknown names get a best-effort fallback based on the prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ReasoningStyle = Literal["none", "effort", "adaptive", "budget", "level"]
"""How a model accepts a thinking / reasoning hint:

* ``none``     — non-reasoning model; ignore any ``thinking_budget``.
* ``effort``   — OpenAI Responses API ``reasoning.effort`` (``low/medium/high/xhigh``).
* ``adaptive`` — Anthropic 4.6+ ``thinking={"type":"adaptive"}`` (no explicit budget).
* ``budget``   — Legacy ``budget_tokens`` int (Claude 4.5 / Gemini 2.x).
* ``level``    — Gemini 3.x ``thinking_level`` (``low/medium/high``).
"""


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    context_window: int = 0
    max_output_tokens: int = 0
    supports_reasoning: bool = False
    reasoning_style: ReasoningStyle = "none"
    notes: str = ""


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
OPENAI_MODELS: list[ModelSpec] = [
    ModelSpec("openai", "gpt-5.5",         200_000, 128_000, True,  "effort", "Latest flagship (2026-04-23)"),
    ModelSpec("openai", "gpt-5.5-pro",     200_000, 128_000, True,  "effort", "Higher-accuracy variant"),
    ModelSpec("openai", "gpt-5.4-mini",    200_000,  64_000, True,  "effort", "Smaller, faster"),
    ModelSpec("openai", "gpt-5.4-nano",    400_000,  64_000, True,  "effort", "Cheapest 5.4 (2026-03-17)"),
    ModelSpec("openai", "gpt-5",           200_000, 128_000, True,  "effort", ""),
    ModelSpec("openai", "gpt-5-mini",      200_000,  64_000, True,  "effort", ""),
    ModelSpec("openai", "o4-mini",         128_000,  64_000, True,  "effort", ""),
    ModelSpec("openai", "o3",              128_000,  64_000, True,  "effort", ""),
    ModelSpec("openai", "gpt-4o",          128_000,  16_000, False, "none",   "Non-reasoning"),
    ModelSpec("openai", "gpt-4o-mini",     128_000,  16_000, False, "none",   "Non-reasoning"),
]

# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
ANTHROPIC_MODELS: list[ModelSpec] = [
    ModelSpec("anthropic", "claude-opus-4-7",       200_000, 64_000, True, "adaptive", "Most capable"),
    ModelSpec("anthropic", "claude-sonnet-4-6",     200_000, 64_000, True, "adaptive", "Balanced (recommended)"),
    ModelSpec("anthropic", "claude-haiku-4-5",      200_000, 32_000, True, "adaptive", "Fast + cheap"),
    ModelSpec("anthropic", "claude-sonnet-4-5",     200_000, 64_000, True, "budget",   "Legacy budget mode"),
    ModelSpec("anthropic", "claude-opus-4",         200_000, 32_000, True, "budget",   "Legacy"),
]

# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------
GOOGLE_MODELS: list[ModelSpec] = [
    ModelSpec("google", "gemini-3-pro",                  1_000_000, 65_000, True, "level",  "Flagship reasoning"),
    ModelSpec("google", "gemini-3.1-pro-preview",        1_000_000, 65_000, True, "level",  "Preview"),
    ModelSpec("google", "gemini-3.5-flash",              1_048_576, 64_000, True, "level",  "Newest Flash flagship (GA 2026-05-19)"),
    ModelSpec("google", "gemini-3-flash",                1_000_000, 65_000, True, "level",  "Fast tier"),
    ModelSpec("google", "gemini-3.1-flash-lite",         1_000_000, 65_000, True, "level",  "Cheapest 3.x (GA)"),
    ModelSpec("google", "gemini-3.1-flash-lite-preview", 1_000_000, 65_000, True, "level",  "Older preview ID"),
    ModelSpec("google", "gemini-2.5-pro",                1_000_000, 65_000, True, "budget", "Legacy budget mode"),
    ModelSpec("google", "gemini-2.5-flash",              1_000_000, 65_000, True, "budget", "Legacy"),
]

PROVIDER_MODELS: dict[str, list[ModelSpec]] = {
    "openai": OPENAI_MODELS,
    "anthropic": ANTHROPIC_MODELS,
    "google": GOOGLE_MODELS,
}

CATALOG: dict[str, ModelSpec] = {
    m.model: m for spec_list in PROVIDER_MODELS.values() for m in spec_list
}


def lookup(model: str) -> ModelSpec | None:
    """Resolve ``model`` to a :class:`ModelSpec`.

    Exact catalog match wins; otherwise a heuristic fallback infers the
    provider and reasoning style from the model name prefix so newer
    releases work without a catalog update.
    """
    if not model:
        return None
    if model in CATALOG:
        return CATALOG[model]

    # Anthropic 4.6+ → adaptive thinking
    if model.startswith("claude-"):
        if any(tag in model for tag in ("-4-6", "-4-7", "-4-8", "-4-9", "-5-")):
            return ModelSpec("anthropic", model, supports_reasoning=True, reasoning_style="adaptive")
        return ModelSpec("anthropic", model, supports_reasoning=True, reasoning_style="budget")

    # Gemini 3.x → thinking_level; 2.x → thinking_budget
    if model.startswith("gemini-3"):
        return ModelSpec("google", model, supports_reasoning=True, reasoning_style="level")
    if model.startswith("gemini-"):
        return ModelSpec("google", model, supports_reasoning=True, reasoning_style="budget")

    # OpenAI gpt-5.x / o-series → reasoning_effort
    if model.startswith("gpt-5") or (model.startswith("o") and len(model) > 1 and model[1].isdigit()):
        return ModelSpec("openai", model, supports_reasoning=True, reasoning_style="effort")
    if model.startswith("gpt-"):
        return ModelSpec("openai", model)

    return None


def models_for(provider: str) -> list[ModelSpec]:
    """Return the curated model list for a provider (empty for unknown / openai_compat)."""
    return list(PROVIDER_MODELS.get(provider, []))


__all__ = [
    "ModelSpec",
    "ReasoningStyle",
    "OPENAI_MODELS",
    "ANTHROPIC_MODELS",
    "GOOGLE_MODELS",
    "PROVIDER_MODELS",
    "CATALOG",
    "lookup",
    "models_for",
]
