"""Pure functions that decide which (provider, model, thinking_budget) an agent runs on."""

from __future__ import annotations

from typing import Tuple

from .config import Config, RoutingConfig, TierConfig
from .models import PROVIDER_MODELS
from .providers.base import ThinkingBudget


# --------------------------------------------------------------------- profiles


PROFILE_NAMES = ("max", "std", "quick")


PROFILES: dict[str, RoutingConfig] = {
    # Most capable models everywhere, max thinking. Slow + expensive.
    "max": RoutingConfig(
        tier_high=TierConfig(
            provider="anthropic", model="claude-opus-4-7", thinking_budget="auto",
        ),
        tier_standard=TierConfig(
            provider="openai", model="gpt-5.5", thinking_budget="high",
        ),
        tier_basic=TierConfig(
            provider="google", model="gemini-3-pro", thinking_budget="high",
        ),
    ),
    # Balanced — what `config_example.yaml` ships as the default routing.
    "std": RoutingConfig(
        tier_high=TierConfig(
            provider="anthropic", model="claude-sonnet-4-6", thinking_budget="auto",
        ),
        tier_standard=TierConfig(
            provider="openai", model="gpt-5.4-mini", thinking_budget="medium",
        ),
        tier_basic=TierConfig(
            provider="google", model="gemini-3.5-flash", thinking_budget="medium",
        ),
    ),
    # Cheapest+fastest everywhere, no extended thinking. Smoke-grade quality.
    "quick": RoutingConfig(
        tier_high=TierConfig(
            provider="anthropic", model="claude-haiku-4-5", thinking_budget=None,
        ),
        tier_standard=TierConfig(
            provider="openai", model="gpt-5.4-nano", thinking_budget="low",
        ),
        tier_basic=TierConfig(
            provider="google", model="gemini-3.1-flash-lite", thinking_budget="low",
        ),
    ),
}


# ------------------------------------------------------------------ auto-mode

# Cross-provider equivalents per tier slot. Derived by intensity-matching
# across the curated PROFILES: tier_high → flagship reasoning model,
# tier_standard → mid-tier, tier_basic → cheapest/fastest. Used when a tier
# points to a provider the user did not supply a key for.
_TIER_EQUIVALENTS: dict[str, dict[str, str]] = {
    "tier_high": {
        "openai":    "gpt-5.5",
        "anthropic": "claude-opus-4-7",
        "google":    "gemini-3-pro",
    },
    "tier_standard": {
        "openai":    "gpt-5.4-mini",
        "anthropic": "claude-sonnet-4-6",
        "google":    "gemini-3.5-flash",
    },
    "tier_basic": {
        "openai":    "gpt-5.4-nano",
        "anthropic": "claude-haiku-4-5",
        "google":    "gemini-3.1-flash-lite",
    },
}

_TIER_LABELS: dict[str, str] = {
    "tier_high":     "High tier",
    "tier_standard": "Standard tier",
    "tier_basic":    "Basic tier",
}


def _pick_fallback_provider(tier_slot: str, available: set[str]) -> str | None:
    """Pick a target provider from ``available`` for ``tier_slot``.

    Preference order:
      1. an available provider that has an explicit ``_TIER_EQUIVALENTS`` entry
         for this slot — guarantees we pick a model that fits the tier role;
      2. any other available provider (deterministic alphabetical sort) — used
         only when the user supplies a non-canonical provider (e.g. a local
         Ollama endpoint registered under a custom name).
    """
    if not available:
        return None
    known = _TIER_EQUIVALENTS[tier_slot]
    canonical = sorted(p for p in available if p in known)
    if canonical:
        return canonical[0]
    return sorted(available)[0]


def _pick_fallback_model(tier_slot: str, provider: str) -> str:
    """Resolve the model string for ``(tier_slot, provider)``.

    Falls back to the catalog's first model when the provider has no curated
    equivalent — keeps custom / local providers working out of the box.
    """
    explicit = _TIER_EQUIVALENTS[tier_slot].get(provider)
    if explicit is not None:
        return explicit
    catalog = PROVIDER_MODELS.get(provider) or []
    if catalog:
        return catalog[0].model
    return ""  # caller will surface this; provider registry rejects empty model


def apply_auto_mode(
    routing: RoutingConfig,
    available_providers: set[str],
) -> tuple[RoutingConfig, list[str]]:
    """Remap any tier whose provider is unavailable to a provider the user has.

    Returns a deep-copied :class:`RoutingConfig` (the original is never
    mutated) plus a list of human-readable warnings — one per remapped tier —
    suitable for surfacing in the UI. When every tier already points to an
    available provider, the warning list is empty. When ``available_providers``
    is empty (no keys at all), the routing is returned unchanged with no
    warnings — the calling layer is expected to surface the missing-keys
    failure separately.

    The remapped tier keeps its ``thinking_budget`` field intact; each
    provider translates ``"auto"`` / integer / level strings on its own, so a
    sane budget on one vendor stays sane on another.
    """
    if not available_providers:
        return routing, []

    new_routing = routing.model_copy(deep=True)
    warnings: list[str] = []

    for tier_slot in ("tier_high", "tier_standard", "tier_basic"):
        tier: TierConfig = getattr(new_routing, tier_slot)
        if tier.provider in available_providers:
            continue

        fallback_provider = _pick_fallback_provider(tier_slot, available_providers)
        if fallback_provider is None:
            continue  # impossible (we checked available_providers above) but defensive
        fallback_model = _pick_fallback_model(tier_slot, fallback_provider)

        original_provider = tier.provider
        original_model = tier.model
        tier.provider = fallback_provider
        tier.model = fallback_model

        warnings.append(
            f"{original_provider.capitalize()} not available "
            f"→ mapped {_TIER_LABELS[tier_slot]} from "
            f"{original_provider}/{original_model} to "
            f"{fallback_provider}/{fallback_model}"
        )

    return new_routing, warnings


# ------------------------------------------------------------------ profiles


def get_profile(name: str) -> RoutingConfig | None:
    """Return a deep-copied RoutingConfig for the named profile, or None."""
    if not name:
        return None
    base = PROFILES.get(name.strip().lower())
    if base is None:
        return None
    return base.model_copy(deep=True)


def get_routing(config: Config) -> RoutingConfig:
    """Return the active routing config.

    Precedence:
        1. ``config.routing`` block if present (preferred).
        2. Legacy ``model_powerful / model_standard / model_basic`` fields,
           all pinned to the ``openai`` provider.
    """
    if config.routing is not None:
        return config.routing
    return RoutingConfig(
        tier_high=TierConfig(provider="openai", model=config.model_powerful),
        tier_standard=TierConfig(provider="openai", model=config.model_standard),
        tier_basic=TierConfig(provider="openai", model=config.model_basic),
    )


def _pick_tier(routing: RoutingConfig, combined_score: float) -> TierConfig:
    if combined_score >= 0.65:
        return routing.tier_high
    if combined_score >= 0.45:
        return routing.tier_standard
    return routing.tier_basic


def route_agent(
    agent_name: str,
    base_complexity: float,
    paper_complexity_score: float,
    config: Config,
) -> Tuple[str, str, ThinkingBudget, float]:
    """Resolve the (provider, model, thinking_budget, combined_score) for an agent.

    Combined complexity = 40% paper score + 60% per-agent base. Per-agent overrides
    in ``config.agents.<name>`` take precedence over the tier defaults.
    """
    routing = get_routing(config)
    combined = paper_complexity_score * 0.4 + base_complexity * 0.6
    tier = _pick_tier(routing, combined)

    override = config.agents.get(agent_name)
    if override is None:
        return tier.provider, tier.model, tier.thinking_budget, combined

    provider = override.provider or tier.provider
    model = override.model or tier.model
    thinking: ThinkingBudget = (
        override.thinking_budget if override.thinking_budget is not None else tier.thinking_budget
    )
    return provider, model, thinking, combined
