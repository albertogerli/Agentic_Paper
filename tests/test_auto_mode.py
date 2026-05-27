"""Unit tests for routing.apply_auto_mode and the BYOK auto-mode plumbing."""

from __future__ import annotations

from agentic_paper.config import RoutingConfig, TierConfig
from agentic_paper.routing import apply_auto_mode, get_profile


def _std_routing() -> RoutingConfig:
    """The 'std' profile: anthropic / openai / google across the three tiers."""
    return get_profile("std")  # type: ignore[return-value]


# --------------------------------------------------------------- no-op path


def test_auto_mode_noop_when_all_providers_available() -> None:
    routing = _std_routing()
    available = {"openai", "anthropic", "google"}

    new_routing, warnings = apply_auto_mode(routing, available)

    assert warnings == []
    assert new_routing.tier_high.provider == "anthropic"
    assert new_routing.tier_standard.provider == "openai"
    assert new_routing.tier_basic.provider == "google"


def test_auto_mode_noop_when_no_providers_available() -> None:
    """Empty available set returns routing unchanged + no warnings.

    The calling layer is expected to surface the missing-keys failure
    separately — we don't second-guess what we can't fix."""
    routing = _std_routing()

    new_routing, warnings = apply_auto_mode(routing, set())

    assert warnings == []
    assert new_routing.tier_high.provider == "anthropic"


def test_auto_mode_does_not_mutate_input() -> None:
    routing = _std_routing()
    original_high_model = routing.tier_high.model

    apply_auto_mode(routing, {"google"})

    assert routing.tier_high.provider == "anthropic"
    assert routing.tier_high.model == original_high_model


# --------------------------------------------------------------- remap path


def test_auto_mode_remaps_all_tiers_to_google_only() -> None:
    """Realistic case: user gave only GEMINI_API_KEY but picked 'std'."""
    routing = _std_routing()

    new_routing, warnings = apply_auto_mode(routing, {"google"})

    assert new_routing.tier_high.provider == "google"
    assert new_routing.tier_standard.provider == "google"
    assert new_routing.tier_basic.provider == "google"
    # tier_basic was already google → no warning for it.
    assert len(warnings) == 2
    assert any("Anthropic" in w and "High tier" in w for w in warnings)
    assert any("Openai" in w and "Standard tier" in w for w in warnings)


def test_auto_mode_picks_per_tier_equivalent_model() -> None:
    """Each remap should land on a model that fits the tier role, not random."""
    routing = _std_routing()

    new_routing, _ = apply_auto_mode(routing, {"google"})

    # tier_high (was anthropic flagship) → google flagship reasoning model
    assert new_routing.tier_high.model == "gemini-3-pro"
    # tier_standard (was openai mid) → google mid (Flash)
    assert new_routing.tier_standard.model == "gemini-3.5-flash"


def test_auto_mode_preserves_thinking_budget() -> None:
    routing = RoutingConfig(
        tier_high=TierConfig(
            provider="anthropic", model="claude-opus-4-7", thinking_budget="auto"
        ),
        tier_standard=TierConfig(
            provider="openai", model="gpt-5.4-mini", thinking_budget="medium"
        ),
        tier_basic=TierConfig(
            provider="google", model="gemini-3.1-flash-lite", thinking_budget=None
        ),
    )

    new_routing, _ = apply_auto_mode(routing, {"google"})

    # thinking_budget passes through unchanged; the target provider's
    # translation layer is responsible for honouring it.
    assert new_routing.tier_high.thinking_budget == "auto"
    assert new_routing.tier_standard.thinking_budget == "medium"
    assert new_routing.tier_basic.thinking_budget is None


def test_auto_mode_remaps_only_unavailable_tiers() -> None:
    routing = _std_routing()  # anthropic / openai / google

    new_routing, warnings = apply_auto_mode(routing, {"openai", "google"})

    # Only tier_high (anthropic) was missing.
    assert len(warnings) == 1
    assert "Anthropic" in warnings[0]
    assert new_routing.tier_high.provider in {"openai", "google"}
    assert new_routing.tier_standard.provider == "openai"
    assert new_routing.tier_basic.provider == "google"


def test_auto_mode_warning_format_is_human_readable() -> None:
    routing = _std_routing()

    _, warnings = apply_auto_mode(routing, {"google"})

    # User-visible string carries every concrete element needed to debug a
    # surprising routing decision.
    for w in warnings:
        assert "→ mapped" in w
        assert "tier" in w.lower()
        assert "/" in w  # provider/model
