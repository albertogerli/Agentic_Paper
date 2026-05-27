"""Profile presets (max / std / quick) — routing + CLI + web."""

from __future__ import annotations

from agentic_paper.routing import PROFILE_NAMES, PROFILES, get_profile


def test_three_profiles_exist() -> None:
    assert set(PROFILES.keys()) == set(PROFILE_NAMES)
    assert PROFILE_NAMES == ("max", "std", "quick")


def test_each_profile_has_all_three_tiers() -> None:
    for name, cfg in PROFILES.items():
        assert cfg.tier_high.model, f"{name} missing tier_high.model"
        assert cfg.tier_standard.model, f"{name} missing tier_standard.model"
        assert cfg.tier_basic.model, f"{name} missing tier_basic.model"
        for tier_name in ("tier_high", "tier_standard", "tier_basic"):
            tier = getattr(cfg, tier_name)
            assert tier.provider in {"openai", "anthropic", "google"}


def test_profile_max_uses_flagship_models() -> None:
    p = PROFILES["max"]
    assert p.tier_high.model == "claude-opus-4-7"
    assert p.tier_high.thinking_budget == "auto"
    assert p.tier_standard.model == "gpt-5.5"
    assert p.tier_basic.model == "gemini-3-pro"


def test_profile_quick_uses_cheapest_per_provider() -> None:
    p = PROFILES["quick"]
    assert p.tier_high.model == "claude-haiku-4-5"
    assert p.tier_high.thinking_budget is None       # no thinking on quick
    assert p.tier_standard.model == "gpt-5.4-nano"
    assert p.tier_basic.model == "gemini-3.1-flash-lite"


def test_get_profile_returns_deep_copy() -> None:
    a = get_profile("max")
    b = get_profile("max")
    assert a is not b
    a.tier_high.model = "tampered"
    assert PROFILES["max"].tier_high.model == "claude-opus-4-7"
    assert b.tier_high.model == "claude-opus-4-7"


def test_get_profile_unknown_returns_none() -> None:
    assert get_profile("") is None
    assert get_profile(None) is None  # type: ignore[arg-type]
    assert get_profile("super-deluxe") is None


def test_get_profile_is_case_insensitive() -> None:
    assert get_profile("MAX") is not None
    assert get_profile("  std  ") is not None
