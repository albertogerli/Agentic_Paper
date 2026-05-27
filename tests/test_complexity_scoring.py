"""Complexity scoring + routing determinism tests."""

from __future__ import annotations

import asyncio

import pytest

from agentic_paper.agents import ALL_AGENT_MODULES
from agentic_paper.config import Config, RoutingConfig, TierConfig
from agentic_paper.paper import assess_paper_complexity
from agentic_paper.routing import route_agent


def test_agent_base_complexity_in_unit_interval() -> None:
    for module in ALL_AGENT_MODULES:
        score = module.BASE_COMPLEXITY
        assert 0.0 <= score <= 1.0, f"{module.KEY}: {score} not in [0, 1]"


def test_assess_paper_complexity_returns_default_when_no_api_key() -> None:
    # No key → guaranteed default 0.5, no network call.
    cfg = Config(api_key="")
    text = "Hello world. " * 100
    score_a = asyncio.run(assess_paper_complexity(text, cfg))
    score_b = asyncio.run(assess_paper_complexity(text, cfg))
    assert score_a == 0.5
    assert score_b == 0.5, "must be deterministic without an API key"
    assert 0.0 <= score_a <= 1.0


@pytest.mark.parametrize(
    "agent_base, paper_score, expected_tier_model",
    [
        # combined = paper*0.4 + base*0.6
        (0.9, 0.5,   "high-model"),       # 0.74 → high
        (0.4, 0.5,   "basic-model"),       # 0.44 → basic
        (0.8, 0.3,   "standard-model"),    # 0.60 → standard
        (1.0, 0.5,   "high-model"),       # 0.80 → high
        (0.6, 0.4,   "standard-model"),    # 0.52 → standard
        (0.1, 0.1,   "basic-model"),       # 0.10 → basic
    ],
)
def test_route_agent_picks_expected_tier(
    agent_base: float, paper_score: float, expected_tier_model: str
) -> None:
    cfg = Config(
        api_key="dummy",
        routing=RoutingConfig(
            tier_high=TierConfig(provider="anthropic", model="high-model"),
            tier_standard=TierConfig(provider="openai", model="standard-model"),
            tier_basic=TierConfig(provider="google", model="basic-model"),
        ),
    )
    provider, model, thinking, combined = route_agent(
        agent_name="x",
        base_complexity=agent_base,
        paper_complexity_score=paper_score,
        config=cfg,
    )
    assert model == expected_tier_model, f"combined={combined:.3f}"


def test_route_agent_is_deterministic() -> None:
    cfg = Config(api_key="dummy")  # legacy fallback → all-openai
    out1 = route_agent("methodology", 0.9, 0.5, cfg)
    out2 = route_agent("methodology", 0.9, 0.5, cfg)
    assert out1 == out2


def test_route_agent_combined_score_in_range() -> None:
    cfg = Config(api_key="dummy")
    for base in (0.0, 0.5, 1.0):
        for paper in (0.0, 0.5, 1.0):
            _, _, _, combined = route_agent("x", base, paper, cfg)
            assert 0.0 <= combined <= 1.0
