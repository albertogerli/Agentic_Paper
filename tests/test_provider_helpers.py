"""Provider request-shape + reasoning-budget mapping (no network)."""

from __future__ import annotations

import pytest

from agentic_paper.providers.anthropic_provider import (
    AnthropicProvider,
    _build_tool_from_schema,
    _resolve_thinking as anthropic_resolve,
)
from agentic_paper.providers.google_provider import (
    _budget_to_level,
    _resolve_thinking_config as google_resolve,
)
from agentic_paper.providers.openai_provider import (
    OpenAIProvider,
    _is_reasoning_model,
    _reasoning_effort_from_budget,
)
from agentic_paper.schemas import Review


# ---------------------------------------------------------------- OpenAI helpers


@pytest.mark.parametrize(
    "budget, expected",
    [
        (None, None),
        ("auto", "medium"),
        ("xhigh", "xhigh"),
        ("none", "none"),
        (0, "none"),
        (3_000, "low"),
        (15_000, "medium"),
        (40_000, "high"),
        (80_000, "xhigh"),
        ("invalid", None),
    ],
)
def test_openai_reasoning_effort_mapping(budget, expected) -> None:
    assert _reasoning_effort_from_budget(budget) == expected


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gpt-5.5", True),
        ("gpt-5.4-mini", True),
        ("o3", True),
        ("o4-mini", True),
        ("gpt-4o", False),
        ("gpt-4o-mini", False),
    ],
)
def test_openai_is_reasoning_model(model: str, expected: bool) -> None:
    assert _is_reasoning_model(model) is expected


def test_openai_builds_responses_request_shape() -> None:
    p = OpenAIProvider(api_key="dummy")
    req = p._build_responses_kwargs(
        instructions="sys", message="hi", model="gpt-5.5",
        temperature=1.0, max_tokens=128, thinking_budget="high",
    )
    assert req["model"] == "gpt-5.5"
    assert req["instructions"] == "sys"
    assert req["input"] == "hi"
    assert req["reasoning"] == {"effort": "high"}
    assert "temperature" not in req


def test_openai_builds_chat_request_for_non_reasoning() -> None:
    p = OpenAIProvider(api_key="dummy")
    req = p._build_chat_kwargs(
        instructions="sys", message="hi", model="gpt-4o",
        temperature=0.3, max_tokens=128, thinking_budget=None,
    )
    assert req["messages"][0]["role"] == "system"
    assert req["temperature"] == 0.3
    assert "reasoning_effort" not in req


# ------------------------------------------------------------- Anthropic helpers


def test_anthropic_resolve_adaptive_for_4_6_plus() -> None:
    assert anthropic_resolve("claude-opus-4-7", "auto") == {"type": "adaptive"}
    assert anthropic_resolve("claude-sonnet-4-6", "auto") == {"type": "adaptive"}


def test_anthropic_resolve_legacy_enabled_budget() -> None:
    assert anthropic_resolve("claude-sonnet-4-5", "auto") == {
        "type": "enabled", "budget_tokens": 10000,
    }
    assert anthropic_resolve("claude-haiku-4-5", 5000) == {
        "type": "enabled", "budget_tokens": 5000,
    }


def test_anthropic_resolve_disabled() -> None:
    assert anthropic_resolve("claude-opus-4-7", None) is None
    assert anthropic_resolve("claude-opus-4-7", 0) is None
    assert anthropic_resolve("claude-opus-4-7", "junk") is None


def test_anthropic_force_legacy_flag_overrides_adaptive() -> None:
    # With force_legacy=True (used when forced tool is present), adaptive flips to enabled.
    out = anthropic_resolve("claude-opus-4-7", "auto", force_legacy=True)
    assert out == {"type": "enabled", "budget_tokens": 10000}


def test_anthropic_build_tool_from_pydantic_schema() -> None:
    tool = _build_tool_from_schema(Review)
    assert tool["name"] == "submit_review"
    schema = tool["input_schema"]
    assert schema["type"] == "object"
    assert "summary" in schema["properties"]
    assert "recommendation" in schema["properties"]


def test_anthropic_build_request_with_schema_attaches_tool_choice() -> None:
    p = AnthropicProvider(api_key="dummy")
    req = p._build_request(
        instructions="sys", message="hi", model="claude-opus-4-7",
        temperature=0.4, max_tokens=2048, use_caching=False,
        thinking_budget="auto", response_schema=Review,
    )
    assert req["tool_choice"] == {"type": "tool", "name": "submit_review"}
    # Forced legacy thinking when tool is present
    assert req["thinking"] == {"type": "enabled", "budget_tokens": 10000}
    assert req["temperature"] == 1.0  # enforced by thinking


def test_anthropic_build_request_caching_wraps_system_and_user() -> None:
    p = AnthropicProvider(api_key="dummy")
    req = p._build_request(
        instructions="sys", message="hi", model="claude-haiku-4-5",
        temperature=1.0, max_tokens=2048, use_caching=True,
        thinking_budget=None, response_schema=None,
    )
    assert isinstance(req["system"], list)
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert isinstance(req["messages"][0]["content"], list)
    assert req["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------- Google helpers


@pytest.mark.parametrize(
    "budget, expected",
    [
        ("auto", "high"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("garbage", "medium"),
        (0, "low"),
        (3_000, "low"),
        (15_000, "medium"),
        (40_000, "high"),
    ],
)
def test_google_budget_to_level(budget, expected: str) -> None:
    assert _budget_to_level(budget) == expected


def test_google_resolve_thinking_level_for_3x() -> None:
    cfg = google_resolve("gemini-3.1-flash-lite", "auto")
    assert cfg is not None
    # SDK normalises to ThinkingLevel enum; compare via .value
    level = getattr(cfg.thinking_level, "value", cfg.thinking_level)
    assert str(level).lower() == "high"


def test_google_resolve_legacy_budget_for_2x() -> None:
    cfg = google_resolve("gemini-2.5-flash", 8000)
    assert cfg.thinking_budget == 8000
    assert cfg.thinking_level is None


def test_google_resolve_disabled() -> None:
    assert google_resolve("gemini-3-pro", None) is None
