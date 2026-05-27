"""Drive every agent against the StubProvider and assert schema validity."""

from __future__ import annotations

from typing import Type

import pytest
from pydantic import BaseModel

from agentic_paper.agents import ALL_AGENT_MODULES
from agentic_paper.agents.base import Agent
from agentic_paper.providers.stub_provider import StubProvider
from agentic_paper.schemas import (
    AuthorEditorSummary,
    CoordinatorAssessment,
    EditorDecision,
    Review,
)


def _build_agent(module, provider: StubProvider) -> Agent:
    return Agent(
        name=module.NAME,
        instructions=module.INSTRUCTIONS,
        model="stub-model",
        provider=provider,
        temperature=1.0,
        max_output_tokens=1024,
        use_caching=False,
        thinking_budget=None,
        schema=module.SCHEMA,
    )


@pytest.mark.parametrize("module", ALL_AGENT_MODULES, ids=lambda m: m.KEY)
def test_every_agent_returns_its_schema(module, stub_provider: StubProvider) -> None:
    agent = _build_agent(module, stub_provider)
    result = agent.run("dummy paper text")
    assert isinstance(result, BaseModel), f"{module.KEY} returned {type(result).__name__}"
    assert isinstance(result, module.SCHEMA), (
        f"{module.KEY} returned {type(result).__name__}, expected {module.SCHEMA.__name__}"
    )


def test_review_schema_round_trips_through_json(stub_provider: StubProvider) -> None:
    from agentic_paper.agents import methodology
    agent = _build_agent(methodology, stub_provider)
    result = agent.run("paper")
    assert isinstance(result, Review)
    blob = result.model_dump_json()
    again = Review.model_validate_json(blob)
    assert again == result


def test_stub_provider_records_schema_per_call(stub_provider: StubProvider) -> None:
    from agentic_paper.agents import methodology, coordinator, editor, author_editor_summary

    schemas_seen: list[Type[BaseModel]] = [Review, CoordinatorAssessment, EditorDecision, AuthorEditorSummary]
    for module, expected in zip(
        (methodology, coordinator, editor, author_editor_summary), schemas_seen
    ):
        _build_agent(module, stub_provider).run("paper")
    seen = [call["schema"] for call in stub_provider.calls]
    assert seen[-4:] == ["Review", "CoordinatorAssessment", "EditorDecision", "AuthorEditorSummary"]


def test_agent_without_schema_returns_text() -> None:
    provider = StubProvider(text_when_no_schema="raw text path")
    agent = Agent(
        name="plain", instructions="be brief", model="stub-model",
        provider=provider, schema=None,
    )
    result = agent.run("hi")
    assert isinstance(result, str)
    assert result == "raw text path"
