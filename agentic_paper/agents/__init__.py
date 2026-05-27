"""Agent factory + per-role agent modules.

Each per-role module exposes:
    KEY              — config / dict key
    NAME             — display name passed to the LLM as identity
    BASE_COMPLEXITY  — float used for routing
    INSTRUCTIONS     — system prompt
"""

from __future__ import annotations

import logging

from ..config import Config
from ..providers import ProviderRegistry
from ..routing import route_agent
from . import (
    ai_origin,
    author_editor_summary,
    citation_validator,
    contradiction,
    coordinator,
    editor,
    ethics,
    hallucination,
    impact,
    literature,
    methodology,
    results,
    revision_assessor,
    statcheck_validator,
    structure,
)
from .base import Agent, AsyncAgent, CachingAsyncAgent

logger = logging.getLogger(__name__)

REVIEWER_MODULES = (
    methodology,
    results,
    literature,
    structure,
    impact,
    contradiction,
    ethics,
    ai_origin,
    hallucination,
    citation_validator,
    statcheck_validator,
    revision_assessor,
)

POST_REVIEW_MODULES = (coordinator, editor, author_editor_summary)

ALL_AGENT_MODULES = REVIEWER_MODULES + POST_REVIEW_MODULES


class AgentFactory:
    """Build the full agent roster, resolving (provider, model, thinking) per agent."""

    def __init__(
        self,
        config: Config,
        paper_complexity_score: float,
        registry: ProviderRegistry,
    ) -> None:
        self.config = config
        self.paper_complexity_score = paper_complexity_score
        self.registry = registry

    @property
    def AGENT_BASE_COMPLEXITY(self) -> dict[str, float]:
        return {m.KEY: m.BASE_COMPLEXITY for m in ALL_AGENT_MODULES}

    def _get_temperature(self, agent_name: str) -> float:
        attr = f"temperature_{agent_name}"
        if hasattr(self.config, attr):
            return float(getattr(self.config, attr))
        return 1.0

    def _build(self, module) -> Agent:
        provider_name, model, thinking, combined = route_agent(
            module.KEY, module.BASE_COMPLEXITY, self.paper_complexity_score, self.config
        )
        try:
            provider = self.registry.get(provider_name)
        except KeyError:
            available = self.registry.names()
            if not available:
                raise RuntimeError(
                    f"No providers registered; agent '{module.KEY}' needs '{provider_name}'."
                ) from None
            fallback = available[0]
            logger.warning(
                "Provider '%s' not registered for agent '%s'; falling back to '%s'.",
                provider_name, module.KEY, fallback,
            )
            provider = self.registry.get(fallback)
            provider_name = fallback
            from ..models import PROVIDER_MODELS
            if fallback in PROVIDER_MODELS and PROVIDER_MODELS[fallback]:
                model = PROVIDER_MODELS[fallback][0].model

        logger.info(
            "Routing agent '%s' → %s/%s (score=%.2f, thinking=%s)",
            module.KEY, provider_name, model, combined, thinking,
        )

        return Agent(
            name=module.NAME,
            instructions=module.INSTRUCTIONS,
            model=model,
            provider=provider,
            temperature=self._get_temperature(module.KEY),
            max_output_tokens=self.config.max_output_tokens,
            use_caching=self.config.use_prompt_caching,
            thinking_budget=thinking,
            schema=getattr(module, "SCHEMA", None),
            seed=self.config.seed,
        )

    def create_all_agents(self) -> dict[str, Agent]:
        return {m.KEY: self._build(m) for m in ALL_AGENT_MODULES}


__all__ = [
    "Agent",
    "AsyncAgent",
    "CachingAsyncAgent",
    "AgentFactory",
    "ALL_AGENT_MODULES",
    "REVIEWER_MODULES",
    "POST_REVIEW_MODULES",
]
