"""Pydantic-based configuration + logging setup."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

ThinkingBudgetField = Union[int, str, None]


def setup_logging(log_level: str = "INFO", log_file: str = "paper_review_system.log") -> None:
    """Configure the agentic_paper logger tree.

    Console handler honours ``log_level``; file handler always logs DEBUG.
    Idempotent: safe to call multiple times (e.g. CLI re-applies after parsing args).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    root = logging.getLogger("agentic_paper")
    root.setLevel(level)

    legacy = logging.getLogger("paper_review_system")
    legacy.setLevel(level)

    if root.handlers:
        for handler in root.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(level)
        return

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)
    legacy.addHandler(console)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    legacy.addHandler(file_handler)


class ProviderConfig(BaseModel):
    """Per-provider connection settings."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None


class TierConfig(BaseModel):
    """Routing target for a single complexity tier."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    provider: str = "openai"
    model: str = "gpt-5.4-mini"
    thinking_budget: ThinkingBudgetField = None


class RoutingConfig(BaseModel):
    """Tier-based routing: low / standard / high complexity → (provider, model)."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    tier_high: TierConfig = Field(default_factory=lambda: TierConfig(provider="openai", model="gpt-5.4-mini"))
    tier_standard: TierConfig = Field(default_factory=lambda: TierConfig(provider="openai", model="gpt-5.4-mini"))
    tier_basic: TierConfig = Field(default_factory=lambda: TierConfig(provider="openai", model="gpt-5.4-mini"))


class AgentOverride(BaseModel):
    """Optional per-agent overrides; unset fields fall back to the tier default."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    provider: str | None = None
    model: str | None = None
    thinking_budget: ThinkingBudgetField = None


class Config(BaseModel):
    """Centralized configuration for the system.

    Field names preserved for backward compatibility with v1 config.yaml files.
    Unknown YAML keys are tolerated (extra='ignore') rather than crashing.

    Multi-provider extensions (Prompt 3):
        * ``providers`` — explicit per-provider connection config.
        * ``routing``   — tier→(provider, model, thinking_budget) mapping.
        * ``agents``    — per-agent overrides.
    """

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    # Legacy OpenAI key — still honoured for back-compat.
    api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))

    # Legacy tier model names (used when ``routing`` is not provided).
    model_powerful: str = "gpt-5.4-mini"
    model_standard: str = "gpt-5.4-mini"
    model_basic: str = "gpt-5.4-mini"

    output_dir: str = "output_paper_review"
    max_parallel_agents: int = 6
    agent_timeout: int = 600

    # Per-agent temperatures.
    temperature_methodology: float = 1.0
    temperature_results: float = 1.0
    temperature_literature: float = 1.0
    temperature_structure: float = 1.0
    temperature_impact: float = 1.0
    temperature_contradiction: float = 1.0
    temperature_ethics: float = 1.0
    temperature_coordinator: float = 1.0
    temperature_editor: float = 1.0
    temperature_ai_origin: float = 1.0
    temperature_hallucination: float = 1.0

    max_output_tokens: int = 16000
    use_prompt_caching: bool = True

    # Carried for backward CLI compatibility; not currently consumed.
    reasoning_effort: str = "medium"

    # ------------------ Multi-provider extensions ------------------

    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    routing: RoutingConfig | None = None
    agents: dict[str, AgentOverride] = Field(default_factory=dict)

    # ------------------ Reproducibility ------------------
    seed: int | None = None
    """When set, propagated to every provider that supports deterministic
    sampling (OpenAI, Google). Anthropic ignores it (logged once)."""

    # ------------------ External enrichment ------------------
    enrich_with_citations: bool = True
    """When True, the orchestrator queries OpenAlex (free, no API key) to
    validate every citation extracted from the paper and attaches the report
    to the initial reviewer message. Disable for offline-only runs."""

    enrich_with_statcheck: bool = True
    """When True, the orchestrator runs the R package ``statcheck`` against
    the paper text to flag numerical reporting errors in p-values. Requires
    R + the ``statcheck`` package installed on the host. Disable to skip the
    R subprocess entirely."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        try:
            with open(path) as f:
                data: dict[str, Any] = yaml.safe_load(f) or {}
            return cls(**data)
        except FileNotFoundError:
            logger.warning("Config file %s not found, using defaults", path)
            return cls()
        except Exception as e:
            logger.error("Error loading config: %s", e)
            return cls()

    def validate(self) -> bool:
        """Validate the configuration.

        Requires *some* provider to be reachable: either the legacy
        ``api_key`` field is set, an explicit ``providers:`` block is
        provided, or a vendor env var (ANTHROPIC_API_KEY, GOOGLE_API_KEY) is exported.
        """
        if self.api_key:
            return True
        if self.providers:
            return True
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            return True
        raise ValueError(
            "No API key configured. Set OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY, "
            "or provide a `providers:` block in config.yaml."
        )
