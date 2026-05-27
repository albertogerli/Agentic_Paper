"""Provider registry — collects configured providers and exposes them by name."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Callable

from .anthropic_provider import AnthropicProvider
from .base import LLMProvider
from .google_provider import GoogleProvider
from .openai_compat_provider import OpenAICompatProvider
from .openai_provider import OpenAIProvider

if TYPE_CHECKING:
    from ..config import Config, ProviderConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- dispatch tables
#
# Centralising the per-vendor builders + env-var lookups in two tables turns the
# four duplicated `if name == "X": _try_register(...)` blocks of the previous
# implementation into a single typed dispatch step. Knock-on benefit: adding a
# new first-party provider (say "azure") = one entry per table; the resolution
# logic itself never changes.

# Dispatch table: vendor name → callable that builds the adapter from an api_key.
# Used by both the explicit-providers path and the env-var-fallback path.
_VENDOR_FACTORIES: dict[str, Callable[[str], LLMProvider]] = {
    "openai":    lambda key: OpenAIProvider(api_key=key),
    "anthropic": lambda key: AnthropicProvider(api_key=key),
    "google":    lambda key: GoogleProvider(api_key=key),
}

# Provider keys reserved for the first-party vendor adapters.
# Kept as a public-ish module constant for callers that want to ask
# "is X a known vendor or a custom openai_compat endpoint?".
_VENDOR_PROVIDERS: frozenset[str] = frozenset(_VENDOR_FACTORIES)

# Auto-detect fallback: vendor name → ordered tuple of env-var names. The first
# non-empty value wins. OpenAI is handled separately via the legacy
# ``config.api_key`` path (which itself defaults to ``OPENAI_API_KEY``).
_ENV_FALLBACKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("anthropic", ("ANTHROPIC_API_KEY",)),
    ("google",    ("GOOGLE_API_KEY", "GEMINI_API_KEY")),
)


class ProviderRegistry:
    """In-memory map of provider name → :class:`LLMProvider` instance."""

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}

    def register(self, name: str, provider: LLMProvider) -> None:
        if name in self._providers:
            logger.debug("Replacing provider '%s' in registry", name)
        self._providers[name] = provider

    def get(self, name: str) -> LLMProvider:
        if name not in self._providers:
            raise KeyError(
                f"Provider '{name}' not registered. Available: {sorted(self._providers)}"
            )
        return self._providers[name]

    def has(self, name: str) -> bool:
        return name in self._providers

    def names(self) -> list[str]:
        return sorted(self._providers)

    def items(self):
        return self._providers.items()

    def __len__(self) -> int:
        return len(self._providers)

    def __bool__(self) -> bool:
        return bool(self._providers)


# ---------------------------------------------------------------- small helpers


def _resolve_api_key(env_var: str | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    if env_var:
        return os.environ.get(env_var, "")
    return ""


def _try_register(
    registry: ProviderRegistry, name: str, factory: Callable[[], LLMProvider],
) -> None:
    """Call ``factory()`` and register; log + skip on ImportError/ValueError.

    Kept deliberately small + tolerant: missing SDKs (e.g. anthropic not
    installed) or bad config values must never abort the whole registry build.
    """
    try:
        provider = factory()
    except ImportError as e:
        logger.warning("Provider '%s' skipped — SDK not installed (%s)", name, e)
        return
    except Exception as e:
        logger.warning("Provider '%s' skipped — %s", name, e)
        return
    registry.register(name, provider)
    logger.info("Registered provider: %s", name)


def _first_env_key(env_vars: tuple[str, ...]) -> str:
    """Return the first non-empty value among ``env_vars`` in os.environ, else ''."""
    for var in env_vars:
        value = os.environ.get(var, "")
        if value:
            return value
    return ""


# ---------------------------------------------------------------- pass helpers
#
# Three small helpers, one per resolution layer, in priority order. Each is
# safe to call standalone and an early-return tour: nothing nested.


def _register_one_explicit(
    registry: ProviderRegistry, name: str, pc: "ProviderConfig",
) -> None:
    """Register a single entry from ``config.providers``.

    Known vendor names (``openai`` / ``anthropic`` / ``google``) go through
    :data:`_VENDOR_FACTORIES`. Any other name is treated as a custom
    OpenAI-compatible endpoint and requires a ``base_url``.
    """
    api_key = _resolve_api_key(pc.api_key_env, pc.api_key)

    factory = _VENDOR_FACTORIES.get(name)
    if factory is not None:
        # First-party vendor branch.
        if not api_key:
            return  # silently skip — no key, no registration
        _try_register(registry, name, lambda f=factory, k=api_key: f(k))
        return

    # Treat anything else as an OpenAI-compatible custom endpoint.
    if not pc.base_url:
        logger.warning(
            "Provider '%s' has no base_url; treat as OpenAI-compat — skipping", name,
        )
        return
    _try_register(
        registry,
        name,
        lambda key=api_key, base_url=pc.base_url, name=name: OpenAICompatProvider(
            api_key=key, base_url=base_url, name=name,
        ),
    )


def _register_explicit_providers(
    registry: ProviderRegistry, config: "Config",
) -> None:
    """Priority 1: walk ``config.providers`` and register each declared entry."""
    for name, pc in config.providers.items():
        _register_one_explicit(registry, name, pc)


def _register_legacy_openai(
    registry: ProviderRegistry, config: "Config",
) -> None:
    """Priority 2: legacy top-level ``config.api_key`` (OpenAI-only fallback).

    Preserves the v1 single-key behaviour for users who never adopted the
    ``providers:`` block in their config.yaml. No-ops if OpenAI is already
    registered or if no legacy key is set.
    """
    if registry.has("openai"):
        return
    if not config.api_key:
        return
    _try_register(
        registry, "openai",
        lambda key=config.api_key: OpenAIProvider(api_key=key),
    )


def _register_env_fallbacks(registry: ProviderRegistry) -> None:
    """Priority 3: auto-detect vendor env vars (ANTHROPIC_API_KEY / GOOGLE_API_KEY).

    Only fills in providers that nobody upstream has already registered, so a
    user-supplied ``providers:`` entry always wins over an env-var auto-detect.
    """
    for name, env_vars in _ENV_FALLBACKS:
        if registry.has(name):
            continue
        key = _first_env_key(env_vars)
        if not key:
            continue
        factory = _VENDOR_FACTORIES[name]
        _try_register(registry, name, lambda f=factory, k=key: f(k))


# ---------------------------------------------------------------- public entry


def build_registry(config: "Config") -> ProviderRegistry:
    """Build a :class:`ProviderRegistry` from a :class:`Config`.

    Recognized sources, in order:
        1. ``config.providers`` explicit block (most specific).
        2. Legacy top-level ``config.api_key`` (OpenAI fallback).
        3. ``ANTHROPIC_API_KEY`` / ``GOOGLE_API_KEY`` env vars (auto-detect).
    """
    registry = ProviderRegistry()
    _register_explicit_providers(registry, config)
    _register_legacy_openai(registry, config)
    _register_env_fallbacks(registry)

    if not registry:
        logger.warning(
            "No providers registered. Set OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY "
            "or supply a `providers:` block in config.yaml."
        )

    return registry
