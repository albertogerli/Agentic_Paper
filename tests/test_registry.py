"""ProviderRegistry + build_registry tests (no real network calls)."""

from __future__ import annotations

import pytest

from agentic_paper.config import Config, ProviderConfig
from agentic_paper.providers import build_registry
from agentic_paper.providers.registry import ProviderRegistry
from agentic_paper.providers.stub_provider import StubProvider


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(k, raising=False)


def test_empty_registry_is_falsy_and_len_zero() -> None:
    r = ProviderRegistry()
    assert bool(r) is False
    assert len(r) == 0
    assert r.names() == []


def test_register_get_has_names() -> None:
    r = ProviderRegistry()
    p = StubProvider()
    r.register("stub", p)
    assert r.has("stub") is True
    assert r.has("missing") is False
    assert r.get("stub") is p
    assert r.names() == ["stub"]
    assert bool(r) is True
    assert len(r) == 1


def test_get_missing_provider_raises() -> None:
    r = ProviderRegistry()
    with pytest.raises(KeyError):
        r.get("nope")


def test_build_registry_with_no_keys_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    cfg = Config(api_key="")
    r = build_registry(cfg)
    assert len(r) == 0


def test_build_registry_picks_up_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-google")
    cfg = Config(api_key="dummy-openai")
    r = build_registry(cfg)
    assert set(r.names()) == {"openai", "anthropic", "google"}


def test_build_registry_explicit_providers_block(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("MY_OPENAI", "k1")
    cfg = Config(
        api_key="",
        providers={
            "openai": ProviderConfig(api_key_env="MY_OPENAI"),
        },
    )
    r = build_registry(cfg)
    assert r.has("openai")


def test_build_registry_openai_compat_via_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    cfg = Config(
        api_key="",
        providers={
            "ollama": ProviderConfig(api_key="ollama", base_url="http://localhost:11434/v1"),
        },
    )
    r = build_registry(cfg)
    assert r.has("ollama")
    assert r.get("ollama").name == "ollama"


def test_build_registry_skips_provider_without_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    cfg = Config(
        api_key="",
        providers={
            "weirdo": ProviderConfig(api_key="abc"),  # neither vendor name nor base_url
        },
    )
    r = build_registry(cfg)
    assert not r.has("weirdo")
