"""Connectivity smoke check.

Run as: ``python -m agentic_paper.providers.smoke [--config CONFIG]``

For every registered provider, send a 1-token request and report
``{provider, model, latency_ms, ok}``. Used in CI and by humans before
kicking off a real review run.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from ..config import Config, setup_logging
from ..routing import get_routing
from . import ProviderRegistry, build_registry

logger = logging.getLogger(__name__)

# Cheapest model per provider for the smoke ping; used when routing
# doesn't pin a model for that provider.
_DEFAULT_PROBE_MODEL = {
    "openai": "gpt-5.4-mini",
    "anthropic": "claude-haiku-4-5",
    "google": "gemini-3.1-flash-lite",
}


def _probe_model(registry: ProviderRegistry, config: Config, provider_name: str) -> str:
    """Pick a small model to ping this provider with."""
    routing = get_routing(config)
    for tier in (routing.tier_basic, routing.tier_standard, routing.tier_high):
        if tier.provider == provider_name and tier.model:
            return tier.model
    return _DEFAULT_PROBE_MODEL.get(provider_name, "")


def _ping_one(name: str, provider, model: str) -> dict:
    start = time.time()
    try:
        if not model:
            return {
                "provider": name,
                "model": "?",
                "ok": False,
                "latency_ms": None,
                "error": "no probe model resolved (set a routing tier or _DEFAULT_PROBE_MODEL)",
            }
        # max_tokens kept generous (256) because reasoning / adaptive thinking
        # models consume part of the budget on internal reasoning tokens; a
        # tiny cap can return an empty text completion.
        response = provider.generate(
            instructions="You are a helpful assistant.",
            message="Reply with the single word: ok",
            model=model,
            temperature=1.0,
            max_tokens=256,
        )
        elapsed = int((time.time() - start) * 1000)
        return {
            "provider": name,
            "model": model,
            "ok": True,
            "latency_ms": elapsed,
            "text": (response.text or "").strip()[:40],
        }
    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        return {
            "provider": name,
            "model": model,
            "ok": False,
            "latency_ms": elapsed,
            "error": str(e)[:200],
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Provider connectivity smoke check.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    config = Config.from_yaml(args.config)
    registry = build_registry(config)

    if not registry:
        print("No providers configured.")
        print("Set OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY in your environment,")
        print("or define a `providers:` block in config.yaml.")
        return 1

    print(f"{'Provider':<18} {'Model':<32} {'Latency':<10} Status")
    print("-" * 80)

    overall_ok = True
    for name, provider in sorted(registry.items()):
        model = _probe_model(registry, config, name)
        result = _ping_one(name, provider, model)
        if result["ok"]:
            print(
                f"{result['provider']:<18} "
                f"{result['model']:<32} "
                f"{str(result['latency_ms']) + 'ms':<10} "
                f"ok  ({result.get('text', '')!r})"
            )
        else:
            overall_ok = False
            latency = f"{result['latency_ms']}ms" if result["latency_ms"] is not None else "—"
            print(
                f"{result['provider']:<18} "
                f"{result['model']:<32} "
                f"{latency:<10} "
                f"FAIL  {result.get('error', '')}"
            )

    return 0 if overall_ok else 2


if __name__ == "__main__":
    sys.exit(main())
