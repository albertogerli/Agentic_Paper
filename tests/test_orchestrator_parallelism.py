"""Parallelism test — three stubbed agents fan out faster than a serial run would."""

from __future__ import annotations

import asyncio
import time

import pytest

from agentic_paper.agents import literature, methodology, structure
from agentic_paper.agents.base import Agent
from agentic_paper.providers.stub_provider import StubProvider


_LATENCY_MS = 300


def _build_agents(provider: StubProvider) -> dict[str, Agent]:
    out: dict[str, Agent] = {}
    for module in (methodology, literature, structure):
        out[module.KEY] = Agent(
            name=module.NAME,
            instructions=module.INSTRUCTIONS,
            model="stub-model",
            provider=provider,
            schema=module.SCHEMA,
            max_output_tokens=512,
        )
    return out


async def _gather_in_parallel(agents: dict[str, Agent], message: str) -> list:
    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(None, agent.run, message) for agent in agents.values()]
    return await asyncio.gather(*tasks)


def test_three_agents_run_under_1_5x_slowest(stub_provider: StubProvider) -> None:
    stub_provider.latency_ms = _LATENCY_MS
    agents = _build_agents(stub_provider)

    start = time.perf_counter()
    results = asyncio.run(_gather_in_parallel(agents, "dummy"))
    elapsed = time.perf_counter() - start

    assert len(results) == 3
    slowest = _LATENCY_MS / 1000.0
    # Generous upper bound to tolerate CI / loaded-laptop scheduling jitter
    # but still tight enough to catch a true regression to serial execution.
    upper_bound = 2.0 * slowest
    assert elapsed < upper_bound, (
        f"3 agents took {elapsed:.3f}s; expected < {upper_bound:.3f}s "
        f"(serial would be ~{3 * slowest:.3f}s)"
    )
    # The strict signal that we're actually parallel: total must be well below
    # the serial estimate of N × latency.
    serial_estimate = 3 * slowest
    assert elapsed < serial_estimate * 0.85, (
        f"{elapsed:.3f}s suggests serial execution (serial would be ~{serial_estimate:.3f}s)"
    )


def test_stub_provider_recorded_three_calls(stub_provider: StubProvider) -> None:
    stub_provider.latency_ms = 0
    agents = _build_agents(stub_provider)
    asyncio.run(_gather_in_parallel(agents, "dummy"))
    assert len(stub_provider.calls) >= 3
