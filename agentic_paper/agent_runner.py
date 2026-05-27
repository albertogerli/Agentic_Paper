"""Concurrent fan-out of reviewer agents.

The orchestrator used to drive ``asyncio.gather`` + ``run_in_executor``
directly, mixing business logic (who to call, what to save) with concurrency
mechanics (task lifecycle, exception unwrap, thread pool routing for sync
agents). This module isolates the latter behind one small surface:

    runner = ConcurrentAgentRunner(agents=..., storage=..., audit=..., event_bus=..., run_id=...)
    reviews, errors = await runner.run_batch(["methodology", "results", ...], initial_message)

``run_batch`` always returns a ``(reviews, errors)`` pair with the same shape
the orchestrator emitted before — the keys are agent identifiers, the values
are :class:`AnnotatedReview` or human-readable error strings. No exception is
ever propagated out of a batch; per-agent failures land in ``errors``.

The internal helper :func:`_run_single_safe` wraps each agent invocation in
its own try/except, so :func:`run_batch` itself never has to call
``asyncio.gather(..., return_exceptions=True)``. That makes the concurrent
queue future-friendly for per-agent timeouts, partial retries, or a swap to
an external broker (e.g. Celery) — each agent's lifecycle is already a
self-contained coroutine.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from tenacity import RetryError

from .agents import Agent
from .agents.base import AsyncAgent
from .audit import AuditLogger
from .schemas import AnnotatedReview, Review
from .storage import StorageProvider

logger = logging.getLogger(__name__)


def format_exception(exc: BaseException) -> str:
    """Surface the most useful message for a (possibly nested) exception.

    Tenacity wraps the final failure in a ``RetryError`` that stringifies as
    ``RetryError[<Future ...>]`` and hides the root cause. We unwrap it to
    show the original exception's class + message in the report and the SSE
    feed.
    """
    if isinstance(exc, RetryError):
        try:
            exc.last_attempt.result()
        except BaseException as inner:  # noqa: BLE001 — we want any subclass
            return format_exception(inner)
        return "RetryError (no inner exception captured)"
    msg = str(exc) or exc.__class__.__name__
    return f"{type(exc).__name__}: {msg}"


def annotate(annotated_cls, raw: BaseModel, agent: Agent) -> BaseModel:
    """Wrap a bare LLM output in its Annotated* counterpart with provenance."""
    return annotated_cls(
        agent=agent.name,
        model_used=agent.model,
        **raw.model_dump(),
    )


@dataclass
class _RunResult:
    """Outcome of one agent invocation. Exactly one of ``review``/``error`` is set."""

    agent_key: str
    review: Review | None
    error: str | None


class ConcurrentAgentRunner:
    """Fans out a list of agent keys, returning ``(reviews, errors)``.

    The runner owns:
        * the ``agents`` dict (instantiated by ``AgentFactory``),
        * the ``storage`` backend (so it can persist each saved review),
        * the ``audit`` logger (passed through to every agent call),
        * the optional ``event_bus`` (for live thinking-stream events).

    It does NOT own coordinator / editor / summary execution — those remain
    in the orchestrator because they're serial and depend on each other.
    """

    _FLUSH_AT = 120  # characters before a thinking buffer is flushed to SSE

    def __init__(
        self,
        *,
        agents: dict[str, Agent],
        storage: StorageProvider,
        audit: AuditLogger,
        event_bus: Any = None,
        run_id: str,
    ) -> None:
        self.agents = agents
        self.storage = storage
        self.audit = audit
        self.event_bus = event_bus
        self.run_id = run_id

    # ------------------------------------------------------------------ public

    async def run_batch(
        self, agent_keys: list[str], message: str,
    ) -> tuple[dict[str, AnnotatedReview], dict[str, str]]:
        """Run every agent in ``agent_keys`` concurrently.

        Missing keys (no entry in ``self.agents``) are silently skipped, just
        like the legacy ``_batch_process_agents`` behaviour.
        """
        valid_keys = [k for k in agent_keys if k in self.agents]
        if not valid_keys:
            return {}, {}

        tasks = [
            asyncio.create_task(self._run_single_safe(k, message))
            for k in valid_keys
        ]
        results = await asyncio.gather(*tasks)

        reviews: dict[str, AnnotatedReview] = {}
        errors: dict[str, str] = {}
        for r in results:
            if r.error is not None:
                errors[r.agent_key] = r.error
                continue
            agent = self.agents[r.agent_key]
            annotated = annotate(AnnotatedReview, r.review, agent)
            reviews[r.agent_key] = annotated
            self.storage.save_review(r.agent_key, annotated.model_dump_json(indent=2))
        return reviews, errors

    # ------------------------------------------------------------------ internals

    async def _run_single_safe(self, agent_key: str, message: str) -> _RunResult:
        """Invoke one agent and translate every failure into a ``_RunResult``.

        Guarantees: this coroutine never raises. Any exception from the agent
        — including ``RetryError`` and unexpected return types — is captured
        into ``_RunResult.error`` so the gather() in :meth:`run_batch` can
        stay simple and exception-free.
        """
        agent = self.agents[agent_key]
        try:
            callback = self._make_thinking_callback(agent_key) if self.event_bus else None
            if isinstance(agent, AsyncAgent):
                result = await agent.arun(message, audit=self.audit, on_thinking=callback)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: agent.run(message, audit=self.audit, on_thinking=callback),
                )
        except Exception as e:  # noqa: BLE001 — convert *any* failure to a message
            err_msg = format_exception(e)
            logger.error("Error in agent %s: %s", agent_key, err_msg)
            return _RunResult(agent_key=agent_key, review=None, error=err_msg)

        if not isinstance(result, Review):
            err_msg = f"unexpected return type {type(result).__name__}"
            logger.error("Agent %s returned non-Review object: %r", agent_key, type(result))
            return _RunResult(agent_key=agent_key, review=None, error=err_msg)

        return _RunResult(agent_key=agent_key, review=result, error=None)

    def _make_thinking_callback(self, agent_key: str):
        """Build a thinking-delta forwarder bound to this run + agent.

        Each chunk is buffered up to a handful of words to avoid flooding the
        SSE channel with single-token events. We flush on whitespace breaks
        or every ~120 chars, whichever comes first.
        """
        if self.event_bus is None:
            return None
        # Local import: avoid hard dependency on the web layer from the
        # runner (which is also used by the CLI).
        from .web.event_bus import LiveEvent

        bus = self.event_bus
        run_id = self.run_id
        flush_at = self._FLUSH_AT
        bus.push(run_id, LiveEvent(kind="thinking_start", agent=agent_key))

        buf: list[str] = []
        buf_len = [0]

        def flush() -> None:
            if not buf:
                return
            chunk = "".join(buf)
            buf.clear()
            buf_len[0] = 0
            bus.push(run_id, LiveEvent(kind="thinking_chunk", agent=agent_key, text=chunk))

        def cb(text: str) -> None:
            buf.append(text)
            buf_len[0] += len(text)
            if buf_len[0] >= flush_at or "\n" in text or text.endswith((".", "!", "?")):
                flush()

        return cb


__all__ = ["ConcurrentAgentRunner", "format_exception", "annotate"]
