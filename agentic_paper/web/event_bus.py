"""Per-run in-memory event bus for live ``thinking`` deltas.

The Anthropic Messages streaming API emits ``thinking_delta`` events while
adaptive / extended thinking is enabled. We pipe those through this bus so
the SSE endpoint can fan them out to the browser in real time, alongside the
already-existing audit.jsonl tail.

Out of scope (for now):
    * OpenAI Responses reasoning summaries — different SDK surface.
    * Gemini thinking — needs different streaming hooks.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class LiveEvent:
    kind: str          # "thinking_start" | "thinking_chunk" | "thinking_end" | "info"
    agent: str
    text: str = ""


class _RunChannel:
    """Single-producer, multi-consumer-friendly bounded queue with a sentinel."""

    SENTINEL = object()

    def __init__(self, maxsize: int = 2000) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._closed = False
        self._lock = threading.Lock()

    def push(self, evt: LiveEvent) -> None:
        if self._closed:
            return
        try:
            self._q.put_nowait(evt)
        except queue.Full:
            # Drop on overflow rather than block — the SSE consumer may have left.
            logger.debug("RunChannel overflow; dropping event %s", evt.kind)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._q.put_nowait(self.SENTINEL)
            except queue.Full:
                pass

    def drain(self, timeout: float = 0.5) -> Iterator[LiveEvent]:
        """Yield available events. Stops yielding when the sentinel arrives.
        Blocks up to ``timeout`` seconds per get; caller loops as needed."""
        while True:
            try:
                item = self._q.get(timeout=timeout)
            except queue.Empty:
                return
            if item is self.SENTINEL:
                return
            yield item  # type: ignore[misc]


class RunEventBus:
    """Map of ``run_id -> _RunChannel``. Cleared by :meth:`drop` on completion."""

    def __init__(self) -> None:
        self._channels: dict[str, _RunChannel] = {}
        self._lock = threading.Lock()

    def channel(self, run_id: str) -> _RunChannel:
        with self._lock:
            ch = self._channels.get(run_id)
            if ch is None:
                ch = _RunChannel()
                self._channels[run_id] = ch
            return ch

    def push(self, run_id: str, evt: LiveEvent) -> None:
        self.channel(run_id).push(evt)

    def close(self, run_id: str) -> None:
        with self._lock:
            ch = self._channels.get(run_id)
            if ch is not None:
                ch.close()

    def drop(self, run_id: str) -> None:
        with self._lock:
            ch = self._channels.pop(run_id, None)
            if ch is not None:
                ch.close()
