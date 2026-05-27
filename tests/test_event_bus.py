"""RunEventBus — in-memory live-thinking channel used by Anthropic streaming."""

from __future__ import annotations

import threading
import time

from agentic_paper.web.event_bus import LiveEvent, RunEventBus


def test_bus_creates_channels_per_run() -> None:
    bus = RunEventBus()
    a = bus.channel("run-A")
    b = bus.channel("run-A")
    c = bus.channel("run-B")
    assert a is b, "same run_id must reuse the same channel"
    assert a is not c


def test_push_then_drain_yields_events_in_order() -> None:
    bus = RunEventBus()
    bus.push("r", LiveEvent("thinking_start", "agent_x"))
    bus.push("r", LiveEvent("thinking_chunk", "agent_x", "hello"))
    bus.push("r", LiveEvent("thinking_chunk", "agent_x", "world"))
    bus.close("r")

    events = list(bus.channel("r").drain(timeout=0.05))
    kinds = [e.kind for e in events]
    texts = [e.text for e in events]
    assert kinds == ["thinking_start", "thinking_chunk", "thinking_chunk"]
    assert texts == ["", "hello", "world"]


def test_close_terminates_drain_via_sentinel() -> None:
    """A late close() must wake up a parked drain() consumer."""
    bus = RunEventBus()
    out: list[LiveEvent] = []

    def consumer():
        for evt in bus.channel("r").drain(timeout=2.0):
            out.append(evt)

    t = threading.Thread(target=consumer, daemon=True)
    t.start()
    bus.push("r", LiveEvent("thinking_chunk", "x", "z"))
    time.sleep(0.05)
    bus.close("r")
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert len(out) == 1 and out[0].text == "z"


def test_push_after_close_is_dropped_silently() -> None:
    bus = RunEventBus()
    bus.push("r", LiveEvent("thinking_chunk", "a", "first"))
    bus.close("r")
    bus.push("r", LiveEvent("thinking_chunk", "a", "lost"))   # ignored
    events = list(bus.channel("r").drain(timeout=0.05))
    assert len(events) == 1
    assert events[0].text == "first"


def test_drop_removes_channel_after_close() -> None:
    bus = RunEventBus()
    bus.push("r", LiveEvent("thinking_chunk", "x", "y"))
    bus.drop("r")
    # New channel() on the same run_id should be a fresh one.
    ch2 = bus.channel("r")
    assert list(ch2.drain(timeout=0.01)) == []
