"""Stub LLM provider — test-only. Returns canned :class:`BaseModel` instances
keyed by schema class, with optional synthetic latency for parallelism tests.

Never imported in production code paths.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Type

from pydantic import BaseModel

from .base import LLMProvider, LLMResponse, ThinkingBudget

logger = logging.getLogger(__name__)


class StubProvider(LLMProvider):
    """Returns canned BaseModel responses configured by the test.

    Usage::

        stub = StubProvider(latency_ms=50)
        stub.set_response(Review, my_canned_review)
        # ...wire into AgentFactory via a ProviderRegistry...
    """

    name = "stub"

    def __init__(
        self,
        *,
        responses: dict[Type[BaseModel], BaseModel] | None = None,
        latency_ms: int = 0,
        text_when_no_schema: str = "stub text",
        usage: dict[str, Any] | None = None,
    ) -> None:
        self.responses: dict[Type[BaseModel], BaseModel] = dict(responses or {})
        self.latency_ms = latency_ms
        self.text_when_no_schema = text_when_no_schema
        self.usage = usage or {"total_tokens": 1, "input_tokens": 1, "output_tokens": 0}
        self.calls: list[dict[str, Any]] = []

    def set_response(self, schema_cls: Type[BaseModel], response: BaseModel) -> None:
        if not isinstance(response, schema_cls):
            raise TypeError(
                f"Canned response must be an instance of {schema_cls.__name__}, "
                f"got {type(response).__name__}"
            )
        self.responses[schema_cls] = response

    def _record(
        self, kind: str, model: str, schema: Type[BaseModel] | None, seed: int | None = None
    ) -> None:
        self.calls.append({
            "kind": kind,
            "model": model,
            "schema": schema.__name__ if schema else None,
            "seed": seed,
        })

    def _resolve(self, response_schema: Type[BaseModel] | None) -> tuple[str, BaseModel | None]:
        if response_schema is None:
            return self.text_when_no_schema, None
        parsed = self.responses.get(response_schema)
        if parsed is None:
            raise ValueError(
                f"StubProvider: no canned response registered for schema "
                f"{response_schema.__name__}. Call .set_response() in the test setup."
            )
        return parsed.model_dump_json(), parsed

    def generate(
        self,
        *,
        instructions: str,
        message: str,
        model: str,
        temperature: float,
        max_tokens: int,
        use_caching: bool = False,
        thinking_budget: ThinkingBudget = None,
        response_schema: Type[BaseModel] | None = None,
        seed: int | None = None,
        on_thinking=None,  # noqa: ARG002 — accepted for interface parity, used in tests
    ) -> LLMResponse:
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000.0)
        self._record("sync", model, response_schema, seed)
        text, parsed = self._resolve(response_schema)
        return LLMResponse(
            text=text, usage=dict(self.usage), raw=None,
            provider=self.name, model=model, parsed=parsed,
        )

    async def agenerate(
        self,
        *,
        instructions: str,
        message: str,
        model: str,
        temperature: float,
        max_tokens: int,
        use_caching: bool = False,
        thinking_budget: ThinkingBudget = None,
        response_schema: Type[BaseModel] | None = None,
        seed: int | None = None,
        on_thinking=None,  # noqa: ARG002 — accepted for interface parity, used in tests
    ) -> LLMResponse:
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000.0)
        self._record("async", model, response_schema, seed)
        text, parsed = self._resolve(response_schema)
        return LLMResponse(
            text=text, usage=dict(self.usage), raw=None,
            provider=self.name, model=model, parsed=parsed,
        )
