"""Abstract LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Type, Union

from pydantic import BaseModel

ThinkingCallback = Callable[[str], None]
"""Callable invoked with each incremental thinking chunk during streaming."""

ThinkingBudget = Union[int, str, None]
"""Reasoning/thinking budget hint:
* ``None`` — disabled (default for non-reasoning models).
* ``int`` — token budget (provider maps to its own scheme).
* ``"auto"`` — let the provider pick a sensible default.
* OpenAI: also accepts ``"low" / "medium" / "high" / "xhigh" / "none"``.
"""


@dataclass
class LLMResponse:
    """Normalized response shape returned by every :class:`LLMProvider`."""

    text: str
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    provider: str = ""
    model: str = ""
    parsed: BaseModel | None = None
    """When ``response_schema`` was passed to ``generate``/``agenerate``, the
    provider validates the model's output against the schema and stores the
    result here. ``text`` is still populated where possible for logging."""


class LLMProvider(ABC):
    """Base class for LLM provider adapters."""

    name: str = "base"

    @abstractmethod
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
        on_thinking: "ThinkingCallback | None" = None,
    ) -> LLMResponse:
        """Synchronous single-turn generation.

        If ``response_schema`` is supplied, the provider asks the model to emit
        a structured object matching the schema and returns it via
        :attr:`LLMResponse.parsed`.

        ``seed`` is forwarded to providers that support deterministic sampling
        (OpenAI, Google). Providers that do not (Anthropic) ignore it.

        ``on_thinking`` is invoked with each incremental thinking chunk for
        providers that support live reasoning streaming (Anthropic today;
        OpenAI / Gemini ignore it). Implementations may degrade silently to
        the non-streaming path.
        """

    @abstractmethod
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
        on_thinking: "ThinkingCallback | None" = None,
    ) -> LLMResponse:
        """Async single-turn generation."""
