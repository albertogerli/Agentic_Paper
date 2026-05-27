"""OpenAI provider — Responses API for reasoning models, Chat Completions fallback,
structured outputs via the SDK's ``parse()`` helpers."""

from __future__ import annotations

import logging
from typing import Any, Type

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from ..models import lookup as model_lookup
from .base import LLMProvider, LLMResponse, ThinkingBudget

logger = logging.getLogger(__name__)


def _reasoning_effort_from_budget(budget: ThinkingBudget) -> str | None:
    """Map a generic thinking_budget to OpenAI's ``reasoning.effort`` label."""
    if budget is None:
        return None
    if isinstance(budget, str):
        s = budget.lower()
        if s == "auto":
            return "medium"
        if s in ("none", "low", "medium", "high", "xhigh"):
            return s
        return None
    if isinstance(budget, int):
        if budget <= 0:
            return "none"
        if budget <= 5_000:
            return "low"
        if budget <= 20_000:
            return "medium"
        if budget <= 50_000:
            return "high"
        return "xhigh"
    return None


def _is_reasoning_model(model: str) -> bool:
    spec = model_lookup(model)
    return bool(spec and spec.supports_reasoning)


class OpenAIProvider(LLMProvider):
    """OpenAI adapter that routes between Responses API and Chat Completions."""

    name = "openai"

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("OpenAIProvider requires an api_key")
        self.api_key = api_key
        self._sync_client = OpenAI(api_key=api_key)
        self._responses_available = hasattr(self._sync_client, "responses")
        if not self._responses_available:
            logger.warning(
                "openai SDK does not expose `.responses`; falling back to Chat Completions for all models."
            )

    # ------------------------------------------------------------------ shared

    def _build_chat_kwargs(
        self,
        *,
        instructions: str,
        message: str,
        model: str,
        temperature: float,
        max_tokens: int,
        thinking_budget: ThinkingBudget,
        seed: int | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": message},
            ],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if _is_reasoning_model(model):
            effort = _reasoning_effort_from_budget(thinking_budget)
            if effort:
                kwargs["reasoning_effort"] = effort
        if seed is not None:
            kwargs["seed"] = seed
        return kwargs

    def _build_responses_kwargs(
        self,
        *,
        instructions: str,
        message: str,
        model: str,
        temperature: float,
        max_tokens: int,
        thinking_budget: ThinkingBudget,
        seed: int | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": message,
            "max_output_tokens": max_tokens,
        }
        effort = _reasoning_effort_from_budget(thinking_budget)
        if effort:
            kwargs["reasoning"] = {"effort": effort}
        elif temperature is not None and abs(temperature - 1.0) > 1e-6:
            kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["seed"] = seed
        return kwargs

    @staticmethod
    def _extract_chat_usage(response: Any) -> dict[str, Any]:
        u = getattr(response, "usage", None)
        if u is None:
            return {}
        out: dict[str, Any] = {
            "total_tokens": getattr(u, "total_tokens", 0),
            "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(u, "completion_tokens", 0) or 0,
        }
        prompt_details = getattr(u, "prompt_tokens_details", None)
        cached = getattr(prompt_details, "cached_tokens", None) if prompt_details else None
        if cached is not None:
            out["cached_tokens"] = cached
        return out

    @staticmethod
    def _extract_responses_usage(response: Any) -> dict[str, Any]:
        u = getattr(response, "usage", None)
        if u is None:
            return {}
        out: dict[str, Any] = {
            "input_tokens": getattr(u, "input_tokens", 0) or 0,
            "output_tokens": getattr(u, "output_tokens", 0) or 0,
            "total_tokens": getattr(u, "total_tokens", 0) or 0,
        }
        input_details = getattr(u, "input_tokens_details", None)
        cached = getattr(input_details, "cached_tokens", None) if input_details else None
        if cached is not None:
            out["cached_tokens"] = cached
        out_details = getattr(u, "output_tokens_details", None)
        reasoning_tokens = getattr(out_details, "reasoning_tokens", None) if out_details else None
        if reasoning_tokens is not None:
            out["reasoning_tokens"] = reasoning_tokens
        return out

    @staticmethod
    def _extract_responses_text(response: Any) -> str:
        text = getattr(response, "output_text", None)
        if text:
            return text
        parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for block in getattr(item, "content", []) or []:
                t = getattr(block, "text", None)
                if t:
                    parts.append(t)
        return "".join(parts)

    # ------------------------------------------------------------------ generate

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
        on_thinking=None,  # OpenAI reasoning summaries not streamed in this build
    ) -> LLMResponse:
        if not message or not message.strip():
            raise ValueError("Message content cannot be empty")

        if self._responses_available and _is_reasoning_model(model):
            kwargs = self._build_responses_kwargs(
                instructions=instructions, message=message, model=model,
                temperature=temperature, max_tokens=max_tokens,
                thinking_budget=thinking_budget, seed=seed,
            )
            if response_schema is not None:
                # The SDK helper handles schema → strict mode conversion.
                response = self._sync_client.responses.parse(text_format=response_schema, **kwargs)
                parsed = getattr(response, "output_parsed", None)
            else:
                response = self._sync_client.responses.create(**kwargs)
                parsed = None
            return LLMResponse(
                text=self._extract_responses_text(response),
                usage=self._extract_responses_usage(response),
                raw=response,
                provider=self.name,
                model=model,
                parsed=parsed,
            )

        kwargs = self._build_chat_kwargs(
            instructions=instructions, message=message, model=model,
            temperature=temperature, max_tokens=max_tokens,
            thinking_budget=thinking_budget, seed=seed,
        )
        if response_schema is not None:
            response = self._sync_client.chat.completions.parse(
                response_format=response_schema, **kwargs
            )
            parsed = response.choices[0].message.parsed
            text = response.choices[0].message.content or ""
        else:
            response = self._sync_client.chat.completions.create(**kwargs)
            parsed = None
            text = response.choices[0].message.content or ""
        return LLMResponse(
            text=text,
            usage=self._extract_chat_usage(response),
            raw=response,
            provider=self.name,
            model=model,
            parsed=parsed,
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
        on_thinking=None,  # OpenAI reasoning summaries not streamed in this build
    ) -> LLMResponse:
        if not message or not message.strip():
            raise ValueError("Message content cannot be empty")

        client = AsyncOpenAI(api_key=self.api_key)
        responses_available = hasattr(client, "responses")
        try:
            if responses_available and _is_reasoning_model(model):
                kwargs = self._build_responses_kwargs(
                    instructions=instructions, message=message, model=model,
                    temperature=temperature, max_tokens=max_tokens,
                    thinking_budget=thinking_budget, seed=seed,
                )
                if response_schema is not None:
                    response = await client.responses.parse(text_format=response_schema, **kwargs)
                    parsed = getattr(response, "output_parsed", None)
                else:
                    response = await client.responses.create(**kwargs)
                    parsed = None
                return LLMResponse(
                    text=self._extract_responses_text(response),
                    usage=self._extract_responses_usage(response),
                    raw=response, provider=self.name, model=model, parsed=parsed,
                )
            kwargs = self._build_chat_kwargs(
                instructions=instructions, message=message, model=model,
                temperature=temperature, max_tokens=max_tokens,
                thinking_budget=thinking_budget, seed=seed,
            )
            if response_schema is not None:
                response = await client.chat.completions.parse(
                    response_format=response_schema, **kwargs
                )
                parsed = response.choices[0].message.parsed
                text = response.choices[0].message.content or ""
            else:
                response = await client.chat.completions.create(**kwargs)
                parsed = None
                text = response.choices[0].message.content or ""
            return LLMResponse(
                text=text,
                usage=self._extract_chat_usage(response),
                raw=response, provider=self.name, model=model, parsed=parsed,
            )
        finally:
            await client.close()
