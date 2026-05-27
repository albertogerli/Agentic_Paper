"""Generic OpenAI-compatible endpoint adapter (vLLM, Ollama, DeepSeek, LM Studio…).

Local servers rarely implement OpenAI's strict ``response_format: json_schema``
mode. When a schema is requested we ask for ``json_object`` mode and validate
the text against the pydantic schema on the client side.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Type

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from .base import LLMResponse, ThinkingBudget
from .openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


class OpenAICompatProvider(OpenAIProvider):
    """OpenAI-compatible endpoint at a custom base URL."""

    name = "openai_compat"

    def __init__(self, api_key: str, base_url: str, name: str | None = None) -> None:
        if not base_url:
            raise ValueError("OpenAICompatProvider requires a base_url")
        self.api_key = api_key or "EMPTY"
        self.base_url = base_url
        if name:
            self.name = name
        self._sync_client = OpenAI(api_key=self.api_key, base_url=base_url)
        self._responses_available = False  # never trust local servers with Responses

    # --------------------------------------------------------- request shaping

    def _build_compat_kwargs(
        self,
        *,
        instructions: str,
        message: str,
        model: str,
        temperature: float,
        max_tokens: int,
        response_schema: Type[BaseModel] | None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        if seed is not None:
            kwargs["seed"] = seed
        return kwargs

    @staticmethod
    def _parse_with_schema(text: str, schema: Type[BaseModel]) -> BaseModel | None:
        if not text:
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("openai_compat returned non-JSON despite response_format=json_object")
            return None
        return schema.model_validate(data)

    # ----------------------------------------------------------------- generate

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
        on_thinking=None,  # local servers don't expose reasoning streams
    ) -> LLMResponse:
        if not message or not message.strip():
            raise ValueError("Message content cannot be empty")
        kwargs = self._build_compat_kwargs(
            instructions=instructions, message=message, model=model,
            temperature=temperature, max_tokens=max_tokens, response_schema=response_schema,
            seed=seed,
        )
        response = self._sync_client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        parsed = self._parse_with_schema(text, response_schema) if response_schema else None
        return LLMResponse(
            text=text,
            usage=self._extract_chat_usage(response),
            raw=response, provider=self.name, model=model, parsed=parsed,
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
    ) -> LLMResponse:
        if not message or not message.strip():
            raise ValueError("Message content cannot be empty")
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        try:
            kwargs = self._build_compat_kwargs(
                instructions=instructions, message=message, model=model,
                temperature=temperature, max_tokens=max_tokens, response_schema=response_schema,
            )
            response = await client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or ""
            parsed = self._parse_with_schema(text, response_schema) if response_schema else None
            return LLMResponse(
                text=text,
                usage=self._extract_chat_usage(response),
                raw=response, provider=self.name, model=model, parsed=parsed,
            )
        finally:
            await client.close()
