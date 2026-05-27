"""Anthropic provider — Messages API with adaptive thinking, ephemeral caching,
and structured outputs via forced single-tool use."""

from __future__ import annotations

import logging
from typing import Any, Type

from pydantic import BaseModel

from ..models import lookup as model_lookup
from .base import LLMProvider, LLMResponse, ThinkingBudget, ThinkingCallback

logger = logging.getLogger(__name__)

try:
    from anthropic import Anthropic, AsyncAnthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ANTHROPIC_AVAILABLE = False
    Anthropic = AsyncAnthropic = None  # type: ignore[assignment]


_DEFAULT_AUTO_BUDGET_LEGACY = 10000
_STRUCTURED_TOOL_NAME = "submit_review"

# Module-level so we warn once per process.
_SEED_WARNED = False


def _warn_seed_once() -> None:
    global _SEED_WARNED
    if not _SEED_WARNED:
        _SEED_WARNED = True
        logger.warning(
            "Anthropic Messages API does not support `seed` for deterministic sampling; "
            "the requested seed will be ignored. For reproducibility on this provider, "
            "pair with `temperature=0` where possible."
        )


def _resolve_thinking(model: str, budget: ThinkingBudget, force_legacy: bool = False) -> dict[str, Any] | None:
    """Translate the generic budget hint to Anthropic's ``thinking`` parameter.

    ``force_legacy=True`` forces the ``enabled+budget_tokens`` form even on
    adaptive-capable models; used when combining thinking with forced tool use,
    where adaptive mode is not yet broadly supported.
    """
    if budget is None:
        return None

    spec = model_lookup(model)
    style = spec.reasoning_style if spec else "adaptive"

    if isinstance(budget, str):
        if budget.lower() != "auto":
            return None
        if style == "adaptive" and not force_legacy:
            return {"type": "adaptive"}
        return {"type": "enabled", "budget_tokens": _DEFAULT_AUTO_BUDGET_LEGACY}

    if isinstance(budget, int) and budget > 0:
        return {"type": "enabled", "budget_tokens": budget}

    return None


def _build_tool_from_schema(schema_cls: Type[BaseModel]) -> dict[str, Any]:
    """Build an Anthropic tool whose ``input_schema`` matches a Pydantic model."""
    return {
        "name": _STRUCTURED_TOOL_NAME,
        "description": (
            "Submit your final structured review. Fill every field; do not include any "
            "narrative outside this tool call."
        ),
        "input_schema": schema_cls.model_json_schema(),
    }


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API adapter."""

    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic SDK not installed. Run `pip install anthropic`."
            )
        if not api_key:
            raise ValueError("AnthropicProvider requires an api_key")
        self.api_key = api_key
        self._sync_client = Anthropic(api_key=api_key)

    # ------------------------------------------------------------------ helpers

    def _build_user_content(self, message: str, use_caching: bool) -> Any:
        if use_caching:
            return [
                {
                    "type": "text",
                    "text": message,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        return message

    def _build_system(self, instructions: str, use_caching: bool) -> Any:
        if use_caching:
            return [
                {
                    "type": "text",
                    "text": instructions,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        return instructions

    def _build_request(
        self,
        *,
        instructions: str,
        message: str,
        model: str,
        temperature: float,
        max_tokens: int,
        use_caching: bool,
        thinking_budget: ThinkingBudget,
        response_schema: Type[BaseModel] | None,
    ) -> dict[str, Any]:
        # When using a forced tool we pin to the legacy `enabled` thinking mode;
        # adaptive thinking + forced tool use is not yet broadly available.
        thinking = _resolve_thinking(
            model, thinking_budget, force_legacy=response_schema is not None
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": self._build_system(instructions, use_caching),
            "messages": [
                {"role": "user", "content": self._build_user_content(message, use_caching)}
            ],
        }
        if thinking is not None:
            kwargs["thinking"] = thinking
            kwargs["temperature"] = 1.0  # extended thinking requires temperature=1.0
        else:
            kwargs["temperature"] = temperature

        if response_schema is not None:
            kwargs["tools"] = [_build_tool_from_schema(response_schema)]
            kwargs["tool_choice"] = {"type": "tool", "name": _STRUCTURED_TOOL_NAME}
        return kwargs

    @staticmethod
    def _extract_text(response: Any) -> str:
        parts: list[str] = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", "") or "")
        return "".join(parts)

    @staticmethod
    def _extract_tool_input(response: Any) -> dict[str, Any] | None:
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", "") == _STRUCTURED_TOOL_NAME:
                inp = getattr(block, "input", None)
                if isinstance(inp, dict):
                    return inp
        return None

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, Any]:
        u = getattr(response, "usage", None)
        if u is None:
            return {}
        input_tokens = getattr(u, "input_tokens", 0) or 0
        output_tokens = getattr(u, "output_tokens", 0) or 0
        out: dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        for attr in ("cache_creation_input_tokens", "cache_read_input_tokens"):
            v = getattr(u, attr, None)
            if v is not None:
                out[attr] = v
        return out

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
        on_thinking: ThinkingCallback | None = None,
    ) -> LLMResponse:
        if not message or not message.strip():
            raise ValueError("Message content cannot be empty")
        if seed is not None:
            _warn_seed_once()
        request = self._build_request(
            instructions=instructions, message=message, model=model,
            temperature=temperature, max_tokens=max_tokens,
            use_caching=use_caching, thinking_budget=thinking_budget,
            response_schema=response_schema,
        )
        # Live-thinking path: stream from the SDK and pipe each thinking_delta
        # to the caller's callback. Falls back to non-streaming otherwise.
        if on_thinking is not None and "thinking" in request:
            response = self._generate_streaming(request, on_thinking)
        else:
            response = self._sync_client.messages.create(**request)
        parsed = None
        if response_schema is not None:
            tool_input = self._extract_tool_input(response)
            if tool_input is not None:
                parsed = response_schema.model_validate(tool_input)
        return LLMResponse(
            text=self._extract_text(response),
            usage=self._extract_usage(response),
            raw=response, provider=self.name, model=model, parsed=parsed,
        )

    def _generate_streaming(self, request: dict, on_thinking: ThinkingCallback):
        """Use the Anthropic streaming API so we can forward thinking deltas."""
        try:
            with self._sync_client.messages.stream(**request) as stream:
                for event in stream:
                    delta = getattr(event, "delta", None)
                    if delta is None:
                        continue
                    if getattr(delta, "type", "") == "thinking_delta":
                        chunk = getattr(delta, "thinking", "") or ""
                        if chunk:
                            try:
                                on_thinking(chunk)
                            except Exception as e:  # noqa: BLE001
                                logger.debug("on_thinking callback raised: %s", e)
                return stream.get_final_message()
        except Exception as e:  # noqa: BLE001
            # Streaming failed — fall back to plain create() so the run continues.
            logger.warning("Anthropic streaming path failed (%s); falling back to non-streaming.", e)
            return self._sync_client.messages.create(**request)

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
        on_thinking: ThinkingCallback | None = None,
    ) -> LLMResponse:
        if not message or not message.strip():
            raise ValueError("Message content cannot be empty")
        client = AsyncAnthropic(api_key=self.api_key)
        try:
            request = self._build_request(
                instructions=instructions, message=message, model=model,
                temperature=temperature, max_tokens=max_tokens,
                use_caching=use_caching, thinking_budget=thinking_budget,
                response_schema=response_schema,
            )
            response = await client.messages.create(**request)
            parsed = None
            if response_schema is not None:
                tool_input = self._extract_tool_input(response)
                if tool_input is not None:
                    parsed = response_schema.model_validate(tool_input)
            return LLMResponse(
                text=self._extract_text(response),
                usage=self._extract_usage(response),
                raw=response, provider=self.name, model=model, parsed=parsed,
            )
        finally:
            await client.close()
