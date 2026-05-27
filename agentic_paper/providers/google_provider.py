"""Google provider — google-genai SDK with thinking_level (3.x) / thinking_budget (2.x)
and structured outputs via response_schema."""

from __future__ import annotations

import json
import logging
from typing import Any, Type

from pydantic import BaseModel

from ..models import lookup as model_lookup
from .base import LLMProvider, LLMResponse, ThinkingBudget

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GENAI_AVAILABLE = False
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]


def _sanitize_schema_for_gemini(schema: dict) -> dict:
    """Translate a Pydantic-generated JSON Schema into a Gemini-compatible
    ``response_schema`` dict.

    Gemini's schema validator rejects fields that Pydantic emits by default:

      * ``additionalProperties: false`` — Pydantic emits this when
        ``model_config = ConfigDict(extra="forbid")``. Gemini does not accept
        it (HTTP 400 ``Unknown name "additional_properties"``).
      * ``$ref`` / ``$defs`` — Pydantic uses these for nested models; Gemini
        wants the schema inlined.
      * ``anyOf: [{type: T}, {type: null}]`` — Pydantic emits this for
        ``Optional[T]``; Gemini expects ``nullable: true`` with the inner type
        promoted to top-level.

    All other JSON Schema fields (``type``, ``properties``, ``items``,
    ``required``, ``enum``, ``description``, ``title``) pass through.
    """
    defs = schema.get("$defs", {})

    def _walk(node):
        if isinstance(node, list):
            return [_walk(item) for item in node]
        if not isinstance(node, dict):
            return node

        # 1. Resolve $ref by inlining the target.
        if "$ref" in node:
            ref_name = node["$ref"].rsplit("/", 1)[-1]
            target = defs.get(ref_name, {})
            inlined = _walk(target)
            # Merge any sibling keys onto the inlined target.
            for k, v in node.items():
                if k != "$ref":
                    inlined.setdefault(k, _walk(v))
            return inlined

        # 2. Collapse anyOf: [T, null] → nullable + T's fields.
        if "anyOf" in node and isinstance(node["anyOf"], list):
            variants = node["anyOf"]
            non_null = [v for v in variants if not (isinstance(v, dict) and v.get("type") == "null")]
            has_null = len(non_null) != len(variants)
            if has_null and len(non_null) == 1:
                merged = {k: _walk(v) for k, v in node.items() if k != "anyOf"}
                merged.update(_walk(non_null[0]))
                merged["nullable"] = True
                return merged

        # 3. Drop fields Gemini rejects; recurse into the rest.
        out: dict = {}
        for k, v in node.items():
            if k in ("additionalProperties", "$defs", "$schema"):
                continue
            out[k] = _walk(v)
        return out

    return _walk(schema)


def _budget_to_level(budget: ThinkingBudget) -> str:
    if isinstance(budget, str):
        s = budget.lower()
        if s == "auto":
            return "high"
        if s in ("low", "medium", "high"):
            return s
        return "medium"
    if isinstance(budget, int):
        if budget <= 0:
            return "low"
        if budget <= 5_000:
            return "low"
        if budget <= 20_000:
            return "medium"
        return "high"
    return "medium"


def _resolve_thinking_config(model: str, budget: ThinkingBudget):
    if budget is None:
        return None
    spec = model_lookup(model)
    style = spec.reasoning_style if spec else "level"

    if style == "level":
        return genai_types.ThinkingConfig(thinking_level=_budget_to_level(budget))

    if isinstance(budget, str) and budget.lower() == "auto":
        return genai_types.ThinkingConfig(thinking_budget=-1)
    if isinstance(budget, int) and budget > 0:
        return genai_types.ThinkingConfig(thinking_budget=budget)
    return None


class GoogleProvider(LLMProvider):
    """Google Gemini adapter."""

    name = "google"

    def __init__(self, api_key: str) -> None:
        if not _GENAI_AVAILABLE:
            raise ImportError(
                "google-genai SDK not installed. Run `pip install google-genai`."
            )
        if not api_key:
            raise ValueError("GoogleProvider requires an api_key")
        self.api_key = api_key
        self._client = genai.Client(api_key=api_key)

    def _build_config(
        self,
        *,
        instructions: str,
        temperature: float,
        max_tokens: int,
        model: str,
        thinking_budget: ThinkingBudget,
        response_schema: Type[BaseModel] | None,
        seed: int | None = None,
    ):
        config_kwargs: dict[str, Any] = {
            "system_instruction": instructions,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        thinking_cfg = _resolve_thinking_config(model, thinking_budget)
        if thinking_cfg is not None:
            config_kwargs["thinking_config"] = thinking_cfg
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            # google-genai accepts a Pydantic class directly, but passing one
            # ends up sending Pydantic's `additionalProperties: false` (from
            # ConfigDict(extra="forbid")) plus `$defs/$ref`, which Gemini's
            # schema validator rejects. We sanitize the JSON Schema ourselves
            # and pass a plain dict.
            raw = response_schema.model_json_schema()
            config_kwargs["response_schema"] = _sanitize_schema_for_gemini(raw)
        if seed is not None:
            config_kwargs["seed"] = seed
        return genai_types.GenerateContentConfig(**config_kwargs)

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, Any]:
        m = getattr(response, "usage_metadata", None)
        if m is None:
            return {}
        out: dict[str, Any] = {
            "input_tokens": getattr(m, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(m, "candidates_token_count", 0) or 0,
            "total_tokens": getattr(m, "total_token_count", 0) or 0,
        }
        for src, dst in (
            ("thoughts_token_count", "thoughts_tokens"),
            ("cached_content_token_count", "cached_tokens"),
        ):
            v = getattr(m, src, None)
            if v is not None:
                out[dst] = v
        return out

    @staticmethod
    def _extract_parsed(
        response: Any, response_schema: Type[BaseModel] | None
    ) -> BaseModel | None:
        if response_schema is None:
            return None
        # 1. Convenience attribute available on recent google-genai versions.
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            if isinstance(parsed, response_schema):
                return parsed
            if isinstance(parsed, dict):
                return response_schema.model_validate(parsed)
        # 2. Fallback: text is JSON; decode + validate.
        text = getattr(response, "text", "") or ""
        if not text:
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Google response was not valid JSON despite response_schema being set")
            return None
        return response_schema.model_validate(data)

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
        on_thinking=None,  # Gemini thinking not streamed in this build
    ) -> LLMResponse:
        if not message or not message.strip():
            raise ValueError("Message content cannot be empty")
        config = self._build_config(
            instructions=instructions, temperature=temperature,
            max_tokens=max_tokens, model=model, thinking_budget=thinking_budget,
            response_schema=response_schema, seed=seed,
        )
        response = self._client.models.generate_content(
            model=model, contents=message, config=config,
        )
        return LLMResponse(
            text=getattr(response, "text", "") or "",
            usage=self._extract_usage(response),
            raw=response, provider=self.name, model=model,
            parsed=self._extract_parsed(response, response_schema),
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
        on_thinking=None,  # Gemini thinking not streamed in this build
    ) -> LLMResponse:
        if not message or not message.strip():
            raise ValueError("Message content cannot be empty")
        config = self._build_config(
            instructions=instructions, temperature=temperature,
            max_tokens=max_tokens, model=model, thinking_budget=thinking_budget,
            response_schema=response_schema, seed=seed,
        )
        response = await self._client.aio.models.generate_content(
            model=model, contents=message, config=config,
        )
        return LLMResponse(
            text=getattr(response, "text", "") or "",
            usage=self._extract_usage(response),
            raw=response, provider=self.name, model=model,
            parsed=self._extract_parsed(response, response_schema),
        )
