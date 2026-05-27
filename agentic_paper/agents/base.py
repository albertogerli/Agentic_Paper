"""Agent base classes — synchronous, async, and caching async variants."""

from __future__ import annotations

import logging
import time
from typing import Optional, Type

from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..audit import AuditLogger
from ..providers import LLMProvider
from ..providers.base import ThinkingBudget

logger = logging.getLogger(__name__)


class Agent:
    """Synchronous agent backed by an :class:`LLMProvider`.

    With ``schema`` set, ``run()`` returns a validated pydantic model;
    without one, it returns the raw text. Reviewers always supply a schema
    after Prompt 4.
    """

    def __init__(
        self,
        name: str,
        instructions: str,
        model: str,
        provider: LLMProvider,
        temperature: float = 1.0,
        max_output_tokens: int = 16000,
        use_caching: bool = True,
        thinking_budget: ThinkingBudget = None,
        schema: Type[BaseModel] | None = None,
        seed: int | None = None,
    ) -> None:
        self.name = name
        self.instructions = instructions
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.use_caching = use_caching
        self.thinking_budget = thinking_budget
        self.schema = schema
        self.seed = seed

    def _full_prompt(self, message: str) -> str:
        return f"=== SYSTEM ===\n{self.instructions}\n\n=== USER ===\n{message}\n"

    def _response_text_for_audit(self, response) -> str:
        if response.parsed is not None:
            try:
                return response.parsed.model_dump_json(indent=2)
            except Exception:
                pass
        return response.text or ""

    def _record_audit(self, audit: AuditLogger | None, response, elapsed_ms: int, message: str) -> None:
        if audit is None:
            return
        try:
            audit.record(
                agent=self.name,
                provider=self.provider.name if self.provider else "unknown",
                model=self.model,
                prompt=self._full_prompt(message),
                response_text=self._response_text_for_audit(response),
                usage=response.usage,
                latency_ms=elapsed_ms,
                thinking_enabled=self.thinking_budget is not None,
                seed=self.seed,
            )
        except Exception as e:  # never break the pipeline because audit failed
            logger.warning("Audit record failed for agent %s: %s", self.name, e)

    def _log_usage(self, usage: dict, kind: str = "Agent") -> None:
        cached = usage.get("cached_tokens", usage.get("cache_read_input_tokens", 0))
        if cached:
            logger.info(
                "%s %s completed - Tokens: %s (cached: %s) [%s/%s]",
                kind, self.name, usage.get("total_tokens", 0), cached,
                self.provider.name, self.model,
            )
        else:
            logger.info(
                "%s %s completed - Tokens: %s [%s/%s]",
                kind, self.name, usage.get("total_tokens", 0),
                self.provider.name, self.model,
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=120),
        retry=retry_if_exception_type((Exception,)),
    )
    def run(
        self,
        message: str,
        audit: AuditLogger | None = None,
        on_thinking=None,
    ) -> BaseModel | str:
        if self.provider is None:
            raise ValueError("Provider not initialized")
        if not message or not message.strip():
            raise ValueError("Message content cannot be empty")
        start = time.perf_counter()
        try:
            response = self.provider.generate(
                instructions=self.instructions,
                message=message,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                use_caching=self.use_caching,
                thinking_budget=self.thinking_budget,
                response_schema=self.schema,
                seed=self.seed,
                on_thinking=on_thinking,
            )
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            self._log_usage(response.usage, kind="Agent")
            self._record_audit(audit, response, elapsed_ms, message)
            if self.schema is not None:
                if response.parsed is None:
                    raise ValueError(
                        f"Agent {self.name}: provider returned no parsed object for schema "
                        f"{self.schema.__name__}; raw text was: {response.text[:200]!r}"
                    )
                return response.parsed
            return response.text
        except Exception as e:
            logger.error(
                "Error in agent %s [%s/%s]: %s",
                self.name, self.provider.name, self.model, e,
            )
            raise


class AsyncAgent(Agent):
    """Asynchronous variant of :class:`Agent`."""

    async def arun(
        self,
        message: str,
        audit: AuditLogger | None = None,
        on_thinking=None,
    ) -> BaseModel | str:
        if self.provider is None:
            raise ValueError("Provider not initialized")
        if not message or not message.strip():
            raise ValueError("Message content cannot be empty")
        start = time.perf_counter()
        try:
            response = await self.provider.agenerate(
                instructions=self.instructions,
                message=message,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                use_caching=self.use_caching,
                thinking_budget=self.thinking_budget,
                response_schema=self.schema,
                seed=self.seed,
                on_thinking=on_thinking,
            )
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            self._log_usage(response.usage, kind="Async agent")
            self._record_audit(audit, response, elapsed_ms, message)
            if self.schema is not None:
                if response.parsed is None:
                    raise ValueError(
                        f"Async agent {self.name}: provider returned no parsed object for schema "
                        f"{self.schema.__name__}; raw text was: {response.text[:200]!r}"
                    )
                return response.parsed
            return response.text
        except Exception as e:
            logger.error(
                "Error in async agent %s [%s/%s]: %s",
                self.name, self.provider.name, self.model, e,
            )
            raise


class CachingAsyncAgent(AsyncAgent):
    """:class:`AsyncAgent` with in-memory result caching keyed by message hash."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cache: dict[int, BaseModel | str] = {}

    async def arun(
        self,
        message: str,
        audit: AuditLogger | None = None,
        on_thinking=None,
    ) -> BaseModel | str:
        key = hash(message)
        if key in self._cache:
            logger.info("Using cached result for agent %s", self.name)
            return self._cache[key]
        result = await super().arun(message, audit=audit, on_thinking=on_thinking)
        self._cache[key] = result
        return result
