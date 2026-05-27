"""Per-run audit trail: run_id, prompts, raw responses, JSONL log + cost estimate.

The output of one review run lives entirely under ``output/<run_id>/``::

    output/<run_id>/
    ├── audit.jsonl                    # one record per provider call
    ├── prompts/<agent>.txt            # exact prompt sent
    ├── responses/<agent>.json         # raw response text + usage
    ├── reviews/<agent>.json           # validated AnnotatedReview
    ├── paper_info.json
    ├── review_report_*.md             # human report (incl. cost summary)
    ├── review_results_*.json
    ├── dashboard_*.html
    └── paper_review_system.log
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Per-(provider, model) USD pricing per 1M tokens.
# Rates as of May 2026; update via pricing pages of each vendor before publication.
# Keys are matched longest-prefix-first within a provider.
COST_RATES: dict[tuple[str, str], tuple[float, float]] = {
    # ---- OpenAI ----------------------------------------------------------
    ("openai", "gpt-5.5-pro"):              (15.0, 75.0),
    ("openai", "gpt-5.5"):                  ( 5.0, 25.0),
    ("openai", "gpt-5.4-mini"):             ( 0.50, 2.00),
    ("openai", "gpt-5.4-nano"):             ( 0.20, 1.25),   # 2026-03-17
    ("openai", "gpt-5-mini"):               ( 0.40, 1.60),
    ("openai", "gpt-5"):                    ( 5.0, 25.0),
    ("openai", "o4-mini"):                  ( 3.0, 12.0),
    ("openai", "o3"):                       (15.0, 60.0),
    ("openai", "gpt-4o-mini"):              ( 0.15, 0.60),
    ("openai", "gpt-4o"):                   ( 2.5, 10.0),
    # ---- Anthropic -------------------------------------------------------
    ("anthropic", "claude-opus-4-7"):       (15.0, 75.0),
    ("anthropic", "claude-sonnet-4-6"):     ( 3.0, 15.0),
    ("anthropic", "claude-haiku-4-5"):      ( 1.0, 5.0),
    ("anthropic", "claude-sonnet-4-5"):     ( 3.0, 15.0),
    ("anthropic", "claude-opus-4"):         (15.0, 75.0),
    # ---- Google ----------------------------------------------------------
    ("google", "gemini-3.1-pro-preview"):       ( 2.0, 10.0),
    ("google", "gemini-3.1-flash-lite"):        ( 0.25, 1.50),
    ("google", "gemini-3.5-flash"):             ( 1.50, 9.00),  # GA 2026-05-19
    ("google", "gemini-3-pro"):                 ( 2.0, 10.0),
    ("google", "gemini-3-flash"):               ( 0.30, 1.20),
    ("google", "gemini-2.5-pro"):               ( 1.25, 10.0),
    ("google", "gemini-2.5-flash"):             ( 0.075, 0.30),
}


def generate_run_id() -> str:
    """Compact reproducible identifier: ``YYYYMMDD-HHMMSS-XXXXXX``."""
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def hash_prompt(text: str) -> str:
    """Stable 16-hex-char SHA-256 prefix for a prompt body."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost estimate using :data:`COST_RATES`. Longest-prefix match wins."""
    if input_tokens <= 0 and output_tokens <= 0:
        return 0.0
    if (provider, model) in COST_RATES:
        in_rate, out_rate = COST_RATES[(provider, model)]
    else:
        candidates = [k for k in COST_RATES if k[0] == provider and model.startswith(k[1])]
        if not candidates:
            return 0.0
        in_rate, out_rate = COST_RATES[max(candidates, key=lambda k: len(k[1]))]
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in (name or ""))


class AuditLogger:
    """Thread-safe writer for ``output/<run_id>/`` audit artefacts."""

    def __init__(self, run_dir: str | Path, run_id: str | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "prompts").mkdir(exist_ok=True)
        (self.run_dir / "responses").mkdir(exist_ok=True)
        self.audit_path = self.run_dir / "audit.jsonl"
        self.run_id = run_id or self.run_dir.name
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def record(
        self,
        *,
        agent: str,
        provider: str,
        model: str,
        prompt: str,
        response_text: str,
        usage: dict[str, Any],
        latency_ms: int,
        thinking_enabled: bool,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Append one provenance row + persist the prompt + raw response."""
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        cost = estimate_cost(provider, model, input_tokens, output_tokens)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "agent": agent,
            "provider": provider,
            "model": model,
            "prompt_hash": hash_prompt(prompt),
            "input_token_count": input_tokens,
            "output_token_count": output_tokens,
            "latency_ms": int(latency_ms),
            "cost_estimate_usd": round(cost, 6),
            "thinking_mode_enabled": bool(thinking_enabled),
            "seed": seed,
        }
        safe = _safe_name(agent)
        with self._lock:
            self.records.append(record)
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            (self.run_dir / "prompts" / f"{safe}.txt").write_text(prompt, encoding="utf-8")
            payload = {
                "agent": agent,
                "provider": provider,
                "model": model,
                "usage": usage,
                "text": response_text,
                "seed": seed,
            }
            (self.run_dir / "responses" / f"{safe}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return record

    def summary(self) -> dict[str, Any]:
        """Aggregate totals for the markdown / JSON report cost section."""
        total_in = sum(r["input_token_count"] for r in self.records)
        total_out = sum(r["output_token_count"] for r in self.records)
        total_cost = sum(r["cost_estimate_usd"] for r in self.records)
        per_provider: dict[str, dict[str, Any]] = {}
        per_agent: dict[str, dict[str, Any]] = {}
        for r in self.records:
            p = r["provider"]
            per_provider.setdefault(p, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
            per_provider[p]["calls"] += 1
            per_provider[p]["input_tokens"] += r["input_token_count"]
            per_provider[p]["output_tokens"] += r["output_token_count"]
            per_provider[p]["cost_usd"] += r["cost_estimate_usd"]
            a = r["agent"]
            per_agent[a] = {
                "provider": p,
                "model": r["model"],
                "input_tokens": r["input_token_count"],
                "output_tokens": r["output_token_count"],
                "latency_ms": r["latency_ms"],
                "cost_usd": r["cost_estimate_usd"],
                "thinking_mode_enabled": r["thinking_mode_enabled"],
            }
        return {
            "run_id": self.run_id,
            "total_calls": len(self.records),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_cost_usd": round(total_cost, 4),
            "per_provider": {
                p: {**v, "cost_usd": round(v["cost_usd"], 4)} for p, v in per_provider.items()
            },
            "per_agent": per_agent,
        }


__all__ = [
    "AuditLogger",
    "COST_RATES",
    "estimate_cost",
    "generate_run_id",
    "hash_prompt",
]
