"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from .audit import generate_run_id
from .config import Config, setup_logging
from .orchestrator import ReviewOrchestrator
from .paper import FileManager
from .providers import build_registry
from .routing import PROFILE_NAMES, get_profile

logger = logging.getLogger(__name__)


def system_health_check(config: Config) -> dict[str, Any]:
    """Inventory check — confirms storage exists and at least one provider is registered.

    Does NOT call any LLM API; use ``python -m agentic_paper.providers.smoke`` for
    a true connectivity ping.
    """
    registry = build_registry(config)
    return {
        "storage_ok": Path(config.output_dir).exists(),
        "providers": registry.names(),
        "providers_ok": bool(registry),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Advanced Multi-Agent System for Scientific Paper Review",
    )
    parser.add_argument("paper_path", help="Path to the paper file to review")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning effort (carried for backward compat; currently unused).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Deterministic sampling seed. Forwarded to OpenAI + Google providers. "
            "Anthropic ignores it (logged once)."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Override the auto-generated run identifier (lands under output/<run_id>/).",
    )
    parser.add_argument(
        "--profile",
        default=None,
        choices=list(PROFILE_NAMES),
        help=(
            "Pre-baked routing bundle: 'max' (flagships + max thinking, ~$5+/run), "
            "'std' (balanced, ~$0.50-2/run), 'quick' (cheapest, ~$0.02-0.10/run). "
            "Overrides the routing in config.yaml for this run."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Resolve run_id + scoped log path before configuring logging so the log
    # lands under the same directory as the audit artefacts.
    run_id = args.run_id or generate_run_id()
    try:
        config_preview = Config.from_yaml(args.config)
        base_output = args.output_dir or config_preview.output_dir
    except Exception:
        base_output = args.output_dir or "output_paper_review"
    run_dir = Path(base_output) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level, log_file=str(run_dir / "paper_review_system.log"))

    try:
        config = Config.from_yaml(args.config)
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.reasoning_effort:
            config.reasoning_effort = args.reasoning_effort
        if args.seed is not None:
            config.seed = args.seed
        if args.profile:
            profile_routing = get_profile(args.profile)
            if profile_routing is not None:
                config.routing = profile_routing
                logger.info("Applied profile '%s' (overrides routing).", args.profile)

        config.validate()

        health = system_health_check(config)
        logger.info("System health: %s", health)

        file_manager = FileManager(config.output_dir)
        if args.paper_path.lower().endswith(".pdf"):
            paper_text = file_manager.extract_text_from_pdf(args.paper_path)
        else:
            paper_text = file_manager.read_paper(args.paper_path)

        if not paper_text:
            logger.error("Failed to read paper file")
            return 1

        logger.info("Paper loaded successfully. Length: %s characters", f"{len(paper_text):,}")

        orchestrator = ReviewOrchestrator(config, run_id=run_id)
        orchestrator.execute_review_process(paper_text)

        logger.info("Review process completed. Results saved in: %s", orchestrator.run_dir)
        logger.info("✅ PROCESS COMPLETED SUCCESSFULLY!")
        return 0

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
