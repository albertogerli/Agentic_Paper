"""Review orchestrator — parallel reviewer fan-out then serial coordinator/editor/summary.

All reviewer outputs are typed pydantic models (Review, CoordinatorAssessment,
EditorDecision, AuthorEditorSummary). The orchestrator wraps each one with
provenance into an Annotated* model before storage / rendering.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from pathlib import Path

from .agent_runner import ConcurrentAgentRunner, annotate as _annotate, format_exception as _format_exception
from .agents import Agent, AgentFactory
from .audit import AuditLogger, generate_run_id
from .config import Config
from .external.citations import (
    extract_citations,
    format_report_for_agent,
    validate_citations,
)
from .external.statcheck import (
    format_report_for_agent as format_statcheck_for_agent,
    run_statcheck,
)
from .diff_utils import format_diff_for_agent, paper_diff
from .paper import FileManager, PaperAnalyzer, PaperInfo, PDFExtractor, assess_paper_complexity
from .providers import build_registry
from .reports import render_executive_summary, render_html, render_markdown
from .storage import LocalFileStorage, StorageProvider
from .schemas import (
    AnnotatedAuthorEditorSummary,
    AnnotatedCoordinatorAssessment,
    AnnotatedEditorDecision,
    AnnotatedReview,
    AuthorEditorSummary,
    CoordinatorAssessment,
    EditorDecision,
)

logger = logging.getLogger(__name__)


REVIEWER_KEYS = (
    "methodology",
    "results",
    "literature",
    "structure",
    "impact",
    "contradiction",
    "ethics",
    "ai_origin",
    "hallucination",
    "citation_validator",
    "statcheck_validator",
    "revision_assessor",
)


def _run_context_preamble(today: str | None = None) -> str:
    """Date-anchoring preamble injected into every agent message.

    LLMs have training cutoffs; a paper that talks about "June 2025" can be
    misread as a forecast when the model thinks "today" is sometime in 2024.
    Pinning the current date stops that whole class of misreadings.
    """
    if today is None:
        today = datetime.now().strftime("%Y-%m-%d")
    return (
        "=== RUN CONTEXT ===\n"
        f"Today's date is {today}.\n"
        "The paper below was authored before today; treat dates, events, and findings\n"
        "inside it as factual past or already-completed work unless the paper itself\n"
        "explicitly frames them as predictions, forecasts, or future plans.\n"
        "If a year inside the paper looks 'in the future' relative to your training\n"
        "data, you can assume it has already happened from the reviewer's perspective.\n"
    )


def _format_review_for_consumer(key: str, r: AnnotatedReview) -> str:
    """Human/LLM-friendly text rendering of a single typed review."""
    lines = [
        f"=== {key.upper()} REVIEW ===",
        f"Reviewer: {r.agent} ({r.model_used})",
        f"Recommendation: {r.recommendation} (confidence: {r.confidence:.2f})",
        "",
        f"Summary: {r.summary}",
    ]
    if r.strengths:
        lines.append("\nStrengths:")
        lines.extend(f"  - {s}" for s in r.strengths)
    if r.concerns:
        lines.append("\nConcerns:")
        for c in r.concerns:
            page = f"p.{c.page}" if c.page is not None else "—"
            lines.append(f"  [{c.severity.upper()}] {c.section} ({page}): {c.issue}")
            if c.suggested_fix:
                lines.append(f"    Suggested fix: {c.suggested_fix}")
    else:
        lines.append("\nConcerns: (none flagged)")
    return "\n".join(lines)


def _format_coordinator_for_consumer(c: AnnotatedCoordinatorAssessment) -> str:
    lines = [
        f"=== COORDINATOR ASSESSMENT ({c.agent} / {c.model_used}) ===",
        f"Final recommendation: {c.final_recommendation}",
        f"Overall score: {c.overall_score:.2f}",
        f"\nExecutive summary: {c.executive_summary}",
    ]
    if c.consensus_strengths:
        lines.append("\nConsensus strengths:")
        lines.extend(f"  - {s}" for s in c.consensus_strengths)
    if c.consensus_concerns:
        lines.append("\nConsensus concerns:")
        for cc in c.consensus_concerns:
            page = f"p.{cc.page}" if cc.page is not None else "—"
            lines.append(f"  [{cc.severity.upper()}] {cc.section} ({page}): {cc.issue}")
    if c.disagreements:
        lines.append("\nDisagreements:")
        lines.extend(f"  - {d}" for d in c.disagreements)
    if c.revision_priorities:
        lines.append("\nRevision priorities (ordered):")
        lines.extend(f"  {i}. {p}" for i, p in enumerate(c.revision_priorities, 1))
    return "\n".join(lines)


class ReviewOrchestrator:
    """Main orchestrator for the review process."""

    def __init__(
        self,
        config: Config,
        run_id: str | None = None,
        *,
        event_bus=None,
        storage: StorageProvider | None = None,
    ) -> None:
        self.config = config
        self.run_id = run_id or generate_run_id()
        self.event_bus = event_bus
        # Scope every artefact under output/<run_id>/ — reproducibility-friendly,
        # concurrent / repeated runs no longer clobber each other.
        self.run_dir = Path(config.output_dir) / self.run_id
        # Storage backend is injectable; default keeps v1 behaviour (local FS).
        # ``file_manager`` is kept as a thin facade for v1 callers (web/runner,
        # tests) — new code should reach for ``self.storage`` directly.
        self.storage: StorageProvider = storage or LocalFileStorage(self.run_dir)
        self.file_manager = FileManager(str(self.run_dir))
        self.pdf_extractor = PDFExtractor()
        self.paper_analyzer = PaperAnalyzer(config)
        self.registry = build_registry(config)
        logger.info("Run id: %s (output dir: %s)", self.run_id, self.run_dir)
        logger.info("Provider registry: %s", self.registry.names() or "<empty>")
        if config.seed is not None:
            logger.info("Deterministic mode: seed=%d will be forwarded to providers that support it.", config.seed)
        self.audit = AuditLogger(self.run_dir, run_id=self.run_id)
        self.agent_factory: AgentFactory | None = None
        self.agents: dict[str, Agent] = {}
        # ConcurrentAgentRunner is built lazily once agents exist (see
        # ``_init_runner``); both ``execute_review_process`` and
        # ``retry_failed_agents`` re-instantiate the agents and the runner.
        self.runner: ConcurrentAgentRunner | None = None

    # ------------------------------------------------------------------ run

    def execute_review_process(
        self, paper_text: str, *, v1_text: str | None = None,
    ) -> dict[str, Any]:
        """Run the full multi-agent pipeline and emit reports.

        ``v1_text`` (optional) is the text of an earlier version of the paper.
        When provided, the orchestrator computes a section-level diff and
        attaches it to the initial reviewer message so the Revision Assessor
        agent has something concrete to chew on.
        """
        try:
            complexity_score = asyncio.run(assess_paper_complexity(paper_text, self.config))
            self.agent_factory = AgentFactory(self.config, complexity_score, self.registry)
            self.agents = self.agent_factory.create_all_agents()
            self._init_runner()

            paper_info = self.paper_analyzer.extract_info(paper_text)
            self.storage.save_json(paper_info.to_dict(), "paper_info.json")
            # Persist paper text + minimal state so `retry_failed` can rerun
            # individual agents without re-uploading the PDF / paying for
            # complexity scoring again.
            self.storage.save_text(paper_text, "paper.txt")
            self.storage.save_json(
                {
                    "run_id": self.run_id,
                    "complexity_score": complexity_score,
                    "v1_present": v1_text is not None,
                },
                "_run_state.json",
            )
            if v1_text is not None:
                self.storage.save_text(v1_text, "paper_v1.txt")

            logger.info("Starting multi-agent peer review process…")
            citation_block = self._build_citation_block(paper_text)
            statcheck_block = self._build_statcheck_block(paper_text)
            diff_block = self._build_diff_block(v1_text, paper_text) if v1_text else ""
            initial_message = self._prepare_initial_message(
                paper_info, paper_text,
                citation_block=citation_block,
                statcheck_block=statcheck_block,
                diff_block=diff_block,
            )

            reviews, errors = self._execute_main_reviewers(initial_message)
            coordinator = self._execute_coordinator(reviews)
            summary = self._execute_author_editor_summary(reviews, coordinator)
            editor = self._execute_editor(reviews, coordinator, summary)

            final_results = self._synthesize_results(
                paper_info=paper_info,
                reviews=reviews,
                errors=errors,
                coordinator=coordinator,
                summary=summary,
                editor=editor,
            )
            self._generate_reports(final_results)
            return final_results

        except Exception as e:
            logger.error("Critical error in review process: %s", e)
            raise
        finally:
            if self.event_bus is not None:
                self.event_bus.close(self.run_id)

    # ------------------------------------------------------------------ stages

    def _prepare_initial_message(
        self,
        paper_info: PaperInfo,
        paper_text: str,
        *,
        citation_block: str = "",
        statcheck_block: str = "",
        diff_block: str = "",
    ) -> str:
        if len(paper_text) > 50_000:
            logger.info(
                "Paper text is %d characters; this may exceed some model limits "
                "(recommended <= 50,000).",
                len(paper_text),
            )
        msg = (
            _run_context_preamble()
            + "\nPaper to be analyzed:\n\n"
            f"Title: {paper_info.title}\n"
            f"Authors: {paper_info.authors}\n"
            f"Abstract: {paper_info.abstract}\n\n"
            "Please conduct a comprehensive and thorough review of this scientific paper.\n"
            "All reviewers should provide their comments IN ENGLISH.\n"
            "Each reviewer should analyze the paper from their own expert perspective.\n\n"
            "The paper content is as follows:\n\n"
            f"{paper_text}\n"
        )
        if citation_block:
            msg += "\n=== CITATION VALIDATION REPORT ===\n" + citation_block + "\n"
        if statcheck_block:
            msg += "\n=== STATCHECK REPORT (R) ===\n" + statcheck_block + "\n"
        if diff_block:
            msg += "\n=== PAPER VERSION DIFF (v1 → v2) ===\n" + diff_block + "\n"
        return msg

    def retry_failed_agents(self, only_keys: list[str] | None = None) -> dict[str, Any]:
        """Re-run only the agents in ``only_keys`` (or, when None, every
        reviewer whose previous attempt is missing from the run dir).

        Reads the paper text saved at the original run, rebuilds the agents,
        and re-runs the failing subset in parallel. Existing successful
        reviews are preserved; coordinator / summary / editor are NOT
        re-run automatically — the run dir's reports stay until the user
        explicitly re-renders them.
        """
        # Recover paper text + complexity score from the persisted state.
        # Reads go through ``self.storage`` so a future S3/DB backend works
        # without changes here; only the "list saved reviews" lookup below
        # still touches the filesystem directly.
        run_dir = self.run_dir
        paper_text = self.storage.read_text("paper.txt")
        if paper_text is None:
            raise FileNotFoundError(
                f"No paper.txt under {run_dir}; cannot retry without the source text."
            )
        state_text = self.storage.read_text("_run_state.json")
        try:
            state = json.loads(state_text) if state_text else {}
            complexity_score = float(state.get("complexity_score", 0.5))
        except json.JSONDecodeError:
            complexity_score = 0.5

        # Reuse the registry the orchestrator was built with (in the web layer
        # the orchestrator is fresh per /retry call, so its constructor has
        # already loaded any env-var key changes).
        if not self.registry:
            self.registry = build_registry(self.config)
        self.agent_factory = AgentFactory(self.config, complexity_score, self.registry)
        self.agents = self.agent_factory.create_all_agents()
        self._init_runner()

        # Determine target keys: explicit list, or every reviewer with no saved
        # review file. ``list_existing_reviews`` is provided by the storage
        # backend, so this works identically against future remote backends.
        if only_keys is None:
            already = set(self.storage.list_existing_reviews())
            only_keys = [k for k in REVIEWER_KEYS if k not in already]
        if not only_keys:
            return {"retried": [], "succeeded": [], "failed": {}}

        v1_text = self.storage.read_text("paper_v1.txt")

        # Rebuild the initial message identical to the first run.
        paper_info = self.paper_analyzer.extract_info(paper_text)
        citation_block = self._build_citation_block(paper_text)
        statcheck_block = self._build_statcheck_block(paper_text)
        diff_block = self._build_diff_block(v1_text, paper_text) if v1_text else ""
        message = self._prepare_initial_message(
            paper_info, paper_text,
            citation_block=citation_block,
            statcheck_block=statcheck_block,
            diff_block=diff_block,
        )

        logger.info("Retrying %d agents: %s", len(only_keys), only_keys)
        assert self.runner is not None  # _init_runner just ran
        reviews, errors = asyncio.run(self.runner.run_batch(list(only_keys), message))

        return {
            "retried": list(only_keys),
            "succeeded": list(reviews.keys()),
            "failed": errors,
        }

    def _build_diff_block(self, v1_text: str, v2_text: str) -> str:
        """Compute the v1→v2 section diff. Best-effort; returns '' on failure."""
        try:
            d = paper_diff(v1_text, v2_text)
        except Exception as e:  # noqa: BLE001
            logger.warning("paper diff failed: %s", e)
            return ""
        logger.info(
            "Revision diff: overall similarity %.2f · +%d added · -%d removed · ~%d modified.",
            d.overall_similarity, len(d.added_sections),
            len(d.removed_sections), len(d.modified_sections),
        )
        return format_diff_for_agent(d)

    def _build_statcheck_block(self, paper_text: str) -> str:
        """Run the bundled R script. Best-effort; returns '' when disabled."""
        if not getattr(self.config, "enrich_with_statcheck", True):
            return ""
        try:
            report = run_statcheck(paper_text)
        except Exception as e:  # noqa: BLE001
            logger.warning("statcheck run failed: %s", e)
            return ""
        text = format_statcheck_for_agent(report)
        if report.available:
            logger.info(
                "statcheck: %d stats, %d numerical errors, %d decision errors.",
                report.n_stats, report.n_errors, report.n_decision_errors,
            )
        else:
            logger.info("statcheck unavailable: %s", report.reason)
        return text

    def _build_citation_block(self, paper_text: str) -> str:
        """Fetch OpenAlex validation of every citation we can extract.
        Returns '' when disabled in config or when extraction yields nothing.
        Network failures are swallowed and noted — the block is best-effort."""
        if not getattr(self.config, "enrich_with_citations", True):
            return ""
        try:
            refs = extract_citations(paper_text)
        except Exception as e:  # noqa: BLE001
            logger.warning("Citation extraction failed: %s", e)
            return ""
        if not refs:
            return "No citations could be extracted from the paper text."
        try:
            report = asyncio.run(validate_citations(refs))
        except Exception as e:  # noqa: BLE001
            logger.warning("OpenAlex validation failed: %s", e)
            return "Citation extraction succeeded but OpenAlex validation could not run."
        try:
            text = format_report_for_agent(report)
        except Exception as e:  # noqa: BLE001
            logger.warning("Citation report formatting failed: %s", e)
            return ""
        logger.info(
            "Citation enrichment: %d refs (%d DOI-resolved, %d title-matched, "
            "%d not-found, %d likely-fabricated).",
            report.total, report.by_doi, report.by_title,
            report.not_found, report.likely_fabricated,
        )
        return text

    def _init_runner(self) -> None:
        """Build the ConcurrentAgentRunner once agents exist for this run."""
        self.runner = ConcurrentAgentRunner(
            agents=self.agents,
            storage=self.storage,
            audit=self.audit,
            event_bus=self.event_bus,
            run_id=self.run_id,
        )

    def _execute_main_reviewers(
        self, initial_message: str
    ) -> tuple[dict[str, AnnotatedReview], dict[str, str]]:
        assert self.runner is not None  # _init_runner ran in execute_review_process
        return asyncio.run(self.runner.run_batch(list(REVIEWER_KEYS), initial_message))

    def _execute_coordinator(
        self, reviews: dict[str, AnnotatedReview]
    ) -> AnnotatedCoordinatorAssessment | None:
        coordinator = self.agents.get("coordinator")
        if not coordinator:
            logger.error("Coordinator agent not found")
            return None
        if not reviews:
            logger.warning("No successful reviewer outputs; skipping coordinator")
            return None
        reviews_text = "\n\n".join(_format_review_for_consumer(k, r) for k, r in reviews.items())
        msg = (
            _run_context_preamble()
            + "\nHere are the structured reviewer outputs:\n\n"
            f"{reviews_text}\n\n"
            "Please provide your comprehensive coordinator assessment as a structured object.\n"
        )
        try:
            result = coordinator.run(msg, audit=self.audit)
            if not isinstance(result, CoordinatorAssessment):
                logger.error("Coordinator returned non-CoordinatorAssessment: %r", type(result))
                return None
            annotated = _annotate(AnnotatedCoordinatorAssessment, result, coordinator)
            self.storage.save_review("coordinator", annotated.model_dump_json(indent=2))
            return annotated
        except Exception as e:
            logger.error("Error in coordinator: %s", _format_exception(e))
            return None

    def _execute_author_editor_summary(
        self,
        reviews: dict[str, AnnotatedReview],
        coordinator: AnnotatedCoordinatorAssessment | None,
    ) -> AnnotatedAuthorEditorSummary | None:
        agent = self.agents.get("author_editor_summary")
        if not agent:
            return None
        reviews_text = "\n\n".join(_format_review_for_consumer(k, r) for k, r in reviews.items())
        coord_text = _format_coordinator_for_consumer(coordinator) if coordinator else "(no coordinator output)"
        msg = (
            _run_context_preamble()
            + "\nReviewer outputs:\n\n"
            f"{reviews_text}\n\n"
            "Coordinator assessment:\n\n"
            f"{coord_text}\n\n"
            "Please provide the two requested summaries as a single structured object.\n"
        )
        try:
            result = agent.run(msg, audit=self.audit)
            if not isinstance(result, AuthorEditorSummary):
                logger.error("Summary agent returned non-AuthorEditorSummary: %r", type(result))
                return None
            annotated = _annotate(AnnotatedAuthorEditorSummary, result, agent)
            self.storage.save_review("author_editor_summary", annotated.model_dump_json(indent=2))
            return annotated
        except Exception as e:
            logger.error("Error in author/editor summary agent: %s", _format_exception(e))
            return None

    def _execute_editor(
        self,
        reviews: dict[str, AnnotatedReview],
        coordinator: AnnotatedCoordinatorAssessment | None,
        summary: AnnotatedAuthorEditorSummary | None,
    ) -> AnnotatedEditorDecision | None:
        editor = self.agents.get("editor")
        if not editor:
            return None
        reviews_text = "\n\n".join(_format_review_for_consumer(k, r) for k, r in reviews.items())
        coord_text = _format_coordinator_for_consumer(coordinator) if coordinator else "(no coordinator output)"
        summary_text = (
            (f"\nRecommendation for the author:\n{summary.recommendation_for_author}\n\n"
             f"Confidential recommendation for the editor only:\n{summary.recommendation_for_editor_only}\n")
            if summary else ""
        )
        msg = (
            _run_context_preamble()
            + "\nReviewer outputs:\n\n"
            f"{reviews_text}\n\n"
            "Coordinator assessment:\n\n"
            f"{coord_text}\n"
            f"{summary_text}\n"
            "Please provide your editorial decision as a structured object.\n"
        )
        try:
            result = editor.run(msg, audit=self.audit)
            if not isinstance(result, EditorDecision):
                logger.error("Editor returned non-EditorDecision: %r", type(result))
                return None
            annotated = _annotate(AnnotatedEditorDecision, result, editor)
            self.storage.save_review("editor", annotated.model_dump_json(indent=2))
            return annotated
        except Exception as e:
            logger.error("Error in editor: %s", _format_exception(e))
            return None

    # ------------------------------------------------------------------ output

    def _synthesize_results(
        self,
        *,
        paper_info: PaperInfo,
        reviews: dict[str, AnnotatedReview],
        errors: dict[str, str],
        coordinator: AnnotatedCoordinatorAssessment | None,
        summary: AnnotatedAuthorEditorSummary | None,
        editor: AnnotatedEditorDecision | None,
    ) -> dict[str, Any]:
        per_agent_routing: dict[str, dict[str, Any]] = {}
        for key, agent in self.agents.items():
            per_agent_routing[key] = {
                "provider": agent.provider.name if agent.provider else None,
                "model": agent.model,
                "thinking_budget": agent.thinking_budget,
                "schema": getattr(agent.schema, "__name__", None),
            }
        return {
            "paper_info": paper_info.to_dict(),
            "reviews": {k: r.model_dump() for k, r in reviews.items()},
            "errors": errors,
            "coordinator": coordinator.model_dump() if coordinator else None,
            "author_editor_summary": summary.model_dump() if summary else None,
            "editor_decision": editor.model_dump() if editor else None,
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "config": {
                "models_used": {
                    "powerful": self.config.model_powerful,
                    "standard": self.config.model_standard,
                    "basic": self.config.model_basic,
                },
                "num_reviewers": len(reviews),
                "providers": self.registry.names(),
                "routing": per_agent_routing,
                "seed": self.config.seed,
            },
            "audit_summary": self.audit.summary(),
        }

    def _generate_reports(self, results: dict[str, Any]) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.storage.save_text(render_markdown(results), f"review_report_{ts}.md")
        self.storage.save_json(results, f"review_results_{ts}.json")
        self.storage.save_text(
            render_executive_summary(results), f"executive_summary_{ts}.md"
        )
        self.storage.save_text(render_html(results), f"dashboard_{ts}.html")
