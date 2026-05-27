"""Background runner that owns one orchestrator invocation per ``run_id``.

Keeps the heavy lifting where it already lives (`agentic_paper.orchestrator`)
and only adds enough bookkeeping to let the web layer fan out status events.
"""

from __future__ import annotations

import logging
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..audit import generate_run_id
from ..config import Config, setup_logging
from .event_bus import RunEventBus

logger = logging.getLogger(__name__)


@dataclass
class RunStatus:
    run_id: str
    pdf_path: Path
    state: str = "queued"  # queued | running | completed | failed
    error: Optional[str] = None
    output_dir: Optional[Path] = None
    pdf_v1_path: Optional[Path] = None
    # Human-readable warnings produced by ``routing.apply_auto_mode`` when the
    # user's submitted keys did not cover every tier of the selected profile.
    # Empty by default; populated by ``RunRegistry.start`` when the web layer
    # detects BYOK gaps. Rendered as a banner on the run page.
    auto_mode_warnings: list[str] = field(default_factory=list)


# Public protocol for the runner so tests can stub it cheaply.
RunnerFn = Callable[[Path, str, Config], None]


def _extract_text(pdf_path: Path, file_manager) -> str:
    if str(pdf_path).lower().endswith(".pdf"):
        return file_manager.extract_text_from_pdf(str(pdf_path))
    return file_manager.read_paper(str(pdf_path)) or ""


def _default_runner(
    pdf_path: Path,
    run_id: str,
    config: Config,
    *,
    pdf_v1_path: Path | None = None,
    event_bus: RunEventBus | None = None,
) -> None:
    """Real runner — calls into the existing ReviewOrchestrator."""
    # Local import so the web module does not pull the world at import time.
    from ..orchestrator import ReviewOrchestrator

    orch = ReviewOrchestrator(config, run_id=run_id, event_bus=event_bus)
    text = _extract_text(pdf_path, orch.file_manager)
    if not text:
        raise RuntimeError(f"Could not extract text from {pdf_path}")
    v1_text: str | None = None
    if pdf_v1_path is not None:
        v1_text = _extract_text(pdf_v1_path, orch.file_manager) or None
        if not v1_text:
            logger.warning("v1 file %s yielded empty text — running single-version review", pdf_v1_path)
    orch.execute_review_process(text, v1_text=v1_text)


class RunRegistry:
    """In-process registry of background runs."""

    def __init__(
        self,
        config_path: str = "config.yaml",
        max_workers: int = 2,
        runner: RunnerFn | None = None,
    ) -> None:
        self.config_path = config_path
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="agentic-paper-runner"
        )
        self.runs: dict[str, RunStatus] = {}
        self._lock = threading.Lock()
        self.runner: RunnerFn = runner or _default_runner
        self.event_bus = RunEventBus()

    def start(
        self,
        pdf_path: Path,
        *,
        config: Config | None = None,
        cleanup_pdf: bool = False,
        pdf_v1_path: Path | None = None,
        auto_mode_warnings: list[str] | None = None,
    ) -> str:
        """Kick off a background run; returns the new run_id immediately.

        When ``cleanup_pdf`` is True, the source ``pdf_path`` (and ``pdf_v1_path``
        if given) is unlinked after the worker finishes (success or failure).

        ``auto_mode_warnings`` carries any tier-remap notices produced by
        :func:`agentic_paper.routing.apply_auto_mode` so the run page can show
        them in a banner.
        """
        run_id = generate_run_id()
        cfg = config or Config.from_yaml(self.config_path)
        run_dir = Path(cfg.output_dir) / run_id
        status = RunStatus(
            run_id=run_id, pdf_path=Path(pdf_path), output_dir=run_dir,
            pdf_v1_path=Path(pdf_v1_path) if pdf_v1_path else None,
            auto_mode_warnings=list(auto_mode_warnings or []),
        )
        with self._lock:
            self.runs[run_id] = status

        def _go() -> None:
            # Each worker thread sets up its own logger sink under the run dir.
            run_dir.mkdir(parents=True, exist_ok=True)
            setup_logging("INFO", log_file=str(run_dir / "paper_review_system.log"))
            try:
                status.state = "running"
                # The default runner accepts pdf_v1_path + event_bus kwargs;
                # custom test runners may not — pass only when applicable.
                if self.runner is _default_runner:
                    self.runner(
                        Path(pdf_path), run_id, cfg,
                        pdf_v1_path=Path(pdf_v1_path) if pdf_v1_path else None,
                        event_bus=self.event_bus,
                    )
                elif pdf_v1_path is not None:
                    self.runner(Path(pdf_path), run_id, cfg, pdf_v1_path=Path(pdf_v1_path))
                else:
                    self.runner(Path(pdf_path), run_id, cfg)
                status.state = "completed"
            except Exception as e:
                status.state = "failed"
                status.error = f"{type(e).__name__}: {e}"
                logger.error("Run %s failed: %s\n%s", run_id, e, traceback.format_exc())
            finally:
                if cleanup_pdf:
                    for path_to_clean in (pdf_path, pdf_v1_path):
                        if path_to_clean is None:
                            continue
                        try:
                            Path(path_to_clean).unlink(missing_ok=True)
                        except OSError as e:
                            logger.warning("Could not delete temp upload %s: %s", path_to_clean, e)

        self.executor.submit(_go)
        return run_id

    def get(self, run_id: str) -> RunStatus | None:
        with self._lock:
            return self.runs.get(run_id)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
