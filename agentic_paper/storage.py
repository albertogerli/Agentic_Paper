"""Storage abstraction for review artefacts.

The orchestrator persists everything by *filename* (e.g. ``paper_info.json``,
``review_methodology.txt``, ``review_report_<ts>.md``) inside a run-scoped
namespace. This module hides where that namespace lives.

Concrete implementations bundled here:
    * :class:`LocalFileStorage` — writes under a single output directory on
      the local filesystem. Bit-for-bit compatible with the v1 ``FileManager``
      so existing run-dir consumers (web layer, CLI, tests, audit log tail)
      keep working unchanged.

Future implementations will plug in without touching the orchestrator:
    * ``S3Storage(bucket, prefix)`` — push to S3 for shared / archival runs.
    * ``DatabaseStorage(conn, run_id)`` — review rows in Postgres.
    * ``InMemoryStorage()`` — for fast unit tests that don't want disk I/O.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StorageProvider(ABC):
    """Abstract sink for review artefacts.

    Every method works on *filenames* (relative to whatever run-scoped
    namespace the implementation owns). The orchestrator never knows whether
    files end up on disk, S3, a DB, or RAM.
    """

    @abstractmethod
    def save_json(self, data: Any, filename: str) -> bool:
        """Persist ``data`` as pretty-printed UTF-8 JSON under ``filename``.
        Returns True on success, False on any I/O / serialization error."""

    @abstractmethod
    def save_text(self, text: str, filename: str) -> bool:
        """Persist ``text`` as UTF-8 under ``filename``. Returns True on success."""

    @abstractmethod
    def read_text(self, filename: str) -> str | None:
        """Read ``filename`` back as UTF-8 text. Returns ``None`` when the
        file is missing or unreadable — callers branch on None, never on
        exceptions raised from this method."""

    @abstractmethod
    def save_review(self, reviewer_name: str, content: str) -> str:
        """Persist one reviewer's output. The filename mapping is an
        implementation detail; callers receive a human-readable status string.

        For the local backend, this writes ``review_<reviewer_name>.txt``
        with spaces replaced by underscores."""

    @abstractmethod
    def get_reviews(self) -> dict[str, str]:
        """Return all reviews stored so far as ``{reviewer_name: content}``."""

    @abstractmethod
    def list_existing_reviews(self) -> list[str]:
        """Return the agent-key part of every saved review (e.g.
        ``methodology``, ``citation_validator``). Preserves underscores as
        stored — this is the inverse of ``save_review`` for keys that don't
        contain spaces, which is the case for every key in ``REVIEWER_KEYS``.

        Used by ``retry_failed_agents`` to decide which reviewers still need
        to run; remote backends (S3/DB) override with a native list."""


class LocalFileStorage(StorageProvider):
    """Reference implementation: a single output directory on local disk.

    Bit-for-bit compatible with the v1 ``FileManager`` storage operations —
    same filenames, same JSON formatting, same log messages on save / error.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    # ------------------------------------------------------------------ writes

    def save_json(self, data: Any, filename: str) -> bool:
        filepath = self.output_dir / filename
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("JSON saved: %s", filepath)
            return True
        except Exception as e:
            logger.error("Error saving JSON %s: %s", filepath, e)
            return False

    def save_text(self, text: str, filename: str) -> bool:
        filepath = self.output_dir / filename
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
            logger.info("Text file saved: %s", filepath)
            return True
        except Exception as e:
            logger.error("Error saving text file %s: %s", filepath, e)
            return False

    def save_review(self, reviewer_name: str, content: str) -> str:
        filename = f"review_{reviewer_name.replace(' ', '_')}.txt"
        if self.save_text(content, filename):
            return f"Review successfully saved in {filename}"
        return f"Error saving review for {reviewer_name}"

    # ------------------------------------------------------------------ reads

    def read_text(self, filename: str) -> str | None:
        filepath = self.output_dir / filename
        try:
            return filepath.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            # Missing file is the common case for the retry path; downgrade
            # to debug so we don't spam logs.
            logger.debug("read_text(%s) failed: %s", filepath, e)
            return None

    def list_existing_reviews(self) -> list[str]:
        if not self.output_dir.exists():
            return []
        return [
            p.stem.removeprefix("review_")
            for p in self.output_dir.glob("review_*.txt")
        ]

    def get_reviews(self) -> dict[str, str]:
        reviews: dict[str, str] = {}
        if not self.output_dir.exists():
            logger.warning("Output directory does not exist")
            return reviews
        try:
            for filepath in self.output_dir.glob("review_*.txt"):
                # NOTE: spaces-to-underscores in save_review is lossy in this
                # direction; preserved verbatim for v1 callers that already
                # rely on the "Methodology Expert" reverse-mapping.
                reviewer_name = filepath.stem[7:].replace("_", " ")
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        reviews[reviewer_name] = f.read()
                except Exception as e:
                    logger.error("Error reading review %s: %s", filepath, e)
        except Exception as e:
            logger.error("Error accessing reviews: %s", e)
        return reviews


__all__ = ["StorageProvider", "LocalFileStorage"]
