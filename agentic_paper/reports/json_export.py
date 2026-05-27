"""Persist the structured review-result dict as JSON."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def export_json(results: dict[str, Any], filepath: str | Path) -> bool:
    """Write ``results`` to ``filepath`` as pretty-printed UTF-8 JSON."""
    filepath = Path(filepath)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("JSON saved: %s", filepath)
        return True
    except Exception as e:
        logger.error("Error saving JSON %s: %s", filepath, e)
        return False
