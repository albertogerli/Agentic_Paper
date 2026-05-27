"""Multi-Agent System for Scientific Paper Review.

Public entry points:
    from agentic_paper import Config, ReviewOrchestrator
    from agentic_paper.cli import main
"""

from __future__ import annotations

__version__ = "2.1.0"

from .config import Config, setup_logging
from .orchestrator import ReviewOrchestrator

__all__ = ["Config", "ReviewOrchestrator", "setup_logging", "__version__"]
