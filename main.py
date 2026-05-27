"""Backward-compatible CLI shim.

Historic entry point — kept so existing shell invocations (`python main.py paper.pdf`)
keep working. The canonical entry point is :mod:`agentic_paper.cli`.
"""

from __future__ import annotations

import sys

from agentic_paper.cli import main

if __name__ == "__main__":
    sys.exit(main())
