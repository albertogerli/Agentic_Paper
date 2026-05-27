"""External-data helpers used by the citation-validator agent.

Currently wraps OpenAlex (https://openalex.org), which is fully open, requires
no API key, and rate-limits politely at ~10 req/s per IP for unauthenticated
clients. The optional ``mailto`` URL parameter gets you into a "polite pool"
with higher limits — set the ``OPENALEX_MAILTO`` env var if you want it.
"""

from __future__ import annotations

__all__: list[str] = []
