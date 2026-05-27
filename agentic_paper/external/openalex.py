"""Thin async client for OpenAlex (https://api.openalex.org).

OpenAlex is a free, open-access scholarly database with no API-key requirement.
We use two endpoints:
    - GET /works/doi:<DOI>          — resolve a DOI to a Work
    - GET /works?search=<title>     — fuzzy title search, top match returned

The optional ``mailto`` URL parameter (or ``OPENALEX_MAILTO`` env var) puts
you in the "polite pool" with higher rate limits — recommended for production.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org"
DEFAULT_TIMEOUT = 8.0


@dataclass
class OpenAlexWork:
    """A subset of OpenAlex's Work record, normalized for our use."""

    id: str                       # https://openalex.org/W...
    doi: str | None
    title: str
    year: int | None
    authors: list[str] = field(default_factory=list)
    cited_by_count: int = 0
    venue: str | None = None
    open_access: bool = False
    raw: dict[str, Any] | None = None

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> "OpenAlexWork":
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in payload.get("authorships", [])
        ]
        venue = (payload.get("primary_location") or {}).get("source", {}) or {}
        return cls(
            id=payload.get("id", "") or "",
            doi=payload.get("doi"),
            title=payload.get("title") or payload.get("display_name") or "",
            year=payload.get("publication_year"),
            authors=[a for a in authors if a],
            cited_by_count=payload.get("cited_by_count", 0) or 0,
            venue=venue.get("display_name") if isinstance(venue, dict) else None,
            open_access=(payload.get("open_access") or {}).get("is_oa", False),
            raw=payload,
        )


class OpenAlexClient:
    """Async OpenAlex client. Single shared httpx.AsyncClient per instance."""

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        mailto: str | None = None,
        user_agent: str = "agentic-paper/2.0 (https://github.com/albertogerli/Agentic_Paper)",
    ) -> None:
        self.timeout = timeout
        self.mailto = mailto or os.environ.get("OPENALEX_MAILTO") or None
        self.user_agent = user_agent
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "OpenAlexClient":
        params = {"mailto": self.mailto} if self.mailto else {}
        self._client = httpx.AsyncClient(
            base_url=OPENALEX_BASE,
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent},
            params=params,
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------ search

    async def get_by_doi(self, doi: str) -> OpenAlexWork | None:
        """Resolve a DOI to a Work. Returns None on 404 / network error."""
        if not self._client:
            raise RuntimeError("Use `async with OpenAlexClient()` first")
        path = "/works/doi:" + doi.lstrip("/").lstrip("doi:")
        try:
            r = await self._client.get(path)
        except httpx.HTTPError as e:
            logger.warning("OpenAlex DOI lookup failed for %s: %s", doi, e)
            return None
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            logger.warning("OpenAlex DOI lookup %s returned %s", doi, r.status_code)
            return None
        try:
            return OpenAlexWork.from_api(r.json())
        except Exception as e:  # noqa: BLE001
            logger.warning("OpenAlex DOI response parse failed: %s", e)
            return None

    async def search_title(self, title: str, *, per_page: int = 3) -> list[OpenAlexWork]:
        """Fuzzy title search; returns up to ``per_page`` works, best first."""
        if not self._client:
            raise RuntimeError("Use `async with OpenAlexClient()` first")
        if not title.strip():
            return []
        try:
            r = await self._client.get(
                "/works",
                params={"search": title.strip()[:200], "per-page": per_page},
            )
        except httpx.HTTPError as e:
            logger.warning("OpenAlex title search failed for %r: %s", title[:40], e)
            return []
        if r.status_code != 200:
            return []
        try:
            results = r.json().get("results", []) or []
        except Exception:  # noqa: BLE001
            return []
        return [OpenAlexWork.from_api(w) for w in results]
