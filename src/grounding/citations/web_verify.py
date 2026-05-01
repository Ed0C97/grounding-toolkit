"""Web citation verifier.

Migrated from Sentinel's
``sentinel.cross_phase.citation_verifier`` (P16 hard cutover).  The
verifier fetches the cited URL via a consumer-supplied async fetcher
and checks whether the quoted text appears (exact substring or
SequenceMatcher fuzzy match above a threshold).
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Awaitable, Callable, Optional


@dataclass
class CitationVerdict:
    """Outcome of a single web-citation verification."""

    url: str
    reachable: bool
    quote_found: bool
    similarity: float
    verdict: str  # "verified" | "unverified" | "hallucinated"


HttpFetcher = Callable[[str], Awaitable[Optional[str]]]


async def verify_citation(
    url: str,
    *,
    quote: str,
    fetcher: HttpFetcher,
    similarity_threshold: float = 0.85,
) -> CitationVerdict:
    """Verify that ``quote`` is present at ``url``.

    Steps:
    1. ``fetcher(url)`` returns the page body or ``None`` (unreachable).
    2. If reachable and the quote is non-empty, exact substring match
       wins (similarity 1.0, verdict ``"verified"``).
    3. Otherwise, SequenceMatcher.quick_ratio is computed against the
       first ``len(quote) * 4`` chars of the body (cheap upper bound).
       The verdict is ``"verified"`` when the similarity meets the
       threshold, else ``"unverified"``.
    4. An empty quote returns ``"unverified"``.
    5. An unreachable URL returns ``"hallucinated"``.
    """
    body = await fetcher(url)
    if body is None:
        return CitationVerdict(
            url=url,
            reachable=False,
            quote_found=False,
            similarity=0.0,
            verdict="hallucinated",
        )

    if not quote.strip():
        return CitationVerdict(
            url=url,
            reachable=True,
            quote_found=False,
            similarity=0.0,
            verdict="unverified",
        )

    needle = quote.lower().strip()
    haystack = body.lower()
    if needle in haystack:
        return CitationVerdict(
            url=url,
            reachable=True,
            quote_found=True,
            similarity=1.0,
            verdict="verified",
        )
    similarity = SequenceMatcher(
        None, needle, haystack[: len(needle) * 4]
    ).quick_ratio()
    return CitationVerdict(
        url=url,
        reachable=True,
        quote_found=False,
        similarity=similarity,
        verdict=(
            "verified"
            if similarity >= similarity_threshold
            else "unverified"
        ),
    )


__all__ = ["CitationVerdict", "HttpFetcher", "verify_citation"]
