"""Tests for grounding.citations.web_verify."""

from __future__ import annotations

import pytest

from grounding.citations.web_verify import (
    CitationVerdict,
    verify_citation,
)


@pytest.mark.asyncio
async def test_unreachable_url_marked_hallucinated() -> None:
    async def _fetcher(url):  # noqa: ARG001
        return None

    out = await verify_citation(
        "https://example.com",
        quote="anything",
        fetcher=_fetcher,
    )
    assert out.verdict == "hallucinated"
    assert not out.reachable


@pytest.mark.asyncio
async def test_exact_quote_marked_verified() -> None:
    async def _fetcher(url):  # noqa: ARG001
        return "the page says EBITDA grew strongly this year"

    out = await verify_citation(
        "https://example.com",
        quote="EBITDA grew strongly",
        fetcher=_fetcher,
    )
    assert out.verdict == "verified"
    assert out.quote_found
    assert out.similarity == 1.0


@pytest.mark.asyncio
async def test_fuzzy_above_threshold_verified() -> None:
    async def _fetcher(url):  # noqa: ARG001
        return "the document says EBITDA was strong"

    out = await verify_citation(
        "https://example.com",
        quote="EBITDA was strong",
        fetcher=_fetcher,
        similarity_threshold=0.5,
    )
    assert out.verdict == "verified"


@pytest.mark.asyncio
async def test_fuzzy_below_threshold_unverified() -> None:
    async def _fetcher(url):  # noqa: ARG001
        return "completely unrelated content"

    out = await verify_citation(
        "https://example.com",
        quote="EBITDA grew strongly",
        fetcher=_fetcher,
        similarity_threshold=0.85,
    )
    assert out.verdict == "unverified"


@pytest.mark.asyncio
async def test_empty_quote_marked_unverified() -> None:
    async def _fetcher(url):  # noqa: ARG001
        return "any body"

    out = await verify_citation(
        "https://example.com",
        quote="",
        fetcher=_fetcher,
    )
    assert out.verdict == "unverified"
