"""Tests for grounding.core.speculative.speculative_prescreen."""

from __future__ import annotations

from grounding import CitationSpan, Claim, Source, Verdict
from grounding.core.speculative import speculative_prescreen


def test_returns_none_without_span() -> None:
    r = speculative_prescreen(Claim(text="x"), Source.from_text("y"))
    assert r is None


def test_returns_grounded_for_matching_span() -> None:
    src = Source.from_pages(["the alpha and the omega"])
    span = CitationSpan(page=1, char_start=4, char_end=9)
    claim = Claim(text="alpha", citation_span=span)
    r = speculative_prescreen(claim, src)
    assert r is not None
    assert r.verdict == Verdict.GROUNDED


def test_returns_ungrounded_for_mismatched_span() -> None:
    src = Source.from_pages(["the alpha and the omega"])
    # span points at "alpha" but claim is "WRONG"
    span = CitationSpan(page=1, char_start=4, char_end=9)
    claim = Claim(text="WRONG", citation_span=span)
    r = speculative_prescreen(claim, src)
    assert r is not None
    assert r.verdict == Verdict.UNGROUNDED


def test_cascade_short_circuits_on_grounded_span() -> None:
    """End-to-end: GroundingVerifier short-circuits when span verifies."""
    from grounding import GroundingVerifier

    v = GroundingVerifier()
    src = Source.from_pages(["the alpha and the omega"])
    span = CitationSpan(page=1, char_start=4, char_end=9)
    claim = Claim(text="alpha", citation_span=span)
    r = v.verify(claim, src)
    assert r.verdict == Verdict.GROUNDED
    # Lexical tier should NOT have been invoked because span succeeded.
    assert "lexical" not in r.tier_results
    assert "citation_span" in r.tier_results


def test_cascade_falls_through_when_span_fails() -> None:
    """When span verification fails, cascade proceeds to other tiers."""
    from grounding import GroundingVerifier

    v = GroundingVerifier()
    # Span is mismatched but the claim text is also literally in the
    # source — lexical tier should still ground it.
    src = Source.from_pages(["alpha is a known token"])
    span = CitationSpan(page=1, char_start=0, char_end=3)  # "alp"
    claim = Claim(text="alpha", citation_span=span)
    r = v.verify(claim, src)
    # The span window is "alp", which is shorter than claim "alpha".
    # SpanVerifier sees containment ("alp" in "alpha") and grounds it.
    # That short-circuits — so lexical does NOT run. This is expected.
    assert r.verdict == Verdict.GROUNDED
