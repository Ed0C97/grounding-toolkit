"""Tests for grounding.citations.span.SpanVerifier."""

from __future__ import annotations

from grounding import CitationSpan, Claim, Source, Verdict
from grounding.citations.span import SpanVerifier


def _src_with_pages(*pages: str, doc_id: str = "d") -> Source:
    return Source.from_pages(list(pages), doc_id=doc_id)


def test_skipped_when_no_span() -> None:
    v = SpanVerifier()
    r = v.verify(Claim(text="x"), Source.from_text("y"))
    assert r.verdict == Verdict.SKIPPED


def test_grounded_for_exact_span_match() -> None:
    v = SpanVerifier()
    page_text = "The total debt is EUR 8.4 million as of 2025-12-31."
    src = _src_with_pages(page_text)
    span = CitationSpan(page=1, char_start=4, char_end=29)
    # cited window: "total debt is EUR 8.4 mil" — claim within
    claim = Claim(text="total debt is EUR 8.4 mil", citation_span=span)
    r = v.verify(claim, src)
    assert r.verdict == Verdict.GROUNDED
    assert r.score == 1.0
    assert r.evidence
    assert r.evidence[0].page == 1
    assert r.evidence[0].char_start == 4
    assert r.evidence[0].char_end == 29
    assert r.evidence[0].doc_id == "d"


def test_grounded_when_cited_text_contains_claim() -> None:
    v = SpanVerifier()
    page_text = "Section 4: total debt is EUR 8.4 million; ..."
    src = _src_with_pages(page_text)
    # span covers the entire sentence; claim is shorter
    span = CitationSpan(page=1, char_start=0, char_end=len(page_text))
    claim = Claim(
        text="total debt is EUR 8.4 million",
        citation_span=span,
    )
    r = v.verify(claim, src)
    assert r.verdict == Verdict.GROUNDED


def test_grounded_for_fuzzy_span_match() -> None:
    v = SpanVerifier(similarity_threshold=0.85)
    page_text = "The quick brown fox jumps over the lazy dog."
    src = _src_with_pages(page_text)
    # Span exactly fits "quick brown fox" (chars 4..19)
    span = CitationSpan(page=1, char_start=4, char_end=19)
    # claim has minor drift
    claim = Claim(text="quick brown fxo", citation_span=span)
    r = v.verify(claim, src)
    # 14/15 chars match exactly; ratio with one swap is >= 0.85
    assert r.verdict == Verdict.GROUNDED


def test_ungrounded_when_span_text_differs() -> None:
    v = SpanVerifier(similarity_threshold=0.85)
    page_text = "The benign source corpus contains nothing relevant."
    src = _src_with_pages(page_text)
    span = CitationSpan(page=1, char_start=0, char_end=20)
    claim = Claim(
        text="totally fabricated claim", citation_span=span,
    )
    r = v.verify(claim, src)
    assert r.verdict == Verdict.UNGROUNDED


def test_ungrounded_when_page_missing() -> None:
    v = SpanVerifier()
    src = _src_with_pages("only one page")
    span = CitationSpan(page=99, char_start=0, char_end=5)
    claim = Claim(text="x", citation_span=span)
    r = v.verify(claim, src)
    assert r.verdict == Verdict.UNGROUNDED
    assert "page 99 not in source" in r.detail


def test_ungrounded_when_span_out_of_bounds() -> None:
    v = SpanVerifier()
    src = _src_with_pages("short page")
    span = CitationSpan(page=1, char_start=0, char_end=999)
    claim = Claim(text="x", citation_span=span)
    r = v.verify(claim, src)
    assert r.verdict == Verdict.UNGROUNDED
    assert "out of page bounds" in r.detail


def test_ungrounded_when_span_malformed() -> None:
    v = SpanVerifier()
    src = _src_with_pages("page text")
    span = CitationSpan(page=1, char_start=10, char_end=5)
    claim = Claim(text="x", citation_span=span)
    r = v.verify(claim, src)
    assert r.verdict == Verdict.UNGROUNDED
    assert "malformed" in r.detail


def test_ungrounded_when_page_zero() -> None:
    v = SpanVerifier()
    src = _src_with_pages("page text")
    span = CitationSpan(page=0, char_start=0, char_end=4)
    claim = Claim(text="x", citation_span=span)
    r = v.verify(claim, src)
    assert r.verdict == Verdict.UNGROUNDED


def test_falls_back_to_text_when_no_pages_and_page_one() -> None:
    """Source without pages but with text on page 1 still verifies."""
    v = SpanVerifier()
    src = Source.from_text("the entire document text")
    span = CitationSpan(page=1, char_start=4, char_end=10)
    claim = Claim(text="entire", citation_span=span)
    r = v.verify(claim, src)
    assert r.verdict == Verdict.GROUNDED
