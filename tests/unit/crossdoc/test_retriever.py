"""Tests for grounding.crossdoc.retriever.CrossDocVerifier."""

from __future__ import annotations

from grounding import Claim, Source, Verdict
from grounding.crossdoc.linker import DocumentRef
from grounding.crossdoc.retriever import CrossDocVerifier


def test_skipped_for_empty_corpus() -> None:
    v = CrossDocVerifier()
    r = v.verify(Claim(text="x"), [])
    assert r.verdict == Verdict.SKIPPED


def test_skipped_when_no_link() -> None:
    v = CrossDocVerifier()
    corpus = [
        DocumentRef(
            doc_id="d1",
            name="Loan Agreement",
            source=Source.from_text("loan content"),
        )
    ]
    r = v.verify(Claim(text="unrelated narrative"), corpus)
    assert r.verdict == Verdict.SKIPPED


def test_grounded_via_linked_doc() -> None:
    v = CrossDocVerifier()
    corpus = [
        DocumentRef(
            doc_id="d1",
            name="Loan Agreement",
            source=Source.from_text(
                "the rate per annum is 4.5%"
            ),
        )
    ]
    claim = Claim(
        text="As per the Loan Agreement, the rate per annum is 4.5%",
    )
    r = v.verify(claim, corpus)
    assert r.verdict == Verdict.GROUNDED
    assert r.evidence
    assert r.evidence[0].doc_id == "d1"


def test_ungrounded_when_link_exists_but_content_absent() -> None:
    v = CrossDocVerifier()
    corpus = [
        DocumentRef(
            doc_id="d1",
            name="Loan Agreement",
            source=Source.from_text("only mentions equity"),
        )
    ]
    claim = Claim(
        text="As per the Loan Agreement, debt equals 8 million",
    )
    r = v.verify(claim, corpus)
    assert r.verdict == Verdict.UNGROUNDED


def test_uses_retriever_when_provided() -> None:
    """When a RetrievalFn is injected the verifier narrows the search."""
    calls = []

    def _retriever(*, query, top_k=5):
        calls.append(query)
        return [{"text": "rate 4.5% specified"}]

    v = CrossDocVerifier(retriever=_retriever)
    corpus = [
        DocumentRef(
            doc_id="d1",
            name="Loan Agreement",
            source=Source.from_text("a much larger document text..."),
        )
    ]
    claim = Claim(text="Loan Agreement: rate 4.5% specified")
    r = v.verify(claim, corpus)
    assert r.verdict == Verdict.GROUNDED
    assert calls  # retriever was invoked


def test_retriever_failure_falls_back_to_doc_source() -> None:
    def _broken_retriever(*, query, top_k=5):
        raise RuntimeError("boom")

    v = CrossDocVerifier(retriever=_broken_retriever)
    corpus = [
        DocumentRef(
            doc_id="d1",
            name="Loan Agreement",
            source=Source.from_text("rate 4.5% specified"),
        )
    ]
    claim = Claim(text="Loan Agreement: rate 4.5% specified")
    r = v.verify(claim, corpus)
    assert r.verdict == Verdict.GROUNDED
