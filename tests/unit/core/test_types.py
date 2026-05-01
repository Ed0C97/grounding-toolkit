"""Tests for grounding.core.types."""

from __future__ import annotations

from grounding.core.types import (
    CitationSpan,
    Claim,
    ConsensusPrior,
    EvidencePointer,
    GroundingResult,
    Source,
    Table,
    TierVerdict,
    Verdict,
)


def test_verdict_enum_values() -> None:
    assert Verdict.GROUNDED.value == "GROUNDED"
    assert Verdict.UNGROUNDED.value == "UNGROUNDED"
    assert Verdict.UNCERTAIN.value == "UNCERTAIN"
    assert Verdict.SKIPPED.value == "SKIPPED"


def test_consensus_prior_values() -> None:
    assert ConsensusPrior.CONFIRMED.value == "CONFIRMED"
    assert ConsensusPrior.SINGLE.value == "SINGLE"
    assert ConsensusPrior.DISAGREEMENT.value == "DISAGREEMENT"
    assert ConsensusPrior.UNKNOWN.value == "UNKNOWN"


def test_citation_span_length() -> None:
    span = CitationSpan(page=1, char_start=10, char_end=30)
    assert span.length() == 20


def test_evidence_pointer_length() -> None:
    ep = EvidencePointer(doc_id="d", page=1, char_start=5, char_end=12)
    assert ep.length() == 7


def test_evidence_pointer_negative_length_clamped() -> None:
    ep = EvidencePointer(doc_id="d", page=None, char_start=10, char_end=5)
    assert ep.length() == 0


def test_claim_defaults() -> None:
    c = Claim(text="hello")
    assert c.text == "hello"
    assert c.page is None
    assert c.citation_span is None
    assert c.metadata == {}


def test_source_from_text() -> None:
    s = Source.from_text("hello world", doc_id="x", language="it")
    assert s.text == "hello world"
    assert s.doc_id == "x"
    assert s.language == "it"
    assert s.tables == []
    assert s.kv_pairs == {}


def test_table_defaults() -> None:
    t = Table()
    assert t.page is None
    assert t.headers == []
    assert t.rows == []


def test_tier_verdict_defaults() -> None:
    tv = TierVerdict(name="x", verdict=Verdict.GROUNDED)
    assert tv.score == 0.0
    assert tv.threshold_used == 0.0
    assert tv.evidence == []
    assert tv.detail == ""


def test_grounding_result_defaults() -> None:
    r = GroundingResult(claim_text="c", verdict=Verdict.UNCERTAIN)
    assert r.confidence == 0.0
    assert r.tier_results == {}
    assert r.evidence_pointers == []
    assert r.conflict_pointers == []
    assert r.reasoning_trace == []
    assert r.merkle_proof is None
