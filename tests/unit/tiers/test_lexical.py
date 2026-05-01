"""Tests for grounding.tiers.lexical."""

from __future__ import annotations

from grounding import Claim, Source, Verdict
from grounding.tiers.lexical import LexicalTier, compute_text_overlap


def test_exact_substring_grounded_with_score_1() -> None:
    tier = LexicalTier()
    r = tier.verify(
        Claim(text="hello world"),
        Source.from_text("foo hello world bar"),
        threshold=0.85,
    )
    assert r.verdict == Verdict.GROUNDED
    assert r.score == 1.0
    assert r.evidence
    assert r.evidence[0].char_start == 4
    assert r.evidence[0].char_end == 15
    assert r.detail == "exact substring match"


def test_fuzzy_above_threshold_grounded() -> None:
    tier = LexicalTier()
    # 90% of the claim appears contiguously in the source.
    r = tier.verify(
        Claim(text="The quick brown fox"),
        Source.from_text("Once: The quick brown fo... and more text"),
        threshold=0.85,
    )
    # longest match should be "The quick brown fo" = 18 of 19 chars ≈ 0.947
    assert r.verdict == Verdict.GROUNDED
    assert r.score >= 0.85


def test_fuzzy_below_threshold_ungrounded() -> None:
    tier = LexicalTier()
    r = tier.verify(
        Claim(text="completely different sentence here"),
        Source.from_text("Random unrelated content."),
        threshold=0.85,
    )
    assert r.verdict == Verdict.UNGROUNDED
    assert r.score < 0.85
    assert r.evidence == []


def test_empty_claim_skipped() -> None:
    tier = LexicalTier()
    r = tier.verify(
        Claim(text=""),
        Source.from_text("non-empty"),
        threshold=0.85,
    )
    assert r.verdict == Verdict.SKIPPED
    assert "empty claim" in r.detail


def test_empty_source_skipped() -> None:
    tier = LexicalTier()
    r = tier.verify(
        Claim(text="non-empty"),
        Source.from_text(""),
        threshold=0.85,
    )
    assert r.verdict == Verdict.SKIPPED
    assert "empty source" in r.detail


def test_threshold_used_recorded() -> None:
    tier = LexicalTier()
    r = tier.verify(
        Claim(text="x"),
        Source.from_text("y"),
        threshold=0.42,
    )
    assert r.threshold_used == 0.42


def test_evidence_pointer_doc_id_propagates() -> None:
    tier = LexicalTier()
    r = tier.verify(
        Claim(text="abc"),
        Source.from_text("xx abc yy", doc_id="d-99"),
        threshold=0.85,
    )
    assert r.verdict == Verdict.GROUNDED
    assert r.evidence[0].doc_id == "d-99"


def test_compute_text_overlap_identical() -> None:
    assert compute_text_overlap("hello world", "hello world") == 1.0


def test_compute_text_overlap_disjoint() -> None:
    assert compute_text_overlap("alpha beta", "gamma delta") == 0.0


def test_compute_text_overlap_partial() -> None:
    # tokens: {alpha, beta} vs {beta, gamma} → inter=1, union=3 → 1/3
    v = compute_text_overlap("alpha beta", "beta gamma")
    assert abs(v - (1.0 / 3.0)) < 1e-9


def test_compute_text_overlap_empty() -> None:
    assert compute_text_overlap("", "anything") == 0.0
    assert compute_text_overlap("anything", "") == 0.0
    assert compute_text_overlap("", "") == 0.0
