"""Tests for grounding.tiers.semantic."""

from __future__ import annotations

from grounding import Claim, Source, Verdict
from grounding.testing import StubEmbeddingFn
from grounding.tiers.semantic import SemanticTier


def test_skipped_without_embedding_fn() -> None:
    tier = SemanticTier()
    r = tier.verify(Claim(text="anything"), Source.from_text("source"))
    assert r.verdict == Verdict.SKIPPED
    assert "no embedding function" in r.detail


def test_skipped_with_empty_claim() -> None:
    tier = SemanticTier(embedding_fn=StubEmbeddingFn())
    r = tier.verify(Claim(text=""), Source.from_text("source"))
    assert r.verdict == Verdict.SKIPPED


def test_skipped_with_empty_source() -> None:
    tier = SemanticTier(embedding_fn=StubEmbeddingFn())
    r = tier.verify(Claim(text="x"), Source.from_text(""))
    assert r.verdict == Verdict.SKIPPED


def test_grounded_for_identical_text() -> None:
    """The stub embedding is deterministic per text, so identical
    claim+chunk produce identical vectors and cosine == 1.0."""
    tier = SemanticTier(embedding_fn=StubEmbeddingFn(dim=8))
    text = "alpha beta gamma"
    r = tier.verify(Claim(text=text), Source.from_text(text), threshold=0.80)
    assert r.verdict == Verdict.GROUNDED
    assert r.score >= 0.99


def test_evidence_pointer_carries_doc_id() -> None:
    tier = SemanticTier(embedding_fn=StubEmbeddingFn(dim=8))
    text = "alpha beta gamma"
    src = Source.from_text(text, doc_id="d-77")
    r = tier.verify(Claim(text=text), src, threshold=0.80)
    assert r.verdict == Verdict.GROUNDED
    assert r.evidence
    assert r.evidence[0].doc_id == "d-77"


def test_skipped_when_embedding_raises() -> None:
    class _RaisingFn:
        def __call__(self, texts):  # noqa: D401, ARG001
            raise RuntimeError("boom")

    tier = SemanticTier(embedding_fn=_RaisingFn())
    r = tier.verify(Claim(text="x"), Source.from_text("y"))
    assert r.verdict == Verdict.SKIPPED
    assert "boom" in r.detail


def test_skipped_when_embedding_returns_wrong_count() -> None:
    class _BrokenFn:
        def __call__(self, texts):  # noqa: D401
            # always returns one fewer than requested
            return [[0.0] * 4 for _ in texts[1:]]

    tier = SemanticTier(embedding_fn=_BrokenFn())
    r = tier.verify(Claim(text="x"), Source.from_text("y"))
    assert r.verdict == Verdict.SKIPPED
