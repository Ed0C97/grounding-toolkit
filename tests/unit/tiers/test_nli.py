"""Tests for grounding.tiers.nli."""

from __future__ import annotations

from grounding import Claim, Source, Verdict
from grounding.testing import StubNLIFn
from grounding.tiers.nli import NLITier


def test_skipped_without_nli_fn() -> None:
    tier = NLITier()
    r = tier.verify(Claim(text="x"), Source.from_text("y"))
    assert r.verdict == Verdict.SKIPPED


def test_grounded_when_claim_in_source() -> None:
    tier = NLITier(nli_fn=StubNLIFn())
    r = tier.verify(
        Claim(text="brown fox"),
        Source.from_text("The quick brown fox jumps."),
        threshold=0.5,
    )
    assert r.verdict == Verdict.GROUNDED
    assert r.score >= 0.5


def test_ungrounded_when_neutral() -> None:
    tier = NLITier(nli_fn=StubNLIFn())
    r = tier.verify(
        Claim(text="totally absent"),
        Source.from_text("benign source corpus"),
        threshold=0.5,
    )
    assert r.verdict == Verdict.UNGROUNDED


def test_contradiction_detected() -> None:
    """When the stub flags a contradiction term, tier emits UNGROUNDED
    with a conflict pointer (carried via evidence list)."""
    tier = NLITier(nli_fn=StubNLIFn(contradictions=["WRONG_VALUE"]))
    r = tier.verify(
        Claim(text="WRONG_VALUE"),
        Source.from_text("the actual value is correct"),
    )
    assert r.verdict == Verdict.UNGROUNDED
    assert r.evidence  # conflict pointer attached


def test_empty_claim_skipped() -> None:
    tier = NLITier(nli_fn=StubNLIFn())
    r = tier.verify(Claim(text=""), Source.from_text("y"))
    assert r.verdict == Verdict.SKIPPED


def test_empty_source_skipped() -> None:
    tier = NLITier(nli_fn=StubNLIFn())
    r = tier.verify(Claim(text="x"), Source.from_text(""))
    assert r.verdict == Verdict.SKIPPED


def test_skipped_when_nli_raises() -> None:
    class _RaisingFn:
        def __call__(self, *, claim, source):  # noqa: ARG002
            raise RuntimeError("boom")

    tier = NLITier(nli_fn=_RaisingFn())
    r = tier.verify(Claim(text="x"), Source.from_text("y"))
    assert r.verdict == Verdict.SKIPPED
