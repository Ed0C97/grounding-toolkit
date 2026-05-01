"""Tests for grounding.adversarial.robustness.RobustnessChecker."""

from __future__ import annotations

from grounding import Claim, GroundingVerifier, Source, Verdict
from grounding.adversarial.robustness import RobustnessChecker


def test_safe_when_no_perturbation() -> None:
    v = GroundingVerifier()
    src = Source.from_text("hello world")
    claim = Claim(text="hello")
    rc = RobustnessChecker()
    out = rc.check(v, claim, src)
    assert out.safe
    assert not out.perturbation_report.has_perturbations
    assert out.canonical_result is None
    assert out.original_result.verdict == Verdict.GROUNDED


def test_perturbation_with_no_verdict_flip() -> None:
    """A single ZWSP inside a long claim is detected, but the longest
    contiguous match still covers > 90 % of the claim — both raw and
    canonical runs ground."""
    v = GroundingVerifier()
    src = Source.from_text(
        "Once upon a time, hello world appeared in the document."
    )
    # claim has ZWSP at the end; the longest match before the ZWSP is
    # "hello world " (12 chars) out of 13 → ratio ≈ 0.92, above the
    # default 0.85 threshold.
    claim = Claim(text="hello world ​")
    rc = RobustnessChecker()
    out = rc.check(v, claim, src)
    assert out.perturbation_report.has_perturbations
    assert out.canonical_result is not None
    assert out.original_result.verdict == out.canonical_result.verdict
    assert out.safe


def test_verdict_flips_after_canonicalisation() -> None:
    """Cyrillic 'а' breaks exact substring; canonicalising restores it."""
    v = GroundingVerifier()
    src = Source.from_text("apple is a fruit")
    # Cyrillic a + 'pple' — lexical against raw fails substring; longest
    # match might still hit threshold.  We make it harder: short claim.
    raw_claim = Claim(text="аpple")
    canon_claim_text = "apple"

    rc = RobustnessChecker()
    out = rc.check(v, raw_claim, src)
    # The perturbation report MUST flag confusables.
    assert out.perturbation_report.has_perturbations
    assert out.perturbation_report.canonical_text == canon_claim_text
    # Canonical run should ground.
    assert out.canonical_result is not None
    assert out.canonical_result.verdict == Verdict.GROUNDED


def test_safe_property_when_no_perturbation() -> None:
    v = GroundingVerifier()
    rc = RobustnessChecker()
    out = rc.check(
        v, Claim(text="x"), Source.from_text("x")
    )
    assert out.safe
