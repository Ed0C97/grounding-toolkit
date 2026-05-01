"""Tests for grounding.confidence.bayesian."""

from __future__ import annotations

from grounding import (
    Claim,
    GroundingResult,
    GroundingVerifier,
    Source,
    TierVerdict,
    Verdict,
)
from grounding.confidence.bayesian import (
    ConfidenceCalibration,
    TierWeights,
    posterior_for_verdicts,
    posterior_grounded,
)


def test_no_tiers_returns_prior() -> None:
    cal = ConfidenceCalibration(prior_log_odds=0.0)
    r = GroundingResult(claim_text="x", verdict=Verdict.UNCERTAIN)
    p = posterior_grounded(r, calibration=cal)
    assert abs(p - 0.5) < 1e-9


def test_grounded_tiers_increase_posterior() -> None:
    r = GroundingResult(
        claim_text="x",
        verdict=Verdict.GROUNDED,
        tier_results={
            "lexical": TierVerdict(
                name="lexical", verdict=Verdict.GROUNDED
            ),
        },
    )
    p = posterior_grounded(r)
    assert p > 0.5


def test_ungrounded_tiers_decrease_posterior() -> None:
    r = GroundingResult(
        claim_text="x",
        verdict=Verdict.UNGROUNDED,
        tier_results={
            "lexical": TierVerdict(
                name="lexical", verdict=Verdict.UNGROUNDED
            ),
        },
    )
    p = posterior_grounded(r)
    assert p < 0.5


def test_skipped_tiers_neutral() -> None:
    cal = ConfidenceCalibration(prior_log_odds=0.0)
    r = GroundingResult(
        claim_text="x",
        verdict=Verdict.UNCERTAIN,
        tier_results={
            "lexical": TierVerdict(
                name="lexical", verdict=Verdict.SKIPPED
            ),
        },
    )
    p = posterior_grounded(r, calibration=cal)
    assert abs(p - 0.5) < 1e-9


def test_multiple_grounded_pushes_higher() -> None:
    r1 = GroundingResult(
        claim_text="x",
        verdict=Verdict.GROUNDED,
        tier_results={
            "lexical": TierVerdict(
                name="lexical", verdict=Verdict.GROUNDED
            ),
        },
    )
    r2 = GroundingResult(
        claim_text="x",
        verdict=Verdict.GROUNDED,
        tier_results={
            "lexical": TierVerdict(
                name="lexical", verdict=Verdict.GROUNDED
            ),
            "semantic": TierVerdict(
                name="semantic", verdict=Verdict.GROUNDED
            ),
        },
    )
    p1 = posterior_grounded(r1)
    p2 = posterior_grounded(r2)
    assert p2 > p1


def test_citation_span_dominates() -> None:
    """Citation-span GROUNDED should produce a very high posterior."""
    r = GroundingResult(
        claim_text="x",
        verdict=Verdict.GROUNDED,
        tier_results={
            "citation_span": TierVerdict(
                name="citation_span", verdict=Verdict.GROUNDED
            ),
        },
    )
    p = posterior_grounded(r)
    assert p > 0.9


def test_unknown_tier_uses_default_weights() -> None:
    r = GroundingResult(
        claim_text="x",
        verdict=Verdict.GROUNDED,
        tier_results={
            "totally_new_tier": TierVerdict(
                name="totally_new_tier", verdict=Verdict.GROUNDED
            ),
        },
    )
    p = posterior_grounded(r)
    assert p > 0.5  # default_grounded > 0


def test_posterior_for_verdicts() -> None:
    p = posterior_for_verdicts(
        {"lexical": Verdict.GROUNDED, "semantic": Verdict.GROUNDED}
    )
    assert p > 0.8


def test_custom_calibration_overrides_default() -> None:
    weights = TierWeights(default_grounded=10.0)
    cal = ConfidenceCalibration(weights=weights)
    p = posterior_for_verdicts(
        {"unknown_tier": Verdict.GROUNDED}, calibration=cal
    )
    # Strong grounded weight pushes posterior near 1.
    assert p > 0.99


def test_real_cascade_result() -> None:
    """End-to-end via GroundingVerifier."""
    v = GroundingVerifier()
    r = v.verify(
        Claim(text="hello"),
        Source.from_text("hello world"),
    )
    p = posterior_grounded(r)
    assert p > 0.5
