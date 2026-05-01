"""Bayesian confidence calibration.

Combines per-tier verdicts into a single posterior probability of
groundedness using a simple log-odds sum.  Each tier has an associated
``log_likelihood_ratio`` (LLR) that measures the strength of its
evidence; the posterior is computed as::

    posterior = sigmoid(prior_log_odds + sum(weight * LLR_per_tier))

Defaults are conservative (LLR magnitudes around 1.0-3.0) and tuned to
prevent any single tier from dominating.  The consumer overrides
defaults via :class:`ConfidenceCalibration`.

This module ships **framework + sensible defaults**; a real calibration
against a gold-truth dataset (Phase 13) replaces the defaults with
empirically-derived values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from grounding.core.types import GroundingResult, Verdict


@dataclass
class TierWeights:
    """Per-tier log-likelihood-ratio contributions.

    A positive value pushes the posterior toward GROUNDED; a negative
    value pushes it toward UNGROUNDED.  Magnitude reflects evidence
    strength.
    """

    citation_span_grounded: float = 4.0       # very strong
    citation_span_ungrounded: float = -3.0
    lexical_grounded: float = 2.5
    lexical_ungrounded: float = -1.5
    semantic_grounded: float = 1.5
    semantic_ungrounded: float = -1.0
    nli_grounded: float = 2.0
    nli_ungrounded: float = -2.0
    llm_judge_grounded: float = 2.5
    llm_judge_ungrounded: float = -2.0
    multimodal_grounded: float = 1.5
    multimodal_ungrounded: float = -1.0
    derivation_grounded: float = 1.5
    derivation_ungrounded: float = -2.5
    temporal_grounded: float = 1.0
    temporal_ungrounded: float = -1.5
    definitional_grounded: float = 1.0
    definitional_ungrounded: float = -1.5
    crossdoc_grounded: float = 1.0
    crossdoc_ungrounded: float = -0.5
    multilingual_grounded: float = 1.0
    multilingual_ungrounded: float = -0.5
    default_grounded: float = 1.0
    default_ungrounded: float = -0.5


@dataclass
class ConfidenceCalibration:
    """Calibration parameters for the Bayesian combiner."""

    prior_log_odds: float = 0.0
    weights: TierWeights = field(default_factory=TierWeights)

    def llr_for(self, tier_name: str, verdict: Verdict) -> float:
        """Return the LLR contribution for a tier verdict."""
        prefix = tier_name.lower()
        # Map tier name to weight attribute
        attr_grounded = f"{prefix}_grounded"
        attr_ungrounded = f"{prefix}_ungrounded"
        if not hasattr(self.weights, attr_grounded):
            attr_grounded = "default_grounded"
            attr_ungrounded = "default_ungrounded"
        if verdict == Verdict.GROUNDED:
            return float(getattr(self.weights, attr_grounded))
        if verdict == Verdict.UNGROUNDED:
            return float(getattr(self.weights, attr_ungrounded))
        return 0.0  # SKIPPED / UNCERTAIN contribute zero


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def posterior_grounded(
    result: GroundingResult,
    *,
    calibration: Optional[ConfidenceCalibration] = None,
) -> float:
    """Compute the posterior probability the claim is grounded.

    Inputs:
        result: a :class:`GroundingResult` with populated ``tier_results``.
        calibration: optional :class:`ConfidenceCalibration`; defaults
            to the conservative shipped values.

    Returns:
        A float in [0, 1].
    """
    cal = calibration or ConfidenceCalibration()
    log_odds = float(cal.prior_log_odds)
    for tv in result.tier_results.values():
        log_odds += cal.llr_for(tv.name, tv.verdict)
    return _sigmoid(log_odds)


def posterior_for_verdicts(
    tier_verdicts: Dict[str, Verdict],
    *,
    calibration: Optional[ConfidenceCalibration] = None,
    prior_log_odds: float = 0.0,
) -> float:
    """Compute the posterior from a flat ``{tier_name: Verdict}`` map.

    Convenience helper for callers who don't have a full
    :class:`GroundingResult` available.
    """
    cal = calibration or ConfidenceCalibration()
    log_odds = float(prior_log_odds or cal.prior_log_odds)
    for name, verdict in tier_verdicts.items():
        log_odds += cal.llr_for(name, verdict)
    return _sigmoid(log_odds)


__all__ = [
    "ConfidenceCalibration",
    "TierWeights",
    "posterior_grounded",
    "posterior_for_verdicts",
]
