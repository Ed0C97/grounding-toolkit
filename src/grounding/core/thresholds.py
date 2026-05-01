"""Threshold modulation by the consensus prior.

The cascade uses a base :class:`ThresholdProfile` and modulates it by
the consensus prior of the claim:

- ``CONFIRMED`` (multiple generators agree)  → multiplier 0.85 (looser)
- ``SINGLE`` / ``UNKNOWN``                   → multiplier 1.00
- ``DISAGREEMENT``                            → multiplier 1.10 (stricter)

A multiplier > 1 raises every threshold (harder to clear), making the
verifier more sceptical when the upstream generators disagreed.

These multipliers are deliberately conservative defaults; once
calibrated against a gold-truth dataset (see ``grounding/calibration``),
the consumer can override them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from grounding.core.types import ConsensusPrior


@dataclass(frozen=True)
class ThresholdProfile:
    """Per-tier base thresholds.

    All thresholds are in the unit interval [0, 1].
    """

    fuzzy: float = 0.85
    semantic: float = 0.80
    nli: float = 0.65
    llm_judge: float = 0.50


_PRIOR_MULTIPLIER: Dict[ConsensusPrior, float] = {
    ConsensusPrior.CONFIRMED: 0.85,
    ConsensusPrior.SINGLE: 1.00,
    ConsensusPrior.UNKNOWN: 1.00,
    ConsensusPrior.DISAGREEMENT: 1.10,
}


def modulate(profile: ThresholdProfile, prior: ConsensusPrior) -> ThresholdProfile:
    """Return a new :class:`ThresholdProfile` scaled by the prior multiplier.

    Multipliers > 1 produce thresholds capped at 1.0; multipliers < 1
    produce thresholds floored at 0.0 (the floor never matters in
    practice for our default values).
    """
    mult = _PRIOR_MULTIPLIER.get(prior, 1.0)
    return ThresholdProfile(
        fuzzy=_clamp01(profile.fuzzy * mult),
        semantic=_clamp01(profile.semantic * mult),
        nli=_clamp01(profile.nli * mult),
        llm_judge=_clamp01(profile.llm_judge * mult),
    )


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


__all__ = ["ThresholdProfile", "modulate"]
