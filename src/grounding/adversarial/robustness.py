"""Robustness verification — re-run grounding with canonicalised text.

Wraps any tier (or the full cascade) so that:

1. The claim is checked for adversarial perturbations.
2. If perturbations are detected, the verifier is re-run on the
   canonicalised text.
3. Results from both runs are compared; if the canonicalised run flips
   a verdict, the report flags the perturbation as a probable attack.

This is a *post-hoc* defensive layer.  The first line of defence is
canonicalising input early in the pipeline, but a robustness check
catches cases where canonicalisation was forgotten or incomplete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from grounding.adversarial.perturbation import (
    PerturbationDetector,
    PerturbationReport,
)
from grounding.core.types import (
    Claim,
    GroundingResult,
    Source,
    Verdict,
)


class _VerifierLike(Protocol):
    def verify(self, claim: Claim, source: Source) -> GroundingResult: ...


@dataclass
class RobustnessResult:
    """Output of :meth:`RobustnessChecker.check`."""

    perturbation_report: PerturbationReport
    original_result: GroundingResult
    canonical_result: Optional[GroundingResult]
    verdict_flipped: bool

    @property
    def safe(self) -> bool:
        """True if either no perturbation was found, or the verdict
        agrees on both raw and canonical claim text."""
        if not self.perturbation_report.has_perturbations:
            return True
        return not self.verdict_flipped


@dataclass
class RobustnessChecker:
    """Run grounding twice (raw + canonical) and report disagreement."""

    detector: PerturbationDetector = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.detector is None:
            self.detector = PerturbationDetector()

    def check(
        self,
        verifier: _VerifierLike,
        claim: Claim,
        source: Source,
    ) -> RobustnessResult:
        report = self.detector.detect(claim.text)
        original = verifier.verify(claim, source)
        canonical: Optional[GroundingResult] = None
        flipped = False
        if report.has_perturbations:
            canonical_claim = Claim(
                text=report.canonical_text,
                page=claim.page,
                citation_span=claim.citation_span,
                metadata=dict(claim.metadata),
            )
            canonical = verifier.verify(canonical_claim, source)
            flipped = canonical.verdict != original.verdict
        return RobustnessResult(
            perturbation_report=report,
            original_result=original,
            canonical_result=canonical,
            verdict_flipped=flipped,
        )


__all__ = ["RobustnessChecker", "RobustnessResult"]
