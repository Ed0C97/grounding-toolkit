"""Cascade orchestrator.

Runs verification tiers in order, applies threshold modulation from the
consensus prior, accumulates evidence, and short-circuits on GROUNDED.

Phase 1 cascade: consensus prior  →  lexical tier.

Later phases extend the cascade with multimodal / numerical / temporal /
definitional / crossdoc / semantic / NLI / LLM-judge tiers, all wired
through the same orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from grounding.core.thresholds import ThresholdProfile, modulate
from grounding.core.types import (
    Claim,
    GroundingResult,
    Source,
    TierVerdict,
    Verdict,
)
from grounding.tiers.consensus import ConsensusTier
from grounding.tiers.lexical import LexicalTier


@dataclass
class GroundingVerifier:
    """Cascade orchestrator.

    Phase 1 wires only the consensus prior + lexical tier.  Future
    phases append additional tiers.
    """

    consensus_tier: ConsensusTier = field(default_factory=ConsensusTier)
    lexical_tier: LexicalTier = field(default_factory=LexicalTier)
    base_thresholds: ThresholdProfile = field(default_factory=ThresholdProfile)

    def verify(self, claim: Claim, source: Source) -> GroundingResult:
        trace: List[str] = []
        tier_results: dict[str, TierVerdict] = {}

        # Tier −1: consensus prior
        prior = self.consensus_tier.extract_prior(claim)
        consensus_result = self.consensus_tier.verify(claim, source)
        tier_results[consensus_result.name] = consensus_result
        trace.append(f"consensus prior: {prior.value}")

        # Modulate thresholds
        thresholds = modulate(self.base_thresholds, prior)
        trace.append(
            f"thresholds: fuzzy={thresholds.fuzzy:.3f}"
        )

        # Tier 0+1: lexical
        lex_result = self.lexical_tier.verify(
            claim, source, threshold=thresholds.fuzzy
        )
        tier_results[lex_result.name] = lex_result
        trace.append(
            f"lexical: {lex_result.verdict.value} "
            f"score={lex_result.score:.3f}"
        )

        # Aggregate
        if lex_result.verdict == Verdict.GROUNDED:
            verdict = Verdict.GROUNDED
            confidence = min(1.0, lex_result.score)
            evidence = list(lex_result.evidence)
        elif lex_result.verdict == Verdict.SKIPPED:
            verdict = Verdict.UNCERTAIN
            confidence = 0.0
            evidence = []
        else:
            verdict = Verdict.UNGROUNDED
            confidence = max(0.0, min(1.0, 1.0 - lex_result.score))
            evidence = []

        return GroundingResult(
            claim_text=claim.text,
            verdict=verdict,
            confidence=confidence,
            tier_results=tier_results,
            evidence_pointers=evidence,
            conflict_pointers=[],
            reasoning_trace=trace,
        )


__all__ = ["GroundingVerifier"]
