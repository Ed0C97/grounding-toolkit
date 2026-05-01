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

from grounding.citations.span import SpanVerifier
from grounding.core.speculative import speculative_prescreen
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

    Pipeline (Phase 3):

    1. Speculative pre-screen — if the claim has a citation_span and
       the span deterministically matches, short-circuit GROUNDED.
    2. Tier −1 — extract consensus prior from claim metadata.
    3. Threshold modulation — scale per-tier thresholds by the prior.
    4. Tier 0+1 — substring + lexical fuzzy.
    5. Aggregate verdict + confidence.

    Future phases append multimodal / numerical / temporal /
    definitional / crossdoc / semantic / NLI / LLM-judge tiers, all
    wired through the same orchestrator.
    """

    consensus_tier: ConsensusTier = field(default_factory=ConsensusTier)
    lexical_tier: LexicalTier = field(default_factory=LexicalTier)
    span_verifier: SpanVerifier = field(default_factory=SpanVerifier)
    base_thresholds: ThresholdProfile = field(default_factory=ThresholdProfile)

    def verify(self, claim: Claim, source: Source) -> GroundingResult:
        trace: List[str] = []
        tier_results: dict[str, TierVerdict] = {}

        # Step 1 — speculative pre-screen
        span_result = speculative_prescreen(
            claim, source, verifier=self.span_verifier
        )
        if span_result is not None:
            tier_results[span_result.name] = span_result
            trace.append(
                f"citation_span: {span_result.verdict.value} "
                f"score={span_result.score:.3f}"
            )
            if span_result.verdict == Verdict.GROUNDED:
                return GroundingResult(
                    claim_text=claim.text,
                    verdict=Verdict.GROUNDED,
                    confidence=min(1.0, span_result.score),
                    tier_results=tier_results,
                    evidence_pointers=list(span_result.evidence),
                    conflict_pointers=[],
                    reasoning_trace=trace,
                )

        # Step 2 — consensus prior
        prior = self.consensus_tier.extract_prior(claim)
        consensus_result = self.consensus_tier.verify(claim, source)
        tier_results[consensus_result.name] = consensus_result
        trace.append(f"consensus prior: {prior.value}")

        # Step 3 — threshold modulation
        thresholds = modulate(self.base_thresholds, prior)
        trace.append(f"thresholds: fuzzy={thresholds.fuzzy:.3f}")

        # Step 4 — lexical
        lex_result = self.lexical_tier.verify(
            claim, source, threshold=thresholds.fuzzy
        )
        tier_results[lex_result.name] = lex_result
        trace.append(
            f"lexical: {lex_result.verdict.value} "
            f"score={lex_result.score:.3f}"
        )

        # Step 5 — aggregate
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
