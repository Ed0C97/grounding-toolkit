"""Tier 4 — LLM-as-judge.

Final fallback tier.  The consumer supplies an
:class:`grounding.core.ports.LLMJudgeFn` that, given the claim and
source, returns a verdict + rationale dict.  Costly; only invoked when
earlier tiers are inconclusive (cascade orchestration handles ordering).

If no judge function is injected the tier returns SKIPPED.

Sentinel will inject a local-LLM judge after the LLM → local
migration is complete (Phase D5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from grounding.core.ports import LLMJudgeFn
from grounding.core.types import Claim, Source, TierVerdict, Verdict


@dataclass
class LLMJudgeTier:
    """Tier 4 — LLM-as-judge."""

    judge_fn: Optional[LLMJudgeFn] = None
    name: str = "llm_judge"

    def verify(
        self,
        claim: Claim,
        source: Source,
        *,
        threshold: float = 0.50,
    ) -> TierVerdict:
        if self.judge_fn is None:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="no LLM-judge function injected",
            )
        if not claim.text or not source.text:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="empty claim or source",
            )
        try:
            out = self.judge_fn(claim=claim.text, source=source.text)
        except Exception as exc:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail=f"judge_fn raised: {exc!r}",
            )

        verdict_str = str(out.get("verdict", "")).strip().upper()
        rationale = str(out.get("rationale", ""))
        confidence = float(
            out.get("confidence", 1.0 if verdict_str else 0.0)
        )

        if verdict_str == "GROUNDED":
            return TierVerdict(
                name=self.name,
                verdict=Verdict.GROUNDED,
                score=confidence,
                threshold_used=threshold,
                detail=f"LLM-judge: {rationale[:120]}",
            )
        if verdict_str == "UNGROUNDED":
            return TierVerdict(
                name=self.name,
                verdict=Verdict.UNGROUNDED,
                score=max(0.0, 1.0 - confidence),
                threshold_used=threshold,
                detail=f"LLM-judge: {rationale[:120]}",
            )

        return TierVerdict(
            name=self.name,
            verdict=Verdict.UNCERTAIN,
            score=0.5,
            threshold_used=threshold,
            detail=f"LLM-judge unrecognised verdict: {verdict_str!r}",
        )


__all__ = ["LLMJudgeTier"]
