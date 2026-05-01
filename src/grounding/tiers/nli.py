"""Tier 3 — NLI entailment / contradiction scoring.

The tier delegates to the consumer-supplied
:class:`grounding.core.ports.NLIFn` to score the claim against the
source.  We interpret the label probabilities as follows:

- ``contradiction > 0.5``  → UNGROUNDED + a conflict pointer
- ``entailment >= threshold`` → GROUNDED + an evidence pointer
- otherwise → UNGROUNDED with low score

If no NLI function is injected the tier returns SKIPPED, allowing the
cascade to proceed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from grounding.core.ports import NLIFn
from grounding.core.types import (
    Claim,
    EvidencePointer,
    Source,
    TierVerdict,
    Verdict,
)


@dataclass
class NLITier:
    """Tier 3 — entailment scoring."""

    nli_fn: Optional[NLIFn] = None
    contradiction_threshold: float = 0.5
    name: str = "nli"

    def verify(
        self,
        claim: Claim,
        source: Source,
        *,
        threshold: float = 0.65,
    ) -> TierVerdict:
        if self.nli_fn is None:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="no NLI function injected",
            )
        if not claim.text or not source.text:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="empty claim or source",
            )
        try:
            scores = self.nli_fn(claim=claim.text, source=source.text)
        except Exception as exc:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail=f"nli_fn raised: {exc!r}",
            )

        ent = float(scores.get("entailment", 0.0))
        contra = float(scores.get("contradiction", 0.0))

        if contra > self.contradiction_threshold:
            ev = EvidencePointer(
                doc_id=source.doc_id,
                page=None,
                char_start=0,
                char_end=min(len(source.text), 200),
            )
            return TierVerdict(
                name=self.name,
                verdict=Verdict.UNGROUNDED,
                score=contra,
                threshold_used=threshold,
                evidence=[ev],
                detail=f"contradiction prob={contra:.3f}",
            )

        if ent >= threshold:
            ev = EvidencePointer(
                doc_id=source.doc_id,
                page=None,
                char_start=0,
                char_end=min(len(source.text), 200),
            )
            return TierVerdict(
                name=self.name,
                verdict=Verdict.GROUNDED,
                score=ent,
                threshold_used=threshold,
                evidence=[ev],
                detail=f"entailment prob={ent:.3f}",
            )

        return TierVerdict(
            name=self.name,
            verdict=Verdict.UNGROUNDED,
            score=ent,
            threshold_used=threshold,
            detail=f"entailment={ent:.3f} below {threshold:.3f}",
        )


__all__ = ["NLITier"]
