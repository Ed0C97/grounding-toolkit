"""Verify claims against ``Source.figures``.

Currently a thin scaffold: the toolkit accepts an ``image_caption_fn``
Protocol-shaped callable that maps a figure dict to a caption / extracted
text payload, then delegates to lexical similarity against the caption.

Most of the value of figure verification comes from a captioner, which
the toolkit deliberately does NOT bundle (consumer responsibility).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from grounding.core.types import (
    Claim,
    EvidencePointer,
    Source,
    TierVerdict,
    Verdict,
)
from grounding.tiers.lexical import LexicalTier


CaptionFn = Callable[[dict], str]


def _default_caption_fn(figure: dict) -> str:
    """Fall back to the ``caption`` / ``alt`` / ``text`` field, in order."""
    for key in ("caption", "alt", "text", "title"):
        v = figure.get(key)
        if v:
            return str(v)
    return ""


@dataclass
class FigureVerifier:
    """Verify claims against extracted figure captions."""

    caption_fn: Optional[CaptionFn] = None
    fuzzy_threshold: float = 0.75
    name: str = "figures"

    def verify(
        self,
        claim: Claim,
        source: Source,
        *,
        threshold: float = 0.75,
    ) -> TierVerdict:
        if not source.figures:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="no figures in source",
            )
        if not claim.text:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="empty claim",
            )

        cf = self.caption_fn or _default_caption_fn
        lex = LexicalTier()

        evidence: List[EvidencePointer] = []
        best_score = 0.0
        for fig in source.figures:
            caption = cf(fig).strip()
            if not caption:
                continue
            sub_source = Source(
                text=caption,
                doc_id=source.doc_id,
                page_count=source.page_count,
            )
            sub_result = lex.verify(claim, sub_source, threshold=threshold)
            if sub_result.score > best_score:
                best_score = sub_result.score
            if sub_result.verdict == Verdict.GROUNDED:
                evidence.extend(sub_result.evidence)

        if evidence:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.GROUNDED,
                score=best_score,
                threshold_used=threshold,
                evidence=evidence,
                detail=f"matched figure caption(s); best={best_score:.3f}",
            )
        return TierVerdict(
            name=self.name,
            verdict=Verdict.UNGROUNDED,
            score=best_score,
            threshold_used=threshold,
            detail=f"best caption similarity={best_score:.3f}",
        )


__all__ = ["FigureVerifier", "CaptionFn"]
