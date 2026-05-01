"""Deterministic verification of LLM-emitted citation spans.

When the LLM is instructed (via the structured signature in
``citations/structured_signature.py``) to attach a
:class:`CitationSpan(page, char_start, char_end)` to every claim, this
module verifies the tuple deterministically:

1. The span is well-formed (``char_start < char_end``, ``page >= 1``).
2. The page text exists (``Source.page_text(span.page)`` is non-None).
3. The text within the span window matches the claim text either
   exactly (substring containment in either direction) or with high
   fuzzy similarity (configurable threshold, default 0.85).

A successful match is the strongest groundedness signal: cascade
orchestration uses :func:`grounding.core.speculative.speculative_prescreen`
to short-circuit the rest of the pipeline.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from grounding.core.types import (
    Claim,
    EvidencePointer,
    Source,
    TierVerdict,
    Verdict,
)


@dataclass
class SpanVerifier:
    """Deterministic span verifier."""

    similarity_threshold: float = 0.85
    name: str = "citation_span"

    def verify(self, claim: Claim, source: Source) -> TierVerdict:
        if claim.citation_span is None:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                detail="no citation_span emitted",
            )

        span = claim.citation_span

        if span.char_start >= span.char_end:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.UNGROUNDED,
                detail="malformed span (char_start >= char_end)",
            )
        if span.page < 1:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.UNGROUNDED,
                detail=f"invalid page {span.page}",
            )

        page_text = source.page_text(span.page)
        if page_text is None:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.UNGROUNDED,
                detail=(
                    f"page {span.page} not in source "
                    f"(page_count={source.page_count})"
                ),
            )

        if span.char_start < 0 or span.char_end > len(page_text):
            return TierVerdict(
                name=self.name,
                verdict=Verdict.UNGROUNDED,
                detail=(
                    f"span out of page bounds: "
                    f"[{span.char_start}, {span.char_end}) vs "
                    f"page len {len(page_text)}"
                ),
            )

        cited_text = page_text[span.char_start : span.char_end]
        if not claim.text or not cited_text:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.UNGROUNDED,
                detail="empty cited text or claim",
            )

        if claim.text in cited_text or cited_text in claim.text:
            ev = EvidencePointer(
                doc_id=source.doc_id,
                page=span.page,
                char_start=span.char_start,
                char_end=span.char_end,
            )
            return TierVerdict(
                name=self.name,
                verdict=Verdict.GROUNDED,
                score=1.0,
                threshold_used=self.similarity_threshold,
                evidence=[ev],
                detail="exact span match",
            )

        sm = difflib.SequenceMatcher(
            None, claim.text, cited_text, autojunk=False
        )
        ratio = sm.ratio()
        if ratio >= self.similarity_threshold:
            ev = EvidencePointer(
                doc_id=source.doc_id,
                page=span.page,
                char_start=span.char_start,
                char_end=span.char_end,
            )
            return TierVerdict(
                name=self.name,
                verdict=Verdict.GROUNDED,
                score=ratio,
                threshold_used=self.similarity_threshold,
                evidence=[ev],
                detail=f"fuzzy span match ratio={ratio:.3f}",
            )

        return TierVerdict(
            name=self.name,
            verdict=Verdict.UNGROUNDED,
            score=ratio,
            threshold_used=self.similarity_threshold,
            detail=(
                f"span similarity={ratio:.3f} below "
                f"{self.similarity_threshold:.3f}"
            ),
        )


__all__ = ["SpanVerifier"]
