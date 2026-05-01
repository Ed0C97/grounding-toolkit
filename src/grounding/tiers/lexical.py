"""Tier 0+1 — substring exact + lexical fuzzy match.

Pure standard library.  Zero network egress.  Free to run on every
claim.  Handles the common case where the LLM either copies the source
verbatim or paraphrases lightly with small drift / typos / OCR noise.

- Tier 0: ``claim.text`` is a literal substring of ``source.text`` →
  GROUNDED with score 1.0 and a precise evidence pointer.
- Tier 1: longest contiguous match between claim and source covers at
  least ``threshold`` ratio of ``claim.text`` → GROUNDED with score
  equal to the coverage ratio.

Anything below the threshold returns UNGROUNDED.  An empty claim or
empty source returns SKIPPED.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from grounding.core.types import (
    Claim,
    EvidencePointer,
    Source,
    TierVerdict,
    Verdict,
)


# Word-character tokenizer used by :func:`compute_text_overlap` so
# trailing punctuation does not break Jaccard agreement.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class LexicalTier:
    """Tier 0+1 combined: substring + difflib fuzzy."""

    name: str = "lexical"

    def verify(
        self,
        claim: Claim,
        source: Source,
        *,
        threshold: float = 0.85,
    ) -> TierVerdict:
        if not claim.text:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                score=0.0,
                threshold_used=threshold,
                detail="empty claim",
            )
        if not source.text:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                score=0.0,
                threshold_used=threshold,
                detail="empty source",
            )

        # Tier 0: exact substring
        idx = source.text.find(claim.text)
        if idx >= 0:
            ev = EvidencePointer(
                doc_id=source.doc_id,
                page=None,
                char_start=idx,
                char_end=idx + len(claim.text),
            )
            return TierVerdict(
                name=self.name,
                verdict=Verdict.GROUNDED,
                score=1.0,
                threshold_used=threshold,
                evidence=[ev],
                detail="exact substring match",
            )

        # Tier 1: longest contiguous fuzzy match
        sm = difflib.SequenceMatcher(
            None, claim.text, source.text, autojunk=False
        )
        match = sm.find_longest_match(
            0, len(claim.text), 0, len(source.text)
        )
        ratio = match.size / max(len(claim.text), 1)
        if ratio >= threshold:
            ev = EvidencePointer(
                doc_id=source.doc_id,
                page=None,
                char_start=match.b,
                char_end=match.b + match.size,
            )
            return TierVerdict(
                name=self.name,
                verdict=Verdict.GROUNDED,
                score=ratio,
                threshold_used=threshold,
                evidence=[ev],
                detail=f"fuzzy match ratio={ratio:.3f}",
            )

        return TierVerdict(
            name=self.name,
            verdict=Verdict.UNGROUNDED,
            score=ratio,
            threshold_used=threshold,
            detail=(
                f"longest match ratio={ratio:.3f} below {threshold:.3f}"
            ),
        )


def compute_text_overlap(a: str, b: str) -> float:
    """Token-set Jaccard overlap between two strings.

    Tokenisation uses ``\\w+`` so adjacent punctuation does not cause
    spurious mismatches (``"strong."`` matches ``"strong"``).

    Pre-migration parity helper for Sentinel's
    ``sentinel.utils.definition_finder.compute_text_overlap`` (P16
    hard-cutover).  Returns 0..1.
    """
    if not a or not b:
        return 0.0
    sa = set(_TOKEN_RE.findall(a.lower()))
    sb = set(_TOKEN_RE.findall(b.lower()))
    inter = sa & sb
    union = sa | sb
    return len(inter) / len(union) if union else 0.0


__all__ = ["LexicalTier", "compute_text_overlap"]
