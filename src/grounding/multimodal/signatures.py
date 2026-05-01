"""Verify claims about document signatories against ``Source.signatures``.

Each signature in ``Source.signatures`` is a free-form dict.  Recognised
keys: ``name``, ``role``, ``date``, ``page``.  Matching uses regex
word-boundary search (case-insensitive) against ``name`` and ``role`` so
single-character or short tokens don't spuriously match common words
("y" in "by", "a" in "and").  Tokens shorter than 2 characters are
ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from grounding.core.types import (
    Claim,
    EvidencePointer,
    Source,
    TierVerdict,
    Verdict,
)


def _word_boundary_match(needle: str, haystack: str) -> bool:
    if not needle or len(needle.strip()) < 2:
        return False
    pattern = r"\b" + re.escape(needle.strip()) + r"\b"
    return re.search(pattern, haystack, flags=re.IGNORECASE) is not None


@dataclass
class SignatureVerifier:
    """Match claim text against signatory metadata."""

    name: str = "signatures"

    def verify(
        self,
        claim: Claim,
        source: Source,
        *,
        threshold: float = 0.0,
    ) -> TierVerdict:
        if not source.signatures:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="no signatures in source",
            )
        if not claim.text:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="empty claim",
            )

        evidence: List[EvidencePointer] = []
        for sig in source.signatures:
            for key in ("name", "role"):
                v = sig.get(key)
                if not v:
                    continue
                if _word_boundary_match(str(v), claim.text):
                    evidence.append(
                        EvidencePointer(
                            doc_id=source.doc_id,
                            page=sig.get("page"),
                            char_start=0,
                            char_end=len(str(v)),
                        )
                    )
                    break

        if evidence:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.GROUNDED,
                score=1.0,
                threshold_used=threshold,
                evidence=evidence,
                detail=f"matched {len(evidence)} signatory record(s)",
            )
        return TierVerdict(
            name=self.name,
            verdict=Verdict.UNGROUNDED,
            score=0.0,
            threshold_used=threshold,
            detail="no signatory match",
        )


__all__ = ["SignatureVerifier"]
