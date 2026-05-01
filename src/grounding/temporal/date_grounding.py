"""Verify temporal claims against a document timeline.

Two complementary checks:

1. **Date span check** — given a claim with a date and an optional
   ``timeline`` (list of date tuples covering the document's reference
   period), verify the claim's date falls inside the timeline.

2. **Date presence check** — given a claim with a date, verify that the
   same date appears somewhere in the source text or in
   :class:`Source.kv_pairs`.

The verifier is locale-aware via the underlying
:class:`grounding.numerical.NumberExtractor` which understands
``dd/mm/yyyy``, ``yyyy-mm-dd``, and 4-digit years.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Sequence, Tuple

from grounding.core.types import (
    Claim,
    EvidencePointer,
    Source,
    TierVerdict,
    Verdict,
)
from grounding.numerical.number_extraction import (
    ExtractedNumber,
    NumberExtractor,
)


def _to_date(num: ExtractedNumber) -> Optional[date]:
    if num.unit == "date":
        try:
            return date(
                int(num.extras.get("year", 0)),
                int(num.extras.get("month", 1)),
                int(num.extras.get("day", 1)),
            )
        except (ValueError, TypeError):
            return None
    if num.unit == "year":
        try:
            return date(int(num.value), 1, 1)
        except (ValueError, TypeError):
            return None
    return None


def _claim_dates(extractor: NumberExtractor, text: str) -> List[Tuple[ExtractedNumber, date]]:
    out: List[Tuple[ExtractedNumber, date]] = []
    for n in extractor.extract(text):
        d = _to_date(n)
        if d is not None:
            out.append((n, d))
    return out


def _date_in_text(d: date, extractor: NumberExtractor, text: str) -> bool:
    for n in extractor.extract(text):
        d2 = _to_date(n)
        if d2 is None:
            continue
        if d2 == d:
            return True
        # Year-level match: claim year matches stored date with same year
        if n.unit == "year" and d2.year == d.year:
            return True
        if n.unit == "date" and d2.year == d.year and d2.month == d.month and d2.day == d.day:
            return True
    return False


@dataclass
class DateTimeline:
    """Inclusive date-range intervals describing the document scope."""

    spans: List[Tuple[date, date]] = field(default_factory=list)

    def covers(self, d: date) -> bool:
        for start, end in self.spans:
            if start <= d <= end:
                return True
        return False


@dataclass
class TemporalVerifier:
    """Verify temporal claims (dates / years) against the source."""

    extractor: NumberExtractor = None  # type: ignore[assignment]
    timeline: Optional[DateTimeline] = None
    name: str = "temporal"

    def __post_init__(self) -> None:
        if self.extractor is None:
            self.extractor = NumberExtractor()

    def verify(
        self,
        claim: Claim,
        source: Source,
        *,
        threshold: float = 1.0,
    ) -> TierVerdict:
        if not claim.text:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="empty claim",
            )

        claim_dates = _claim_dates(self.extractor, claim.text)
        if not claim_dates:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="no temporal tokens in claim",
            )

        evidence: List[EvidencePointer] = []
        ungrounded: List[str] = []
        for n, d in claim_dates:
            grounded = False

            # Path 1: explicit timeline
            if self.timeline is not None:
                if self.timeline.covers(d):
                    grounded = True

            # Path 2: presence in source.text
            if not grounded and source.text:
                if _date_in_text(d, self.extractor, source.text):
                    grounded = True

            # Path 3: presence in KV pairs
            if not grounded and source.kv_pairs:
                for v in source.kv_pairs.values():
                    if v is None:
                        continue
                    if _date_in_text(d, self.extractor, str(v)):
                        grounded = True
                        break

            if grounded:
                evidence.append(
                    EvidencePointer(
                        doc_id=source.doc_id,
                        page=None,
                        char_start=n.char_start,
                        char_end=n.char_end,
                    )
                )
            else:
                ungrounded.append(n.raw)

        if ungrounded:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.UNGROUNDED,
                threshold_used=threshold,
                evidence=evidence,
                detail=f"dates not found in source: {ungrounded[:5]}",
            )
        return TierVerdict(
            name=self.name,
            verdict=Verdict.GROUNDED,
            score=1.0,
            threshold_used=threshold,
            evidence=evidence,
            detail=f"all {len(claim_dates)} dates grounded",
        )


__all__ = ["DateTimeline", "TemporalVerifier"]
