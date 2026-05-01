"""Verify numeric claims against ``Source.tables``.

For every numeric token extracted from the claim text, search every
table cell for a value that matches within tolerance.  A cell is also
checked when its raw string contains the original token (e.g.
``"€ 8.4 M"``) so locale-mismatched normalisation doesn't drop matches.

The table data is consumer-populated.  This module never imports
``ocr-toolkit`` — Sentinel passes already-extracted tables as
:class:`grounding.core.types.Table` instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from grounding.core.types import (
    Claim,
    EvidencePointer,
    Source,
    Table,
    TierVerdict,
    Verdict,
)
from grounding.numerical.number_extraction import (
    ExtractedNumber,
    NumberExtractor,
    numbers_match,
)


def _cells(table: Table) -> List[str]:
    out: List[str] = []
    for row in table.rows:
        for cell in row:
            if cell is None:
                continue
            out.append(str(cell))
    return out


def _table_contains_number(
    extractor: NumberExtractor,
    table: Table,
    target: ExtractedNumber,
    *,
    tolerance: float,
) -> bool:
    target_raw_lower = target.raw.strip().lower()
    for cell in _cells(table):
        cell_lower = cell.strip().lower()
        if target_raw_lower and target_raw_lower in cell_lower:
            return True
        for cell_num in extractor.extract(cell):
            # Units must match: a percentage in claim should match a
            # percentage in table, not a monetary value with same digit.
            if cell_num.unit != target.unit:
                continue
            if numbers_match(target.value, cell_num.value, tolerance=tolerance):
                return True
    return False


@dataclass
class TableVerifier:
    """Verify numeric claims against ``Source.tables``."""

    extractor: NumberExtractor = None  # type: ignore[assignment]
    tolerance: float = 0.05
    name: str = "tables"

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
        if not source.tables:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="no tables in source",
            )
        if not claim.text:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="empty claim",
            )

        claim_numbers = self.extractor.extract(claim.text)
        if not claim_numbers:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="no numbers in claim",
            )

        evidence: List[EvidencePointer] = []
        ungrounded: List[str] = []
        for n in claim_numbers:
            grounded = False
            for t in source.tables:
                if _table_contains_number(
                    self.extractor, t, n, tolerance=self.tolerance
                ):
                    grounded = True
                    evidence.append(
                        EvidencePointer(
                            doc_id=source.doc_id,
                            page=t.page,
                            char_start=n.char_start,
                            char_end=n.char_end,
                        )
                    )
                    break
            if not grounded:
                ungrounded.append(n.raw)

        if ungrounded:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.UNGROUNDED,
                score=0.0,
                threshold_used=threshold,
                evidence=evidence,
                detail=(
                    "numbers not in any table: "
                    f"{ungrounded[:5]}"
                ),
            )
        return TierVerdict(
            name=self.name,
            verdict=Verdict.GROUNDED,
            score=1.0,
            threshold_used=threshold,
            evidence=evidence,
            detail=f"all {len(claim_numbers)} numbers matched a table cell",
        )


__all__ = ["TableVerifier"]
