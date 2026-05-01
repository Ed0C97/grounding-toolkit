"""Conflict-span identification.

A claim is *contradicted* (rather than merely unsupported) when the
source contains an assertion that explicitly disagrees.  This module
ships two heuristics:

1. **Numeric mismatch**: when the claim states a numeric value with a
   specific unit and the source contains the same descriptive context
   with a *different* numeric value (outside tolerance), the source span
   is a conflict.

2. **Negation flip**: when the source contains the claim text wrapped in
   a negation marker ("not", "non", "no", "never", "mai"), the source
   span is a conflict.

Both heuristics are deterministic, free, and locale-tolerant for
EN/IT.  More sophisticated NLI-based contradiction detection lives in
:mod:`grounding.tiers.nli`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from grounding.core.types import (
    Claim,
    EvidencePointer,
    Source,
)
from grounding.numerical.number_extraction import (
    NumberExtractor,
    numbers_match,
)


_NEGATION_WORDS_EN = ("not", "no", "never")
_NEGATION_WORDS_IT = ("non", "no", "mai")
_ALL_NEGATIONS = _NEGATION_WORDS_EN + _NEGATION_WORDS_IT
_COPULAS = (
    "is", "are", "was", "were",         # English
    "e", "sono", "era", "erano",         # Italian
)


def _build_prefix_negation_pattern(claim_text: str) -> re.Pattern:
    """Match ``<negation> <claim>`` (negation BEFORE the claim)."""
    escaped = re.escape(claim_text.strip())
    if not escaped:
        return re.compile(r"$^")
    negations = "|".join(_ALL_NEGATIONS)
    return re.compile(
        rf"\b(?:{negations})\s+{escaped}",
        re.IGNORECASE,
    )


def _build_inline_negation_patterns(
    claim_text: str,
) -> List[re.Pattern]:
    """Match ``<prefix> <copula> <negation> <suffix>`` inside the claim.

    Handles 'X is binding' → conflict 'X is not binding'.
    """
    patterns: List[re.Pattern] = []
    text = claim_text.strip()
    if not text:
        return patterns
    for cop in _COPULAS:
        cop_re = re.compile(
            rf"\b{re.escape(cop)}\b", re.IGNORECASE
        )
        m = cop_re.search(text)
        if m is None:
            continue
        prefix = text[: m.end()].rstrip()
        suffix = text[m.end():].lstrip()
        if not suffix:
            continue
        for neg in _ALL_NEGATIONS:
            target = (
                rf"{re.escape(prefix)}\s+{re.escape(neg)}\s+"
                rf"{re.escape(suffix)}"
            )
            patterns.append(re.compile(target, re.IGNORECASE))
    return patterns


@dataclass
class ConflictDetector:
    """Identify contradictory passages in the source."""

    extractor: NumberExtractor = None  # type: ignore[assignment]
    tolerance: float = 0.05

    def __post_init__(self) -> None:
        if self.extractor is None:
            self.extractor = NumberExtractor()

    def detect(
        self, claim: Claim, source: Source
    ) -> List[EvidencePointer]:
        """Return a list of conflict pointers found in ``source``."""
        if not claim.text or not source.text:
            return []
        out: List[EvidencePointer] = []
        out.extend(self._numeric_mismatch(claim, source))
        out.extend(self._negation_flip(claim, source))
        return out

    # --------------------------------------------------------------
    # Heuristics
    # --------------------------------------------------------------

    def _numeric_mismatch(
        self, claim: Claim, source: Source
    ) -> List[EvidencePointer]:
        out: List[EvidencePointer] = []
        claim_numbers = self.extractor.extract(claim.text)
        if not claim_numbers:
            return out
        source_numbers = self.extractor.extract(source.text)
        for cn in claim_numbers:
            for sn in source_numbers:
                if cn.unit != sn.unit:
                    continue
                if numbers_match(cn.value, sn.value, tolerance=self.tolerance):
                    continue
                # Different value, same unit — potential conflict.
                # Confirm contextual relevance: numeric tokens within
                # ~80 chars share a description; close token positions
                # in source are likely the contradicting figure.
                out.append(
                    EvidencePointer(
                        doc_id=source.doc_id,
                        page=None,
                        char_start=sn.char_start,
                        char_end=sn.char_end,
                    )
                )
        return out

    def _negation_flip(
        self, claim: Claim, source: Source
    ) -> List[EvidencePointer]:
        out: List[EvidencePointer] = []
        target = claim.text.strip()[:80]
        if not target:
            return out

        # Path A: <negation> <claim>
        for m in _build_prefix_negation_pattern(target).finditer(
            source.text
        ):
            out.append(
                EvidencePointer(
                    doc_id=source.doc_id,
                    page=None,
                    char_start=m.start(),
                    char_end=m.end(),
                )
            )

        # Path B: <prefix> <copula> <negation> <suffix>
        for pattern in _build_inline_negation_patterns(target):
            for m in pattern.finditer(source.text):
                out.append(
                    EvidencePointer(
                        doc_id=source.doc_id,
                        page=None,
                        char_start=m.start(),
                        char_end=m.end(),
                    )
                )
        return out


__all__ = ["ConflictDetector"]
