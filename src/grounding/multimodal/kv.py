"""Verify claims against ``Source.kv_pairs``.

Two complementary checks:

1. **Key-value assertion**: when a claim has the shape
   ``"<key>: <value>"`` (or ``"<key> is <value>"``, ``"<key> = <value>"``),
   the verifier looks up the key in ``Source.kv_pairs`` and compares
   the asserted value to the stored value (case-insensitive substring
   match for strings; numeric tolerance match for numbers).

2. **Numeric reference**: when the claim contains a numeric token, the
   verifier scans the KV values for a matching number — useful when the
   claim references a fact that's only present in extracted KV pairs
   (e.g. forms, headers, signatures).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from grounding.core.types import (
    Claim,
    EvidencePointer,
    Source,
    TierVerdict,
    Verdict,
)
from grounding.numerical.number_extraction import (
    NumberExtractor,
    numbers_match,
)


# Match an explicit "<key>: <value>" or "<key> = <value>" assertion.
# We deliberately do NOT match the English copula "is" — too many
# false positives in free-form descriptive sentences ("this is a ...",
# "the figure is ...").  Consumers who want copula-style matching can
# pre-normalise the claim text.
_KV_CLAIM_RE = re.compile(
    r"^(?P<key>[A-Za-z][A-Za-z0-9_\- ]{1,50}?)\s*"
    r"[:=]\s*"
    r"(?P<value>.+?)\s*$",
)


def _find_key_value(claim_text: str) -> Optional[Tuple[str, str]]:
    m = _KV_CLAIM_RE.match(claim_text.strip())
    if m is None:
        return None
    key = m.group("key").strip()
    value = m.group("value").strip().rstrip(".,;")
    if not key or not value:
        return None
    return key, value


def _ci_lookup(kv: dict, key: str) -> Optional[str]:
    """Case-insensitive lookup with whitespace tolerance."""
    target = key.lower().strip()
    for k, v in kv.items():
        if k.lower().strip() == target:
            return str(v) if v is not None else None
    return None


@dataclass
class KVVerifier:
    """Verify claims against ``Source.kv_pairs``."""

    extractor: NumberExtractor = None  # type: ignore[assignment]
    tolerance: float = 0.05
    name: str = "kv"

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
        if not source.kv_pairs:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="no KV pairs in source",
            )
        if not claim.text:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="empty claim",
            )

        # Path 1: explicit "key: value" assertion
        kv_pair = _find_key_value(claim.text)
        if kv_pair is not None:
            key, asserted_value = kv_pair
            stored = _ci_lookup(source.kv_pairs, key)
            if stored is None:
                return TierVerdict(
                    name=self.name,
                    verdict=Verdict.UNGROUNDED,
                    threshold_used=threshold,
                    detail=f"key {key!r} not in KV pairs",
                )
            if self._values_match(asserted_value, stored):
                return TierVerdict(
                    name=self.name,
                    verdict=Verdict.GROUNDED,
                    score=1.0,
                    threshold_used=threshold,
                    evidence=[
                        EvidencePointer(
                            doc_id=source.doc_id,
                            page=None,
                            char_start=0,
                            char_end=len(stored),
                        )
                    ],
                    detail=f"KV match: {key} = {stored}",
                )
            return TierVerdict(
                name=self.name,
                verdict=Verdict.UNGROUNDED,
                threshold_used=threshold,
                detail=(
                    f"KV mismatch: claimed {asserted_value!r} vs "
                    f"stored {stored!r}"
                ),
            )

        # Path 2: numeric reference scan
        claim_numbers = self.extractor.extract(claim.text)
        if not claim_numbers:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="no key-value pattern and no numeric tokens",
            )

        evidence: List[EvidencePointer] = []
        ungrounded: List[str] = []
        for n in claim_numbers:
            grounded = False
            for v in source.kv_pairs.values():
                if v is None:
                    continue
                v_str = str(v)
                if n.raw.lower() in v_str.lower():
                    grounded = True
                    evidence.append(
                        EvidencePointer(
                            doc_id=source.doc_id,
                            page=None,
                            char_start=n.char_start,
                            char_end=n.char_end,
                        )
                    )
                    break
                for vn in self.extractor.extract(v_str):
                    if vn.unit != n.unit:
                        continue
                    if numbers_match(
                        vn.value, n.value, tolerance=self.tolerance
                    ):
                        grounded = True
                        evidence.append(
                            EvidencePointer(
                                doc_id=source.doc_id,
                                page=None,
                                char_start=n.char_start,
                                char_end=n.char_end,
                            )
                        )
                        break
                if grounded:
                    break
            if not grounded:
                ungrounded.append(n.raw)

        if ungrounded:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.UNGROUNDED,
                threshold_used=threshold,
                evidence=evidence,
                detail=f"numbers not in KV pairs: {ungrounded[:5]}",
            )
        return TierVerdict(
            name=self.name,
            verdict=Verdict.GROUNDED,
            score=1.0,
            threshold_used=threshold,
            evidence=evidence,
            detail=f"all {len(claim_numbers)} numbers matched KV values",
        )

    def _values_match(self, asserted: str, stored: str) -> bool:
        if asserted.lower().strip() == stored.lower().strip():
            return True
        if asserted.lower() in stored.lower():
            return True
        if stored.lower() in asserted.lower():
            return True
        # Numeric match
        a_nums = self.extractor.extract(asserted)
        s_nums = self.extractor.extract(stored)
        for an in a_nums:
            for sn in s_nums:
                if an.unit == sn.unit and numbers_match(
                    an.value, sn.value, tolerance=self.tolerance
                ):
                    return True
        return False


__all__ = ["KVVerifier"]
