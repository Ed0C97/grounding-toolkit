"""Term-definition consistency checks.

Two checks:

1. **Definition presence** — given a claim that mentions a term and a
   ``glossary`` mapping (``term -> definition``), verify the term is
   defined.  Useful for catching invented acronyms (``EXPECTED_LOSS``)
   that appear nowhere in the document glossary.

2. **Definition coherence** — given a definition assertion ("X means
   ..."), verify the asserted definition is consistent with the
   glossary entry (case-insensitive substring containment in either
   direction, or token-set overlap above a threshold).

Both checks are domain-agnostic: glossaries are supplied by the
consumer (Sentinel, in our case, populates them via Resolver agent).

Also exports :func:`definition_text_overlap` — an asymmetric
definition-grounded ratio that mirrors Sentinel's pre-migration
``sentinel.utils.definition_finder.compute_text_overlap`` (3+ char
tokens, division by definition-token count not by union).  Distinct
from :func:`grounding.tiers.lexical.compute_text_overlap` (Jaccard).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from grounding.core.types import (
    Claim,
    EvidencePointer,
    Source,
    TierVerdict,
    Verdict,
)
from grounding.tiers.lexical import compute_text_overlap


# Asymmetric definition-grounded ratio: fraction of definition tokens
# (>= 3 chars) that also appear in the source text.
_DEF_TOKEN_RE = re.compile(r"\b\w{3,}\b")


def definition_text_overlap(
    definition: str, source_text: str
) -> float:
    """Asymmetric definition-grounded ratio.

    Returns ``|def_tokens & source_tokens| / |def_tokens|``, with
    tokens normalised to lower-case and filtered to length >= 3.

    Migrated bit-for-bit from Sentinel's
    ``sentinel.utils.definition_finder.compute_text_overlap`` so the
    Resolver's 0.30 grounding threshold keeps its exact semantics
    after the hard cutover.
    """
    if not definition or not source_text:
        return 0.0
    def_tokens = set(_DEF_TOKEN_RE.findall(definition.lower()))
    if not def_tokens:
        return 0.0
    src_tokens = set(_DEF_TOKEN_RE.findall(source_text.lower()))
    return len(def_tokens & src_tokens) / len(def_tokens)


# Patterns to extract candidate "definitional" terms from claim text:
# - **Bold** segments (Markdown style)
# - ALL_CAPS_WITH_DIGITS_AND_UNDERSCORES (≥ 4 chars)
_BOLD_RE = re.compile(r"\*\*(?P<term>[^*]{4,}?)\*\*")
_CAPS_RE = re.compile(r"\b(?P<term>[A-Z][A-Z0-9_]{3,})\b")

# Common false-positive ALL CAPS that aren't actually domain terms.
_NOISE = frozenset(
    {
        "TODO",
        "FIXME",
        "XXX",
        "NOTE",
        "WARNING",
        "HTTP",
        "HTTPS",
        "URL",
        "URI",
        "API",
        "JSON",
        "YAML",
        "XML",
        "PDF",
        "OCR",
        "ID",
    }
)

# "<term> means/is/refers to <definition>" detector
_DEFN_RE = re.compile(
    r"^(?P<term>[A-Z][A-Za-z0-9_ \-]{2,40}?)\s+"
    r"(?:means|is defined as|refers to|stands for)\s+"
    r"(?P<defn>.+?)\s*[.\n]?\s*$",
    re.IGNORECASE,
)


def _candidate_terms(text: str) -> List[str]:
    seen: List[str] = []
    seen_set: set[str] = set()
    for m in _BOLD_RE.finditer(text):
        t = m.group("term").strip()
        key = t.upper()
        if key in _NOISE or len(t) <= 3 or key in seen_set:
            continue
        seen_set.add(key)
        seen.append(t)
    for m in _CAPS_RE.finditer(text):
        t = m.group("term").strip()
        key = t.upper()
        if key in _NOISE or len(t) <= 3 or key in seen_set:
            continue
        seen_set.add(key)
        seen.append(t)
    return seen


def _ci_lookup(glossary: Dict[str, str], term: str) -> Optional[str]:
    target = term.lower().strip()
    for k, v in glossary.items():
        if k.lower().strip() == target:
            return v
    return None


@dataclass
class DefinitionalVerifier:
    """Term + definition consistency verifier.

    Inputs:
    - The claim under test.
    - A consumer-supplied ``glossary`` (passed in via :meth:`verify`).
    """

    overlap_threshold: float = 0.30
    name: str = "definitional"

    def verify_terms(
        self,
        claim: Claim,
        glossary: Dict[str, str],
        source: Source,
        *,
        threshold: float = 1.0,
    ) -> TierVerdict:
        """Check that every candidate term in the claim is in glossary."""
        if not claim.text:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="empty claim",
            )
        terms = _candidate_terms(claim.text)
        if not terms:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="no candidate terms in claim",
            )

        undefined: List[str] = []
        evidence: List[EvidencePointer] = []
        glossary_keys_lower = {k.lower() for k in glossary.keys()}
        for term in terms:
            if term.lower() in glossary_keys_lower:
                evidence.append(
                    EvidencePointer(
                        doc_id=source.doc_id,
                        page=None,
                        char_start=0,
                        char_end=len(term),
                    )
                )
            else:
                undefined.append(term)

        if undefined:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.UNGROUNDED,
                threshold_used=threshold,
                evidence=evidence,
                detail=f"undefined terms: {undefined[:5]}",
            )
        return TierVerdict(
            name=self.name,
            verdict=Verdict.GROUNDED,
            score=1.0,
            threshold_used=threshold,
            evidence=evidence,
            detail=f"all {len(terms)} terms in glossary",
        )

    def verify_assertion(
        self,
        claim: Claim,
        glossary: Dict[str, str],
        source: Source,
        *,
        threshold: float = 1.0,
    ) -> TierVerdict:
        """Verify a "<term> means/is/refers to <definition>" claim."""
        if not claim.text:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="empty claim",
            )
        m = _DEFN_RE.match(claim.text.strip())
        if m is None:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="no definitional assertion pattern",
            )
        term = m.group("term").strip()
        asserted = m.group("defn").strip()
        stored = _ci_lookup(glossary, term)
        if stored is None:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.UNGROUNDED,
                threshold_used=threshold,
                detail=f"term {term!r} not in glossary",
            )
        # Substring containment fast path.
        if (
            asserted.lower() in stored.lower()
            or stored.lower() in asserted.lower()
        ):
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
                detail=f"definition match: {term} = {stored[:80]}",
            )
        # Token overlap.
        overlap = compute_text_overlap(asserted, stored)
        if overlap >= self.overlap_threshold:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.GROUNDED,
                score=overlap,
                threshold_used=threshold,
                evidence=[
                    EvidencePointer(
                        doc_id=source.doc_id,
                        page=None,
                        char_start=0,
                        char_end=len(stored),
                    )
                ],
                detail=f"definition overlap={overlap:.3f}",
            )
        return TierVerdict(
            name=self.name,
            verdict=Verdict.UNGROUNDED,
            score=overlap,
            threshold_used=threshold,
            detail=(
                f"asserted definition diverges from glossary "
                f"(overlap={overlap:.3f})"
            ),
        )


__all__ = ["DefinitionalVerifier"]
