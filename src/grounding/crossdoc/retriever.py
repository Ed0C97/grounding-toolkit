"""Cross-document verifier.

Given a claim, a primary source, and a corpus of related documents:

1. Use :class:`grounding.crossdoc.linker.DocumentLinker` to identify
   which corpus documents the claim references.
2. For each linked document, run a **bidirectional lexical match**:
   substring containment in either direction, falling back to longest
   contiguous match normalised by the SHORTER of the two strings.  This
   is the right choice for cross-doc verification because the claim
   typically wraps a quoted snippet from the linked document with
   reference framing ("As per the Loan Agreement, ...") — normalising
   by the longer string would unfairly penalise the match.
3. If an injected :class:`grounding.core.ports.RetrievalFn` is provided,
   retrieve top-k passages from each linked source instead of
   considering the full doc text (faster on large corpora).
4. Aggregate verdicts: GROUNDED if ANY linked source grounds the claim;
   UNGROUNDED otherwise; SKIPPED when the linker finds no references.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from grounding.core.ports import RetrievalFn
from grounding.core.types import (
    Claim,
    EvidencePointer,
    Source,
    TierVerdict,
    Verdict,
)
from grounding.crossdoc.linker import DocumentLinker, DocumentRef


def _bidirectional_match(claim_text: str, doc_text: str) -> tuple[float, int, int]:
    """Return (ratio, char_start, char_end) of the best match.

    - ``ratio`` is in [0, 1].  1.0 means substring containment in either
      direction; lower values come from
      :meth:`difflib.SequenceMatcher.find_longest_match` normalised by
      ``min(len(claim_text), len(doc_text))``.
    - ``char_start`` / ``char_end`` index into ``doc_text`` and locate
      the best matching window.
    """
    if not claim_text or not doc_text:
        return (0.0, 0, 0)
    idx = doc_text.find(claim_text)
    if idx >= 0:
        return (1.0, idx, idx + len(claim_text))
    idx = claim_text.find(doc_text)
    if idx >= 0:
        return (1.0, 0, len(doc_text))
    sm = difflib.SequenceMatcher(
        None, claim_text, doc_text, autojunk=False
    )
    match = sm.find_longest_match(
        0, len(claim_text), 0, len(doc_text)
    )
    shorter = min(len(claim_text), len(doc_text))
    if shorter == 0:
        return (0.0, match.b, match.b + match.size)
    ratio = match.size / shorter
    return (ratio, match.b, match.b + match.size)


@dataclass
class CrossDocVerifier:
    """Verify a claim against a linked corpus of related documents."""

    linker: DocumentLinker = field(default_factory=DocumentLinker)
    retriever: Optional[RetrievalFn] = None
    top_k: int = 5
    name: str = "crossdoc"

    def verify(
        self,
        claim: Claim,
        corpus: Sequence[DocumentRef],
        *,
        threshold: float = 0.85,
    ) -> TierVerdict:
        if not corpus:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="empty corpus",
            )
        if not claim.text:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="empty claim",
            )

        linked = self.linker.link(claim.text, corpus)
        if not linked:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="claim references no document in corpus",
            )

        evidence: List[EvidencePointer] = []
        best_ratio = 0.0
        for doc in linked:
            sub_source = self._build_subsource(claim, doc)
            ratio, c_start, c_end = _bidirectional_match(
                claim.text, sub_source.text
            )
            if ratio >= threshold:
                evidence.append(
                    EvidencePointer(
                        doc_id=doc.doc_id,
                        page=None,
                        char_start=c_start,
                        char_end=c_end,
                    )
                )
            if ratio > best_ratio:
                best_ratio = ratio

        if evidence:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.GROUNDED,
                score=best_ratio,
                threshold_used=threshold,
                evidence=evidence,
                detail=(
                    f"grounded in linked doc(s): "
                    f"{[d.doc_id for d in linked][:5]}"
                ),
            )
        return TierVerdict(
            name=self.name,
            verdict=Verdict.UNGROUNDED,
            score=best_ratio,
            threshold_used=threshold,
            detail=(
                f"linked {len(linked)} doc(s); best ratio "
                f"{best_ratio:.3f} below {threshold:.3f}"
            ),
        )

    def _build_subsource(
        self, claim: Claim, doc: DocumentRef
    ) -> Source:
        """Either return ``doc.source`` directly or a retrieval-narrowed
        sub-source built from top-k passages."""
        if self.retriever is None:
            return doc.source
        try:
            passages = self.retriever(query=claim.text, top_k=self.top_k)
        except Exception:
            return doc.source
        if not passages:
            return doc.source
        retrieved_text = "\n\n".join(
            str(p.get("text", "")) for p in passages
        )
        return Source(
            text=retrieved_text,
            doc_id=doc.source.doc_id or doc.doc_id,
            language=doc.source.language,
        )


__all__ = ["CrossDocVerifier"]
