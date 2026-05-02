"""Tier 2 — semantic similarity via injected EmbeddingFn.

The tier chunks the source text, embeds claim + chunks via the consumer-
supplied :class:`grounding.core.ports.EmbeddingFn`, computes cosine
similarity between the claim vector and each chunk vector, and accepts
the claim if the maximum similarity meets the threshold.

If no embedding function is injected the tier returns SKIPPED, allowing
the cascade to proceed through subsequent tiers.

Sentinel will inject a local-model EmbeddingFn after the LLM → local
migration is complete (Phase D5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from grounding.core.ports import EmbeddingFn
from grounding.core.types import (
    Claim,
    EvidencePointer,
    Source,
    TierVerdict,
    Verdict,
)


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class SemanticTier:
    """Tier 2 — cosine similarity over chunked source."""

    embedding_fn: Optional[EmbeddingFn] = None
    chunk_size: int = 500
    chunk_stride: int = 250
    name: str = "semantic"

    def verify(
        self,
        claim: Claim,
        source: Source,
        *,
        threshold: float = 0.80,
    ) -> TierVerdict:
        if self.embedding_fn is None:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="no embedding function injected",
            )
        if not claim.text or not source.text:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="empty claim or source",
            )

        chunks = self._chunk(source.text)
        all_texts = [claim.text] + chunks
        try:
            vectors = self.embedding_fn(all_texts)
        except Exception as exc:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail=f"embedding_fn raised: {exc!r}",
            )
        if len(vectors) != len(all_texts):
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="embedding_fn returned wrong number of vectors",
            )

        claim_vec = vectors[0]
        chunk_vecs = vectors[1:]
        best_score = 0.0
        best_idx = 0
        for i, cv in enumerate(chunk_vecs):
            s = _cosine(claim_vec, cv)
            if s > best_score:
                best_score = s
                best_idx = i

        char_start = best_idx * self.chunk_stride
        char_end = min(char_start + self.chunk_size, len(source.text))

        if best_score >= threshold:
            ev = EvidencePointer(
                doc_id=source.doc_id,
                page=None,
                char_start=char_start,
                char_end=char_end,
            )
            return TierVerdict(
                name=self.name,
                verdict=Verdict.GROUNDED,
                score=best_score,
                threshold_used=threshold,
                evidence=[ev],
                detail=f"semantic similarity={best_score:.3f}",
            )

        return TierVerdict(
            name=self.name,
            verdict=Verdict.UNGROUNDED,
            score=best_score,
            threshold_used=threshold,
            detail=(
                f"max similarity={best_score:.3f} below {threshold:.3f}"
            ),
        )

    def _chunk(self, text: str) -> List[str]:
        if not text:
            return []
        chunks: List[str] = []
        i = 0
        while i < len(text):
            chunks.append(text[i : i + self.chunk_size])
            if i + self.chunk_size >= len(text):
                break
            i += self.chunk_stride
        return chunks if chunks else [text]


__all__ = ["SemanticTier"]
