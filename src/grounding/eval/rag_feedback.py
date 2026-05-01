"""RAG feedback evaluation.

The Sentinel-side ``observability/feedback_funcs.grounding_score`` ships
a deterministic Jaccard-based groundedness check used by the implicit
feedback collector.  Phase 16 hard-cutovers Sentinel imports to point
here; the implementation matches Sentinel's contract bit-for-bit.

The metric is intentionally simple: per-sentence max Jaccard against
sources, averaged over all answer sentences.  Token-level Jaccard is
the only operation, so the function is extremely fast and dependency-free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / len(union)


def grounding_score(
    answer: str,
    sources: Sequence[str],
) -> float:
    """Average max-Jaccard of every answer sentence against any source.

    Mirrors :func:`sentinel.observability.feedback_funcs.grounding_score`.
    Returns 0.0 for empty inputs.
    """
    sentences = _split_sentences(answer)
    if not sentences or not sources:
        return 0.0
    total = 0.0
    for s in sentences:
        best = 0.0
        for src in sources:
            score = _jaccard(s, src)
            if score > best:
                best = score
        total += best
    return total / len(sentences)


@dataclass
class BundleEvaluation:
    """Summary returned by :func:`evaluate_bundle`."""

    grounding_score: float
    n_sentences: int
    n_sources: int


def evaluate_bundle(
    answer: str,
    sources: Sequence[str],
) -> BundleEvaluation:
    """Convenience wrapper: returns score + counts."""
    return BundleEvaluation(
        grounding_score=grounding_score(answer, sources),
        n_sentences=len(_split_sentences(answer)),
        n_sources=len(sources),
    )


__all__ = ["BundleEvaluation", "evaluate_bundle", "grounding_score"]
