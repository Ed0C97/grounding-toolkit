"""RAGAS-compatible groundedness metrics.

Pure-Python implementations of the three core RAGAS retrieval-augmented
generation metrics, computed without any external dependency:

- **Faithfulness**: fraction of claim sentences that are entailed by
  at least one retrieved context passage.
- **Context Precision**: fraction of retrieved contexts that the
  ground-truth answer actually uses.
- **Context Recall**: fraction of ground-truth-answer sentences that
  are present in the retrieved contexts.

The toolkit's metrics deliberately use lexical proxy entailment (token
overlap thresholds) so they run without a model.  When the consumer
passes an :class:`grounding.core.ports.NLIFn`, the entailment check
upgrades to NLI-based scoring automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from grounding.core.ports import NLIFn
from grounding.tiers.lexical import compute_text_overlap


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _entails(
    claim_sentence: str,
    context: str,
    *,
    nli_fn: Optional[NLIFn],
    overlap_threshold: float,
) -> bool:
    """Return True when ``context`` entails ``claim_sentence``."""
    if nli_fn is not None:
        try:
            scores = nli_fn(claim=claim_sentence, source=context)
        except Exception:
            scores = {"entailment": 0.0}
        return float(scores.get("entailment", 0.0)) >= overlap_threshold
    return compute_text_overlap(claim_sentence, context) >= overlap_threshold


@dataclass
class FaithfulnessResult:
    score: float
    n_claim_sentences: int
    n_supported: int


def faithfulness(
    answer: str,
    contexts: Sequence[str],
    *,
    nli_fn: Optional[NLIFn] = None,
    overlap_threshold: float = 0.30,
) -> FaithfulnessResult:
    """Fraction of answer sentences supported by ANY context.

    Without an NLI function, "supported" reduces to token-set overlap
    above ``overlap_threshold`` (Jaccard, lower-cased token sets).
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return FaithfulnessResult(score=0.0, n_claim_sentences=0, n_supported=0)
    n_supported = 0
    for s in sentences:
        for ctx in contexts:
            if _entails(
                s,
                ctx,
                nli_fn=nli_fn,
                overlap_threshold=overlap_threshold,
            ):
                n_supported += 1
                break
    return FaithfulnessResult(
        score=n_supported / len(sentences),
        n_claim_sentences=len(sentences),
        n_supported=n_supported,
    )


@dataclass
class ContextPrecisionResult:
    score: float
    n_contexts: int
    n_relevant: int


def context_precision(
    answer: str,
    contexts: Sequence[str],
    *,
    overlap_threshold: float = 0.20,
) -> ContextPrecisionResult:
    """Fraction of contexts that share content with the answer.

    Approximates "did the model actually use this passage?" via token
    overlap.
    """
    if not contexts:
        return ContextPrecisionResult(score=0.0, n_contexts=0, n_relevant=0)
    n_relevant = sum(
        1
        for ctx in contexts
        if compute_text_overlap(ctx, answer) >= overlap_threshold
    )
    return ContextPrecisionResult(
        score=n_relevant / len(contexts),
        n_contexts=len(contexts),
        n_relevant=n_relevant,
    )


@dataclass
class ContextRecallResult:
    score: float
    n_truth_sentences: int
    n_recalled: int


def context_recall(
    ground_truth_answer: str,
    contexts: Sequence[str],
    *,
    overlap_threshold: float = 0.30,
) -> ContextRecallResult:
    """Fraction of ground-truth-answer sentences present in contexts."""
    sentences = _split_sentences(ground_truth_answer)
    if not sentences:
        return ContextRecallResult(score=0.0, n_truth_sentences=0, n_recalled=0)
    n_recalled = 0
    for s in sentences:
        for ctx in contexts:
            if compute_text_overlap(s, ctx) >= overlap_threshold:
                n_recalled += 1
                break
    return ContextRecallResult(
        score=n_recalled / len(sentences),
        n_truth_sentences=len(sentences),
        n_recalled=n_recalled,
    )


__all__ = [
    "ContextPrecisionResult",
    "ContextRecallResult",
    "FaithfulnessResult",
    "context_precision",
    "context_recall",
    "faithfulness",
]
