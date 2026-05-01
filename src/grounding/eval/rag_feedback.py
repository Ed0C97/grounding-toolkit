"""RAG feedback evaluation.

Migrated from Sentinel's
``sentinel.observability.feedback_funcs`` (P16 hard cutover).

Three TruLens-style feedback functions:

- :func:`grounding_score` — claim ↔ source coverage (per-sentence max
  Jaccard, averaged).
- :func:`relevance_score` — query ↔ answer Jaccard.
- :func:`harmfulness_score` — keyword-based harm detector.

All three return a :class:`FeedbackScore` dataclass with
``name`` / ``score`` (clamped to [0, 1]) / ``explanation`` / ``backend``.

:func:`evaluate_bundle` runs all three and returns a dict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass
class FeedbackScore:
    """Single feedback metric."""

    name: str
    score: float  # clamped into [0, 1] by ``to_dict``
    explanation: str = ""
    backend: str = "heuristic"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score": round(max(0.0, min(1.0, self.score)), 4),
            "explanation": self.explanation,
            "backend": self.backend,
        }


# ---------------------------------------------------------------------------
# Tokenisation + helpers
# ---------------------------------------------------------------------------


_PUNCT = ",.:;!?()[]{}\"'`"
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _tokens(text: str) -> set:
    out: set = set()
    for raw in text.split():
        cleaned = raw.strip(_PUNCT).lower()
        if cleaned:
            out.add(cleaned)
    return out


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    return [
        s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()
    ]


def _best_coverage(claims: Iterable[str], sources: Sequence[str]) -> float:
    claim_list = list(claims)
    if not claim_list or not sources:
        return 0.0
    total = 0.0
    for claim in claim_list:
        total += max(_jaccard(claim, s) for s in sources)
    return total / len(claim_list)


_HARMFUL_PATTERNS = [
    r"\bkill\b",
    r"\bexplode\b",
    r"\bsuicide\b",
    r"\bweapon\b",
    r"\bhack(?:er|ing|)\b",
    r"\bdrug(?:s)?\b",
    r"\bharass",
]


def _harm_hits(text: str) -> int:
    n = 0
    for pattern in _HARMFUL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def grounding_score(
    answer: str, sources: Sequence[str]
) -> FeedbackScore:
    """How well is each sentence of ``answer`` grounded in ``sources``?"""
    if not answer.strip():
        return FeedbackScore(
            name="grounding", score=0.0, explanation="empty answer"
        )
    sentences = _split_sentences(answer)
    if not sentences:
        sentences = [answer]
    score = _best_coverage(sentences, sources)
    explanation = (
        f"{len(sentences)} claims vs {len(sources)} sources"
    )
    return FeedbackScore(
        name="grounding", score=score, explanation=explanation
    )


def relevance_score(query: str, answer: str) -> FeedbackScore:
    """How relevant is ``answer`` to ``query``?"""
    if not query.strip() or not answer.strip():
        return FeedbackScore(
            name="relevance", score=0.0, explanation="missing inputs"
        )
    score = _jaccard(query, answer)
    return FeedbackScore(
        name="relevance",
        score=score,
        explanation="jaccard(query, answer)",
    )


def harmfulness_score(answer: str) -> FeedbackScore:
    """Simple keyword detector. Higher score → more suspected harm."""
    hits = _harm_hits(answer)
    if hits == 0:
        return FeedbackScore(
            name="harmfulness",
            score=0.0,
            explanation="no harmful patterns matched",
        )
    score = min(1.0, hits / 3.0)
    return FeedbackScore(
        name="harmfulness",
        score=score,
        explanation=f"{hits} harmful pattern(s) matched",
    )


def evaluate_bundle(
    *,
    query: str,
    answer: str,
    sources: Sequence[str],
) -> Dict[str, FeedbackScore]:
    """Run all three feedback functions and return a dict of scores."""
    return {
        "grounding": grounding_score(answer, list(sources)),
        "relevance": relevance_score(query, answer),
        "harmfulness": harmfulness_score(answer),
    }


__all__ = [
    "FeedbackScore",
    "evaluate_bundle",
    "grounding_score",
    "harmfulness_score",
    "relevance_score",
]
