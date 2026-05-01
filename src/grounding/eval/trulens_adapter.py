"""TruLens-compatible feedback function adapter.

TruLens (https://github.com/truera/trulens) wires arbitrary callables
into its eval pipeline as "feedback functions" — float-returning
callables of ``(query, response, [contexts])``.  This adapter exposes
our metrics under that signature.
"""

from __future__ import annotations

from typing import Sequence

from grounding.eval.rag_feedback import grounding_score
from grounding.eval.ragas_metrics import (
    context_precision,
    context_recall,
    faithfulness,
)


def trulens_groundedness(
    query: str,
    response: str,
    contexts: Sequence[str],
) -> float:
    """Return the deterministic grounding score of ``response`` against
    ``contexts``."""
    _ = query  # unused, kept for TruLens signature compatibility
    return grounding_score(response, contexts)


def trulens_faithfulness(
    query: str,
    response: str,
    contexts: Sequence[str],
) -> float:
    _ = query
    return faithfulness(response, contexts).score


def trulens_context_precision(
    query: str,
    response: str,
    contexts: Sequence[str],
) -> float:
    _ = query
    return context_precision(response, contexts).score


def trulens_context_recall(
    expected: str,
    contexts: Sequence[str],
) -> float:
    """Note: TruLens context_recall takes only ``(expected, contexts)`` —
    the query is irrelevant for recall, which compares ground-truth
    answer sentences to retrieved contexts."""
    return context_recall(expected, contexts).score


__all__ = [
    "trulens_context_precision",
    "trulens_context_recall",
    "trulens_faithfulness",
    "trulens_groundedness",
]
