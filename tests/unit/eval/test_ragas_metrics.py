"""Tests for grounding.eval.ragas_metrics."""

from __future__ import annotations

from grounding.eval.ragas_metrics import (
    context_precision,
    context_recall,
    faithfulness,
)


def test_faithfulness_perfect() -> None:
    answer = "The total debt is 8 million."
    contexts = ["The total debt is 8 million euros as of today."]
    r = faithfulness(answer, contexts)
    assert r.score == 1.0
    assert r.n_supported == 1


def test_faithfulness_empty_answer() -> None:
    r = faithfulness("", ["any context"])
    assert r.score == 0.0
    assert r.n_claim_sentences == 0


def test_faithfulness_no_support() -> None:
    answer = "Completely random fabricated claim."
    contexts = ["A totally unrelated passage about something else."]
    r = faithfulness(answer, contexts)
    assert r.score < 0.5


def test_context_precision_all_relevant() -> None:
    answer = "EBITDA grew this year"
    contexts = ["EBITDA growth strong this year"]
    r = context_precision(answer, contexts)
    assert r.score >= 0.5


def test_context_precision_empty_contexts() -> None:
    r = context_precision("answer", [])
    assert r.score == 0.0
    assert r.n_contexts == 0


def test_context_recall_perfect() -> None:
    truth = "EBITDA was strong. Revenue grew."
    contexts = [
        "EBITDA was strong this year.",
        "Revenue grew significantly.",
    ]
    r = context_recall(truth, contexts)
    assert r.score == 1.0


def test_context_recall_partial() -> None:
    truth = "EBITDA was strong. Revenue grew."
    contexts = ["EBITDA was strong this year."]
    r = context_recall(truth, contexts)
    assert 0.0 < r.score < 1.0


def test_context_recall_no_overlap() -> None:
    r = context_recall("foo bar", ["completely different content"])
    assert r.score < 0.5


def test_faithfulness_with_nli_fn() -> None:
    """When an NLI fn is supplied, entailment dominates."""
    def _nli(*, claim, source):  # noqa: ARG001
        return {"entailment": 0.95, "contradiction": 0.0, "neutral": 0.05}

    r = faithfulness(
        "any sentence.",
        ["unrelated source"],
        nli_fn=_nli,
        overlap_threshold=0.5,
    )
    assert r.score == 1.0
