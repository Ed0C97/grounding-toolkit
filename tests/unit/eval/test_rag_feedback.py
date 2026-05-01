"""Tests for grounding.eval.rag_feedback."""

from __future__ import annotations

from grounding.eval.rag_feedback import (
    evaluate_bundle,
    grounding_score,
)


def test_grounding_score_perfect() -> None:
    score = grounding_score(
        "The total debt is 8 million euros.",
        ["The total debt is 8 million euros as of today."],
    )
    assert score >= 0.5


def test_grounding_score_empty_inputs() -> None:
    assert grounding_score("", ["x"]) == 0.0
    assert grounding_score("x", []) == 0.0


def test_grounding_score_unrelated() -> None:
    score = grounding_score(
        "absolutely fabricated nonsense.",
        ["a benign source about something else."],
    )
    assert score < 0.5


def test_evaluate_bundle() -> None:
    out = evaluate_bundle(
        "The total debt is 8 million euros.",
        ["The total debt is 8 million euros."],
    )
    assert out.grounding_score >= 0.5
    assert out.n_sentences >= 1
    assert out.n_sources == 1
