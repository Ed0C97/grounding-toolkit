"""Tests for grounding.eval.rag_feedback."""

from __future__ import annotations

from grounding.eval.rag_feedback import (
    FeedbackScore,
    evaluate_bundle,
    grounding_score,
    harmfulness_score,
    relevance_score,
)


def test_feedback_score_to_dict_clamps() -> None:
    fs = FeedbackScore(name="x", score=2.0)
    out = fs.to_dict()
    assert out["score"] == 1.0
    fs2 = FeedbackScore(name="y", score=-1.0)
    assert fs2.to_dict()["score"] == 0.0


def test_grounding_score_returns_feedback_score() -> None:
    fs = grounding_score(
        "EBITDA was strong this year.",
        ["EBITDA was strong this year and last."],
    )
    assert isinstance(fs, FeedbackScore)
    assert fs.name == "grounding"
    assert fs.score >= 0.5


def test_grounding_score_empty_answer() -> None:
    fs = grounding_score("", ["x"])
    assert fs.score == 0.0
    assert "empty" in fs.explanation


def test_grounding_score_no_sources() -> None:
    fs = grounding_score("any answer", [])
    assert fs.score == 0.0


def test_relevance_score() -> None:
    fs = relevance_score("EBITDA grew", "EBITDA grew strongly")
    assert fs.name == "relevance"
    assert fs.score > 0.0


def test_relevance_score_missing_inputs() -> None:
    assert relevance_score("", "answer").score == 0.0
    assert relevance_score("query", "").score == 0.0


def test_harmfulness_score_clean() -> None:
    fs = harmfulness_score("a benign harmless statement")
    assert fs.score == 0.0


def test_harmfulness_score_hit() -> None:
    fs = harmfulness_score("how to hack a server")
    assert fs.score > 0.0


def test_evaluate_bundle_returns_dict() -> None:
    bundle = evaluate_bundle(
        query="EBITDA?",
        answer="EBITDA grew strongly.",
        sources=["EBITDA grew strongly this year."],
    )
    assert set(bundle.keys()) == {"grounding", "relevance", "harmfulness"}
    for v in bundle.values():
        assert isinstance(v, FeedbackScore)
