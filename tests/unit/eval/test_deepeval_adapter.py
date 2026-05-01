"""Tests for grounding.eval.deepeval_adapter."""

from __future__ import annotations

from grounding.eval.deepeval_adapter import (
    ContextPrecisionMetric,
    ContextRecallMetric,
    DeepEvalTestCase,
    FaithfulnessMetric,
)


def test_faithfulness_metric_measure() -> None:
    tc = DeepEvalTestCase(
        actual_output="EBITDA was strong",
        retrieval_context=["EBITDA was strong this year"],
    )
    m = FaithfulnessMetric(threshold=0.5)
    score = m.measure(tc)
    assert score >= 0.5
    assert m.success


def test_faithfulness_metric_below_threshold() -> None:
    tc = DeepEvalTestCase(
        actual_output="something fabricated",
        retrieval_context=["totally unrelated content"],
    )
    m = FaithfulnessMetric(threshold=0.95)
    score = m.measure(tc)
    assert score < 0.95
    assert not m.success


def test_context_precision_metric() -> None:
    tc = DeepEvalTestCase(
        actual_output="alpha beta",
        retrieval_context=["alpha beta gamma", "unrelated"],
    )
    m = ContextPrecisionMetric(threshold=0.5)
    score = m.measure(tc)
    assert 0.0 <= score <= 1.0


def test_context_recall_metric() -> None:
    tc = DeepEvalTestCase(
        expected_output="alpha and beta and gamma.",
        retrieval_context=["alpha and beta and gamma are present"],
    )
    m = ContextRecallMetric(threshold=0.5)
    score = m.measure(tc)
    assert score >= 0.5
