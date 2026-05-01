"""Tests for grounding.eval.trulens_adapter."""

from __future__ import annotations

from grounding.eval.trulens_adapter import (
    trulens_context_precision,
    trulens_context_recall,
    trulens_faithfulness,
    trulens_groundedness,
)


def test_groundedness() -> None:
    s = trulens_groundedness(
        "q",
        "The debt is 8 million.",
        ["The debt is 8 million euros."],
    )
    assert s >= 0.5


def test_faithfulness() -> None:
    s = trulens_faithfulness(
        "q",
        "EBITDA was strong.",
        ["EBITDA was strong this year"],
    )
    assert s >= 0.5


def test_context_precision() -> None:
    s = trulens_context_precision(
        "q",
        "alpha beta",
        ["alpha beta gamma", "x y z"],
    )
    assert 0.0 <= s <= 1.0


def test_context_recall() -> None:
    s = trulens_context_recall(
        "alpha beta gamma.",
        ["alpha beta gamma in context"],
    )
    assert s >= 0.5
