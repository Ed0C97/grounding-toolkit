"""DeepEval-compatible adapter.

DeepEval (https://github.com/confident-ai/deepeval) defines metric
classes with a ``measure(test_case)`` API that returns a numeric score.
This adapter exposes our RAGAS-style metrics under the same interface
so consumers already wired into DeepEval can swap in our deterministic
implementations without code changes.

We deliberately do NOT take ``deepeval`` as a runtime dependency — the
adapter mimics the interface but works standalone.  If ``deepeval`` is
present, the consumer can register these as custom metrics; if not,
they're still callable via ``.measure()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from grounding.eval.ragas_metrics import (
    context_precision,
    context_recall,
    faithfulness,
)


@dataclass
class DeepEvalTestCase:
    """Minimal test-case shape mirroring DeepEval's LLMTestCase."""

    input: str = ""
    actual_output: str = ""
    expected_output: str = ""
    retrieval_context: List[str] = field(default_factory=list)


class FaithfulnessMetric:
    """DeepEval-shaped wrapper over :func:`faithfulness`."""

    def __init__(self, *, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.score: Optional[float] = None
        self.success: Optional[bool] = None

    def measure(self, test_case: DeepEvalTestCase) -> float:
        result = faithfulness(
            test_case.actual_output, test_case.retrieval_context
        )
        self.score = result.score
        self.success = self.score >= self.threshold
        return self.score


class ContextPrecisionMetric:
    """DeepEval-shaped wrapper over :func:`context_precision`."""

    def __init__(self, *, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.score: Optional[float] = None
        self.success: Optional[bool] = None

    def measure(self, test_case: DeepEvalTestCase) -> float:
        result = context_precision(
            test_case.actual_output, test_case.retrieval_context
        )
        self.score = result.score
        self.success = self.score >= self.threshold
        return self.score


class ContextRecallMetric:
    """DeepEval-shaped wrapper over :func:`context_recall`."""

    def __init__(self, *, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.score: Optional[float] = None
        self.success: Optional[bool] = None

    def measure(self, test_case: DeepEvalTestCase) -> float:
        result = context_recall(
            test_case.expected_output, test_case.retrieval_context
        )
        self.score = result.score
        self.success = self.score >= self.threshold
        return self.score


__all__ = [
    "ContextPrecisionMetric",
    "ContextRecallMetric",
    "DeepEvalTestCase",
    "FaithfulnessMetric",
]
