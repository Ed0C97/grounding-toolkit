"""grounding.eval — RAGAS / DeepEval / TruLens adapters + benchmark runner."""

from __future__ import annotations

from grounding.eval.benchmark_runner import (
    BenchmarkReport,
    run_benchmark,
)
from grounding.eval.deepeval_adapter import (
    ContextPrecisionMetric,
    ContextRecallMetric,
    DeepEvalTestCase,
    FaithfulnessMetric,
)
from grounding.eval.rag_feedback import (
    FeedbackScore,
    evaluate_bundle,
    grounding_score,
    harmfulness_score,
    relevance_score,
)
from grounding.eval.ragas_metrics import (
    ContextPrecisionResult,
    ContextRecallResult,
    FaithfulnessResult,
    context_precision,
    context_recall,
    faithfulness,
)
from grounding.eval.trulens_adapter import (
    trulens_context_precision,
    trulens_context_recall,
    trulens_faithfulness,
    trulens_groundedness,
)

__all__ = [
    # Metrics
    "context_precision",
    "context_recall",
    "faithfulness",
    "ContextPrecisionResult",
    "ContextRecallResult",
    "FaithfulnessResult",
    # RAG feedback
    "FeedbackScore",
    "evaluate_bundle",
    "grounding_score",
    "harmfulness_score",
    "relevance_score",
    # DeepEval
    "ContextPrecisionMetric",
    "ContextRecallMetric",
    "DeepEvalTestCase",
    "FaithfulnessMetric",
    # TruLens
    "trulens_context_precision",
    "trulens_context_recall",
    "trulens_faithfulness",
    "trulens_groundedness",
    # Benchmark
    "BenchmarkReport",
    "run_benchmark",
]
