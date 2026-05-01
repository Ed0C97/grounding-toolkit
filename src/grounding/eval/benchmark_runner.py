"""Benchmark runner.

Iterates over a :class:`grounding.calibration.dataset_schema.GoldDataset`
and a consumer-supplied ``predict_fn`` that returns a
:class:`GroundingResult` for each record.  Aggregates per-tier and
overall metrics: accuracy, precision, recall, F1, plus the calibration
metrics (Brier, ECE) from :mod:`grounding.confidence.uncertainty`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from grounding.calibration.dataset_schema import GoldDataset, GoldRecord
from grounding.confidence.bayesian import (
    ConfidenceCalibration,
    posterior_grounded,
)
from grounding.confidence.uncertainty import (
    brier_score,
    expected_calibration_error,
)
from grounding.core.types import GroundingResult, Verdict


PredictResultFn = Callable[[GoldRecord], GroundingResult]


@dataclass
class BenchmarkReport:
    """Aggregate metrics over a dataset run."""

    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    brier: float
    ece: float
    per_tier_accuracy: Dict[str, float] = field(default_factory=dict)


def _label_to_verdict(label: str) -> Optional[Verdict]:
    if label == "GROUNDED":
        return Verdict.GROUNDED
    if label == "UNGROUNDED":
        return Verdict.UNGROUNDED
    return None


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def run_benchmark(
    dataset: GoldDataset,
    predict_fn: PredictResultFn,
    *,
    calibration: Optional[ConfidenceCalibration] = None,
) -> BenchmarkReport:
    """Run ``predict_fn`` on every dataset record and aggregate metrics."""
    cal = calibration or ConfidenceCalibration()
    pairs: List[Tuple[float, bool]] = []
    correct = 0
    total = 0
    tp = fp = fn = tn = 0

    per_tier_correct: Dict[str, int] = {}
    per_tier_total: Dict[str, int] = {}

    for rec in dataset.records:
        label_verdict = _label_to_verdict(rec.label)
        if label_verdict is None:
            continue
        result = predict_fn(rec)
        total += 1
        if result.verdict == label_verdict:
            correct += 1
        if (
            label_verdict == Verdict.GROUNDED
            and result.verdict == Verdict.GROUNDED
        ):
            tp += 1
        elif (
            label_verdict == Verdict.UNGROUNDED
            and result.verdict == Verdict.GROUNDED
        ):
            fp += 1
        elif (
            label_verdict == Verdict.GROUNDED
            and result.verdict == Verdict.UNGROUNDED
        ):
            fn += 1
        elif (
            label_verdict == Verdict.UNGROUNDED
            and result.verdict == Verdict.UNGROUNDED
        ):
            tn += 1
        # calibration pair
        prob = posterior_grounded(result, calibration=cal)
        pairs.append((prob, label_verdict == Verdict.GROUNDED))
        # per-tier accuracy
        for tier_name, tv in result.tier_results.items():
            per_tier_total[tier_name] = (
                per_tier_total.get(tier_name, 0) + 1
            )
            if tv.verdict == label_verdict:
                per_tier_correct[tier_name] = (
                    per_tier_correct.get(tier_name, 0) + 1
                )

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return BenchmarkReport(
        n=total,
        accuracy=_safe_div(correct, total),
        precision=precision,
        recall=recall,
        f1=f1,
        brier=brier_score(pairs),
        ece=expected_calibration_error(pairs),
        per_tier_accuracy={
            t: _safe_div(per_tier_correct.get(t, 0), per_tier_total[t])
            for t in per_tier_total
        },
    )


__all__ = ["BenchmarkReport", "run_benchmark"]
