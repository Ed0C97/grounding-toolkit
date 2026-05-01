"""Uncertainty quantification.

Brier score and Expected Calibration Error (ECE) for evaluating how
well a calibrated confidence aligns with empirical groundedness on a
gold-truth dataset.

These are pure-math utilities that take a list of
``(predicted_prob_grounded, actual_grounded_bool)`` pairs and return
calibration metrics.  Used by Phase 13 (calibration framework) to
validate that the confidence module is doing its job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass
class CalibrationMetrics:
    """Output of :func:`evaluate_calibration`."""

    n: int
    brier: float
    ece: float
    accuracy: float


def brier_score(
    pairs: Sequence[Tuple[float, bool]],
) -> float:
    """Mean squared error between predicted probability and outcome.

    Lower is better.  0.0 = perfect calibration + perfect prediction;
    0.25 = uninformative (always predict 0.5).
    """
    if not pairs:
        return 0.0
    total = 0.0
    for prob, actual in pairs:
        a = 1.0 if actual else 0.0
        total += (prob - a) ** 2
    return total / len(pairs)


def expected_calibration_error(
    pairs: Sequence[Tuple[float, bool]],
    *,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) over equal-width bins.

    Returns a non-negative float in [0, 1].  Lower is better.
    """
    if not pairs:
        return 0.0
    n_bins = max(1, n_bins)
    buckets: List[List[Tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for prob, actual in pairs:
        idx = min(n_bins - 1, max(0, int(prob * n_bins)))
        buckets[idx].append((prob, actual))

    total = float(len(pairs))
    ece = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        avg_prob = sum(p for p, _ in bucket) / len(bucket)
        avg_actual = sum(
            1.0 if a else 0.0 for _, a in bucket
        ) / len(bucket)
        weight = len(bucket) / total
        ece += weight * abs(avg_prob - avg_actual)
    return ece


def accuracy(
    pairs: Sequence[Tuple[float, bool]],
    *,
    threshold: float = 0.5,
) -> float:
    """Accuracy when thresholding the probability at ``threshold``."""
    if not pairs:
        return 0.0
    correct = 0
    for prob, actual in pairs:
        predicted = prob >= threshold
        if predicted == bool(actual):
            correct += 1
    return correct / len(pairs)


def evaluate_calibration(
    pairs: Sequence[Tuple[float, bool]],
    *,
    n_bins: int = 10,
    threshold: float = 0.5,
) -> CalibrationMetrics:
    """One-shot calibration evaluation."""
    return CalibrationMetrics(
        n=len(pairs),
        brier=brier_score(pairs),
        ece=expected_calibration_error(pairs, n_bins=n_bins),
        accuracy=accuracy(pairs, threshold=threshold),
    )


__all__ = [
    "CalibrationMetrics",
    "accuracy",
    "brier_score",
    "evaluate_calibration",
    "expected_calibration_error",
]
