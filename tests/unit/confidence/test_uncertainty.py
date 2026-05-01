"""Tests for grounding.confidence.uncertainty."""

from __future__ import annotations

from grounding.confidence.uncertainty import (
    accuracy,
    brier_score,
    evaluate_calibration,
    expected_calibration_error,
)


def test_brier_perfect_score() -> None:
    pairs = [(1.0, True), (0.0, False), (1.0, True)]
    assert brier_score(pairs) == 0.0


def test_brier_worst_score() -> None:
    pairs = [(0.0, True), (1.0, False)]
    # Each error is 1, mean = 1.0
    assert brier_score(pairs) == 1.0


def test_brier_uninformative() -> None:
    pairs = [(0.5, True), (0.5, False), (0.5, True), (0.5, False)]
    assert abs(brier_score(pairs) - 0.25) < 1e-9


def test_brier_empty() -> None:
    assert brier_score([]) == 0.0


def test_ece_perfect() -> None:
    # 100 calibrated predictions: prob 0.7 with 70% positives
    pairs = [(0.7, i < 70) for i in range(100)]
    e = expected_calibration_error(pairs, n_bins=10)
    assert e < 0.05


def test_ece_uncalibrated() -> None:
    # Predict 0.9 always but only 10% positive
    pairs = [(0.9, i < 10) for i in range(100)]
    e = expected_calibration_error(pairs, n_bins=10)
    assert e > 0.7


def test_ece_empty() -> None:
    assert expected_calibration_error([]) == 0.0


def test_accuracy_threshold() -> None:
    pairs = [(0.9, True), (0.1, False), (0.9, False), (0.1, True)]
    a = accuracy(pairs, threshold=0.5)
    assert a == 0.5  # 2/4 correct


def test_accuracy_empty() -> None:
    assert accuracy([]) == 0.0


def test_evaluate_calibration_aggregates() -> None:
    pairs = [(0.9, True), (0.1, False), (0.9, True), (0.1, False)]
    m = evaluate_calibration(pairs, n_bins=10)
    assert m.n == 4
    assert m.brier < 0.05
    assert m.accuracy == 1.0
