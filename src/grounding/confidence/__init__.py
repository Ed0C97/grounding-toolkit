"""grounding.confidence — Bayesian posterior + uncertainty quantification."""

from __future__ import annotations

from grounding.confidence.bayesian import (
    ConfidenceCalibration,
    TierWeights,
    posterior_for_verdicts,
    posterior_grounded,
)
from grounding.confidence.uncertainty import (
    CalibrationMetrics,
    accuracy,
    brier_score,
    evaluate_calibration,
    expected_calibration_error,
)

__all__ = [
    "CalibrationMetrics",
    "ConfidenceCalibration",
    "TierWeights",
    "accuracy",
    "brier_score",
    "evaluate_calibration",
    "expected_calibration_error",
    "posterior_for_verdicts",
    "posterior_grounded",
]
