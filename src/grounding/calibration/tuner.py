"""Threshold + weight tuner.

Given a :class:`GoldDataset` and the cascade output for every record,
finds per-tier thresholds and weights that minimise a target objective
(default: ECE; alternative: Brier).  Uses a simple grid search over a
bounded parameter space — sufficient for the toolkit's level of
parameterisation and reproducible without external optimisers.

Future enhancements (post-toolkit-release): plug in
``scikit-optimize`` / ``optuna`` for proper Bayesian optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from grounding.calibration.dataset_schema import GoldDataset, GoldRecord
from grounding.confidence.bayesian import (
    ConfidenceCalibration,
    TierWeights,
    posterior_for_verdicts,
)
from grounding.confidence.uncertainty import (
    brier_score,
    expected_calibration_error,
)
from grounding.core.types import Verdict


PredictFn = Callable[[GoldRecord], Dict[str, Verdict]]


@dataclass
class TuningSpec:
    """Bounded grid-search parameter space."""

    prior_log_odds_grid: Tuple[float, ...] = (-1.0, 0.0, 1.0)
    multiplier_grid: Tuple[float, ...] = (0.5, 1.0, 1.5)
    """Multiplier applied uniformly to every grounded weight (and the
    inverse to ungrounded weights).  Keeps the search space small."""

    objective: str = "ece"


@dataclass
class TuningResult:
    """Best calibration found by the tuner."""

    calibration: ConfidenceCalibration
    objective_value: float
    objective_name: str
    extra_metrics: Dict[str, float] = field(default_factory=dict)


def _scale_weights(base: TierWeights, multiplier: float) -> TierWeights:
    inv = 1.0 / multiplier if multiplier > 0 else 1.0
    fields = base.__dataclass_fields__.keys()
    new_values: Dict[str, float] = {}
    for f in fields:
        v = float(getattr(base, f))
        if f.endswith("_grounded"):
            new_values[f] = v * multiplier
        elif f.endswith("_ungrounded"):
            new_values[f] = v * inv
        else:
            new_values[f] = v
    return TierWeights(**new_values)


def _label_to_bool(label: str) -> Optional[bool]:
    if label == "GROUNDED":
        return True
    if label == "UNGROUNDED":
        return False
    return None


def evaluate_calibration_on_dataset(
    calibration: ConfidenceCalibration,
    dataset: GoldDataset,
    predict_fn: PredictFn,
) -> Tuple[float, float]:
    """Return ``(brier, ece)`` for ``calibration`` on ``dataset``."""
    pairs: List[Tuple[float, bool]] = []
    for rec in dataset.records:
        actual = _label_to_bool(rec.label)
        if actual is None:
            continue
        verdicts = predict_fn(rec)
        prob = posterior_for_verdicts(
            verdicts, calibration=calibration
        )
        pairs.append((prob, actual))
    if not pairs:
        return (0.0, 0.0)
    return (brier_score(pairs), expected_calibration_error(pairs))


def tune(
    dataset: GoldDataset,
    predict_fn: PredictFn,
    *,
    spec: Optional[TuningSpec] = None,
) -> TuningResult:
    """Grid-search the calibration parameters.

    Returns the best :class:`ConfidenceCalibration` plus the objective
    value achieved.  No mutation of dataset or predict_fn.
    """
    spec = spec or TuningSpec()
    base_weights = TierWeights()

    best_value = float("inf")
    best_cal: Optional[ConfidenceCalibration] = None
    best_extra: Dict[str, float] = {}

    for prior in spec.prior_log_odds_grid:
        for mult in spec.multiplier_grid:
            scaled = _scale_weights(base_weights, mult)
            cal = ConfidenceCalibration(
                prior_log_odds=prior, weights=scaled
            )
            brier, ece = evaluate_calibration_on_dataset(
                cal, dataset, predict_fn
            )
            value = ece if spec.objective == "ece" else brier
            if value < best_value:
                best_value = value
                best_cal = cal
                best_extra = {"brier": brier, "ece": ece}

    if best_cal is None:
        best_cal = ConfidenceCalibration()
        best_value = 0.0
    return TuningResult(
        calibration=best_cal,
        objective_value=best_value,
        objective_name=spec.objective,
        extra_metrics=best_extra,
    )


__all__ = [
    "PredictFn",
    "TuningResult",
    "TuningSpec",
    "evaluate_calibration_on_dataset",
    "tune",
]
