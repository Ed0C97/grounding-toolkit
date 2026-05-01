"""Tests for grounding.calibration.tuner."""

from __future__ import annotations

from grounding.calibration.dataset_schema import (
    GoldClaim,
    GoldDataset,
    GoldRecord,
    GoldSource,
)
from grounding.calibration.tuner import (
    TuningSpec,
    evaluate_calibration_on_dataset,
    tune,
)
from grounding.confidence.bayesian import ConfidenceCalibration
from grounding.core.types import Verdict


def _record(label: str, ground: bool) -> GoldRecord:
    return GoldRecord(
        record_id=f"r-{label}-{ground}",
        claim=GoldClaim(text="x"),
        source=GoldSource(),
        label=label,  # type: ignore[arg-type]
    )


def _predict_perfect(rec: GoldRecord):
    """Predict-fn that mirrors the ground truth perfectly."""
    if rec.label == "GROUNDED":
        return {"lexical": Verdict.GROUNDED}
    return {"lexical": Verdict.UNGROUNDED}


def _predict_random(rec: GoldRecord):
    """Always predicts SKIPPED → posterior == prior sigmoid."""
    return {"lexical": Verdict.SKIPPED}


def test_tune_returns_calibration() -> None:
    ds = GoldDataset(
        name="t",
        records=[
            _record("GROUNDED", True),
            _record("UNGROUNDED", False),
        ],
    )
    out = tune(ds, _predict_perfect)
    assert isinstance(out.calibration, ConfidenceCalibration)
    assert out.objective_name == "ece"


def test_perfect_predictor_yields_low_ece() -> None:
    ds = GoldDataset(
        name="t",
        records=[
            _record("GROUNDED", True) for _ in range(10)
        ] + [
            _record("UNGROUNDED", False) for _ in range(10)
        ],
    )
    out = tune(ds, _predict_perfect)
    assert out.objective_value < 0.20


def test_evaluate_directly() -> None:
    ds = GoldDataset(
        name="t",
        records=[_record("GROUNDED", True), _record("UNGROUNDED", False)],
    )
    cal = ConfidenceCalibration()
    brier, ece = evaluate_calibration_on_dataset(cal, ds, _predict_perfect)
    assert 0.0 <= brier <= 1.0
    assert 0.0 <= ece <= 1.0


def test_tune_brier_objective() -> None:
    ds = GoldDataset(
        name="t",
        records=[_record("GROUNDED", True)],
    )
    out = tune(ds, _predict_perfect, spec=TuningSpec(objective="brier"))
    assert out.objective_name == "brier"


def test_skipped_predictions_dont_break_tuner() -> None:
    ds = GoldDataset(
        name="t",
        records=[_record("GROUNDED", True), _record("UNGROUNDED", False)],
    )
    out = tune(ds, _predict_random)
    assert out.calibration is not None


def test_uncertain_records_excluded() -> None:
    ds = GoldDataset(
        name="t",
        records=[
            _record("UNCERTAIN", False),
        ],
    )
    out = tune(ds, _predict_perfect)
    # Empty effective set → tuner returns whatever default
    assert out.objective_value == 0.0
