"""grounding.calibration — gold-truth schema + threshold tuner + feedback loop."""

from __future__ import annotations

from grounding.calibration.dataset_schema import (
    GoldClaim,
    GoldDataset,
    GoldRecord,
    GoldSource,
    load_dataset,
    save_dataset,
)
from grounding.calibration.feedback_loop import (
    FeedbackBuffer,
    FeedbackEvent,
    promote_to_dataset,
    record_feedback,
)
from grounding.calibration.tuner import (
    PredictFn,
    TuningResult,
    TuningSpec,
    evaluate_calibration_on_dataset,
    tune,
)

__all__ = [
    "FeedbackBuffer",
    "FeedbackEvent",
    "GoldClaim",
    "GoldDataset",
    "GoldRecord",
    "GoldSource",
    "PredictFn",
    "TuningResult",
    "TuningSpec",
    "evaluate_calibration_on_dataset",
    "load_dataset",
    "promote_to_dataset",
    "record_feedback",
    "save_dataset",
    "tune",
]
