"""Tests for grounding.eval.benchmark_runner."""

from __future__ import annotations

from grounding import (
    Claim,
    GroundingResult,
    GroundingVerifier,
    Source,
    Verdict,
)
from grounding.calibration.dataset_schema import (
    GoldClaim,
    GoldDataset,
    GoldRecord,
    GoldSource,
)
from grounding.eval.benchmark_runner import run_benchmark


def _record(text: str, source_text: str, label: str) -> GoldRecord:
    return GoldRecord(
        record_id=f"r-{text}",
        claim=GoldClaim(text=text),
        source=GoldSource(text=source_text),
        label=label,  # type: ignore[arg-type]
    )


def test_benchmark_basic() -> None:
    ds = GoldDataset(
        name="t",
        records=[
            _record("hello", "hello world", "GROUNDED"),
            _record("missing", "completely unrelated", "UNGROUNDED"),
        ],
    )

    def _predict(rec: GoldRecord) -> GroundingResult:
        v = GroundingVerifier()
        return v.verify(
            Claim(text=rec.claim.text),
            Source.from_text(rec.source.text),
        )

    report = run_benchmark(ds, _predict)
    assert report.n == 2
    assert report.accuracy > 0.0
    assert "lexical" in report.per_tier_accuracy


def test_benchmark_perfect_predictor() -> None:
    ds = GoldDataset(
        name="t",
        records=[_record("hello", "hello world", "GROUNDED")],
    )

    def _predict(rec: GoldRecord) -> GroundingResult:
        v = GroundingVerifier()
        return v.verify(
            Claim(text=rec.claim.text),
            Source.from_text(rec.source.text),
        )

    report = run_benchmark(ds, _predict)
    assert report.accuracy == 1.0
    assert report.precision == 1.0
    assert report.recall == 1.0


def test_benchmark_excludes_uncertain_labels() -> None:
    ds = GoldDataset(
        name="t",
        records=[_record("x", "y", "UNCERTAIN")],
    )

    def _predict(rec: GoldRecord) -> GroundingResult:
        return GroundingResult(
            claim_text=rec.claim.text, verdict=Verdict.UNCERTAIN
        )

    report = run_benchmark(ds, _predict)
    assert report.n == 0


def test_benchmark_empty_dataset() -> None:
    ds = GoldDataset(name="t")

    def _predict(rec: GoldRecord) -> GroundingResult:
        return GroundingResult(
            claim_text=rec.claim.text, verdict=Verdict.UNCERTAIN
        )

    report = run_benchmark(ds, _predict)
    assert report.n == 0
    assert report.accuracy == 0.0
