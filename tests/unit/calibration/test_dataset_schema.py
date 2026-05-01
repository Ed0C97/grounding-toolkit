"""Tests for grounding.calibration.dataset_schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grounding.calibration.dataset_schema import (
    GoldClaim,
    GoldDataset,
    GoldRecord,
    GoldSource,
    load_dataset,
    save_dataset,
)


def _build_record() -> GoldRecord:
    return GoldRecord(
        record_id="r1",
        claim=GoldClaim(text="hello"),
        source=GoldSource(text="hello world", doc_id="d1"),
        label="GROUNDED",
        annotator="alice",
    )


def test_build_record_validates() -> None:
    rec = _build_record()
    assert rec.label == "GROUNDED"
    assert rec.claim.text == "hello"


def test_label_must_be_known() -> None:
    with pytest.raises(Exception):
        GoldRecord(
            record_id="r1",
            claim=GoldClaim(text="x"),
            source=GoldSource(),
            label="WAT",  # type: ignore[arg-type]
        )


def test_dataset_round_trip(tmp_path: Path) -> None:
    ds = GoldDataset(name="test", records=[_build_record()])
    path = tmp_path / "ds.json"
    save_dataset(ds, path)
    loaded = load_dataset(path)
    assert loaded.name == "test"
    assert len(loaded.records) == 1
    assert loaded.records[0].claim.text == "hello"


def test_dataset_default_locale_en() -> None:
    ds = GoldDataset(name="t")
    assert ds.locale == "en"


def test_records_default_empty() -> None:
    ds = GoldDataset(name="t")
    assert ds.records == []


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "ds.json"
    save_dataset(GoldDataset(name="t"), nested)
    assert nested.exists()
