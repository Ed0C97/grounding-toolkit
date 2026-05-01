"""Tests for grounding.calibration.feedback_loop."""

from __future__ import annotations

import pytest

from grounding import (
    Claim,
    GroundingResult,
    GroundingVerifier,
    Source,
    Verdict,
)
from grounding.calibration.dataset_schema import GoldDataset, GoldSource
from grounding.calibration.feedback_loop import (
    FeedbackBuffer,
    promote_to_dataset,
    record_feedback,
)


def _result() -> GroundingResult:
    v = GroundingVerifier()
    return v.verify(Claim(text="hello"), Source.from_text("hello world"))


def test_record_feedback_returns_event() -> None:
    r = _result()
    event = record_feedback(
        result=r,
        record_id="rec-1",
        analyst_label="GROUNDED",
        source_doc_id="doc-1",
        rationale="confirmed by analyst",
    )
    assert event.record_id == "rec-1"
    assert event.analyst_label == "GROUNDED"
    assert event.predicted_verdict == r.verdict


def test_record_feedback_normalises_label() -> None:
    r = _result()
    event = record_feedback(
        result=r,
        record_id="r1",
        analyst_label="grounded",
    )
    assert event.analyst_label == "GROUNDED"


def test_record_feedback_rejects_unknown_label() -> None:
    r = _result()
    with pytest.raises(ValueError):
        record_feedback(
            result=r,
            record_id="r1",
            analyst_label="MAYBE",
        )


def test_buffer_append() -> None:
    r = _result()
    buf = FeedbackBuffer()
    record_feedback(
        result=r,
        record_id="r1",
        analyst_label="GROUNDED",
        buffer=buf,
    )
    assert len(buf) == 1


def test_promote_to_dataset() -> None:
    r = _result()
    buf = FeedbackBuffer()
    record_feedback(
        result=r,
        record_id="r1",
        analyst_label="GROUNDED",
        source_doc_id="doc-1",
        buffer=buf,
    )
    ds = GoldDataset(name="cal")
    n = promote_to_dataset(buf, ds)
    assert n == 1
    assert len(ds.records) == 1
    assert ds.records[0].label == "GROUNDED"


def test_promote_with_source_lookup() -> None:
    r = _result()
    buf = FeedbackBuffer()
    record_feedback(
        result=r,
        record_id="r1",
        analyst_label="GROUNDED",
        source_doc_id="doc-1",
        buffer=buf,
    )

    def _lookup(doc_id: str):
        return GoldSource(text="full source text", doc_id=doc_id)

    ds = GoldDataset(name="cal")
    n = promote_to_dataset(buf, ds, source_lookup=_lookup)
    assert n == 1
    assert ds.records[0].source.text == "full source text"
