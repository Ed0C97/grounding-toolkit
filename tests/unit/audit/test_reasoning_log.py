"""Tests for grounding.audit.reasoning_log."""

from __future__ import annotations

import json

from grounding import (
    Claim,
    GroundingVerifier,
    Source,
    Verdict,
)
from grounding.audit.reasoning_log import (
    ReasoningLog,
    record_for_result,
)


def test_record_for_result_captures_basics() -> None:
    v = GroundingVerifier()
    r = v.verify(
        Claim(text="hello"),
        Source.from_text("hello world", doc_id="abc"),
    )
    rec = record_for_result(r, source_doc_id="abc")
    assert rec.claim_text == "hello"
    assert rec.source_doc_id == "abc"
    assert rec.final_verdict == "GROUNDED"
    assert rec.evidence_count >= 1
    assert rec.merkle_root != ""


def test_record_id_deterministic_per_inputs() -> None:
    v = GroundingVerifier()
    r = v.verify(Claim(text="x"), Source.from_text("x"))
    rec_a = record_for_result(r, source_doc_id="d", timestamp=1.0)
    rec_b = record_for_result(r, source_doc_id="d", timestamp=1.0)
    assert rec_a.record_id == rec_b.record_id


def test_record_id_differs_per_timestamp() -> None:
    v = GroundingVerifier()
    r = v.verify(Claim(text="x"), Source.from_text("x"))
    rec_a = record_for_result(r, timestamp=1.0)
    rec_b = record_for_result(r, timestamp=2.0)
    assert rec_a.record_id != rec_b.record_id


def test_record_to_json_roundtrip() -> None:
    v = GroundingVerifier()
    r = v.verify(Claim(text="x"), Source.from_text("x"))
    rec = record_for_result(r, timestamp=1.0)
    js = rec.to_json()
    parsed = json.loads(js)
    assert parsed["claim_text"] == "x"
    assert parsed["final_verdict"] == "GROUNDED"
    assert parsed["timestamp"] == 1.0


def test_log_append_and_count() -> None:
    log = ReasoningLog()
    v = GroundingVerifier()
    for i in range(3):
        r = v.verify(
            Claim(text=f"claim-{i}"),
            Source.from_text(f"claim-{i} extra"),
        )
        log.append_result(r)
    assert len(log) == 3


def test_log_to_json_array() -> None:
    log = ReasoningLog()
    v = GroundingVerifier()
    r = v.verify(
        Claim(text="x"),
        Source.from_text("x"),
    )
    log.append_result(r)
    js = log.to_json()
    parsed = json.loads(js)
    assert isinstance(parsed, list)
    assert len(parsed) == 1


def test_metadata_propagates() -> None:
    v = GroundingVerifier()
    r = v.verify(Claim(text="x"), Source.from_text("x"))
    rec = record_for_result(r, metadata={"tenant": "acme"})
    assert rec.metadata["tenant"] == "acme"
