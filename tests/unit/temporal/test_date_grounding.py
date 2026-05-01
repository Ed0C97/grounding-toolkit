"""Tests for grounding.temporal.date_grounding."""

from __future__ import annotations

from datetime import date

from grounding import Claim, Source, Verdict
from grounding.temporal.date_grounding import (
    DateTimeline,
    TemporalVerifier,
)


def test_skipped_when_no_dates_in_claim() -> None:
    v = TemporalVerifier()
    r = v.verify(
        Claim(text="generic statement"),
        Source.from_text("any text"),
    )
    assert r.verdict == Verdict.SKIPPED


def test_grounded_when_date_in_text() -> None:
    v = TemporalVerifier()
    src = Source.from_text("As of 31/12/2025 the figure was final.")
    r = v.verify(Claim(text="As of 31/12/2025"), src)
    assert r.verdict == Verdict.GROUNDED


def test_grounded_when_year_in_text() -> None:
    v = TemporalVerifier()
    src = Source.from_text("Performance during 2025 was strong.")
    r = v.verify(Claim(text="performance in 2025"), src)
    assert r.verdict == Verdict.GROUNDED


def test_ungrounded_when_date_not_in_text() -> None:
    v = TemporalVerifier()
    src = Source.from_text("As of 2024 the figure was provisional.")
    r = v.verify(Claim(text="As of 31/12/1999"), src)
    assert r.verdict == Verdict.UNGROUNDED


def test_grounded_via_kv_pairs() -> None:
    v = TemporalVerifier()
    src = Source(
        text="",
        kv_pairs={"reporting_date": "2025-12-31"},
    )
    r = v.verify(Claim(text="reporting date 31/12/2025"), src)
    assert r.verdict == Verdict.GROUNDED


def test_grounded_via_timeline_coverage() -> None:
    timeline = DateTimeline(spans=[(date(2025, 1, 1), date(2025, 12, 31))])
    v = TemporalVerifier(timeline=timeline)
    # Date is within the timeline; not in source text.
    src = Source.from_text("nothing temporal here")
    r = v.verify(Claim(text="date 15/06/2025"), src)
    assert r.verdict == Verdict.GROUNDED


def test_ungrounded_outside_timeline_and_text() -> None:
    timeline = DateTimeline(spans=[(date(2025, 1, 1), date(2025, 12, 31))])
    v = TemporalVerifier(timeline=timeline)
    src = Source.from_text("nothing temporal")
    r = v.verify(Claim(text="date 15/06/2030"), src)
    assert r.verdict == Verdict.UNGROUNDED


def test_timeline_covers() -> None:
    t = DateTimeline(
        spans=[
            (date(2024, 1, 1), date(2024, 12, 31)),
            (date(2025, 6, 1), date(2025, 12, 31)),
        ]
    )
    assert t.covers(date(2024, 6, 1))
    assert t.covers(date(2025, 7, 1))
    assert not t.covers(date(2025, 3, 1))
    assert not t.covers(date(2030, 1, 1))


def test_empty_claim_skipped() -> None:
    v = TemporalVerifier()
    r = v.verify(Claim(text=""), Source.from_text("anything"))
    assert r.verdict == Verdict.SKIPPED
