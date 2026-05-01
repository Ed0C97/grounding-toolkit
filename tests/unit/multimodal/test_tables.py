"""Tests for grounding.multimodal.tables.TableVerifier."""

from __future__ import annotations

from grounding import Claim, Source, Table, Verdict
from grounding.multimodal.tables import TableVerifier


def _src(*tables: Table) -> Source:
    return Source(text="", tables=list(tables))


def test_skipped_without_tables() -> None:
    v = TableVerifier()
    r = v.verify(Claim(text="value 100"), Source.from_text("nothing"))
    assert r.verdict == Verdict.SKIPPED


def test_skipped_when_no_numbers_in_claim() -> None:
    v = TableVerifier()
    src = _src(Table(headers=["k", "v"], rows=[["x", "y"]]))
    r = v.verify(Claim(text="no digits here"), src)
    assert r.verdict == Verdict.SKIPPED


def test_grounded_when_number_in_table() -> None:
    v = TableVerifier()
    src = _src(
        Table(
            headers=["item", "amount"],
            rows=[["debt", "8,400,000"], ["equity", "5,000,000"]],
        )
    )
    r = v.verify(Claim(text="debt is 8,400,000 EUR"), src)
    assert r.verdict == Verdict.GROUNDED
    assert r.evidence


def test_grounded_with_numeric_tolerance() -> None:
    v = TableVerifier(tolerance=0.05)
    src = _src(
        Table(rows=[["debt", "8,400,000"]])
    )
    # claim says 8.4M which equals 8,400,000 within tolerance
    r = v.verify(Claim(text="debt EUR 8.4M"), src)
    assert r.verdict == Verdict.GROUNDED


def test_ungrounded_when_number_not_in_any_table() -> None:
    v = TableVerifier()
    src = _src(Table(rows=[["debt", "1,000,000"]]))
    r = v.verify(Claim(text="value is 9,999,999 EUR"), src)
    assert r.verdict == Verdict.UNGROUNDED


def test_grounded_with_percentage_match() -> None:
    v = TableVerifier()
    src = _src(Table(rows=[["rate", "4.5%"]]))
    r = v.verify(Claim(text="interest rate 4.5%"), src)
    assert r.verdict == Verdict.GROUNDED


def test_unit_mismatch_does_not_ground() -> None:
    """Percentage 4.5% must not ground a 4.5x ratio."""
    v = TableVerifier()
    src = _src(Table(rows=[["DSCR", "4.5x"]]))
    r = v.verify(Claim(text="rate 4.5%"), src)
    assert r.verdict == Verdict.UNGROUNDED
