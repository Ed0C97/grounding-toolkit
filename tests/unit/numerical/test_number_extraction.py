"""Tests for grounding.numerical.number_extraction."""

from __future__ import annotations

import pytest

from grounding.numerical.number_extraction import (
    NumberExtractor,
    numbers_match,
)


def _values(extractor, text):
    return [n.value for n in extractor.extract(text)]


def _units(extractor, text):
    return [n.unit for n in extractor.extract(text)]


def test_extracts_us_format() -> None:
    e = NumberExtractor()
    out = e.extract("The figure is 1,234.56 dollars.")
    assert any(abs(n.value - 1234.56) < 1e-6 for n in out)


def test_extracts_eu_format() -> None:
    e = NumberExtractor()
    out = e.extract("Il valore e' 1.234,56 euro.")
    assert any(abs(n.value - 1234.56) < 1e-6 for n in out)


def test_extracts_million_multiplier_uppercase() -> None:
    e = NumberExtractor()
    out = e.extract("EUR 8.4M as of today")
    matched = [n for n in out if abs(n.value - 8_400_000) < 1.0]
    assert matched
    assert matched[0].multiplier in ("M", "m")


def test_extracts_billion_multiplier() -> None:
    e = NumberExtractor()
    out = e.extract("Total: 1.5B")
    assert any(abs(n.value - 1_500_000_000) < 1.0 for n in out)


def test_extracts_italian_mln() -> None:
    e = NumberExtractor()
    out = e.extract("Il debito e' 8,4 mln euro.")
    matched = [n for n in out if abs(n.value - 8_400_000) < 1.0]
    assert matched


def test_extracts_currency_prefix() -> None:
    e = NumberExtractor()
    out = e.extract("EUR 1234.56 final.")
    matched = [n for n in out if abs(n.value - 1234.56) < 1e-6]
    assert matched
    assert matched[0].currency == "EUR"


def test_extracts_currency_symbol() -> None:
    e = NumberExtractor()
    out = e.extract("paid € 100.50 today.")
    matched = [n for n in out if abs(n.value - 100.5) < 1e-6]
    assert matched


def test_extracts_percentage() -> None:
    e = NumberExtractor()
    units = _units(e, "interest rate 4.5% nominal")
    assert "%" in units


def test_extracts_ratio() -> None:
    e = NumberExtractor()
    out = e.extract("DSCR is 1.2x baseline")
    pcts = [n for n in out if n.unit == "x"]
    assert pcts
    assert abs(pcts[0].value - 1.2) < 1e-6


def test_extracts_year() -> None:
    e = NumberExtractor()
    out = e.extract("As of 2025 the position was clear.")
    years = [n for n in out if n.unit == "year"]
    assert any(int(n.value) == 2025 for n in years)


def test_extracts_date_dd_mm_yyyy() -> None:
    e = NumberExtractor()
    out = e.extract("Date: 31/12/2025")
    dates = [n for n in out if n.unit == "date"]
    assert dates
    assert dates[0].extras["year"] == 2025
    assert dates[0].extras["month"] == 12
    assert dates[0].extras["day"] == 31


def test_extracts_date_iso() -> None:
    e = NumberExtractor()
    out = e.extract("Booking date 2026-01-15.")
    dates = [n for n in out if n.unit == "date"]
    assert dates
    assert dates[0].extras["year"] == 2026


def test_min_money_value_filter() -> None:
    e = NumberExtractor(min_money_value=1000.0)
    out = e.extract("99 euros plus 5000 euros.")
    # 99 should be filtered out as a small money value;
    # 5000 should be kept.
    money = [n for n in out if n.unit == ""]
    assert all(abs(n.value) >= 1000.0 for n in money)
    assert any(abs(n.value - 5000.0) < 1.0 for n in money)


def test_overlapping_patterns_year_wins_over_money() -> None:
    """A 4-digit year should not be re-extracted as a monetary value."""
    e = NumberExtractor()
    out = e.extract("Year 2025")
    # The token 2025 should appear once with unit=year
    matches = [n for n in out if int(n.value) == 2025]
    assert len(matches) == 1
    assert matches[0].unit == "year"


def test_numbers_match_exact() -> None:
    assert numbers_match(100.0, 100.0)


def test_numbers_match_within_tolerance() -> None:
    assert numbers_match(100.0, 104.0, tolerance=0.05)


def test_numbers_match_outside_tolerance() -> None:
    assert not numbers_match(100.0, 110.0, tolerance=0.05)


def test_numbers_match_handles_zero() -> None:
    assert numbers_match(0.0, 0.0)
    assert not numbers_match(0.0, 1.0)


def test_extractor_returns_sorted_by_position() -> None:
    e = NumberExtractor()
    out = e.extract("First 100, then 200, finally 300 EUR.")
    starts = [n.char_start for n in out]
    assert starts == sorted(starts)
