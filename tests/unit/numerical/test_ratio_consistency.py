"""Tests for grounding.numerical.ratio_consistency."""

from __future__ import annotations

from grounding.numerical.derivation_check import (
    DerivationCheck,
    DerivationFormula,
)
from grounding.numerical.ratio_consistency import (
    RatioConsistencyVerifier,
)


def _check(name: str, expr: str, primitives, claimed):
    return DerivationCheck(
        formula=DerivationFormula(name=name, expression=expr),
        primitives=primitives,
        claimed_value=claimed,
    )


def test_all_consistent() -> None:
    v = RatioConsistencyVerifier()
    report = v.verify(
        [
            _check("R1", "a / b", {"a": 10.0, "b": 5.0}, 2.0),
            _check("R2", "c + d", {"c": 1.0, "d": 2.0}, 3.0),
        ]
    )
    assert report.ok
    assert all(r.ok for r in report.results)


def test_one_inconsistent() -> None:
    v = RatioConsistencyVerifier()
    report = v.verify(
        [
            _check("R1", "a / b", {"a": 10.0, "b": 5.0}, 2.0),
            _check("R2", "c + d", {"c": 1.0, "d": 2.0}, 99.0),
        ]
    )
    assert not report.ok
    failed = report.failed()
    assert len(failed) == 1
    assert failed[0].formula_name == "R2"


def test_summary_when_consistent() -> None:
    v = RatioConsistencyVerifier()
    report = v.verify(
        [_check("R1", "a", {"a": 1.0}, 1.0)]
    )
    assert "consistent" in report.summary()


def test_summary_when_inconsistent() -> None:
    v = RatioConsistencyVerifier()
    report = v.verify(
        [_check("R1", "a", {"a": 1.0}, 99.0)]
    )
    assert "R1" in report.summary()


def test_empty_list() -> None:
    v = RatioConsistencyVerifier()
    report = v.verify([])
    assert report.ok
    assert report.results == []
