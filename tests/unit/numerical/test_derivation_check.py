"""Tests for grounding.numerical.derivation_check."""

from __future__ import annotations

import pytest

from grounding.numerical.derivation_check import (
    DerivationCheck,
    DerivationFormula,
    DerivationVerifier,
)


def _check(expr: str, primitives, claimed):
    return DerivationCheck(
        formula=DerivationFormula(name="X", expression=expr),
        primitives=primitives,
        claimed_value=claimed,
    )


def test_addition() -> None:
    v = DerivationVerifier()
    r = v.verify(_check("a + b", {"a": 2.0, "b": 3.0}, 5.0))
    assert r.ok
    assert r.computed == 5.0


def test_subtraction() -> None:
    v = DerivationVerifier()
    r = v.verify(_check("a - b", {"a": 10.0, "b": 4.0}, 6.0))
    assert r.ok


def test_multiplication() -> None:
    v = DerivationVerifier()
    r = v.verify(_check("a * b", {"a": 3.0, "b": 4.0}, 12.0))
    assert r.ok


def test_division() -> None:
    v = DerivationVerifier()
    r = v.verify(_check("a / b", {"a": 12.0, "b": 10.0}, 1.2))
    assert r.ok


def test_complex_expression() -> None:
    v = DerivationVerifier()
    r = v.verify(
        _check(
            "(ebitda - capex) / debt_service",
            {"ebitda": 100.0, "capex": 20.0, "debt_service": 40.0},
            2.0,
        )
    )
    assert r.ok


def test_within_tolerance() -> None:
    v = DerivationVerifier(tolerance=0.05)
    # claim 1.18 vs computed 1.2 → relative error ≈ 0.0167 → OK
    r = v.verify(_check("a / b", {"a": 12.0, "b": 10.0}, 1.18))
    assert r.ok


def test_outside_tolerance() -> None:
    v = DerivationVerifier(tolerance=0.05)
    r = v.verify(_check("a / b", {"a": 12.0, "b": 10.0}, 5.0))
    assert not r.ok


def test_division_by_zero() -> None:
    v = DerivationVerifier()
    r = v.verify(_check("a / b", {"a": 1.0, "b": 0.0}, 999.0))
    assert not r.ok
    assert "ZeroDivisionError" in r.error
    assert r.computed is None


def test_unknown_identifier() -> None:
    v = DerivationVerifier()
    r = v.verify(_check("a / nope", {"a": 1.0}, 1.0))
    assert not r.ok
    assert "NameError" in r.error
    assert "nope" in r.error


def test_rejects_function_call() -> None:
    v = DerivationVerifier()
    r = v.verify(_check("__import__('os').system('rm -rf /')", {}, 0.0))
    assert not r.ok


def test_rejects_attribute_access() -> None:
    v = DerivationVerifier()
    r = v.verify(_check("a.b", {"a": 1.0}, 1.0))
    assert not r.ok


def test_rejects_subscript() -> None:
    v = DerivationVerifier()
    r = v.verify(_check("a[0]", {"a": 1.0}, 1.0))
    assert not r.ok


def test_rejects_boolean() -> None:
    v = DerivationVerifier()
    r = v.verify(_check("True + 1", {}, 2.0))
    assert not r.ok


def test_unary_minus() -> None:
    v = DerivationVerifier()
    r = v.verify(_check("-a + b", {"a": 5.0, "b": 10.0}, 5.0))
    assert r.ok


def test_power() -> None:
    v = DerivationVerifier()
    r = v.verify(_check("a ** 2", {"a": 4.0}, 16.0))
    assert r.ok


def test_modulo() -> None:
    v = DerivationVerifier()
    r = v.verify(_check("a % 3", {"a": 10.0}, 1.0))
    assert r.ok


def test_floor_div() -> None:
    v = DerivationVerifier()
    r = v.verify(_check("a // 3", {"a": 10.0}, 3.0))
    assert r.ok


def test_empty_expression_rejected() -> None:
    v = DerivationVerifier()
    r = v.verify(_check("", {}, 0.0))
    assert not r.ok


def test_relative_error_property() -> None:
    v = DerivationVerifier()
    r = v.verify(_check("a", {"a": 100.0}, 105.0))
    assert r.relative_error is not None
    assert abs(r.relative_error - (5.0 / 105.0)) < 1e-6


def test_non_numeric_primitive_raises() -> None:
    v = DerivationVerifier()
    r = v.verify(_check("a + 1", {"a": "not a number"}, 2.0))
    assert not r.ok
