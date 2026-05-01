"""Tests for grounding.multimodal.kv.KVVerifier."""

from __future__ import annotations

from grounding import Claim, Source, Verdict
from grounding.multimodal.kv import KVVerifier


def test_skipped_without_kv_pairs() -> None:
    v = KVVerifier()
    r = v.verify(Claim(text="anything"), Source.from_text("text"))
    assert r.verdict == Verdict.SKIPPED


def test_grounded_explicit_key_value() -> None:
    v = KVVerifier()
    src = Source(kv_pairs={"counterparty": "Acme Corp"})
    r = v.verify(Claim(text="Counterparty: Acme Corp"), src)
    assert r.verdict == Verdict.GROUNDED


def test_ungrounded_explicit_key_mismatch() -> None:
    v = KVVerifier()
    src = Source(kv_pairs={"counterparty": "Acme Corp"})
    r = v.verify(Claim(text="Counterparty: Other Inc"), src)
    assert r.verdict == Verdict.UNGROUNDED


def test_ungrounded_unknown_key() -> None:
    v = KVVerifier()
    src = Source(kv_pairs={"counterparty": "Acme"})
    r = v.verify(Claim(text="Lender: BBB"), src)
    assert r.verdict == Verdict.UNGROUNDED


def test_grounded_via_numeric_scan_in_kv() -> None:
    v = KVVerifier()
    src = Source(kv_pairs={"total_debt": "8,400,000 EUR"})
    r = v.verify(Claim(text="value 8,400,000 EUR"), src)
    assert r.verdict == Verdict.GROUNDED


def test_grounded_value_substring_match() -> None:
    """Stored 'EUR 100,000' should ground claim 'amount: 100,000'."""
    v = KVVerifier()
    src = Source(kv_pairs={"amount": "EUR 100,000.00"})
    r = v.verify(Claim(text="amount: 100,000"), src)
    assert r.verdict == Verdict.GROUNDED


def test_skipped_no_kv_pattern_no_numbers() -> None:
    v = KVVerifier()
    src = Source(kv_pairs={"x": "y"})
    r = v.verify(Claim(text="this is purely descriptive text"), src)
    assert r.verdict == Verdict.SKIPPED
