"""Tests for grounding.explainability.conflict.ConflictDetector."""

from __future__ import annotations

from grounding import Claim, Source
from grounding.explainability.conflict import ConflictDetector


def test_no_conflict_for_empty_inputs() -> None:
    d = ConflictDetector()
    assert d.detect(Claim(text=""), Source.from_text("y")) == []
    assert d.detect(Claim(text="x"), Source.from_text("")) == []


def test_numeric_mismatch_detected() -> None:
    d = ConflictDetector()
    claim = Claim(text="The total debt is EUR 8,400,000")
    src = Source.from_text("Records show total debt of EUR 12,500,000.")
    out = d.detect(claim, src)
    assert out
    # Pointer should be in the source side
    for p in out:
        assert p.doc_id == "doc"


def test_no_conflict_when_numbers_match_within_tolerance() -> None:
    d = ConflictDetector()
    claim = Claim(text="value is 1,000,000")
    src = Source.from_text("the value is 1,000,000.")
    out = d.detect(claim, src)
    # Same-value match should not flag a conflict.
    assert out == []


def test_negation_flip_detected_english() -> None:
    d = ConflictDetector()
    claim = Claim(text="the obligation is binding")
    src = Source.from_text("the obligation is not binding by virtue of...")
    out = d.detect(claim, src)
    assert out


def test_negation_flip_detected_italian() -> None:
    d = ConflictDetector()
    claim = Claim(text="impegno garantito")
    src = Source.from_text("non impegno garantito secondo lo statuto")
    out = d.detect(claim, src)
    assert out


def test_unit_mismatch_does_not_flag() -> None:
    """A 4.5% claim should not conflict with a 4.5x ratio in source."""
    d = ConflictDetector()
    claim = Claim(text="rate 4.5%")
    src = Source.from_text("DSCR 4.5x baseline")
    out = d.detect(claim, src)
    # different unit → no comparison made → no conflict
    assert out == []
