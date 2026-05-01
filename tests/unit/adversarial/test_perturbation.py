"""Tests for grounding.adversarial.perturbation.PerturbationDetector."""

from __future__ import annotations

from grounding.adversarial.perturbation import PerturbationDetector


def test_no_perturbation_clean_text() -> None:
    d = PerturbationDetector()
    r = d.detect("Hello, world!")
    assert not r.has_perturbations
    assert r.canonical_text == "Hello, world!"
    assert r.invisible_chars == []
    assert r.confusables_replaced == []


def test_zero_width_space_detected() -> None:
    d = PerturbationDetector()
    r = d.detect("hello​world")
    assert r.has_perturbations
    assert "ZWSP" in r.invisible_chars[0]
    assert r.canonical_text == "helloworld"


def test_zwj_detected() -> None:
    d = PerturbationDetector()
    r = d.detect("a‍b")
    assert r.has_perturbations
    assert "ZWJ" in r.invisible_chars[0]
    assert r.canonical_text == "ab"


def test_bom_detected() -> None:
    d = PerturbationDetector()
    r = d.detect("﻿hello")
    assert r.has_perturbations
    assert r.canonical_text == "hello"


def test_cyrillic_a_replaced() -> None:
    d = PerturbationDetector()
    r = d.detect("аpple")  # cyrillic а + pple
    assert r.has_perturbations
    assert r.confusables_replaced
    assert r.canonical_text == "apple"


def test_greek_o_replaced() -> None:
    d = PerturbationDetector()
    r = d.detect("hοt")  # h + greek omicron + t
    assert r.has_perturbations
    assert r.canonical_text == "hot"


def test_multiple_confusables() -> None:
    d = PerturbationDetector()
    r = d.detect("аео")  # а + е + о (cyrillic)
    assert r.has_perturbations
    assert r.canonical_text == "aeo"
    assert len(r.confusables_replaced) == 3


def test_nfkc_normalisation() -> None:
    """Fullwidth Latin should fold to ASCII via NFKC."""
    d = PerturbationDetector()
    r = d.detect("ＡＢＣ")  # ABC fullwidth
    assert r.has_perturbations
    # Either via _CONFUSABLES or NFKC, the canonical should be "ABC"
    assert r.canonical_text == "ABC"


def test_combined_attack() -> None:
    d = PerturbationDetector()
    payload = "​аpple​"  # zwsp + cyrillic а + pple + zwsp
    r = d.detect(payload)
    assert r.has_perturbations
    assert r.canonical_text == "apple"
    assert r.invisible_chars
    assert r.confusables_replaced
