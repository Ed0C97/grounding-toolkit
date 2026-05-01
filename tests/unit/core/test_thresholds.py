"""Tests for grounding.core.thresholds."""

from __future__ import annotations

from grounding.core.thresholds import ThresholdProfile, modulate
from grounding.core.types import ConsensusPrior


def test_threshold_profile_defaults() -> None:
    p = ThresholdProfile()
    assert p.fuzzy == 0.85
    assert p.semantic == 0.80
    assert p.nli == 0.65
    assert p.llm_judge == 0.50


def test_modulate_unknown_is_identity() -> None:
    p = ThresholdProfile()
    m = modulate(p, ConsensusPrior.UNKNOWN)
    assert m == p


def test_modulate_single_is_identity() -> None:
    p = ThresholdProfile()
    m = modulate(p, ConsensusPrior.SINGLE)
    assert m.fuzzy == p.fuzzy


def test_modulate_confirmed_loosens() -> None:
    p = ThresholdProfile()
    m = modulate(p, ConsensusPrior.CONFIRMED)
    assert m.fuzzy < p.fuzzy
    assert m.semantic < p.semantic
    assert m.nli < p.nli
    assert m.llm_judge < p.llm_judge


def test_modulate_disagreement_tightens() -> None:
    p = ThresholdProfile()
    m = modulate(p, ConsensusPrior.DISAGREEMENT)
    assert m.fuzzy > p.fuzzy
    assert m.semantic > p.semantic
    assert m.nli > p.nli
    assert m.llm_judge > p.llm_judge


def test_modulate_clamps_at_1() -> None:
    p = ThresholdProfile(fuzzy=0.95)
    m = modulate(p, ConsensusPrior.DISAGREEMENT)
    assert m.fuzzy <= 1.0


def test_modulate_clamps_at_0() -> None:
    p = ThresholdProfile(fuzzy=0.0)
    m = modulate(p, ConsensusPrior.CONFIRMED)
    assert m.fuzzy >= 0.0


def test_modulate_returns_new_instance() -> None:
    p = ThresholdProfile()
    m = modulate(p, ConsensusPrior.CONFIRMED)
    assert m is not p
