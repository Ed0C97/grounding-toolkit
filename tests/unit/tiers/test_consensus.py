"""Tests for grounding.tiers.consensus."""

from __future__ import annotations

from grounding import Claim, ConsensusPrior, Source, Verdict
from grounding.tiers.consensus import ConsensusTier


def test_extract_unknown_when_no_metadata() -> None:
    tier = ConsensusTier()
    assert tier.extract_prior(Claim(text="x")) == ConsensusPrior.UNKNOWN


def test_extract_confirmed_variants() -> None:
    tier = ConsensusTier()
    for raw in ("CONFIRMED", "confirmed", "Unanimous", "majority", "agreement"):
        c = Claim(text="x", metadata={"consensus": raw})
        assert tier.extract_prior(c) == ConsensusPrior.CONFIRMED


def test_extract_single_variants() -> None:
    tier = ConsensusTier()
    for raw in ("SINGLE", "single", "None", "ONE"):
        c = Claim(text="x", metadata={"consensus": raw})
        assert tier.extract_prior(c) == ConsensusPrior.SINGLE


def test_extract_disagreement_variants() -> None:
    tier = ConsensusTier()
    for raw in ("DISAGREEMENT", "dissent", "minority_dissent", "CONFLICT"):
        c = Claim(text="x", metadata={"consensus": raw})
        assert tier.extract_prior(c) == ConsensusPrior.DISAGREEMENT


def test_extract_unknown_for_garbage() -> None:
    tier = ConsensusTier()
    c = Claim(text="x", metadata={"consensus": "definitely-not-a-known-value"})
    assert tier.extract_prior(c) == ConsensusPrior.UNKNOWN


def test_extract_alternate_keys() -> None:
    tier = ConsensusTier()
    for key in ("consensus", "moa_consensus", "consensus_prior", "agreement"):
        c = Claim(text="x", metadata={key: "CONFIRMED"})
        assert tier.extract_prior(c) == ConsensusPrior.CONFIRMED


def test_extract_passes_through_enum_value() -> None:
    tier = ConsensusTier()
    c = Claim(
        text="x",
        metadata={"consensus": ConsensusPrior.DISAGREEMENT},
    )
    assert tier.extract_prior(c) == ConsensusPrior.DISAGREEMENT


def test_verify_emits_skipped_with_prior_in_detail() -> None:
    tier = ConsensusTier()
    c = Claim(text="x", metadata={"consensus": "CONFIRMED"})
    s = Source.from_text("anything")
    tv = tier.verify(c, s)
    assert tv.verdict == Verdict.SKIPPED
    assert "CONFIRMED" in tv.detail
