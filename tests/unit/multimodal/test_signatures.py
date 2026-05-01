"""Tests for grounding.multimodal.signatures.SignatureVerifier."""

from __future__ import annotations

from grounding import Claim, Source, Verdict
from grounding.multimodal.signatures import SignatureVerifier


def test_skipped_without_signatures() -> None:
    v = SignatureVerifier()
    r = v.verify(Claim(text="x"), Source.from_text("y"))
    assert r.verdict == Verdict.SKIPPED


def test_grounded_via_name_match() -> None:
    v = SignatureVerifier()
    src = Source(signatures=[{"name": "Acme Corp", "page": 12}])
    r = v.verify(Claim(text="signed by acme corp"), src)
    assert r.verdict == Verdict.GROUNDED
    assert r.evidence
    assert r.evidence[0].page == 12


def test_grounded_via_role_match() -> None:
    v = SignatureVerifier()
    src = Source(signatures=[{"role": "Lender", "page": 1}])
    r = v.verify(Claim(text="The Lender confirms"), src)
    assert r.verdict == Verdict.GROUNDED


def test_ungrounded_when_no_match() -> None:
    v = SignatureVerifier()
    src = Source(signatures=[{"name": "X", "role": "Y"}])
    r = v.verify(Claim(text="signed by Z"), src)
    assert r.verdict == Verdict.UNGROUNDED
