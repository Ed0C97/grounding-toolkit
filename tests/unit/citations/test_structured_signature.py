"""Tests for grounding.citations.structured_signature."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from grounding.citations.structured_signature import (
    GroundedClaim,
    GroundedClaimSpan,
    GroundedFindings,
    grounded_claim_system_prompt,
)


def test_grounded_claim_span_validates_page() -> None:
    s = GroundedClaimSpan(page=1, char_start=0, char_end=10)
    assert s.page == 1


def test_grounded_claim_span_rejects_zero_page() -> None:
    with pytest.raises(ValidationError):
        GroundedClaimSpan(page=0, char_start=0, char_end=10)


def test_grounded_claim_span_rejects_negative_offsets() -> None:
    with pytest.raises(ValidationError):
        GroundedClaimSpan(page=1, char_start=-1, char_end=10)


def test_grounded_claim_round_trip_json() -> None:
    c = GroundedClaim(
        text="hello",
        citation_span=GroundedClaimSpan(
            page=1, char_start=0, char_end=5
        ),
    )
    js = c.model_dump_json()
    c2 = GroundedClaim.model_validate_json(js)
    assert c2.text == "hello"
    assert c2.citation_span.page == 1


def test_grounded_findings_holds_list() -> None:
    f = GroundedFindings(
        findings=[
            GroundedClaim(
                text="a",
                citation_span=GroundedClaimSpan(
                    page=1, char_start=0, char_end=1
                ),
            ),
            GroundedClaim(
                text="b",
                citation_span=GroundedClaimSpan(
                    page=2, char_start=10, char_end=20
                ),
            ),
        ]
    )
    assert len(f.findings) == 2


def test_system_prompt_default_is_english() -> None:
    p = grounded_claim_system_prompt()
    assert "citation_span" in p
    assert "page" in p


def test_system_prompt_italian() -> None:
    p = grounded_claim_system_prompt(language="it")
    assert "citation_span" in p
    assert "pagina" in p.lower()
