"""Tests for grounding.definitional.consistency."""

from __future__ import annotations

from grounding import Claim, Source, Verdict
from grounding.definitional.consistency import DefinitionalVerifier


GLOSSARY = {
    "EBITDA": "Earnings Before Interest Taxes Depreciation Amortization",
    "DSCR": "Debt Service Coverage Ratio",
}


def test_verify_terms_skipped_for_no_terms() -> None:
    v = DefinitionalVerifier()
    r = v.verify_terms(
        Claim(text="generic narrative without uppercase tokens"),
        GLOSSARY,
        Source.from_text("anything"),
    )
    assert r.verdict == Verdict.SKIPPED


def test_verify_terms_grounded_when_all_in_glossary() -> None:
    v = DefinitionalVerifier()
    r = v.verify_terms(
        Claim(text="EBITDA and DSCR are both relevant."),
        GLOSSARY,
        Source.from_text(""),
    )
    assert r.verdict == Verdict.GROUNDED


def test_verify_terms_ungrounded_for_invented_term() -> None:
    v = DefinitionalVerifier()
    r = v.verify_terms(
        Claim(text="EXPECTED_LOSS is high this year"),
        GLOSSARY,
        Source.from_text(""),
    )
    assert r.verdict == Verdict.UNGROUNDED
    assert "EXPECTED_LOSS" in r.detail


def test_verify_terms_handles_bold_markdown() -> None:
    v = DefinitionalVerifier()
    r = v.verify_terms(
        Claim(text="The **EBITDA** drives **FAKE_RATIO**"),
        GLOSSARY,
        Source.from_text(""),
    )
    assert r.verdict == Verdict.UNGROUNDED
    assert "FAKE_RATIO" in r.detail


def test_verify_terms_ignores_noise_tokens() -> None:
    v = DefinitionalVerifier()
    # API / URL / TODO are common but not domain terms.
    r = v.verify_terms(
        Claim(text="Open the URL TODO refresh API"),
        GLOSSARY,
        Source.from_text(""),
    )
    assert r.verdict == Verdict.SKIPPED


def test_verify_assertion_skipped_when_no_pattern() -> None:
    v = DefinitionalVerifier()
    r = v.verify_assertion(
        Claim(text="just a narrative"),
        GLOSSARY,
        Source.from_text(""),
    )
    assert r.verdict == Verdict.SKIPPED


def test_verify_assertion_grounded_when_consistent() -> None:
    v = DefinitionalVerifier()
    r = v.verify_assertion(
        Claim(
            text=(
                "EBITDA means Earnings Before Interest Taxes "
                "Depreciation Amortization"
            )
        ),
        GLOSSARY,
        Source.from_text(""),
    )
    assert r.verdict == Verdict.GROUNDED


def test_verify_assertion_ungrounded_when_diverges() -> None:
    v = DefinitionalVerifier()
    r = v.verify_assertion(
        Claim(text="EBITDA means random unrelated phrase"),
        GLOSSARY,
        Source.from_text(""),
    )
    assert r.verdict == Verdict.UNGROUNDED


def test_verify_assertion_ungrounded_when_term_missing() -> None:
    v = DefinitionalVerifier()
    r = v.verify_assertion(
        Claim(text="ALPHA means random hallucinated text"),
        GLOSSARY,
        Source.from_text(""),
    )
    assert r.verdict == Verdict.UNGROUNDED


def test_verify_assertion_via_overlap() -> None:
    v = DefinitionalVerifier(overlap_threshold=0.20)
    r = v.verify_assertion(
        Claim(
            text=(
                "DSCR refers to a debt service coverage metric"
            )
        ),
        GLOSSARY,
        Source.from_text(""),
    )
    # Asserted: "a debt service coverage metric"
    # Stored:   "Debt Service Coverage Ratio"
    # Token overlap is moderate but should pass.
    assert r.verdict == Verdict.GROUNDED
