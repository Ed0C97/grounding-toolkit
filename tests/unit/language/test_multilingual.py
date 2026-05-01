"""Tests for grounding.language.multilingual."""

from __future__ import annotations

from grounding import Claim, Source, Verdict
from grounding.language.multilingual import (
    LocaleGlossary,
    MultilingualVerifier,
)


def _gloss():
    return LocaleGlossary(
        source_locale="en",
        target_locale="it",
        mappings={
            "loan": ["prestito", "finanziamento"],
            "guarantee": ["garanzia"],
        },
    )


def test_locale_glossary_translates_forward() -> None:
    g = _gloss()
    out = g.translate("loan")
    assert "loan" in out
    assert "prestito" in out
    assert "finanziamento" in out


def test_locale_glossary_translates_backward() -> None:
    g = _gloss()
    out = g.translate("garanzia")
    assert "garanzia" in out
    assert "guarantee" in out


def test_locale_glossary_unknown_term_returns_self() -> None:
    g = _gloss()
    out = g.translate("untranslatable")
    assert out == ["untranslatable"]


def test_grounded_in_same_locale_no_glossary() -> None:
    v = MultilingualVerifier()
    src = Source.from_text("the loan agreement is valid", language="en")
    r = v.verify(Claim(text="the loan agreement is valid"), src)
    assert r.verdict == Verdict.GROUNDED


def test_grounded_after_translation() -> None:
    v = MultilingualVerifier(glossary=_gloss())
    # Source is in Italian; claim is in English mentioning "loan"
    src = Source.from_text(
        "il prestito agreement is valid",
        language="it",
    )
    claim = Claim(
        text="the loan agreement is valid",
        metadata={"language": "en"},
    )
    r = v.verify(claim, src, threshold=0.7)
    # After "loan" → "prestito" translation, the variant should match.
    assert r.verdict == Verdict.GROUNDED


def test_ungrounded_when_no_translation_helps() -> None:
    v = MultilingualVerifier(glossary=_gloss())
    src = Source.from_text(
        "completely unrelated content",
        language="it",
    )
    claim = Claim(
        text="the loan agreement is valid",
        metadata={"language": "en"},
    )
    r = v.verify(claim, src)
    assert r.verdict == Verdict.UNGROUNDED


def test_skipped_for_empty_inputs() -> None:
    v = MultilingualVerifier(glossary=_gloss())
    assert (
        v.verify(Claim(text=""), Source.from_text("y")).verdict
        == Verdict.SKIPPED
    )
    assert (
        v.verify(Claim(text="x"), Source.from_text("")).verdict
        == Verdict.SKIPPED
    )


def test_supports_locale() -> None:
    g = _gloss()
    assert g.supports("en")
    assert g.supports("it")
    assert not g.supports("fr")


def test_falls_back_to_lexical_when_glossary_does_not_support() -> None:
    v = MultilingualVerifier(glossary=_gloss())
    # Source language is French; glossary supports en/it.  Should fall
    # through to plain lexical without translation.
    src = Source.from_text("the loan agreement is valid", language="fr")
    claim = Claim(
        text="the loan agreement is valid",
        metadata={"language": "en"},
    )
    r = v.verify(claim, src)
    assert r.verdict == Verdict.GROUNDED
