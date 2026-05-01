"""Tests for grounding.multimodal.figures.FigureVerifier."""

from __future__ import annotations

from grounding import Claim, Source, Verdict
from grounding.multimodal.figures import FigureVerifier


def test_skipped_without_figures() -> None:
    v = FigureVerifier()
    r = v.verify(Claim(text="x"), Source.from_text("y"))
    assert r.verdict == Verdict.SKIPPED


def test_grounded_via_caption() -> None:
    v = FigureVerifier()
    src = Source(figures=[{"caption": "Total debt over time chart"}])
    r = v.verify(Claim(text="Total debt over time chart"), src)
    assert r.verdict == Verdict.GROUNDED


def test_ungrounded_when_no_caption_match() -> None:
    v = FigureVerifier()
    src = Source(figures=[{"caption": "Unrelated diagram"}])
    r = v.verify(Claim(text="something else entirely"), src)
    assert r.verdict == Verdict.UNGROUNDED


def test_uses_alt_when_no_caption() -> None:
    v = FigureVerifier()
    src = Source(figures=[{"alt": "interest rate trend"}])
    r = v.verify(Claim(text="interest rate trend"), src)
    assert r.verdict == Verdict.GROUNDED


def test_custom_caption_fn() -> None:
    v = FigureVerifier(
        caption_fn=lambda fig: fig.get("description", ""),
    )
    src = Source(figures=[{"description": "amortization schedule chart"}])
    r = v.verify(Claim(text="amortization schedule chart"), src)
    assert r.verdict == Verdict.GROUNDED
