"""Tests for grounding.explainability.reasoning_trace."""

from __future__ import annotations

from grounding import (
    Claim,
    GroundingResult,
    GroundingVerifier,
    Source,
    TierVerdict,
    Verdict,
)
from grounding.explainability.reasoning_trace import (
    ReasoningTrace,
    TraceStep,
)


def test_from_tier_verdict() -> None:
    tv = TierVerdict(
        name="lex",
        verdict=Verdict.GROUNDED,
        score=0.9,
        threshold_used=0.85,
        detail="match",
    )
    s = TraceStep.from_tier_verdict(tv)
    assert s.name == "lex"
    assert s.verdict == Verdict.GROUNDED
    assert s.score == 0.9


def test_to_dict_roundtrip() -> None:
    s = TraceStep(
        name="lex",
        verdict=Verdict.GROUNDED,
        score=0.9,
        threshold_used=0.85,
        detail="match",
    )
    d = s.to_dict()
    assert d["verdict"] == "GROUNDED"
    assert d["score"] == 0.9


def test_from_result_captures_tier_steps() -> None:
    v = GroundingVerifier()
    r = v.verify(
        Claim(text="hello"),
        Source.from_text("hello world"),
    )
    trace = ReasoningTrace.from_result(r)
    assert trace.final_verdict == Verdict.GROUNDED
    assert any(s.name == "lexical" for s in trace.steps)
    assert any(s.name == "consensus" for s in trace.steps)


def test_to_dict_serialises() -> None:
    v = GroundingVerifier()
    r = v.verify(
        Claim(text="hello"),
        Source.from_text("hello world"),
    )
    trace = ReasoningTrace.from_result(r)
    d = trace.to_dict()
    assert d["final_verdict"] == "GROUNDED"
    assert isinstance(d["steps"], list)
    assert isinstance(d["free_text"], list)


def test_to_markdown_includes_verdict() -> None:
    v = GroundingVerifier()
    r = v.verify(
        Claim(text="hello"),
        Source.from_text("hello world"),
    )
    trace = ReasoningTrace.from_result(r)
    md = trace.to_markdown()
    assert "Verdict" in md
    assert "GROUNDED" in md
