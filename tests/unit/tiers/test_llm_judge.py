"""Tests for grounding.tiers.llm_judge."""

from __future__ import annotations

from grounding import Claim, Source, Verdict
from grounding.testing import StubLLMJudgeFn
from grounding.tiers.llm_judge import LLMJudgeTier


def test_skipped_without_judge_fn() -> None:
    tier = LLMJudgeTier()
    r = tier.verify(Claim(text="x"), Source.from_text("y"))
    assert r.verdict == Verdict.SKIPPED


def test_grounded_via_stub() -> None:
    tier = LLMJudgeTier(judge_fn=StubLLMJudgeFn())
    r = tier.verify(
        Claim(text="alpha"),
        Source.from_text("the alpha and the omega"),
    )
    assert r.verdict == Verdict.GROUNDED


def test_ungrounded_via_stub() -> None:
    tier = LLMJudgeTier(judge_fn=StubLLMJudgeFn())
    r = tier.verify(
        Claim(text="absent"),
        Source.from_text("benign text"),
    )
    assert r.verdict == Verdict.UNGROUNDED


def test_uncertain_when_unknown_verdict() -> None:
    class _WeirdFn:
        def __call__(self, *, claim, source):  # noqa: ARG002
            return {"verdict": "BANANA", "rationale": "?"}

    tier = LLMJudgeTier(judge_fn=_WeirdFn())
    r = tier.verify(Claim(text="x"), Source.from_text("y"))
    assert r.verdict == Verdict.UNCERTAIN


def test_skipped_when_judge_raises() -> None:
    class _RaisingFn:
        def __call__(self, *, claim, source):  # noqa: ARG002
            raise RuntimeError("boom")

    tier = LLMJudgeTier(judge_fn=_RaisingFn())
    r = tier.verify(Claim(text="x"), Source.from_text("y"))
    assert r.verdict == Verdict.SKIPPED
