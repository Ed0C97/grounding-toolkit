"""Tests for grounding.answer.verifier.AnswerVerifier."""

from __future__ import annotations

import pytest

from grounding.answer import AnswerVerifier


@pytest.mark.asyncio
async def test_skipped_when_no_chat_fn() -> None:
    v = AnswerVerifier()
    out = await v.verify(
        answer="EBITDA grew",
        chunks=[{"text": "EBITDA grew this year"}],
        query="how was EBITDA?",
    )
    assert out["grounded"] is True
    assert out["ungrounded_claims"] == []


@pytest.mark.asyncio
async def test_grounded_via_judge() -> None:
    async def _fake_judge(*, system_prompt, user_message, model=None, temperature=0.0):
        return {"grounded": True, "ungrounded_claims": []}

    v = AnswerVerifier(chat_fn=_fake_judge)
    out = await v.verify(
        answer="x",
        chunks=[{"text": "y"}],
        query="z",
    )
    assert out["grounded"] is True


@pytest.mark.asyncio
async def test_ungrounded_via_judge() -> None:
    async def _fake_judge(*, system_prompt, user_message, model=None, temperature=0.0):
        return {"grounded": False, "ungrounded_claims": ["claim1", "claim2"]}

    v = AnswerVerifier(chat_fn=_fake_judge)
    out = await v.verify("x", [{"text": "y"}], "z")
    assert out["grounded"] is False
    assert out["ungrounded_claims"] == ["claim1", "claim2"]


@pytest.mark.asyncio
async def test_judge_failure_falls_back_to_grounded_true() -> None:
    async def _fake_judge(*, system_prompt, user_message, model=None, temperature=0.0):
        raise RuntimeError("boom")

    v = AnswerVerifier(chat_fn=_fake_judge)
    out = await v.verify("x", [{"text": "y"}], "z")
    assert out["grounded"] is True


@pytest.mark.asyncio
async def test_feedback_bundle_attached() -> None:
    async def _fake_judge(*, system_prompt, user_message, model=None, temperature=0.0):
        return {"grounded": True, "ungrounded_claims": []}

    v = AnswerVerifier(chat_fn=_fake_judge)
    out = await v.verify("EBITDA grew", [{"text": "EBITDA grew"}], "?")
    assert "feedback_scores" in out
    assert "grounding" in out["feedback_scores"]


@pytest.mark.asyncio
async def test_feedback_bundle_disabled() -> None:
    async def _fake_judge(*, system_prompt, user_message, model=None, temperature=0.0):
        return {"grounded": True, "ungrounded_claims": []}

    v = AnswerVerifier(chat_fn=_fake_judge, attach_feedback_bundle=False)
    out = await v.verify("x", [{"text": "y"}], "z")
    assert "feedback_scores" not in out
