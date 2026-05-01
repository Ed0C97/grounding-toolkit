"""Reasoning-trace structure for grounding decisions.

The cascade emits a free-form list of strings into
:attr:`grounding.core.types.GroundingResult.reasoning_trace`.  This
module provides utilities to convert that informal trace into a
structured :class:`ReasoningTrace` record (plus serialisation helpers)
so consumers can render UI / dossier views deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from grounding.core.types import (
    GroundingResult,
    TierVerdict,
    Verdict,
)


@dataclass
class TraceStep:
    """A single step in the reasoning trace."""

    name: str
    verdict: Verdict
    score: float
    threshold_used: float
    detail: str

    @classmethod
    def from_tier_verdict(cls, tv: TierVerdict) -> "TraceStep":
        return cls(
            name=tv.name,
            verdict=tv.verdict,
            score=tv.score,
            threshold_used=tv.threshold_used,
            detail=tv.detail,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "verdict": self.verdict.value,
            "score": self.score,
            "threshold_used": self.threshold_used,
            "detail": self.detail,
        }


@dataclass
class ReasoningTrace:
    """Structured trace of a single :meth:`GroundingVerifier.verify` call."""

    claim_text: str
    final_verdict: Verdict
    confidence: float
    steps: List[TraceStep] = field(default_factory=list)
    free_text: List[str] = field(default_factory=list)

    @classmethod
    def from_result(cls, result: GroundingResult) -> "ReasoningTrace":
        steps = [
            TraceStep.from_tier_verdict(tv)
            for tv in result.tier_results.values()
        ]
        return cls(
            claim_text=result.claim_text,
            final_verdict=result.verdict,
            confidence=result.confidence,
            steps=steps,
            free_text=list(result.reasoning_trace),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_text": self.claim_text,
            "final_verdict": self.final_verdict.value,
            "confidence": self.confidence,
            "steps": [s.to_dict() for s in self.steps],
            "free_text": list(self.free_text),
        }

    def to_markdown(self) -> str:
        """Render the trace as a short Markdown block."""
        lines: List[str] = [
            f"**Claim**: {self.claim_text}",
            f"**Verdict**: `{self.final_verdict.value}`  ",
            f"**Confidence**: {self.confidence:.3f}",
            "",
            "**Tiers**:",
        ]
        for s in self.steps:
            lines.append(
                f"- `{s.name}` → `{s.verdict.value}` "
                f"(score={s.score:.3f}, "
                f"threshold={s.threshold_used:.3f}) — {s.detail}"
            )
        if self.free_text:
            lines.append("")
            lines.append("**Trace**:")
            for line in self.free_text:
                lines.append(f"- {line}")
        return "\n".join(lines)


__all__ = ["ReasoningTrace", "TraceStep"]
