"""grounding.core — cascade orchestrator + types + Protocols."""

from __future__ import annotations

from grounding.core.cascade import GroundingVerifier
from grounding.core.ports import (
    EmbeddingFn,
    LLMJudgeFn,
    NLIFn,
    RetrievalFn,
    Tier,
)
from grounding.core.thresholds import ThresholdProfile, modulate
from grounding.core.types import (
    CitationSpan,
    Claim,
    ConsensusPrior,
    EvidencePointer,
    GroundingResult,
    Source,
    Table,
    TierVerdict,
    Verdict,
)

__all__ = [
    # Cascade
    "GroundingVerifier",
    # Ports
    "EmbeddingFn",
    "LLMJudgeFn",
    "NLIFn",
    "RetrievalFn",
    "Tier",
    # Thresholds
    "ThresholdProfile",
    "modulate",
    # Types
    "CitationSpan",
    "Claim",
    "ConsensusPrior",
    "EvidencePointer",
    "GroundingResult",
    "Source",
    "Table",
    "TierVerdict",
    "Verdict",
]
