"""grounding-toolkit — Multi-tier groundedness & hallucination detection.

Provider-agnostic, domain-agnostic verification engine for LLM outputs.

Public top-level surface:

- :class:`GroundingVerifier`  — cascade orchestrator
- :class:`Claim`              — what is being verified
- :class:`Source`             — what we verify against
- :class:`GroundingResult`    — the verdict + evidence + confidence
- :class:`EvidencePointer`    — (doc_id, page, char_start, char_end)
- :class:`CitationSpan`       — preventive citation
- :class:`Table`              — structured table
- :class:`TierVerdict`        — single-tier output
- :class:`Verdict`            — GROUNDED | UNGROUNDED | UNCERTAIN | SKIPPED
- :class:`ConsensusPrior`     — inter-generator agreement signal
- :class:`ThresholdProfile`   — per-tier thresholds
- :func:`modulate`            — threshold modulation by prior

Submodules expose finer-grained APIs (tiers, citations, multimodal,
numerical, temporal, definitional, crossdoc, language, explainability,
confidence, audit, adversarial, calibration, eval, constitutional,
consensus, tracking, spatial, answer).
"""

from __future__ import annotations

__version__ = "2026.5.15.0"

from grounding.core.cascade import GroundingVerifier
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
    "__version__",
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
    # Cascade
    "GroundingVerifier",
    # Thresholds
    "ThresholdProfile",
    "modulate",
]
