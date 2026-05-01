"""grounding-toolkit — Multi-tier groundedness & hallucination detection.

Provider-agnostic, domain-agnostic verification engine for LLM outputs.

Public surface (top-level):

- :class:`GroundingVerifier`  — cascade orchestrator
- :class:`Claim`              — what is being verified
- :class:`Source`             — what we verify against
- :class:`GroundingResult`    — the verdict + evidence + confidence
- :class:`EvidencePointer`    — (doc_id, page, char_start, char_end)
- :class:`Verdict`            — GROUNDED | UNGROUNDED | UNCERTAIN

Submodules expose finer-grained APIs (tiers, citations, multimodal,
numerical, temporal, definitional, crossdoc, language, explainability,
confidence, audit, adversarial, calibration, eval, constitutional,
consensus, tracking, spatial, answer).
"""

from __future__ import annotations

__version__ = "2026.5.15.0"

# Public top-level API (populated by later phases).
__all__ = [
    "__version__",
]
