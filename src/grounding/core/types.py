"""Core type definitions for grounding-toolkit.

These dataclasses are the public contract every consumer depends on:

- :class:`Claim`            — what is being verified
- :class:`Source`           — what we verify against
- :class:`Verdict`          — GROUNDED / UNGROUNDED / UNCERTAIN / SKIPPED
- :class:`EvidencePointer`  — locator into source: doc + page + char range
- :class:`CitationSpan`     — preventive citation emitted by the LLM
- :class:`Table`            — structured table (consumer-populated)
- :class:`TierVerdict`      — output of a single tier
- :class:`GroundingResult`  — final cascade output
- :class:`ConsensusPrior`   — inter-generator agreement signal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Verdict(str, Enum):
    """Final groundedness verdict for a claim."""

    GROUNDED = "GROUNDED"
    UNGROUNDED = "UNGROUNDED"
    UNCERTAIN = "UNCERTAIN"
    SKIPPED = "SKIPPED"


class ConsensusPrior(str, Enum):
    """Inter-generator agreement signal that modulates tier thresholds.

    - ``CONFIRMED``: multiple independent generators agree on the claim
      → relax thresholds (low hallucination risk).
    - ``SINGLE``: only one generator produced the claim → default
      thresholds.
    - ``DISAGREEMENT``: generators disagree → tighten thresholds (high
      hallucination risk).
    - ``UNKNOWN``: no consensus metadata available → default.
    """

    CONFIRMED = "CONFIRMED"
    SINGLE = "SINGLE"
    DISAGREEMENT = "DISAGREEMENT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CitationSpan:
    """LLM-emitted preventive citation.

    The LLM is asked (via a structured signature) to attach a span to
    every claim it produces, pointing at the source location it claims
    supports the statement.  The cascade verifies whether the span
    actually contains text matching the claim — when it does, we
    short-circuit the rest of the cascade.
    """

    page: int
    char_start: int
    char_end: int

    def length(self) -> int:
        return max(0, self.char_end - self.char_start)


@dataclass(frozen=True)
class EvidencePointer:
    """Locator into the source corpus.

    ``page`` may be ``None`` for sources that lack pagination.
    ``char_start`` / ``char_end`` are offsets into the page text (or
    ``Source.text`` if no pagination) — half-open: ``[start, end)``.
    """

    doc_id: str
    page: Optional[int]
    char_start: int
    char_end: int

    def length(self) -> int:
        return max(0, self.char_end - self.char_start)


@dataclass
class Table:
    """Structured table extracted from the source.

    Consumer-populated.  The toolkit does not parse documents; it only
    consumes already-extracted data.
    """

    page: Optional[int] = None
    headers: List[str] = field(default_factory=list)
    rows: List[List[Any]] = field(default_factory=list)
    caption: str = ""


@dataclass
class Claim:
    """A statement to verify against a :class:`Source`.

    ``text`` is the natural-language assertion.  Optional fields:
    - ``page``: which page the claim refers to (used by page-bounds
      checks).
    - ``citation_span``: if the LLM emitted a preventive citation, the
      span the LLM claims supports the statement.
    - ``metadata``: free-form bag.  The cascade reads keys
      ``consensus`` / ``moa_consensus`` / ``consensus_prior`` /
      ``agreement`` to determine the inter-generator agreement signal.
    """

    text: str
    page: Optional[int] = None
    citation_span: Optional[CitationSpan] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Source:
    """The ground-truth corpus to verify a :class:`Claim` against.

    Fully populated by the consumer.  The toolkit consumes only already-
    extracted data — it does NOT call ocr-toolkit, pdf-finder, or any
    other parser directly.
    """

    text: str = ""
    tables: List[Table] = field(default_factory=list)
    kv_pairs: Dict[str, str] = field(default_factory=dict)
    figures: List[Dict[str, Any]] = field(default_factory=list)
    signatures: List[Dict[str, Any]] = field(default_factory=list)
    page_count: int = 0
    doc_id: str = "doc"
    language: str = "en"

    @classmethod
    def from_text(cls, text: str, **kwargs: Any) -> "Source":
        """Build a Source from a single text blob (convenience)."""
        return cls(text=text, **kwargs)


@dataclass
class TierVerdict:
    """Output of a single tier in the cascade.

    - ``verdict``: this tier's verdict in isolation.  ``SKIPPED`` means
      the tier didn't apply (e.g. semantic tier without an embedding
      backend).
    - ``score``: 0..1, higher means stronger groundedness signal.
    - ``threshold_used``: the threshold the tier compared ``score``
      against (after modulation by the consensus prior).
    - ``evidence``: zero or more :class:`EvidencePointer` instances if
      the tier identified a supporting passage.
    - ``detail``: short human-readable explanation.
    """

    name: str
    verdict: Verdict
    score: float = 0.0
    threshold_used: float = 0.0
    evidence: List[EvidencePointer] = field(default_factory=list)
    detail: str = ""


@dataclass
class GroundingResult:
    """Final output of :meth:`GroundingVerifier.verify`.

    - ``verdict``: aggregated final verdict.
    - ``confidence``: 0..1, calibrated confidence in the verdict.
    - ``tier_results``: each tier's :class:`TierVerdict`, keyed by name.
    - ``evidence_pointers``: aggregated supporting passages.
    - ``conflict_pointers``: passages that explicitly contradict the
      claim (filled by NLI / LLM-judge tiers).
    - ``reasoning_trace``: per-step trace for explainability.
    - ``merkle_proof``: hex-encoded Merkle root over evidence (P11).
    """

    claim_text: str
    verdict: Verdict
    confidence: float = 0.0
    tier_results: Dict[str, TierVerdict] = field(default_factory=dict)
    evidence_pointers: List[EvidencePointer] = field(default_factory=list)
    conflict_pointers: List[EvidencePointer] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    merkle_proof: Optional[str] = None


__all__ = [
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
