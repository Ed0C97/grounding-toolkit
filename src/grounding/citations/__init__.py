"""grounding.citations — preventive grounding via citation spans."""

from __future__ import annotations

from grounding.citations.provenance import (
    ProvenanceDAG,
    ProvenanceNode,
    claim_id,
)
from grounding.citations.span import SpanVerifier
from grounding.citations.structured_signature import (
    GroundedClaim,
    GroundedClaimSignature,
    GroundedClaimSpan,
    GroundedFindings,
    grounded_claim_system_prompt,
)

__all__ = [
    "SpanVerifier",
    "ProvenanceDAG",
    "ProvenanceNode",
    "claim_id",
    "GroundedClaim",
    "GroundedClaimSignature",
    "GroundedClaimSpan",
    "GroundedFindings",
    "grounded_claim_system_prompt",
]
