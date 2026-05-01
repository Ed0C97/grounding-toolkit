"""grounding.audit — Merkle proof + immutable reasoning log."""

from __future__ import annotations

from grounding.audit.merkle_proof import (
    MerkleProof,
    build_merkle_proof,
    leaf_hash,
    merkle_root,
    merkle_root_for_evidence,
    verify_proof,
)
from grounding.audit.reasoning_log import (
    ReasoningLog,
    ReasoningLogRecord,
    record_for_result,
)

__all__ = [
    "MerkleProof",
    "ReasoningLog",
    "ReasoningLogRecord",
    "build_merkle_proof",
    "leaf_hash",
    "merkle_root",
    "merkle_root_for_evidence",
    "record_for_result",
    "verify_proof",
]
