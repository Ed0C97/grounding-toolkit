"""Merkle proof over evidence pointers.

Computes a deterministic Merkle root over a list of
:class:`grounding.core.types.EvidencePointer` instances (or any list of
hashable string representations).  Used by the consumer to publish a
tamper-evident proof of which evidence supported which claim — the root
goes in the dossier / report, while the leaf hashes (with their order)
are stored for later verification.

The hash function is SHA-256.  Pairs are concatenated as
``left || right`` (raw bytes) before hashing.  Odd numbers of leaves
are handled by duplicating the last element (Bitcoin-style).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from grounding.core.types import EvidencePointer


def _serialise_pointer(p: EvidencePointer) -> str:
    return json.dumps(
        {
            "doc_id": p.doc_id,
            "page": p.page,
            "char_start": p.char_start,
            "char_end": p.char_end,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _h(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def leaf_hash(payload: str) -> str:
    """Hex SHA-256 of a leaf payload."""
    return _h(payload.encode("utf-8")).hex()


def merkle_root(payloads: Sequence[str]) -> str:
    """Compute the hex SHA-256 Merkle root over a list of leaf payloads.

    Returns ``""`` for an empty list (well-defined no-op).
    """
    if not payloads:
        return ""
    level: List[bytes] = [_h(p.encode("utf-8")) for p in payloads]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [
            _h(level[i] + level[i + 1])
            for i in range(0, len(level), 2)
        ]
    return level[0].hex()


def merkle_root_for_evidence(
    pointers: Iterable[EvidencePointer],
) -> str:
    """Convenience wrapper: serialise + sort + Merkle-root."""
    serialised = sorted(_serialise_pointer(p) for p in pointers)
    return merkle_root(serialised)


@dataclass
class MerkleProof:
    """Bundled root + ordered leaf hashes + serialised payloads.

    The consumer stores the ``root`` in their dossier / public log; the
    ``leaves`` (with their original payloads) are kept for later
    verification.
    """

    root: str
    leaves: List[str] = field(default_factory=list)
    payloads: List[str] = field(default_factory=list)

    def verify(self) -> bool:
        """Recompute the Merkle root from payloads and compare to root."""
        if not self.payloads:
            return self.root == ""
        return merkle_root(self.payloads) == self.root


def build_merkle_proof(
    pointers: Iterable[EvidencePointer],
) -> MerkleProof:
    """Build a complete :class:`MerkleProof` over evidence pointers."""
    serialised = sorted(_serialise_pointer(p) for p in pointers)
    leaves = [leaf_hash(p) for p in serialised]
    root = merkle_root(serialised)
    return MerkleProof(
        root=root,
        leaves=leaves,
        payloads=serialised,
    )


def verify_proof(proof: MerkleProof) -> bool:
    return proof.verify()


__all__ = [
    "MerkleProof",
    "build_merkle_proof",
    "leaf_hash",
    "merkle_root",
    "merkle_root_for_evidence",
    "verify_proof",
]
