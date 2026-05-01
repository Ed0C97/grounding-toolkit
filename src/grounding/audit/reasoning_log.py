"""Append-only reasoning log.

A stream of structured records describing every grounding verification
event: input claim, source ref, tier verdicts, final verdict,
confidence, evidence, Merkle root.  Designed for downstream archival,
audit trails, and post-hoc calibration.

The log is append-only (no mutation, no deletion) and JSON-serialisable.
Consumers can stream records to a file, a queue, or a database.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from grounding.audit.merkle_proof import build_merkle_proof
from grounding.core.types import GroundingResult


@dataclass(frozen=True)
class ReasoningLogRecord:
    """A single entry in the reasoning log."""

    timestamp: float
    record_id: str
    claim_text: str
    source_doc_id: str
    final_verdict: str
    confidence: float
    tier_verdicts: Dict[str, str] = field(default_factory=dict)
    evidence_count: int = 0
    merkle_root: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "timestamp": self.timestamp,
                "record_id": self.record_id,
                "claim_text": self.claim_text,
                "source_doc_id": self.source_doc_id,
                "final_verdict": self.final_verdict,
                "confidence": self.confidence,
                "tier_verdicts": dict(self.tier_verdicts),
                "evidence_count": self.evidence_count,
                "merkle_root": self.merkle_root,
                "metadata": dict(self.metadata),
            },
            sort_keys=True,
            separators=(",", ":"),
        )


def _make_record_id(claim_text: str, source_doc_id: str, ts: float) -> str:
    h = hashlib.sha256()
    h.update(b"reasoning-log\x00")
    h.update(claim_text.encode("utf-8"))
    h.update(b"\x00")
    h.update(source_doc_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(f"{ts:.9f}".encode("utf-8"))
    return h.hexdigest()[:16]


def record_for_result(
    result: GroundingResult,
    *,
    source_doc_id: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    timestamp: Optional[float] = None,
) -> ReasoningLogRecord:
    """Build a :class:`ReasoningLogRecord` from a :class:`GroundingResult`.

    Includes a Merkle proof root over the result's evidence pointers.
    """
    ts = timestamp if timestamp is not None else time.time()
    proof = build_merkle_proof(result.evidence_pointers)
    return ReasoningLogRecord(
        timestamp=ts,
        record_id=_make_record_id(
            result.claim_text, source_doc_id, ts
        ),
        claim_text=result.claim_text,
        source_doc_id=source_doc_id,
        final_verdict=result.verdict.value,
        confidence=result.confidence,
        tier_verdicts={
            name: tv.verdict.value
            for name, tv in result.tier_results.items()
        },
        evidence_count=len(result.evidence_pointers),
        merkle_root=proof.root,
        metadata=dict(metadata or {}),
    )


@dataclass
class ReasoningLog:
    """In-memory append-only log.

    For production use the consumer wires this to a persistent sink
    (file, queue, database).  This in-memory implementation suffices
    for unit tests and small jobs.
    """

    records: List[ReasoningLogRecord] = field(default_factory=list)

    def append(self, record: ReasoningLogRecord) -> None:
        self.records.append(record)

    def append_result(
        self,
        result: GroundingResult,
        *,
        source_doc_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReasoningLogRecord:
        record = record_for_result(
            result, source_doc_id=source_doc_id, metadata=metadata
        )
        self.records.append(record)
        return record

    def __len__(self) -> int:
        return len(self.records)

    def to_json(self) -> str:
        return "[" + ",".join(r.to_json() for r in self.records) + "]"


__all__ = [
    "ReasoningLog",
    "ReasoningLogRecord",
    "record_for_result",
]
