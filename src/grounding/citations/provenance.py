"""Provenance DAG for tracing how claims were derived.

Every claim, in the SOTA architecture, has a deterministic provenance
chain that records:

- The originating ``generator`` (agent name, model id).
- The ``parent_ids`` of upstream artefacts that fed into the claim
  (extracted clauses, retrieved passages, prior derivations).
- The optional ``citation_span`` emitted preventively by the LLM.
- The optional ``confidence`` reported alongside the claim.
- A free-form ``metadata`` bag.

The DAG is content-addressed (``claim_id`` is a SHA-256 prefix of the
claim text) so the same claim added twice is idempotent and the JSON
serialisation is canonical — both properties are needed by the Merkle
proof in :mod:`grounding.audit.merkle_proof`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from grounding.core.types import CitationSpan


def claim_id(text: str, *, namespace: str = "grounding") -> str:
    """Deterministic 16-char content-addressed id for a claim text."""
    h = hashlib.sha256()
    h.update(namespace.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


@dataclass
class ProvenanceNode:
    """A single node in the provenance DAG."""

    claim_id: str
    text: str
    generator: str = ""
    parent_ids: List[str] = field(default_factory=list)
    citation_span: Optional[CitationSpan] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvenanceDAG:
    """An append-only DAG of provenance nodes.

    Idempotent: adding the same node twice (by ``claim_id``) is a
    no-op.
    """

    nodes: Dict[str, ProvenanceNode] = field(default_factory=dict)

    def add(self, node: ProvenanceNode) -> str:
        if node.claim_id not in self.nodes:
            self.nodes[node.claim_id] = node
        return node.claim_id

    def add_claim(
        self,
        text: str,
        *,
        generator: str = "",
        parent_ids: Optional[List[str]] = None,
        citation_span: Optional[CitationSpan] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Convenience: build a node from primitives + add."""
        cid = claim_id(text)
        node = ProvenanceNode(
            claim_id=cid,
            text=text,
            generator=generator,
            parent_ids=list(parent_ids or []),
            citation_span=citation_span,
            confidence=confidence,
            metadata=dict(metadata or {}),
        )
        return self.add(node)

    def get(self, cid: str) -> Optional[ProvenanceNode]:
        return self.nodes.get(cid)

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, cid: object) -> bool:
        return isinstance(cid, str) and cid in self.nodes

    def ancestors(self, cid: str) -> List[str]:
        """Return all ancestor ids reachable from ``cid`` (BFS)."""
        seen: set[str] = set()
        out: List[str] = []
        stack: List[str] = [cid]
        while stack:
            current = stack.pop()
            node = self.nodes.get(current)
            if node is None:
                continue
            for pid in node.parent_ids:
                if pid in seen:
                    continue
                seen.add(pid)
                out.append(pid)
                stack.append(pid)
        return out

    def to_json(self) -> str:
        """Serialise the DAG to canonical JSON.

        Deterministic ordering: nodes by ascending ``claim_id``,
        ``parent_ids`` sorted within each node.  Used by
        :mod:`grounding.audit.merkle_proof`.
        """
        items: List[Dict[str, Any]] = []
        for cid in sorted(self.nodes.keys()):
            n = self.nodes[cid]
            items.append(
                {
                    "claim_id": n.claim_id,
                    "text": n.text,
                    "generator": n.generator,
                    "parent_ids": sorted(n.parent_ids),
                    "citation_span": (
                        {
                            "page": n.citation_span.page,
                            "char_start": n.citation_span.char_start,
                            "char_end": n.citation_span.char_end,
                        }
                        if n.citation_span is not None
                        else None
                    ),
                    "confidence": n.confidence,
                    "metadata": n.metadata,
                }
            )
        return json.dumps(items, sort_keys=True, separators=(",", ":"))


__all__ = ["claim_id", "ProvenanceNode", "ProvenanceDAG"]
