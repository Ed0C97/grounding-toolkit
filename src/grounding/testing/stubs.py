"""grounding.testing.stubs — in-process Stub<Backend> implementations.

Every Stub conforms to its associated grounding Protocol so consumers
can exercise grounding-side logic in unit tests without spinning up a
real LLM, embedding model, or retrieval backend.

Example::

    from grounding.testing import StubEmbeddingFn
    from grounding.tiers.semantic import SemanticTier

    tier = SemanticTier(embedding_fn=StubEmbeddingFn(dim=8))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class StubEmbeddingFn:
    """In-memory EmbeddingFn stub.

    Returns a deterministic vector based on character codepoints so two
    runs over the same text are bit-identical.
    """

    dim: int = 8
    calls: List[str] = field(default_factory=list)

    def __call__(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            self.calls.append(t)
            v = [0.0] * self.dim
            for i, ch in enumerate(t.encode("utf-8")):
                v[i % self.dim] += float(ch) / 255.0
            out.append(v)
        return out


@dataclass
class StubNLIFn:
    """In-memory NLIFn stub. Returns "entailment" if claim is a substring
    of source, "contradiction" if explicitly tagged, else "neutral".
    """

    contradictions: List[str] = field(default_factory=list)
    calls: List[Dict[str, str]] = field(default_factory=list)

    def __call__(self, *, claim: str, source: str) -> Dict[str, float]:
        self.calls.append({"claim": claim, "source": source})
        for c in self.contradictions:
            if c in claim:
                return {"entailment": 0.05, "contradiction": 0.9, "neutral": 0.05}
        if claim and claim in source:
            return {"entailment": 0.95, "contradiction": 0.02, "neutral": 0.03}
        return {"entailment": 0.30, "contradiction": 0.10, "neutral": 0.60}


@dataclass
class StubLLMJudgeFn:
    """In-memory LLMJudgeFn stub.

    Returns ``{"verdict": "GROUNDED" | "UNGROUNDED", "rationale": "..."}``
    purely based on whether the claim text appears in the source. Used
    only in unit tests of the cascade orchestrator.
    """

    calls: List[Dict[str, str]] = field(default_factory=list)

    def __call__(self, *, claim: str, source: str) -> Dict[str, Any]:
        self.calls.append({"claim": claim, "source": source})
        if claim and claim in source:
            return {"verdict": "GROUNDED", "rationale": "exact substring match"}
        return {"verdict": "UNGROUNDED", "rationale": "claim not found in source"}


@dataclass
class StubRetrievalFn:
    """In-memory RetrievalFn stub. Returns pre-seeded passages."""

    passages: List[Dict[str, Any]] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)

    def __call__(
        self, *, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        self.calls.append(query)
        return list(self.passages[:top_k])
