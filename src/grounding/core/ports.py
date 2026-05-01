"""Protocol definitions for pluggable verification backends.

The toolkit never imports a provider SDK directly.  Every external
capability is expressed as a Protocol so the consumer injects the
implementation.

Protocols:

- :class:`EmbeddingFn`  — text → vector embedding
- :class:`NLIFn`        — entailment / contradiction / neutral scoring
- :class:`LLMJudgeFn`   — LLM-as-judge verdict
- :class:`RetrievalFn`  — passage retrieval (for crossdoc / multi-hop)
- :class:`Tier`         — one tier in the cascade
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, Sequence, runtime_checkable

from grounding.core.types import Claim, Source, TierVerdict


@runtime_checkable
class EmbeddingFn(Protocol):
    """Map a batch of texts to dense vectors.

    Implementations must be deterministic for a given input and return
    vectors of identical dimensionality.
    """

    def __call__(self, texts: Sequence[str]) -> List[List[float]]: ...


@runtime_checkable
class NLIFn(Protocol):
    """Score the entailment relationship between a claim and a source.

    Returns a label-probability mapping with at least the keys
    ``entailment``, ``contradiction``, ``neutral``.  Probabilities should
    sum to ~1.0.
    """

    def __call__(self, *, claim: str, source: str) -> Dict[str, float]: ...


@runtime_checkable
class LLMJudgeFn(Protocol):
    """LLM-as-judge: produce a verdict + rationale for a claim.

    Returns a dict with at least ``verdict`` (str) and ``rationale``
    (str).  Optional keys: ``confidence`` (float in 0..1), ``evidence``
    (list of strings).
    """

    def __call__(self, *, claim: str, source: str) -> Dict[str, Any]: ...


@runtime_checkable
class RetrievalFn(Protocol):
    """Retrieve top-k passages for a query.

    Returns a list of passage dicts.  Each passage should expose at
    least ``text``; recommended keys: ``id``, ``page``, ``score``.
    """

    def __call__(
        self, *, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]: ...


@runtime_checkable
class Tier(Protocol):
    """A single verification tier."""

    name: str

    def verify(
        self, claim: Claim, source: Source, *, threshold: float
    ) -> TierVerdict: ...


__all__ = ["EmbeddingFn", "NLIFn", "LLMJudgeFn", "RetrievalFn", "Tier"]
