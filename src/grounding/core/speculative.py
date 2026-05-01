"""Speculative pre-screen using LLM-emitted citation spans.

When a claim arrives with a :class:`CitationSpan`, the cascade can
verify it deterministically and short-circuit the rest of the pipeline
when the span match succeeds — the strongest, cheapest groundedness
signal available.

This module is the integration point: it wraps
:class:`grounding.citations.span.SpanVerifier` so the cascade has a
single ``Optional[TierVerdict]`` API for the pre-screen step.
"""

from __future__ import annotations

from typing import Optional

from grounding.citations.span import SpanVerifier
from grounding.core.types import Claim, Source, TierVerdict


def speculative_prescreen(
    claim: Claim,
    source: Source,
    *,
    verifier: Optional[SpanVerifier] = None,
) -> Optional[TierVerdict]:
    """Run speculative pre-screen on a claim.

    Returns:
        - ``TierVerdict`` when the claim has a citation_span (the
          cascade should short-circuit on GROUNDED).
        - ``None`` when no citation_span is available (cascade proceeds
          through the regular tiers).
    """
    if claim.citation_span is None:
        return None
    v = verifier or SpanVerifier()
    return v.verify(claim, source)


__all__ = ["speculative_prescreen"]
