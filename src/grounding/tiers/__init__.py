"""grounding.tiers — verification tiers."""

from __future__ import annotations

from grounding.tiers.consensus import ConsensusTier
from grounding.tiers.lexical import LexicalTier, compute_text_overlap

__all__ = ["ConsensusTier", "LexicalTier", "compute_text_overlap"]
