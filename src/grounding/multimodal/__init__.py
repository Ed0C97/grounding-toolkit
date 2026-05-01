"""grounding.multimodal — verify against tables / KV pairs / figures / signatures."""

from __future__ import annotations

from grounding.multimodal.figures import CaptionFn, FigureVerifier
from grounding.multimodal.kv import KVVerifier
from grounding.multimodal.signatures import SignatureVerifier
from grounding.multimodal.tables import TableVerifier

__all__ = [
    "CaptionFn",
    "FigureVerifier",
    "KVVerifier",
    "SignatureVerifier",
    "TableVerifier",
]
