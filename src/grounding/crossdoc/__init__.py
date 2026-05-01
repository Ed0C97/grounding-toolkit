"""grounding.crossdoc — multi-document grounding."""

from __future__ import annotations

from grounding.crossdoc.linker import DocumentLinker, DocumentRef
from grounding.crossdoc.retriever import CrossDocVerifier

__all__ = ["CrossDocVerifier", "DocumentLinker", "DocumentRef"]
