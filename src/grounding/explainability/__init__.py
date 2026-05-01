"""grounding.explainability — evidence pointers + conflicts + reasoning trace."""

from __future__ import annotations

from grounding.explainability.conflict import ConflictDetector
from grounding.explainability.evidence_pointer import (
    build_pointer,
    extract_text,
    merge_pointers,
    serialise_pointer,
)
from grounding.explainability.reasoning_trace import (
    ReasoningTrace,
    TraceStep,
)

__all__ = [
    "ConflictDetector",
    "ReasoningTrace",
    "TraceStep",
    "build_pointer",
    "extract_text",
    "merge_pointers",
    "serialise_pointer",
]
