"""Document linker — identify which documents a claim references.

A deal / case typically contains multiple linked documents (loan
agreement + bank statements + financial reports + ...).  When a claim
explicitly names another document ("as per the Loan Agreement",
"see Schedule 4"), the linker identifies those references so the
cross-document verifier can fetch evidence from them.

Matching is name-based: the consumer registers each document with a
canonical name plus optional aliases, and the linker uses regex
word-boundary search (case-insensitive) against the claim text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence

from grounding.core.types import Source


@dataclass
class DocumentRef:
    """A named document in the deal corpus."""

    doc_id: str
    name: str
    aliases: List[str] = field(default_factory=list)
    source: Source = field(default_factory=Source)

    def all_names(self) -> List[str]:
        return [self.name] + list(self.aliases)


def _word_boundary_match(needle: str, haystack: str) -> bool:
    if not needle or len(needle.strip()) < 2:
        return False
    pattern = r"\b" + re.escape(needle.strip()) + r"\b"
    return re.search(pattern, haystack, flags=re.IGNORECASE) is not None


@dataclass
class DocumentLinker:
    """Identify referenced documents in a claim via name/alias matching."""

    name: str = "linker"

    def link(
        self, claim_text: str, corpus: Sequence[DocumentRef]
    ) -> List[DocumentRef]:
        if not claim_text:
            return []
        out: List[DocumentRef] = []
        seen: set[str] = set()
        for doc in corpus:
            if doc.doc_id in seen:
                continue
            for n in doc.all_names():
                if _word_boundary_match(n, claim_text):
                    out.append(doc)
                    seen.add(doc.doc_id)
                    break
        return out


__all__ = ["DocumentRef", "DocumentLinker"]
