"""Helpers for working with :class:`EvidencePointer` instances.

Builders, mergers, and span-extraction utilities.  The
:class:`EvidencePointer` dataclass itself lives in
:mod:`grounding.core.types` so it can be returned by every tier without
a circular import.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from grounding.core.types import EvidencePointer, Source


def build_pointer(
    source: Source,
    *,
    page: Optional[int] = None,
    char_start: int = 0,
    char_end: int = 0,
) -> EvidencePointer:
    """Convenience builder."""
    return EvidencePointer(
        doc_id=source.doc_id,
        page=page,
        char_start=char_start,
        char_end=char_end,
    )


def extract_text(
    source: Source, pointer: EvidencePointer
) -> str:
    """Return the source text covered by ``pointer``.

    Falls back to ``source.text`` when the pointer carries no page,
    or when per-page texts are not populated.
    """
    if pointer.page is not None:
        page_text = source.page_text(pointer.page)
        if page_text is not None:
            return page_text[
                max(0, pointer.char_start) : max(0, pointer.char_end)
            ]
    return source.text[
        max(0, pointer.char_start) : max(0, pointer.char_end)
    ]


def merge_pointers(
    pointers: Iterable[EvidencePointer],
) -> List[EvidencePointer]:
    """Merge overlapping or adjacent pointers per (doc_id, page).

    Two pointers are merged when they share doc_id + page and their
    char ranges touch or overlap.  Output is sorted by
    (doc_id, page, char_start).
    """
    bucket: dict = {}
    for p in pointers:
        key = (p.doc_id, p.page)
        bucket.setdefault(key, []).append(p)

    out: List[EvidencePointer] = []
    for (doc_id, page), group in bucket.items():
        group.sort(key=lambda x: (x.char_start, x.char_end))
        merged_start = group[0].char_start
        merged_end = group[0].char_end
        for p in group[1:]:
            if p.char_start <= merged_end:
                merged_end = max(merged_end, p.char_end)
            else:
                out.append(
                    EvidencePointer(
                        doc_id=doc_id,
                        page=page,
                        char_start=merged_start,
                        char_end=merged_end,
                    )
                )
                merged_start = p.char_start
                merged_end = p.char_end
        out.append(
            EvidencePointer(
                doc_id=doc_id,
                page=page,
                char_start=merged_start,
                char_end=merged_end,
            )
        )
    out.sort(
        key=lambda p: (
            p.doc_id,
            (p.page if p.page is not None else -1),
            p.char_start,
        )
    )
    return out


def serialise_pointer(pointer: EvidencePointer) -> dict:
    return {
        "doc_id": pointer.doc_id,
        "page": pointer.page,
        "char_start": pointer.char_start,
        "char_end": pointer.char_end,
    }


__all__ = [
    "build_pointer",
    "extract_text",
    "merge_pointers",
    "serialise_pointer",
]
