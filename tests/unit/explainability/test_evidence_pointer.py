"""Tests for grounding.explainability.evidence_pointer."""

from __future__ import annotations

from grounding import EvidencePointer, Source
from grounding.explainability.evidence_pointer import (
    build_pointer,
    extract_text,
    merge_pointers,
    serialise_pointer,
)


def test_build_pointer_uses_doc_id() -> None:
    src = Source.from_text("text", doc_id="abc")
    p = build_pointer(src, page=1, char_start=0, char_end=4)
    assert p.doc_id == "abc"
    assert p.char_start == 0
    assert p.char_end == 4


def test_extract_text_uses_pages_when_available() -> None:
    src = Source.from_pages(["page one", "page two"])
    p = EvidencePointer(doc_id="d", page=2, char_start=0, char_end=4)
    assert extract_text(src, p) == "page"


def test_extract_text_falls_back_to_full_text() -> None:
    src = Source.from_text("hello world")
    p = EvidencePointer(doc_id="d", page=None, char_start=6, char_end=11)
    assert extract_text(src, p) == "world"


def test_merge_pointers_combines_overlapping() -> None:
    a = EvidencePointer("d", 1, 0, 10)
    b = EvidencePointer("d", 1, 5, 15)
    out = merge_pointers([a, b])
    assert len(out) == 1
    assert out[0].char_start == 0
    assert out[0].char_end == 15


def test_merge_pointers_keeps_separate_when_disjoint() -> None:
    a = EvidencePointer("d", 1, 0, 5)
    b = EvidencePointer("d", 1, 10, 15)
    out = merge_pointers([a, b])
    assert len(out) == 2


def test_merge_pointers_separates_by_doc_and_page() -> None:
    a = EvidencePointer("d1", 1, 0, 5)
    b = EvidencePointer("d2", 1, 0, 5)
    c = EvidencePointer("d1", 2, 0, 5)
    out = merge_pointers([a, b, c])
    assert len(out) == 3


def test_merge_pointers_sorts_output() -> None:
    a = EvidencePointer("d2", 1, 0, 5)
    b = EvidencePointer("d1", 1, 0, 5)
    out = merge_pointers([a, b])
    assert [p.doc_id for p in out] == ["d1", "d2"]


def test_serialise_pointer() -> None:
    p = EvidencePointer("d", 1, 0, 5)
    out = serialise_pointer(p)
    assert out == {"doc_id": "d", "page": 1, "char_start": 0, "char_end": 5}
