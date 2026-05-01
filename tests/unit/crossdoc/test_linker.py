"""Tests for grounding.crossdoc.linker.DocumentLinker."""

from __future__ import annotations

from grounding import Source
from grounding.crossdoc.linker import DocumentLinker, DocumentRef


def test_links_via_name() -> None:
    linker = DocumentLinker()
    corpus = [
        DocumentRef(doc_id="d1", name="Loan Agreement"),
        DocumentRef(doc_id="d2", name="Information Memorandum"),
    ]
    out = linker.link("As per the Loan Agreement, ...", corpus)
    assert [d.doc_id for d in out] == ["d1"]


def test_links_via_alias() -> None:
    linker = DocumentLinker()
    corpus = [
        DocumentRef(
            doc_id="d1",
            name="Loan Agreement",
            aliases=["LA", "Master Loan"],
        ),
    ]
    out = linker.link("see Master Loan section 4", corpus)
    assert [d.doc_id for d in out] == ["d1"]


def test_no_link_when_name_absent() -> None:
    linker = DocumentLinker()
    corpus = [DocumentRef(doc_id="d1", name="Loan Agreement")]
    out = linker.link("nothing referencing any doc here", corpus)
    assert out == []


def test_returns_unique_docs_only() -> None:
    """A doc with multiple aliases that all match returns once."""
    linker = DocumentLinker()
    corpus = [
        DocumentRef(
            doc_id="d1",
            name="Loan Agreement",
            aliases=["LA", "Loan"],
        ),
    ]
    out = linker.link("Loan Agreement and LA and Loan", corpus)
    assert [d.doc_id for d in out] == ["d1"]


def test_word_boundary_avoids_false_positive() -> None:
    """An alias 'LA' should not match 'island' (substring)."""
    linker = DocumentLinker()
    corpus = [DocumentRef(doc_id="d1", name="LA")]
    out = linker.link("the islands and the airplane", corpus)
    assert out == []


def test_case_insensitive_match() -> None:
    linker = DocumentLinker()
    corpus = [DocumentRef(doc_id="d1", name="Loan Agreement")]
    out = linker.link("LOAN AGREEMENT terms", corpus)
    assert [d.doc_id for d in out] == ["d1"]


def test_multiple_docs_linked() -> None:
    linker = DocumentLinker()
    corpus = [
        DocumentRef(doc_id="d1", name="Loan Agreement"),
        DocumentRef(doc_id="d2", name="Schedule 4"),
    ]
    out = linker.link("see Loan Agreement and Schedule 4", corpus)
    ids = sorted(d.doc_id for d in out)
    assert ids == ["d1", "d2"]


def test_short_alias_filtered() -> None:
    """Aliases shorter than 2 chars are ignored."""
    linker = DocumentLinker()
    corpus = [DocumentRef(doc_id="d1", name="X", aliases=["a"])]
    out = linker.link("the alpha and the omega", corpus)
    assert out == []


def test_empty_claim_returns_empty() -> None:
    linker = DocumentLinker()
    corpus = [DocumentRef(doc_id="d1", name="X")]
    assert linker.link("", corpus) == []


def test_all_names_includes_aliases() -> None:
    d = DocumentRef(doc_id="d", name="Main", aliases=["alias1", "alias2"])
    assert d.all_names() == ["Main", "alias1", "alias2"]
