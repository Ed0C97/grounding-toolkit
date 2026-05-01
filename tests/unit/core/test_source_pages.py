"""Tests for the new Source.pages / Source.page_text / Source.from_pages
features added in P3."""

from __future__ import annotations

from grounding import Source


def test_from_pages_populates_text_and_count() -> None:
    s = Source.from_pages(["page1 text", "page2 text"])
    assert s.page_count == 2
    assert "page1 text" in s.text
    assert "page2 text" in s.text


def test_page_text_one_indexed() -> None:
    s = Source.from_pages(["a", "b", "c"])
    assert s.page_text(1) == "a"
    assert s.page_text(2) == "b"
    assert s.page_text(3) == "c"


def test_page_text_out_of_range_returns_none() -> None:
    s = Source.from_pages(["a"])
    assert s.page_text(0) is None
    assert s.page_text(99) is None


def test_page_text_falls_back_to_text_when_no_pages() -> None:
    s = Source.from_text("just text")
    assert s.page_text(1) == "just text"
    assert s.page_text(2) is None


def test_page_text_empty_text_no_pages() -> None:
    s = Source.from_text("")
    assert s.page_text(1) is None


def test_page_count_auto_derived_from_pages() -> None:
    s = Source(pages=["p1", "p2", "p3"])
    assert s.page_count == 3


def test_page_count_explicit_overrides_default() -> None:
    """If consumer passes page_count explicitly, do not override."""
    s = Source(pages=["p1"], page_count=5)
    assert s.page_count == 5
