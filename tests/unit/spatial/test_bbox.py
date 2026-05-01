"""Tests for grounding.spatial.bbox."""

from __future__ import annotations

from grounding.spatial.bbox import (
    block_score,
    find_best_bbox_on_page,
    merge_bboxes,
    normalise,
)


def test_normalise_lowercases_and_collapses() -> None:
    assert normalise("  Hello   WORLD  ") == "hello world"


def test_normalise_empty() -> None:
    assert normalise("") == ""


def test_block_score_short_block_returns_zero() -> None:
    assert block_score("a normalized clause", "x", "anchor") == 0.0


def test_block_score_substring_match_high() -> None:
    clause = normalise("the rate per annum is 4.5%")
    block = "Section 4: the rate per annum is 4.5% per annum."
    score = block_score(clause, block, clause[:80])
    assert score >= 0.5


def test_block_score_unrelated_low() -> None:
    score = block_score(
        normalise("EBITDA grew strongly"),
        "Completely unrelated content about other topics that exceeds ten chars",
        "ebitda grew strongly",
    )
    assert score < 0.5


def test_merge_bboxes_empty_returns_zeros() -> None:
    assert merge_bboxes([]) == [0.0, 0.0, 0.0, 0.0]


def test_merge_bboxes_encompassing() -> None:
    out = merge_bboxes([[0, 0, 5, 5], [3, 3, 10, 10]])
    assert out == [0, 0, 10, 10]


def test_find_best_bbox_skips_noise_zones() -> None:
    blocks = [
        {
            "bbox": [0, 0, 100, 10],
            "text": "this is a header repeated on every page for navigation",
            "category": "Page-header",
        },
        {
            "bbox": [0, 50, 100, 100],
            "text": "the rate per annum is 4.5% per annum",
            "category": "Paragraph",
        },
    ]
    score, bbox = find_best_bbox_on_page(
        normalise("the rate per annum is 4.5%"),
        "the rate per annum is 4.5%",
        blocks,
    )
    assert bbox == [0, 50, 100, 100]
    assert score > 0


def test_find_best_bbox_no_blocks() -> None:
    score, bbox = find_best_bbox_on_page("x", "anchor", [])
    assert score == 0.0
    assert bbox is None


def test_find_best_bbox_below_min_ratio() -> None:
    blocks = [
        {
            "bbox": [0, 0, 10, 10],
            "text": "completely unrelated content here",
            "category": "Paragraph",
        }
    ]
    score, bbox = find_best_bbox_on_page(
        normalise("alpha beta gamma delta"),
        "alpha",
        blocks,
        min_ratio=0.95,
    )
    assert bbox is None


def test_find_best_bbox_skips_invalid_bbox() -> None:
    blocks = [
        {"bbox": None, "text": "valid text"},
        {
            "bbox": [0, 50, 100, 100],
            "text": "alpha beta gamma delta epsilon",
            "category": "Paragraph",
        },
    ]
    score, bbox = find_best_bbox_on_page(
        normalise("alpha beta gamma delta epsilon"),
        "alpha beta gamma",
        blocks,
    )
    assert bbox == [0, 50, 100, 100]
