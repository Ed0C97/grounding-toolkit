"""grounding.spatial — bbox grounding helpers (pure-algorithm portion)."""

from __future__ import annotations

from grounding.spatial.bbox import (
    DEFAULT_ANCHOR_LEN,
    DEFAULT_MIN_RATIO,
    DEFAULT_NOISE_CATEGORIES,
    block_score,
    find_best_bbox_on_page,
    merge_bboxes,
    normalise,
)

__all__ = [
    "DEFAULT_ANCHOR_LEN",
    "DEFAULT_MIN_RATIO",
    "DEFAULT_NOISE_CATEGORIES",
    "block_score",
    "find_best_bbox_on_page",
    "merge_bboxes",
    "normalise",
]
