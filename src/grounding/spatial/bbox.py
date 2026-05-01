"""Block-level bbox grounding helpers.

Migrated from Sentinel's
``sentinel.utils.bbox_grounding`` (P16 hard cutover, partial — only
the **pure-algorithm** portion lives here; the Sentinel-side
orchestrator that wires ``pdf_finder``, OCR factory and project
settings stays in Sentinel as a thin wrapper).

Pure helpers:

- :func:`normalise` — collapse whitespace + lowercase.
- :func:`block_score` — score how well a layout block matches a
  clause text (3-strategy max).
- :func:`merge_bboxes` — encompassing bbox over a list of [x0,y0,x1,y1].
- :func:`find_best_bbox_on_page` — best-scoring layout block on a page,
  ignoring page-header / page-footer noise zones.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple


# Default minimum similarity ratio to accept a block-level match.
DEFAULT_MIN_RATIO: float = 0.25
# Default anchor length for partial substring matching.
DEFAULT_ANCHOR_LEN: int = 80

# Layout categories that represent repeated page noise — never use
# them as grounding targets.
DEFAULT_NOISE_CATEGORIES: frozenset[str] = frozenset(
    {"Page-header", "Page-footer"}
)


def normalise(text: str) -> str:
    """Collapse whitespace and lowercase for comparison."""
    return re.sub(r"\s+", " ", text.strip().lower())


def block_score(
    clause_norm: str,
    block_text: str,
    anchor: str,
) -> float:
    """Score how well a layout block matches a clause text.

    Three strategies; return the max:

    1. Full :class:`difflib.SequenceMatcher` ratio (capped at 500 chars
       both sides) — best for short clauses.
    2. Anchor substring match — first N chars of clause appear in block.
    3. Block-as-substring-of-clause — block is part of a longer clause
       (multi-block scenario).
    """
    block_norm = normalise(block_text)
    if not block_norm or len(block_norm) < 10:
        return 0.0

    scores: List[float] = []

    clause_capped = clause_norm[:500]
    block_capped = block_norm[:500]
    ratio = SequenceMatcher(None, clause_capped, block_capped).ratio()
    scores.append(ratio)

    if anchor and anchor in block_norm:
        scores.append(0.85)

    if len(block_norm) > 20 and block_norm in clause_norm:
        coverage = len(block_norm) / max(len(clause_norm), 1)
        scores.append(0.5 + coverage * 0.4)

    return max(scores) if scores else 0.0


def merge_bboxes(
    bboxes: List[List[float]],
) -> List[float]:
    """Return the encompassing bbox over a list of ``[x0, y0, x1, y1]``."""
    if not bboxes:
        return [0.0, 0.0, 0.0, 0.0]
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return [x0, y0, x1, y1]


def find_best_bbox_on_page(
    clause_norm: str,
    anchor: str,
    page_blocks: List[dict],
    *,
    min_ratio: float = DEFAULT_MIN_RATIO,
    noise_categories: frozenset[str] = DEFAULT_NOISE_CATEGORIES,
) -> Tuple[float, Optional[List[float]]]:
    """Find the best-scoring layout block on a single page.

    Skips blocks tagged as page-header / page-footer noise.  When the
    clause is long (> 200 chars) and multiple adjacent blocks score
    above ``min_ratio``, returns the merged bbox over the top
    contiguous candidates.

    Returns ``(score, bbox)`` or ``(0.0, None)``.
    """
    if not page_blocks:
        return 0.0, None

    scored: List[Tuple[float, int, dict]] = []
    for idx, block in enumerate(page_blocks):
        if not block.get("bbox") or len(block.get("bbox", [])) < 4:
            continue
        text = block.get("text", "")
        if not text or len(str(text).strip()) < 5:
            continue
        cat = (block.get("category") or "").strip()
        if cat in noise_categories:
            continue
        score = block_score(clause_norm, str(text), anchor)
        if score > min_ratio:
            scored.append((score, idx, block))

    if not scored:
        return 0.0, None

    scored.sort(key=lambda x: -x[0])
    best_score, best_idx, best_block = scored[0]

    # Multi-block merge for long clauses.
    if len(clause_norm) > 200 and len(scored) >= 2:
        merge_candidates = [best_block["bbox"]]
        for sc, idx, blk in scored[1:4]:
            if sc > min_ratio and abs(idx - best_idx) <= 3:
                merge_candidates.append(blk["bbox"])
        if len(merge_candidates) > 1:
            return best_score, merge_bboxes(merge_candidates)

    return best_score, list(best_block["bbox"])


__all__ = [
    "DEFAULT_ANCHOR_LEN",
    "DEFAULT_MIN_RATIO",
    "DEFAULT_NOISE_CATEGORIES",
    "block_score",
    "find_best_bbox_on_page",
    "merge_bboxes",
    "normalise",
]
