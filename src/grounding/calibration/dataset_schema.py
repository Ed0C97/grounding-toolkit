"""Pydantic schema for the gold-truth calibration dataset.

A consumer-supplied dataset of human-annotated (claim, source, verdict)
triples used by:

- :mod:`grounding.calibration.tuner` — to optimise per-tier thresholds
  and per-tier log-likelihood-ratio weights.
- :mod:`grounding.confidence.uncertainty` — to compute Brier and ECE
  metrics validating the confidence calibration.

Phase 13 ships only the schema + a JSON loader.  Population of the
dataset is a manual annotation exercise (Phase D6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class GoldClaim(BaseModel):
    """The claim under test."""

    text: str
    page: Optional[int] = None
    citation_span: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GoldSource(BaseModel):
    """The source corpus the claim is verified against."""

    text: str = ""
    pages: List[str] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    kv_pairs: Dict[str, str] = Field(default_factory=dict)
    figures: List[Dict[str, Any]] = Field(default_factory=list)
    signatures: List[Dict[str, Any]] = Field(default_factory=list)
    page_count: int = 0
    doc_id: str = "doc"
    language: str = "en"


class GoldRecord(BaseModel):
    """A single gold-truth annotation."""

    record_id: str
    claim: GoldClaim
    source: GoldSource
    label: Literal["GROUNDED", "UNGROUNDED", "UNCERTAIN"]
    annotator: str = ""
    annotation_date: str = ""
    notes: str = ""


class GoldDataset(BaseModel):
    """A bundle of gold annotations + dataset-level metadata."""

    name: str
    version: str = "0.1.0"
    description: str = ""
    locale: str = "en"
    records: List[GoldRecord] = Field(default_factory=list)


def load_dataset(path: Path | str) -> GoldDataset:
    """Load a :class:`GoldDataset` from a JSON file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return GoldDataset.model_validate(payload)


def save_dataset(dataset: GoldDataset, path: Path | str) -> None:
    """Persist a :class:`GoldDataset` to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(
            dataset.model_dump(),
            fh,
            indent=2,
            sort_keys=True,
        )


__all__ = [
    "GoldClaim",
    "GoldDataset",
    "GoldRecord",
    "GoldSource",
    "load_dataset",
    "save_dataset",
]
