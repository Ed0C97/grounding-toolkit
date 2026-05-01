"""Online feedback loop: capture analyst overrides, build new gold
records, append to a :class:`GoldDataset` for the next tuning cycle.

Workflow:

1. The cascade emits a :class:`GroundingResult` for a claim.
2. A human analyst reviews the result and either confirms or overrides
   the verdict in the consumer's UI.
3. The consumer calls :func:`record_feedback` with the override.
4. Feedback accumulates in an in-memory or persisted
   :class:`FeedbackBuffer` until the next tuning cycle.
5. :func:`promote_to_dataset` converts buffered feedback into
   :class:`GoldRecord` instances appended to the active gold dataset.

The toolkit ships the in-memory buffer.  Persisted backends (file,
queue, database) are the consumer's responsibility — this module
focuses on the data-shape contract.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from grounding.calibration.dataset_schema import (
    GoldClaim,
    GoldDataset,
    GoldRecord,
    GoldSource,
)
from grounding.core.types import GroundingResult, Verdict


@dataclass
class FeedbackEvent:
    """A single analyst override captured at runtime."""

    timestamp: float
    record_id: str
    claim_text: str
    source_doc_id: str
    predicted_verdict: Verdict
    analyst_label: str  # "GROUNDED" | "UNGROUNDED" | "UNCERTAIN"
    rationale: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class FeedbackBuffer:
    """Append-only buffer of :class:`FeedbackEvent`."""

    events: List[FeedbackEvent] = field(default_factory=list)

    def append(self, event: FeedbackEvent) -> None:
        self.events.append(event)

    def __len__(self) -> int:
        return len(self.events)


def record_feedback(
    *,
    result: GroundingResult,
    record_id: str,
    analyst_label: str,
    source_doc_id: str = "",
    rationale: str = "",
    timestamp: Optional[float] = None,
    buffer: Optional[FeedbackBuffer] = None,
    metadata: Optional[dict] = None,
) -> FeedbackEvent:
    """Capture an analyst override.

    If ``buffer`` is provided, the event is appended to it.
    """
    label = analyst_label.upper().strip()
    if label not in {"GROUNDED", "UNGROUNDED", "UNCERTAIN"}:
        raise ValueError(
            f"unsupported analyst_label {analyst_label!r}; "
            "expected GROUNDED / UNGROUNDED / UNCERTAIN"
        )
    event = FeedbackEvent(
        timestamp=timestamp if timestamp is not None else time.time(),
        record_id=record_id,
        claim_text=result.claim_text,
        source_doc_id=source_doc_id,
        predicted_verdict=result.verdict,
        analyst_label=label,
        rationale=rationale,
        metadata=dict(metadata or {}),
    )
    if buffer is not None:
        buffer.append(event)
    return event


def promote_to_dataset(
    buffer: FeedbackBuffer,
    dataset: GoldDataset,
    *,
    source_lookup=None,
) -> int:
    """Convert buffered events into :class:`GoldRecord` instances and
    append to ``dataset``.

    ``source_lookup`` is a consumer-supplied callable
    ``(source_doc_id) -> Optional[GoldSource]``.  When None, we synthesise
    a minimal :class:`GoldSource` with only ``doc_id``.

    Returns the number of records promoted.
    """
    n_promoted = 0
    for event in buffer.events:
        if source_lookup is not None:
            src = source_lookup(event.source_doc_id) or GoldSource(
                doc_id=event.source_doc_id
            )
        else:
            src = GoldSource(doc_id=event.source_doc_id)
        record = GoldRecord(
            record_id=event.record_id,
            claim=GoldClaim(text=event.claim_text),
            source=src,
            label=event.analyst_label,  # type: ignore[arg-type]
            notes=event.rationale,
            annotation_date=str(event.timestamp),
        )
        dataset.records.append(record)
        n_promoted += 1
    return n_promoted


__all__ = [
    "FeedbackBuffer",
    "FeedbackEvent",
    "promote_to_dataset",
    "record_feedback",
]
