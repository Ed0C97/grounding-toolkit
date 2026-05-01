"""Centralised hallucination event tracker.

Migrated from Sentinel's ``sentinel.utils.hallucination_tracker``
(P16 hard cutover, no shim).  Consumers import directly from
:mod:`grounding.tracking` going forward.

API mirrors the Sentinel original to make the cutover mechanical:

- :class:`HallucinationEvent` — frozen-shape dataclass.
- :class:`HallucinationTracker` — singleton-friendly tracker with
  ``record(...)``, ``get_stats(...)``, ``clear()``.
- ``hallucination_tracker`` — module-level singleton.

Per-document accumulation: when ``record()`` is called with a context
object exposing a ``hallucination_events`` list, the same event is
appended to that list for downstream per-document reporting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class HallucinationEvent:
    """A single detected hallucination."""

    agent: str
    type: str
    doc_id: str
    detail: str
    timestamp: datetime = field(default_factory=datetime.now)


class HallucinationTracker:
    """Process-wide tracker of hallucination events."""

    def __init__(self) -> None:
        self._events: List[HallucinationEvent] = []
        self._logger = logging.getLogger("grounding.tracking")

    def record(
        self,
        agent: str,
        type: str,
        doc_id: str,
        detail: str,
        *,
        context: object | None = None,
    ) -> None:
        """Record a hallucination event and emit a warning log.

        Per-document tracking: when ``context`` is supplied and exposes
        a ``hallucination_events`` list attribute, the same event is
        also appended there so downstream reporting (e.g. AAT § 28) can
        compute per-document counts without polling the singleton.
        """
        event = HallucinationEvent(
            agent=agent, type=type, doc_id=doc_id, detail=detail
        )
        self._events.append(event)
        self._logger.warning(
            "[hallucination] agent=%s type=%s doc=%s: %s",
            agent,
            type,
            doc_id,
            detail,
        )
        if context is not None:
            try:
                lst = getattr(context, "hallucination_events", None)
                if lst is not None:
                    lst.append(
                        {
                            "agent": agent,
                            "type": type,
                            "detail": detail,
                        }
                    )
            except Exception:
                pass  # best-effort — never break the agent

    def get_stats(self, window: int = 100) -> dict:
        """Stats for the last *window* events."""
        recent = self._events[-window:] if self._events else []
        by_agent: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        for e in recent:
            by_agent[e.agent] = by_agent.get(e.agent, 0) + 1
            by_type[e.type] = by_type.get(e.type, 0) + 1
        return {
            "total": len(recent),
            "by_agent": by_agent,
            "by_type": by_type,
            "rate": len(recent) / max(window, 1),
        }

    def clear(self) -> None:
        self._events.clear()


# Module-level singleton.
hallucination_tracker = HallucinationTracker()


__all__ = [
    "HallucinationEvent",
    "HallucinationTracker",
    "hallucination_tracker",
]
