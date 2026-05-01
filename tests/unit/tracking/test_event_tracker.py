"""Tests for grounding.tracking.event_tracker."""

from __future__ import annotations

from grounding.tracking.event_tracker import (
    HallucinationTracker,
    hallucination_tracker,
)


def test_record_appends_event() -> None:
    t = HallucinationTracker()
    t.record("dd", "number", "doc-1", "fabricated number")
    stats = t.get_stats()
    assert stats["total"] == 1
    assert stats["by_agent"]["dd"] == 1
    assert stats["by_type"]["number"] == 1


def test_record_with_context_appends_to_context_list() -> None:
    t = HallucinationTracker()

    class _Ctx:
        def __init__(self):
            self.hallucination_events: list = []

    ctx = _Ctx()
    t.record("dd", "page", "doc", "page out of range", context=ctx)
    assert len(ctx.hallucination_events) == 1
    assert ctx.hallucination_events[0]["agent"] == "dd"
    assert ctx.hallucination_events[0]["type"] == "page"


def test_record_with_context_missing_attr_does_not_break() -> None:
    t = HallucinationTracker()

    class _Ctx:
        pass  # no hallucination_events attribute

    t.record("dd", "x", "doc", "detail", context=_Ctx())
    # No raise; event still recorded on the tracker.
    assert t.get_stats()["total"] == 1


def test_clear_resets() -> None:
    t = HallucinationTracker()
    t.record("a", "b", "c", "d")
    t.clear()
    assert t.get_stats()["total"] == 0


def test_window_truncates() -> None:
    t = HallucinationTracker()
    for i in range(150):
        t.record("a", "b", f"doc-{i}", "x")
    stats = t.get_stats(window=100)
    assert stats["total"] == 100


def test_singleton_exists() -> None:
    assert isinstance(hallucination_tracker, HallucinationTracker)


def test_singleton_persists_across_imports() -> None:
    """Re-import the singleton via a fresh path."""
    from grounding.tracking import hallucination_tracker as h2

    assert h2 is hallucination_tracker


def test_stats_rate_is_recent_over_window() -> None:
    t = HallucinationTracker()
    for _ in range(5):
        t.record("a", "b", "c", "d")
    stats = t.get_stats(window=10)
    assert stats["rate"] == 0.5
