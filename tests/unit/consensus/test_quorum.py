"""Tests for grounding.consensus.quorum."""

from __future__ import annotations

from grounding.consensus.quorum import (
    QuorumConfig,
    aggregate,
    aggregate_findings_list,
)


def test_empty_returns_empty() -> None:
    assert aggregate([]) == {}


def test_single_thread_returns_single() -> None:
    out = aggregate([{"severity": "HIGH"}])
    assert out["moa_consensus"] == "SINGLE"
    assert out["severity"] == "HIGH"


def test_unanimous_confirmed() -> None:
    out = aggregate(
        [{"severity": "HIGH"}, {"severity": "HIGH"}, {"severity": "HIGH"}]
    )
    assert out["moa_consensus"] == "CONFIRMED"
    assert out["severity"] == "HIGH"


def test_majority_confirmed_with_minority_notes() -> None:
    # 3-of-4 = 0.75 > 0.667 → CONFIRMED; 2-of-3 ~ 0.6667 < 0.667 →
    # DISAGREEMENT, so we use 4 threads to get a clean CONFIRMED with a
    # minority dissent attached.
    out = aggregate(
        [
            {"severity": "HIGH", "reasoning": "r1"},
            {"severity": "HIGH", "reasoning": "r2"},
            {"severity": "HIGH", "reasoning": "r3"},
            {"severity": "MEDIUM", "reasoning": "minority"},
        ]
    )
    assert out["moa_consensus"] == "CONFIRMED"
    assert "moa_minority_notes" in out
    assert any("minority" in n for n in out["moa_minority_notes"])


def test_disagreement_downgrades_by_default() -> None:
    out = aggregate(
        [
            {"severity": "HIGH"},
            {"severity": "MEDIUM"},
            {"severity": "LOW"},
        ]
    )
    # HIGH is the plurality, but ratio = 1/3 < 0.667 → DISAGREEMENT,
    # downgrade HIGH → MEDIUM.
    assert out["moa_consensus"] == "DISAGREEMENT"
    assert out["severity"] in {"MEDIUM", "LOW"}


def test_symmetric_disagreement_uses_winner_directly() -> None:
    out = aggregate(
        [
            {"severity": "HIGH"},
            {"severity": "MEDIUM"},
            {"severity": "LOW"},
        ],
        config=QuorumConfig(symmetric=True),
    )
    assert out["moa_consensus"] == "DISAGREEMENT"
    assert out["severity"] == "HIGH"


def test_vote_distribution_recorded() -> None:
    out = aggregate(
        [{"severity": "HIGH"}, {"severity": "MEDIUM"}]
    )
    assert "moa_vote_distribution" in out
    assert out["moa_thread_count"] == 2


def test_disagreement_hook_called() -> None:
    calls = []

    def _hook(field, chosen, winning, threads):
        calls.append((field, chosen, winning, len(threads)))

    aggregate(
        [
            {"severity": "HIGH"},
            {"severity": "LOW"},
            {"severity": "MEDIUM"},
        ],
        on_disagreement=_hook,
    )
    assert len(calls) == 1
    assert calls[0][0] == "severity"


def test_hook_not_called_on_confirmed() -> None:
    calls = []

    def _hook(*args):
        calls.append(args)

    aggregate(
        [{"severity": "HIGH"}, {"severity": "HIGH"}, {"severity": "HIGH"}],
        on_disagreement=_hook,
    )
    assert calls == []


def test_aggregate_findings_list_groups_by_ref() -> None:
    # All 3 threads have the same finding → ratio 3/3 = 1.0 → CONFIRMED.
    threads = [
        [{"clause_reference": "C-1", "severity": "HIGH"}],
        [{"clause_reference": "C-1", "severity": "HIGH"}],
        [{"clause_reference": "C-1", "severity": "HIGH"}],
    ]
    out = aggregate_findings_list(threads)
    assert len(out) == 1
    assert out[0]["moa_consensus"] == "CONFIRMED"


def test_aggregate_findings_list_minority_dissent() -> None:
    threads = [
        [{"clause_reference": "C-1", "severity": "HIGH"}],
        [],
        [],
    ]
    out = aggregate_findings_list(threads)
    assert len(out) == 1
    assert out[0]["moa_consensus"] == "MINORITY_DISSENT"


def test_aggregate_findings_list_uses_ref_fallback() -> None:
    threads = [
        [{"ref": "X", "severity": "HIGH"}],
        [{"ref": "X", "severity": "HIGH"}],
        [{"ref": "X", "severity": "HIGH"}],
    ]
    out = aggregate_findings_list(threads)
    assert len(out) == 1


def test_custom_field_aggregation() -> None:
    out = aggregate(
        [
            {"verdict": "PASS"},
            {"verdict": "PASS"},
            {"verdict": "PASS"},
            {"verdict": "FAIL"},
        ],
        config=QuorumConfig(field="verdict", default_value="PASS"),
    )
    # 3-of-4 = 0.75 > 0.667 → CONFIRMED.
    assert out["moa_consensus"] == "CONFIRMED"
    assert out["verdict"] == "PASS"
