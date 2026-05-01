"""Quorum-based consensus aggregator.

Migrated from Sentinel's
``sentinel.agents.supreme_judge.SupremeJudge.aggregate``
(P16 hard cutover).  The toolkit ships the **generic** quorum logic
(majority-vote on a configurable field, single-thread short-circuit,
disagreement downgrade or symmetric majority) and exposes optional
hook callbacks so consumers can inject domain-specific behaviours
(e.g. Sentinel's preference-pair emission for RL training).

Public API:

- :func:`aggregate` — single-result consensus from N parallel threads.
- :func:`aggregate_findings_list` — list-of-results consensus, grouped
  by a key field (e.g. ``clause_reference``).

Both functions are pure (no I/O), deterministic, and
provider-agnostic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence


# Severity ladder used when downgrading on disagreement.  The default
# matches Sentinel's existing semantics (LOW < MEDIUM < HIGH < CRITICAL).
DEFAULT_SEVERITY_ORDER: Dict[str, int] = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}

DEFAULT_SEVERITY_REVERSE: Dict[int, str] = {
    v: k for k, v in DEFAULT_SEVERITY_ORDER.items()
}


@dataclass
class QuorumConfig:
    """Per-call configuration for :func:`aggregate`."""

    threshold: float = 0.667
    """Fraction of threads that must agree for CONFIRMED."""

    field: str = "severity"
    """Result-dict key whose value is being voted on."""

    symmetric: bool = False
    """When True, disagreement uses the majority value (can up- or
    down-grade).  When False, disagreement always downgrades by one
    level on the severity ladder."""

    severity_order: Dict[str, int] = None  # type: ignore[assignment]
    """Optional override of the severity ladder.  Defaults to
    ``DEFAULT_SEVERITY_ORDER``."""

    default_value: str = "MEDIUM"
    """Fallback value when a thread result lacks the voted field."""


# Optional hook signature.  The toolkit calls the hook (if supplied)
# AFTER aggregation; consumer-side side-effects are out of scope for
# the toolkit (Sentinel uses this for preference-pair emission).
DisagreementHook = Callable[
    [
        # field: str
        str,
        # chosen_value: str
        str,
        # winning_value: str
        str,
        # thread_results: list of dicts
        List[Dict[str, Any]],
    ],
    None,
]


def aggregate(
    thread_results: Sequence[Dict[str, Any]],
    *,
    config: Optional[QuorumConfig] = None,
    on_disagreement: Optional[DisagreementHook] = None,
) -> Dict[str, Any]:
    """Quorum-based consensus over N parallel-thread result dicts.

    Returns the winning result enriched with consensus metadata:
    ``moa_consensus`` (``CONFIRMED`` / ``DISAGREEMENT`` / ``SINGLE``),
    ``moa_thread_count``, ``moa_vote_distribution``, optional
    ``moa_minority_notes``.
    """
    cfg = config or QuorumConfig()
    severity_order = cfg.severity_order or DEFAULT_SEVERITY_ORDER
    severity_reverse = (
        {v: k for k, v in severity_order.items()}
        if cfg.severity_order is not None
        else DEFAULT_SEVERITY_REVERSE
    )

    if not thread_results:
        return {}

    if len(thread_results) == 1:
        r = dict(thread_results[0])
        r["moa_consensus"] = "SINGLE"
        r["moa_thread_scores"] = []
        return r

    votes: Counter = Counter()
    for r in thread_results:
        val = (r.get(cfg.field) or cfg.default_value).upper()
        votes[val] += 1

    n = len(thread_results)
    winning_val, winning_count = votes.most_common(1)[0]
    ratio = winning_count / n

    base = next(
        (
            r
            for r in thread_results
            if (r.get(cfg.field) or cfg.default_value).upper()
            == winning_val
        ),
        thread_results[0],
    )
    result = dict(base)

    if ratio >= cfg.threshold:
        result["moa_consensus"] = "CONFIRMED"
        if winning_count != n:
            result["moa_minority_notes"] = [
                r.get("reasoning", "") or r.get("description", "")
                for r in thread_results
                if (r.get(cfg.field) or cfg.default_value).upper()
                != winning_val
            ]
    else:
        if cfg.symmetric:
            # Symmetric — majority value wins regardless of severity
            # direction.
            result[cfg.field] = winning_val
        else:
            # Conservative — downgrade by one severity level.
            current_level = severity_order.get(winning_val, 1)
            downgraded = severity_reverse.get(
                max(0, current_level - 1), "LOW"
            )
            result[cfg.field] = downgraded
        result["moa_consensus"] = "DISAGREEMENT"

    result["moa_thread_count"] = n
    result["moa_vote_distribution"] = dict(votes)

    if on_disagreement is not None and result["moa_consensus"] == "DISAGREEMENT":
        try:
            on_disagreement(
                cfg.field,
                str(result.get(cfg.field) or winning_val),
                winning_val,
                list(thread_results),
            )
        except Exception:
            # Hook side-effects are best-effort.
            pass

    return result


def aggregate_findings_list(
    all_thread_findings: Sequence[Sequence[Dict[str, Any]]],
    *,
    config: Optional[QuorumConfig] = None,
    on_disagreement: Optional[DisagreementHook] = None,
    group_keys: Sequence[str] = ("clause_reference", "ref"),
) -> List[Dict[str, Any]]:
    """Aggregate per-thread finding lists into a single consensus list.

    Findings are grouped by the first non-empty value among
    ``group_keys`` (default: ``clause_reference``, ``ref``).
    """
    cfg = config or QuorumConfig()
    severity_order = cfg.severity_order or DEFAULT_SEVERITY_ORDER
    severity_reverse = (
        {v: k for k, v in severity_order.items()}
        if cfg.severity_order is not None
        else DEFAULT_SEVERITY_REVERSE
    )

    if not all_thread_findings:
        return []

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for findings in all_thread_findings:
        if not isinstance(findings, list):
            continue
        for f in findings:
            ref = ""
            for k in group_keys:
                v = f.get(k)
                if v:
                    ref = str(v)
                    break
            if not ref:
                ref = "__unknown__"
            grouped.setdefault(ref, []).append(f)

    consensus: List[Dict[str, Any]] = []
    n_threads = len(all_thread_findings)

    for ref, variants in grouped.items():
        ratio = len(variants) / n_threads
        if ratio >= cfg.threshold:
            merged = aggregate(
                variants, config=cfg, on_disagreement=on_disagreement
            )
            consensus.append(merged)
        else:
            merged = aggregate(
                variants, config=cfg, on_disagreement=on_disagreement
            )
            merged["moa_consensus"] = "MINORITY_DISSENT"
            sev = (merged.get(cfg.field) or "LOW").upper()
            level = severity_order.get(sev, 0)
            merged[cfg.field] = severity_reverse.get(
                max(0, level - 1), "LOW"
            )
            consensus.append(merged)

    return consensus


__all__ = [
    "DEFAULT_SEVERITY_ORDER",
    "DEFAULT_SEVERITY_REVERSE",
    "DisagreementHook",
    "QuorumConfig",
    "aggregate",
    "aggregate_findings_list",
]
