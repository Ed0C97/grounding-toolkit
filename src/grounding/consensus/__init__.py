"""grounding.consensus — quorum aggregation (migrated from Sentinel SupremeJudge.aggregate)."""

from __future__ import annotations

from grounding.consensus.quorum import (
    DEFAULT_SEVERITY_ORDER,
    DEFAULT_SEVERITY_REVERSE,
    DisagreementHook,
    QuorumConfig,
    aggregate,
    aggregate_findings_list,
)

__all__ = [
    "DEFAULT_SEVERITY_ORDER",
    "DEFAULT_SEVERITY_REVERSE",
    "DisagreementHook",
    "QuorumConfig",
    "aggregate",
    "aggregate_findings_list",
]
