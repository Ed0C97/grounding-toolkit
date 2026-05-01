"""Constitutional rules engine.

A YAML-driven rule loader + matcher.  The toolkit ships **only the
engine** — no rules.  Rule sets are domain-specific and supplied by the
consumer.

Rule shape (YAML)::

    name: financial_claims_must_cite_page
    when:
      claim_text_matches: "(EBITDA|DSCR|LTV|ICR)"
    require:
      - claim_has_citation_span: true
      - source_has_table: true
    severity: high
    rationale: "Financial ratio claims must reference an income statement table."

Engine output: a list of :class:`RuleViolation` for the rules that
matched the claim's ``when`` predicate but failed at least one
``require`` predicate.

The rule predicate vocabulary is intentionally small and string-based
so the YAML is human-editable without surprises.  Consumers extend the
engine by registering new predicate handlers via
:meth:`RulesEngine.register_predicate`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import yaml

from grounding.core.types import Claim, Source


PredicateFn = Callable[[Claim, Source, Any], bool]


# --------------------------------------------------------------------
# Rule data classes
# --------------------------------------------------------------------


@dataclass
class Rule:
    """A single constitutional rule loaded from YAML."""

    name: str
    when: Dict[str, Any] = field(default_factory=dict)
    require: List[Dict[str, Any]] = field(default_factory=list)
    severity: str = "medium"
    rationale: str = ""


@dataclass
class RuleViolation:
    """A failed rule application."""

    rule_name: str
    severity: str
    rationale: str
    failed_predicates: List[str] = field(default_factory=list)


# --------------------------------------------------------------------
# Built-in predicates
# --------------------------------------------------------------------


def _claim_text_matches(claim: Claim, source: Source, value: Any) -> bool:
    pattern = str(value)
    return re.search(pattern, claim.text or "", re.IGNORECASE) is not None


def _claim_has_citation_span(
    claim: Claim, source: Source, value: Any
) -> bool:
    has = claim.citation_span is not None
    return has if value else not has


def _source_has_table(
    claim: Claim, source: Source, value: Any
) -> bool:
    has = bool(source.tables)
    return has if value else not has


def _source_has_kv(
    claim: Claim, source: Source, value: Any
) -> bool:
    has = bool(source.kv_pairs)
    return has if value else not has


def _source_text_matches(
    claim: Claim, source: Source, value: Any
) -> bool:
    pattern = str(value)
    return (
        re.search(pattern, source.text or "", re.IGNORECASE) is not None
    )


def _claim_metadata_eq(
    claim: Claim, source: Source, value: Any
) -> bool:
    if not isinstance(value, dict):
        return False
    meta = claim.metadata or {}
    for k, v in value.items():
        if str(meta.get(k, "")).lower() != str(v).lower():
            return False
    return True


_BUILTIN_PREDICATES: Dict[str, PredicateFn] = {
    "claim_text_matches": _claim_text_matches,
    "claim_has_citation_span": _claim_has_citation_span,
    "source_has_table": _source_has_table,
    "source_has_kv": _source_has_kv,
    "source_text_matches": _source_text_matches,
    "claim_metadata_eq": _claim_metadata_eq,
}


# --------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------


@dataclass
class RulesEngine:
    """Load and apply constitutional rules."""

    rules: List[Rule] = field(default_factory=list)
    predicates: Dict[str, PredicateFn] = field(
        default_factory=lambda: dict(_BUILTIN_PREDICATES)
    )

    def register_predicate(
        self, name: str, fn: PredicateFn
    ) -> None:
        self.predicates[name] = fn

    def load_yaml(self, path: Path | str) -> None:
        p = Path(path)
        with p.open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or []
        if isinstance(payload, dict) and "rules" in payload:
            payload = payload["rules"]
        if not isinstance(payload, list):
            raise RuntimeError(
                f"rules YAML at {p} must be a list (or have a top-level "
                "'rules' key with a list value)"
            )
        for entry in payload:
            self.rules.append(self._parse_rule(entry))

    @staticmethod
    def _parse_rule(entry: Dict[str, Any]) -> Rule:
        if "name" not in entry:
            raise RuntimeError(f"rule missing 'name': {entry!r}")
        return Rule(
            name=str(entry["name"]),
            when=dict(entry.get("when", {})),
            require=list(entry.get("require", [])),
            severity=str(entry.get("severity", "medium")),
            rationale=str(entry.get("rationale", "")),
        )

    def evaluate_predicate(
        self,
        name: str,
        value: Any,
        claim: Claim,
        source: Source,
    ) -> bool:
        if name not in self.predicates:
            raise KeyError(f"unknown predicate: {name}")
        return bool(self.predicates[name](claim, source, value))

    def evaluate(
        self, claim: Claim, source: Source
    ) -> List[RuleViolation]:
        """Apply every rule and return the violations."""
        violations: List[RuleViolation] = []
        for rule in self.rules:
            # Evaluate all 'when' predicates with AND semantics.
            applies = True
            for pred_name, pred_value in rule.when.items():
                if not self.evaluate_predicate(
                    pred_name, pred_value, claim, source
                ):
                    applies = False
                    break
            if not applies:
                continue
            failed: List[str] = []
            for require in rule.require:
                if not isinstance(require, dict):
                    continue
                for pred_name, pred_value in require.items():
                    ok = self.evaluate_predicate(
                        pred_name, pred_value, claim, source
                    )
                    if not ok:
                        failed.append(pred_name)
            if failed:
                violations.append(
                    RuleViolation(
                        rule_name=rule.name,
                        severity=rule.severity,
                        rationale=rule.rationale,
                        failed_predicates=failed,
                    )
                )
        return violations


__all__ = [
    "PredicateFn",
    "Rule",
    "RuleViolation",
    "RulesEngine",
]
