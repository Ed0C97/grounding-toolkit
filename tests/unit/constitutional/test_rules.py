"""Tests for grounding.constitutional.rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from grounding import CitationSpan, Claim, Source, Table
from grounding.constitutional.rules import (
    Rule,
    RulesEngine,
)


def test_engine_no_rules() -> None:
    e = RulesEngine()
    out = e.evaluate(Claim(text="x"), Source.from_text("y"))
    assert out == []


def test_rule_when_doesnt_apply_no_violations() -> None:
    e = RulesEngine(
        rules=[
            Rule(
                name="r1",
                when={"claim_text_matches": "EBITDA"},
                require=[{"claim_has_citation_span": True}],
            )
        ]
    )
    # Claim text does NOT contain EBITDA
    out = e.evaluate(
        Claim(text="something else"),
        Source.from_text("y"),
    )
    assert out == []


def test_rule_applies_and_fails() -> None:
    e = RulesEngine(
        rules=[
            Rule(
                name="r1",
                when={"claim_text_matches": "EBITDA"},
                require=[{"claim_has_citation_span": True}],
                severity="high",
            )
        ]
    )
    out = e.evaluate(
        Claim(text="EBITDA grew"),
        Source.from_text("y"),
    )
    assert len(out) == 1
    assert out[0].rule_name == "r1"
    assert out[0].severity == "high"
    assert "claim_has_citation_span" in out[0].failed_predicates


def test_rule_passes_when_required_satisfied() -> None:
    e = RulesEngine(
        rules=[
            Rule(
                name="r1",
                when={"claim_text_matches": "EBITDA"},
                require=[{"claim_has_citation_span": True}],
            )
        ]
    )
    claim = Claim(
        text="EBITDA grew",
        citation_span=CitationSpan(page=1, char_start=0, char_end=5),
    )
    out = e.evaluate(claim, Source.from_text("y"))
    assert out == []


def test_source_has_table_predicate() -> None:
    e = RulesEngine(
        rules=[
            Rule(
                name="r1",
                when={"claim_text_matches": "EBITDA"},
                require=[{"source_has_table": True}],
            )
        ]
    )
    out = e.evaluate(
        Claim(text="EBITDA grew"),
        Source.from_text("y"),  # no tables
    )
    assert len(out) == 1


def test_source_has_table_satisfied() -> None:
    e = RulesEngine(
        rules=[
            Rule(
                name="r1",
                when={"claim_text_matches": "EBITDA"},
                require=[{"source_has_table": True}],
            )
        ]
    )
    src = Source(tables=[Table(rows=[["a", "b"]])])
    out = e.evaluate(Claim(text="EBITDA grew"), src)
    assert out == []


def test_source_has_kv() -> None:
    e = RulesEngine(
        rules=[
            Rule(
                name="r1",
                when={"claim_text_matches": "."},
                require=[{"source_has_kv": True}],
            )
        ]
    )
    src = Source(kv_pairs={"k": "v"})
    out = e.evaluate(Claim(text="anything"), src)
    assert out == []


def test_source_text_matches() -> None:
    e = RulesEngine(
        rules=[
            Rule(
                name="r1",
                when={"claim_text_matches": "."},
                require=[{"source_text_matches": "Section 4"}],
            )
        ]
    )
    out = e.evaluate(
        Claim(text="x"),
        Source.from_text("Section 4 details"),
    )
    assert out == []


def test_claim_metadata_eq() -> None:
    e = RulesEngine(
        rules=[
            Rule(
                name="r1",
                when={"claim_metadata_eq": {"language": "en"}},
                require=[{"claim_has_citation_span": True}],
            )
        ]
    )
    out = e.evaluate(
        Claim(text="x", metadata={"language": "en"}),
        Source.from_text("y"),
    )
    assert len(out) == 1


def test_unknown_predicate_raises() -> None:
    e = RulesEngine(
        rules=[
            Rule(
                name="r1",
                when={"unknown_pred": True},
                require=[],
            )
        ]
    )
    with pytest.raises(KeyError):
        e.evaluate(Claim(text="x"), Source.from_text("y"))


def test_register_custom_predicate() -> None:
    e = RulesEngine()

    def _custom(claim, source, value):  # noqa: ARG001
        return claim.text == value

    e.register_predicate("claim_text_eq", _custom)
    e.rules.append(
        Rule(
            name="r1",
            when={"claim_text_eq": "exact"},
            require=[{"claim_has_citation_span": True}],
        )
    )
    out = e.evaluate(
        Claim(text="exact"),
        Source.from_text("y"),
    )
    assert len(out) == 1


def test_load_yaml(tmp_path: Path) -> None:
    yaml_text = """
rules:
  - name: financial_must_cite
    when:
      claim_text_matches: "(EBITDA|DSCR)"
    require:
      - claim_has_citation_span: true
    severity: high
    rationale: "Financial ratios must cite a span."
"""
    p = tmp_path / "rules.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    e = RulesEngine()
    e.load_yaml(p)
    assert len(e.rules) == 1
    assert e.rules[0].name == "financial_must_cite"
    assert e.rules[0].severity == "high"


def test_load_yaml_top_level_list(tmp_path: Path) -> None:
    yaml_text = """
- name: r1
  when:
    claim_text_matches: "."
  require: []
"""
    p = tmp_path / "rules.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    e = RulesEngine()
    e.load_yaml(p)
    assert len(e.rules) == 1
