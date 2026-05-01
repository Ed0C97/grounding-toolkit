"""Tests for grounding.citations.provenance."""

from __future__ import annotations

import json

from grounding import CitationSpan
from grounding.citations.provenance import (
    ProvenanceDAG,
    ProvenanceNode,
    claim_id,
)


def test_claim_id_deterministic() -> None:
    a = claim_id("hello world")
    b = claim_id("hello world")
    assert a == b
    assert len(a) == 16
    assert claim_id("different text") != a


def test_claim_id_namespace_changes_hash() -> None:
    a = claim_id("x", namespace="ns1")
    b = claim_id("x", namespace="ns2")
    assert a != b


def test_add_claim_returns_id() -> None:
    dag = ProvenanceDAG()
    cid = dag.add_claim(text="hello")
    assert cid == claim_id("hello")
    assert cid in dag


def test_add_claim_idempotent() -> None:
    dag = ProvenanceDAG()
    a = dag.add_claim(text="hello")
    b = dag.add_claim(text="hello")
    assert a == b
    assert len(dag) == 1


def test_get_returns_node() -> None:
    dag = ProvenanceDAG()
    cid = dag.add_claim(
        text="hello",
        generator="agent-x",
        confidence=0.9,
        citation_span=CitationSpan(page=1, char_start=0, char_end=5),
    )
    node = dag.get(cid)
    assert isinstance(node, ProvenanceNode)
    assert node.generator == "agent-x"
    assert node.confidence == 0.9
    assert node.citation_span is not None
    assert node.citation_span.page == 1


def test_ancestors_walks_dag() -> None:
    dag = ProvenanceDAG()
    a = dag.add_claim(text="A")
    b = dag.add_claim(text="B", parent_ids=[a])
    c = dag.add_claim(text="C", parent_ids=[b])
    ancestors = dag.ancestors(c)
    assert b in ancestors
    assert a in ancestors


def test_ancestors_handles_cycle_safely() -> None:
    """Manually inject a cycle via direct node add and check we don't loop."""
    dag = ProvenanceDAG()
    n1 = ProvenanceNode(claim_id="x", text="X", parent_ids=["y"])
    n2 = ProvenanceNode(claim_id="y", text="Y", parent_ids=["x"])
    dag.add(n1)
    dag.add(n2)
    # Should terminate, returning the other node
    ancestors = dag.ancestors("x")
    assert "y" in ancestors


def test_to_json_canonical() -> None:
    dag = ProvenanceDAG()
    dag.add_claim(text="A", generator="g1")
    dag.add_claim(text="B", generator="g2")
    js1 = dag.to_json()
    js2 = dag.to_json()
    assert js1 == js2  # deterministic
    parsed = json.loads(js1)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    # Sorted by claim_id ascending
    ids = [item["claim_id"] for item in parsed]
    assert ids == sorted(ids)


def test_to_json_serialises_citation_span() -> None:
    dag = ProvenanceDAG()
    dag.add_claim(
        text="A",
        citation_span=CitationSpan(page=2, char_start=10, char_end=20),
    )
    parsed = json.loads(dag.to_json())
    assert parsed[0]["citation_span"] == {
        "page": 2,
        "char_start": 10,
        "char_end": 20,
    }


def test_len_and_contains() -> None:
    dag = ProvenanceDAG()
    assert len(dag) == 0
    assert "anything" not in dag
    cid = dag.add_claim(text="x")
    assert len(dag) == 1
    assert cid in dag
