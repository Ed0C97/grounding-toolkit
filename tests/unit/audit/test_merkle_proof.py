"""Tests for grounding.audit.merkle_proof."""

from __future__ import annotations

from grounding import EvidencePointer
from grounding.audit.merkle_proof import (
    build_merkle_proof,
    leaf_hash,
    merkle_root,
    merkle_root_for_evidence,
    verify_proof,
)


def test_empty_root() -> None:
    assert merkle_root([]) == ""


def test_single_payload_root() -> None:
    r = merkle_root(["a"])
    assert r == leaf_hash("a")


def test_two_payload_root_deterministic() -> None:
    r1 = merkle_root(["a", "b"])
    r2 = merkle_root(["a", "b"])
    assert r1 == r2
    # Different from single
    assert r1 != merkle_root(["a"])


def test_different_payloads_produce_different_roots() -> None:
    assert merkle_root(["a", "b"]) != merkle_root(["a", "c"])


def test_odd_count_handled() -> None:
    # 3 payloads → last duplicated → still valid root
    r = merkle_root(["a", "b", "c"])
    assert r != ""


def test_merkle_root_for_evidence() -> None:
    pointers = [
        EvidencePointer("d", 1, 0, 5),
        EvidencePointer("d", 1, 10, 15),
    ]
    r = merkle_root_for_evidence(pointers)
    assert isinstance(r, str)
    assert len(r) == 64  # sha256 hex


def test_merkle_root_for_evidence_order_independent() -> None:
    p1 = EvidencePointer("d", 1, 0, 5)
    p2 = EvidencePointer("d", 1, 10, 15)
    r1 = merkle_root_for_evidence([p1, p2])
    r2 = merkle_root_for_evidence([p2, p1])
    assert r1 == r2  # canonical sorting before hashing


def test_build_proof_round_trip() -> None:
    pointers = [
        EvidencePointer("d", 1, 0, 5),
        EvidencePointer("d", 1, 10, 15),
    ]
    proof = build_merkle_proof(pointers)
    assert proof.root != ""
    assert len(proof.leaves) == 2
    assert verify_proof(proof)


def test_proof_verifies_after_round_trip() -> None:
    pointers = [EvidencePointer("d", 1, 0, 5)]
    proof = build_merkle_proof(pointers)
    assert proof.verify()


def test_tampered_proof_fails_verification() -> None:
    pointers = [EvidencePointer("d", 1, 0, 5)]
    proof = build_merkle_proof(pointers)
    # Tamper with payloads
    proof.payloads[0] = '{"doc_id":"X","page":1,"char_start":0,"char_end":5}'
    assert not proof.verify()


def test_empty_evidence_list() -> None:
    proof = build_merkle_proof([])
    assert proof.root == ""
    assert proof.leaves == []
    assert proof.verify()
