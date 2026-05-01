# Security policy

## Reporting a vulnerability

Open a private GitHub Security Advisory or email the maintainers
(see `MAINTAINERS.md`).

## Supported versions

- `2026.5.15.x` (latest CalVer lockstep) — security fixes accepted
- Previous calendar tags — best-effort

## Security practices

- No hardcoded secrets — all provider credentials come through injected
  Protocol implementations supplied by the consumer.
- Optional dependencies are gated; the core lexical / consensus cascade
  has zero network egress.
- `py.typed` enables strict static analysis via mypy / pyright.
- `ruff` enforces lint + import order.
- The toolkit does NOT call out to LLM providers directly; the consumer
  injects an `LLMJudgeFn` callable.

## Audit trail

`grounding/audit/merkle_proof.py` produces a deterministic Merkle root
over the evidence-span list of any verification batch. Consumers can
publish the root in their dossier for tamper-evident proofs.
