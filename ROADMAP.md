---
path: ROADMAP.md
section: repo-root
doc-type: roadmap
status: stable
last_updated: 2026-05-03
---
# Roadmap

grounding-toolkit follows the 21-Phase plan documented in
`sentinel/docs/_plans/grounding_plan.md`.

## Phase 0 — Foundation (2026.5.15.0)

- Scaffold repo aligned with the 6 sibling toolkits
- Provisioning bridge for Sentinel test discovery
- Public testing module (stubs, factories, fixtures, matchers, smoke)
- CLI entry point
- Empty submodule scaffolding for all domains

## Phase 1-2 — Lexical cascade (2026.5.15.0+1)

- Tier −1 (consensus prior modulation)
- Tier 0 (substring exact)
- Tier 1 (lexical fuzzy via difflib)
- Stubs for Tier 2/3/4 (semantic / NLI / LLM-judge)
- Threshold modulation framework

## Phase 3 — Preventive grounding

- `citation_span` deterministic verification
- Structured signatures (DSPy) that force span emission
- Provenance DAG

## Phase 4 — Multimodal

- Tables, KV pairs, figures, signatures
- Source-driven (consumer populates `Source.tables` / `kv_pairs` / etc.)

## Phase 5 — Numerical derivation

- Generic `DerivationVerifier`: takes a formula spec from the consumer,
  recomputes from grounded primitives, compares with tolerance.
- No hard-coded financial / domain ratios.

## Phase 6 — Temporal & definitional consistency

## Phase 7 — Cross-document grounding

## Phase 8 — Multilingual (locale-tag driven)

## Phase 9 — Explainability (evidence + conflict + reasoning trace)

## Phase 10 — Confidence calibration

## Phase 11 — Audit (Merkle proof + immutable reasoning log)

## Phase 12 — Adversarial robustness

## Phase 13 — Calibration framework + feedback loop

## Phase 14 — Eval harness (RAGAS / DeepEval / TruLens)

## Phase 15 — Speculative pre-screen + constitutional rules engine

## Phase 16-18 — Sentinel migration + wiring

- Hard cutover migration of grounding code from Sentinel
- Sentinel adapter
- AAT § 28 + ZIP `analisi.json` evidence export

## Phase 19 — Test coverage ≥ 85 %

## Phase 20 — Release v2026.5.15.18

## Future — D5 (post LLM → local migration)

- Implementation bodies of `tiers/semantic.py`, `tiers/nli.py`, `tiers/llm_judge.py`
  using locally-served embedding / NLI / LLM-judge models routed through
  `cognis.core`.

## Future — D6 (annotation + tuning)

- Populate `calibration/golden/` with annotated gold-truth dataset
- Run `tuner.py` to optimize per-domain thresholds

## Future — D7 (UI integration)

- Frontend evidence-span highlighting
- Conflict-span visualisation
- Confidence gauge
