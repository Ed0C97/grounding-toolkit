---
path: docs/PHASE_0_NOTES.md
section: docs
doc-type: docs
status: stable
last_updated: 2026-05-03
---
# Phase 0 — Foundation notes

## What landed in 2026.5.15.0

- Repository scaffold aligned with the 6 existing sibling toolkits.
- `pyproject.toml` with provisioning entry-point, optional dependencies
  per integration (cognis / ocr / dspy / pdf-finder / ragas / deepeval /
  trulens / sbert).
- `provisioning/` bridge with `descriptor()` + `runner()` and
  `manifest.yaml`.
- `testing/` public framework: stubs (Embedding / NLI / LLM-judge /
  Retrieval), factories (Claim / Source), fixtures, matchers, smoke.
- `cli/` entry point with `version`, `smoke`, `manifest` subcommands.
- Empty submodule scaffolding for all P1-P15 domains.
- Tests: smoke, provisioning, version coherence, CLI, parity import
  surface.
- Documentation: README, STRUCTURE, ROADMAP, ARCHITECTURE, plus the
  standard SECURITY / SUPPORT / CONTRIBUTING / CODE_OF_CONDUCT /
  AUTHORS / MAINTAINERS.

## What's deliberately stubbed

- `core/`, `tiers/`, `citations/`, `multimodal/`, `numerical/`,
  `temporal/`, `definitional/`, `crossdoc/`, `language/`,
  `explainability/`, `confidence/`, `audit/`, `adversarial/`,
  `calibration/`, `eval/`, `constitutional/`, `consensus/`,
  `tracking/`, `spatial/`, `answer/` are all docstring-only at P0.
- The substantive logic lands phase by phase (P1 through P15).

## Acceptance criteria for P0

1. `pip install -e .` succeeds.
2. `python -m grounding.testing.smoke` returns True.
3. `pytest tests/unit -q` is zero-skip and zero-fail.
4. `python -m grounding.provisioning` (via the CLI `grounding manifest`)
   prints the manifest as JSON.
5. `make lint` passes.
6. Sentinel `tests/provisioning/_toolkit_bridge.py` discovers the new
   entry-point automatically.

## What's not in P0

- Any actual verification logic — that's P1+.
- Any Sentinel-side wiring — that's P16-P18.
- Any release tag — that's P20.
