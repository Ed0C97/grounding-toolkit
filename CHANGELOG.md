---
path: CHANGELOG.md
section: repo-root
doc-type: changelog
status: stable
last_updated: 2026-05-03
---
# Changelog

All notable changes to grounding-toolkit are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to lockstep CalVer (`YYYY.M.D.N`) with the
Sentinel monorepo.

## [2026.5.15.43] — 2026-05-03 — Doc standardisation lockstep

Lockstep release with Sentinel `v2026.5.15.43`. No runtime API changes;
the standardised YAML frontmatter (`path`, `section`, `doc-type`,
`status`, `last_updated`) was extended to every non-vendored Markdown
file in this repository (CHANGELOG, governance docs, ADRs, examples,
internal notes). Version bumped in lockstep so the calendar tag matches
the rest of the family. Tags from this version onward are SSH-signed.
## [2026.5.15.42] — 2026-05-03 — Lockstep release

Lockstep release with Sentinel `v2026.5.15.42`. No runtime API changes
in this toolkit; the cofounder-maintained READMEs now carry the
standardised YAML frontmatter used across the Sentinel monorepo
(`README.md`, `examples/README.md`). Version bumped in lockstep so the
calendar tag matches the rest of the family.

## [2026.5.15.23] — 2026-05-02 — Lockstep marker (sentinel profile rename)

No grounding-toolkit code changes.  Lockstep with sentinel
`v2026.5.15.23` (legacy profile rename: single_vendor_primary ->
legacy_single_vendor) and cognis-toolkit `v2026.5.15.23`.

## [2026.5.15.22] — 2026-05-02 — Synthetic seed expanded + lockstep

### Changed — D6 calibration synthetic seed

- `src/grounding/calibration/golden/sentinel-dd-synthetic.json`
  v0.1.0 -> v0.2.0.  12 -> 29 records.  Coverage now spans 7
  detector paths (evidence / definition / number / page /
  paraphrase / temporal / derivation), GROUNDED+UNGROUNDED+UNCERTAIN
  labels, IT-EN cross-language pairs, derivation arithmetic checks,
  page-anchor adversarial, contradiction direct.

### Lockstep version refresh

- `pyproject.toml` 2026.5.15.21 -> 2026.5.15.22
- `src/grounding/__init__.py` `__version__`
- `src/grounding/provisioning/manifest.yaml`

## [2026.5.15.21] — 2026-05-02 — Lockstep marker (Sentinel mode=test patch)

No grounding-toolkit code changes.  Lockstep with sentinel
``v2026.5.15.21`` (mode=test + LLM refs cleanup + DSPy recompile
tool) and cognis-toolkit ``v2026.5.15.21``.

### Lockstep version refresh

- `pyproject.toml` 2026.5.15.20 -> 2026.5.15.21
- `src/grounding/__init__.py` `__version__`
- `src/grounding/provisioning/manifest.yaml`

## [2026.5.15.20] — 2026-05-01 — D6 synthetic seed + FASE β lockstep

Lockstep with sentinel `v2026.5.15.20` and cognis-toolkit
`v2026.5.15.20`.

### Added — D6: synthetic seed for the calibration tuner

- `src/grounding/calibration/golden/sentinel-dd-synthetic.json` —
  12-record hand-crafted GoldDataset covering every detector path
  (evidence / number / definition / page / paraphrase / IT-EN /
  contradiction) on every label (GROUNDED / UNGROUNDED / UNCERTAIN).
- Verified end-to-end: `grounding.calibration.tuner.tune()` produces
  Brier ~0.008 / ECE ~0.087 on the seed.

### Lockstep version refresh

- `pyproject.toml` 2026.5.15.19 -> 2026.5.15.20
- `src/grounding/__init__.py` `__version__`
- `src/grounding/provisioning/manifest.yaml`

## [2026.5.15.19] — 2026-05-01 — Lockstep bump for FASE α (Sentinel)

Lockstep version bump to align with Sentinel `v2026.5.15.19` and
cognis-toolkit `v2026.5.15.19`.  No grounding-toolkit code changes —
this release is the marker that the Sentinel-side FASE α (LLM →
local-models migration) shipped.  The Tier 2/3/4 wiring (D5) lands
in FASE β.

### Lockstep version refresh

- `pyproject.toml` 2026.5.15.0 → 2026.5.15.19
- `src/grounding/__init__.py` `__version__`
- `src/grounding/provisioning/manifest.yaml`

## [2026.5.15.0] — 2026-05-01

### Added — Phase 0: foundation

- Scaffold layout aligned with the 6 existing sibling toolkits.
- `pyproject.toml` with provisioning entry-point + optional-dependencies
  for cognis / ocr / dspy / pdf-finder / ragas / deepeval / trulens / sbert.
- `src/grounding/provisioning/` bridge with `descriptor()` + `runner()`
  and `manifest.yaml` exposing test groups to Sentinel.
- `src/grounding/testing/` public test framework: stubs, factories,
  fixtures, matchers, smoke runner.
- `src/grounding/cli/` CLI entry point.
- Empty submodule scaffolding for: `core`, `tiers`, `citations`,
  `multimodal`, `numerical`, `temporal`, `definitional`, `crossdoc`,
  `language`, `explainability`, `confidence`, `audit`, `adversarial`,
  `calibration`, `eval`, `constitutional`, `consensus`, `tracking`,
  `spatial`, `answer`.
- `tests/unit/` skeleton with smoke and version tests.
- Documentation: `README`, `STRUCTURE`, `ROADMAP`, `SECURITY`, `SUPPORT`,
  `CONTRIBUTING`, `CODE_OF_CONDUCT`, `AUTHORS`, `MAINTAINERS`,
  `docs/ARCHITECTURE.md`, `docs/PHASE_0_NOTES.md`.

## Lockstep versioning

grounding-toolkit follows CalVer lockstep with the rest of the Sentinel
monorepo. Calendar tag `2026.5.15.x` shipped together across all 7
sibling toolkits at every release wave.
