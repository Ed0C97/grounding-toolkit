# Changelog

All notable changes to grounding-toolkit are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to lockstep CalVer (`YYYY.M.D.N`) with the
Sentinel monorepo.

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
