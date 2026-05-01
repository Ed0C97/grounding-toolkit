# Contributing

## Development setup

```bash
git clone https://github.com/Ed0C97/grounding-toolkit
cd grounding-toolkit
make install-dev
make test
```

## Conventions

- **CalVer lockstep**: do not bump the version independently; coordinate
  with the rest of the Sentinel monorepo (sentinel-core, cognis-toolkit,
  exchequer-toolkit, dspy-toolkit, rlm-toolkit, ocr-toolkit,
  pdf-finder-toolkit). Update `pyproject.toml`, `manifest.yaml`, and
  `__init__.py` together.
- **Provider-agnostic**: every external dependency must be expressed
  as a Protocol; consumers inject the implementation.
- **Domain-agnostic**: no domain knowledge in `src/grounding/`.
- **Zero-skip default suite**: `pytest tests/` runs unit + parity only;
  integration is opt-in via marker.
- **Lint & format**: `make lint` (ruff), `make format` (ruff format).

## Pull-request workflow

1. Open a PR against `main`
2. CI runs `make check` (lint + test)
3. Ensure version coordination if your change is part of a release wave
4. Squash-merge after review

## Testing layers

- `tests/unit/`: pure-function, offline, default
- `tests/integration/`: live providers, opt-in via `@pytest.mark.integration`
- `tests/parity/`: cross-toolkit contract mirror
- `tests/e2e/`: end-to-end workflows
- `tests/benchmarks/`: performance comparisons
