.PHONY: help install install-dev test lint format check build clean smoke parity

help:
	@echo "grounding-toolkit — Makefile"
	@echo ""
	@echo "Install:"
	@echo "  make install       Install runtime dependencies (editable)"
	@echo "  make install-dev   Install runtime + dev dependencies"
	@echo ""
	@echo "Quality:"
	@echo "  make test          Run pytest suite (unit + parity)"
	@echo "  make smoke         Run 10-second smoke test"
	@echo "  make lint          Run ruff linter"
	@echo "  make format        Run ruff formatter"
	@echo "  make check         Run lint + test"
	@echo "  make parity        Run cross-toolkit parity suite"
	@echo ""
	@echo "Build:"
	@echo "  make build         Build wheel + sdist"
	@echo "  make clean         Remove build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[all,dev]"

test:
	pytest tests/ -ra

smoke:
	python -m grounding.testing.smoke

lint:
	ruff check src tests

format:
	ruff format src tests

check: lint test

parity:
	pytest tests/parity -ra -v

build:
	python -m build

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
