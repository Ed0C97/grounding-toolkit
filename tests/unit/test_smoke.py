"""Smoke tests for grounding-toolkit Phase 0."""

from __future__ import annotations

import grounding
from grounding.testing.smoke import run as smoke_run


def test_smoke_run_returns_true() -> None:
    assert smoke_run() is True


def test_version_string() -> None:
    assert isinstance(grounding.__version__, str)
    assert grounding.__version__.startswith("2026.")


def test_smoke_imports_all_stubs() -> None:
    """Every public stub must import cleanly."""
    from grounding.testing import (
        ClaimFactory,
        SourceFactory,
        StubEmbeddingFn,
        StubLLMJudgeFn,
        StubNLIFn,
        StubRetrievalFn,
        assert_version,
    )

    assert ClaimFactory is not None
    assert SourceFactory is not None
    assert StubEmbeddingFn is not None
    assert StubLLMJudgeFn is not None
    assert StubNLIFn is not None
    assert StubRetrievalFn is not None
    assert assert_version is not None
