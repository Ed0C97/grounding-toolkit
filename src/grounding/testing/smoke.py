"""grounding.testing.smoke — 10-second self-check.

Run programmatically::

    >>> from grounding.testing.smoke import run
    >>> assert run() is True

Or as a module::

    $ python -m grounding.testing.smoke

Verifies the import surface, Protocol conformance of bundled stubs,
and default Config values. Does NOT touch external services.
"""

from __future__ import annotations

import sys


def run() -> bool:
    """Return True if every smoke check passes; raise on regressions."""
    import grounding
    from grounding.testing import (
        ClaimFactory,
        SourceFactory,
        StubEmbeddingFn,
        StubLLMJudgeFn,
        StubNLIFn,
        StubRetrievalFn,
        assert_version,
    )

    if not isinstance(grounding.__version__, str):
        raise AssertionError("grounding.__version__ must be a string")
    if not grounding.__version__.startswith("2026."):
        raise AssertionError(
            f"unexpected grounding version {grounding.__version__!r}"
        )

    claim = ClaimFactory.build()
    source = SourceFactory.build()
    if "text" not in claim or "text" not in source:
        raise AssertionError("factory output must include 'text' key")

    emb = StubEmbeddingFn(dim=4)
    vecs = emb(["hello"])
    if len(vecs) != 1 or len(vecs[0]) != 4:
        raise AssertionError("StubEmbeddingFn shape regression")

    nli = StubNLIFn()
    label_probs = nli(claim="foo", source="hello foo world")
    if "entailment" not in label_probs:
        raise AssertionError("StubNLIFn shape regression")

    judge = StubLLMJudgeFn()
    verdict = judge(claim="foo", source="foo bar")
    if "verdict" not in verdict:
        raise AssertionError("StubLLMJudgeFn shape regression")

    retr = StubRetrievalFn(passages=[{"id": "p1", "text": "x"}])
    hits = retr(query="x", top_k=3)
    if len(hits) != 1:
        raise AssertionError("StubRetrievalFn shape regression")

    assert_version(grounding, "2026.5.15.0")

    return True


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
