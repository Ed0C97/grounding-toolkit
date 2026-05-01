"""grounding.testing — public test framework for grounding-toolkit consumers.

Modelled on numpy.testing and sklearn.utils.estimator_checks: stubs,
factories, fixtures, matchers, and a smoke runner that downstream
consumers (Sentinel and others) plug into their own test suites.

Public modules:

- ``stubs``      : Stub<Backend> classes implementing grounding Protocols.
- ``factories``  : Helpers building canonical dataclass instances.
- ``fixtures``   : pytest plugin (``pytest_plugins = ["grounding.testing.fixtures"]``).
- ``matchers``   : ``assert_*`` helpers for grounding-specific invariants.
- ``smoke``      : ``run() -> bool``, the 10-second self-check.

The fixtures module imports pytest lazily so non-pytest consumers can
import grounding.testing.{stubs, factories, matchers, smoke} without a
test-runner dependency.
"""

from __future__ import annotations

from grounding.testing.factories import (
    ClaimFactory,
    SourceFactory,
)
from grounding.testing.matchers import assert_version
from grounding.testing.stubs import (
    StubEmbeddingFn,
    StubLLMJudgeFn,
    StubNLIFn,
    StubRetrievalFn,
)

__all__ = [
    # Stubs
    "StubEmbeddingFn",
    "StubLLMJudgeFn",
    "StubNLIFn",
    "StubRetrievalFn",
    # Factories
    "ClaimFactory",
    "SourceFactory",
    # Matchers
    "assert_version",
]
