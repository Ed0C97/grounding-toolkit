"""grounding.testing.fixtures — pytest plugin for grounding consumers.

Enable in your project's ``conftest.py``::

    pytest_plugins = ["grounding.testing.fixtures"]

Fixtures provided:

- ``grounding_claim``   : Claim dict, scope=function.
- ``grounding_source``  : Source dict, scope=function.
- ``stub_embedding``    : :class:`StubEmbeddingFn` instance.
- ``stub_nli``          : :class:`StubNLIFn` instance.
- ``stub_llm_judge``    : :class:`StubLLMJudgeFn` instance.
- ``stub_retrieval``    : :class:`StubRetrievalFn` instance.
"""

from __future__ import annotations

import pytest

from grounding.testing.factories import ClaimFactory, SourceFactory
from grounding.testing.stubs import (
    StubEmbeddingFn,
    StubLLMJudgeFn,
    StubNLIFn,
    StubRetrievalFn,
)


@pytest.fixture
def grounding_claim() -> dict:
    return ClaimFactory.build()


@pytest.fixture
def grounding_source() -> dict:
    return SourceFactory.build()


@pytest.fixture
def stub_embedding() -> StubEmbeddingFn:
    return StubEmbeddingFn()


@pytest.fixture
def stub_nli() -> StubNLIFn:
    return StubNLIFn()


@pytest.fixture
def stub_llm_judge() -> StubLLMJudgeFn:
    return StubLLMJudgeFn()


@pytest.fixture
def stub_retrieval() -> StubRetrievalFn:
    return StubRetrievalFn()
