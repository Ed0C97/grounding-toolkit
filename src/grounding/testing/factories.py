"""grounding.testing.factories — canonical test object builders.

Each factory exposes ``build(**overrides)`` returning a fresh instance
with sensible defaults. Until P1 wires ``grounding.core.types``,
factories return plain dicts; consumers can adapt as the type surface
solidifies.
"""

from __future__ import annotations

from typing import Any, Dict


class ClaimFactory:
    """Build canonical Claim-shaped dicts.

    Once ``grounding.core.types.Claim`` is implemented (Phase 1), this
    factory will return instances of that dataclass.
    """

    @staticmethod
    def build(**overrides: Any) -> Dict[str, Any]:
        defaults: Dict[str, Any] = dict(
            text="example claim",
            page=1,
            citation_span=None,
        )
        defaults.update(overrides)
        return defaults

    def __call__(self, **overrides: Any) -> Dict[str, Any]:
        return self.build(**overrides)


class SourceFactory:
    """Build canonical Source-shaped dicts."""

    @staticmethod
    def build(**overrides: Any) -> Dict[str, Any]:
        defaults: Dict[str, Any] = dict(
            text="example source corpus",
            tables=[],
            kv_pairs={},
            page_count=1,
            doc_id="doc-test",
            language="en",
        )
        defaults.update(overrides)
        return defaults

    def __call__(self, **overrides: Any) -> Dict[str, Any]:
        return self.build(**overrides)
