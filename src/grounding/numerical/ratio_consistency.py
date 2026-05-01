"""Cross-formula ratio consistency.

When the consumer asserts multiple derived values at once (e.g. several
ratios computed from a shared set of primitives), this module batches
the verification and reports overall consistency.  Useful as a one-shot
check: "given this set of grounded primitives, are ALL the claimed
ratios internally consistent?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from grounding.numerical.derivation_check import (
    DerivationCheck,
    DerivationResult,
    DerivationVerifier,
)


@dataclass
class RatioConsistencyReport:
    """Aggregated outcome of :meth:`RatioConsistencyVerifier.verify`."""

    ok: bool
    results: List[DerivationResult] = field(default_factory=list)

    def failed(self) -> List[DerivationResult]:
        return [r for r in self.results if not r.ok]

    def summary(self) -> str:
        if self.ok:
            return f"all {len(self.results)} ratios consistent"
        bad = self.failed()
        names = ", ".join(r.formula_name for r in bad[:5])
        return (
            f"{len(bad)}/{len(self.results)} ratios inconsistent: "
            f"{names}"
        )


@dataclass
class RatioConsistencyVerifier:
    """Run a batch of :class:`DerivationCheck` and aggregate the verdict."""

    inner: DerivationVerifier = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.inner is None:
            self.inner = DerivationVerifier()

    def verify(
        self, checks: List[DerivationCheck]
    ) -> RatioConsistencyReport:
        results: List[DerivationResult] = []
        for c in checks:
            results.append(self.inner.verify(c))
        return RatioConsistencyReport(
            ok=all(r.ok for r in results),
            results=results,
        )


__all__ = ["RatioConsistencyReport", "RatioConsistencyVerifier"]
