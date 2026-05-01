"""Tier −1 — consensus prior modulation.

This tier doesn't itself verify the claim; it inspects the claim's
``metadata`` for an inter-generator agreement signal (e.g. "3/3 agree",
"single source", "majority dissent") and returns a
:class:`ConsensusPrior`.  The cascade uses that prior to modulate
downstream tier thresholds.

The actual aggregation logic — running N generators in parallel and
counting their votes — lives elsewhere (the consumer or
``grounding.consensus.quorum`` once P16 lands).  This tier is a pure
mapper: ``metadata → ConsensusPrior``.

Recognised metadata keys (case-insensitive):
    ``consensus``, ``moa_consensus``, ``consensus_prior``, ``agreement``

Recognised values (case-insensitive):
    - CONFIRMED, UNANIMOUS, MAJORITY, AGREE, AGREEMENT → CONFIRMED
    - SINGLE, NONE, ONE                                 → SINGLE
    - DISAGREEMENT, DISSENT, MINORITY_DISSENT, CONFLICT → DISAGREEMENT
    - anything else / missing                            → UNKNOWN
"""

from __future__ import annotations

from dataclasses import dataclass

from grounding.core.types import (
    Claim,
    ConsensusPrior,
    Source,
    TierVerdict,
    Verdict,
)


_KEY_VARIANTS = (
    "consensus",
    "moa_consensus",
    "consensus_prior",
    "agreement",
)

_CONFIRMED = {"CONFIRMED", "UNANIMOUS", "MAJORITY", "AGREE", "AGREEMENT"}
_SINGLE = {"SINGLE", "NONE", "ONE"}
_DISAGREEMENT = {
    "DISAGREEMENT",
    "DISSENT",
    "MINORITY_DISSENT",
    "CONFLICT",
}


@dataclass
class ConsensusTier:
    """Tier −1: consensus prior extractor."""

    name: str = "consensus"

    def extract_prior(self, claim: Claim) -> ConsensusPrior:
        """Read the consensus signal from claim metadata."""
        meta = claim.metadata or {}
        for key in _KEY_VARIANTS:
            if key in meta:
                return self._parse(meta[key])
        return ConsensusPrior.UNKNOWN

    @staticmethod
    def _parse(raw: object) -> ConsensusPrior:
        if isinstance(raw, ConsensusPrior):
            return raw
        s = str(raw).strip().upper()
        if s in _CONFIRMED:
            return ConsensusPrior.CONFIRMED
        if s in _SINGLE:
            return ConsensusPrior.SINGLE
        if s in _DISAGREEMENT:
            return ConsensusPrior.DISAGREEMENT
        return ConsensusPrior.UNKNOWN

    def verify(self, claim: Claim, source: Source) -> TierVerdict:
        """Return a metadata-only :class:`TierVerdict` carrying the prior.

        This tier never decides on its own — the cascade uses the
        extracted prior to modulate downstream thresholds.  We emit a
        SKIPPED verdict so that aggregators ignore this tier when
        computing the final verdict.
        """
        prior = self.extract_prior(claim)
        return TierVerdict(
            name=self.name,
            verdict=Verdict.SKIPPED,
            score=0.0,
            threshold_used=0.0,
            detail=f"consensus prior: {prior.value}",
        )


__all__ = ["ConsensusTier"]
