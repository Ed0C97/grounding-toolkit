"""Locale-tag driven multilingual grounding.

Real-world documents — DD reports especially — mix languages.  The
toolkit needs a way to verify a claim in language A against a source in
language B without baking in any specific language pair.

This module provides:

- :class:`LocaleGlossary` — a consumer-supplied bidirectional dictionary
  ``term_lang_a <-> term_lang_b`` used to translate canonical terms
  before matching.
- :class:`MultilingualVerifier` — runs lexical matching after
  glossary-driven term translation.

The toolkit ships zero hard-coded glossaries.  Sentinel (or any other
consumer) supplies the locale-pair mapping at construction time.

Configuration shape::

    glossary = LocaleGlossary(
        source_locale="en",
        target_locale="it",
        mappings={
            "loan": ["prestito", "finanziamento"],
            "guarantee": ["garanzia", "fideiussione"],
        },
    )

The glossary is bidirectional — supplying mappings under either
direction works.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from grounding.core.types import (
    Claim,
    EvidencePointer,
    Source,
    TierVerdict,
    Verdict,
)
from grounding.tiers.lexical import LexicalTier


@dataclass
class LocaleGlossary:
    """Bidirectional term map between two locales.

    Mappings are supplied as ``source_term -> [target_term1, ...]``.
    Lookup is performed in both directions; case-insensitive.
    """

    source_locale: str
    target_locale: str
    mappings: Dict[str, List[str]] = field(default_factory=dict)

    def translate(self, term: str) -> List[str]:
        """Return translations for ``term``, in either direction.

        Always includes the original term so a claim that already uses
        the target-locale wording still matches.
        """
        if not term:
            return []
        out: List[str] = [term]
        seen: set[str] = {term.lower()}
        target = term.lower().strip()
        for src, dst_list in self.mappings.items():
            if src.lower().strip() == target:
                for d in dst_list:
                    if d.lower() not in seen:
                        seen.add(d.lower())
                        out.append(d)
            for d in dst_list:
                if d.lower().strip() == target:
                    if src.lower() not in seen:
                        seen.add(src.lower())
                        out.append(src)
        return out

    def supports(self, locale: str) -> bool:
        return locale in (self.source_locale, self.target_locale)


def _expand_text(text: str, glossary: LocaleGlossary) -> List[str]:
    """Return a small list of text variants with a single term swapped.

    Each variant replaces one occurrence of a known term with one of its
    translations.  We deliberately bound the explosion to avoid
    combinatorial blow-up (max one substitution per variant; cap on
    total variants).
    """
    out: List[str] = [text]
    seen: set[str] = {text}
    max_variants = 50
    lower = text.lower()
    for src, dst_list in glossary.mappings.items():
        for term in [src] + list(dst_list):
            t = term.strip()
            if not t:
                continue
            pattern = re.compile(
                r"\b" + re.escape(t) + r"\b", re.IGNORECASE
            )
            if not pattern.search(lower):
                continue
            for replacement in [src] + list(dst_list):
                if replacement.lower() == t.lower():
                    continue
                variant = pattern.sub(replacement, text)
                if variant in seen:
                    continue
                seen.add(variant)
                out.append(variant)
                if len(out) >= max_variants:
                    return out
    return out


@dataclass
class MultilingualVerifier:
    """Lexical verification with locale-aware term translation."""

    glossary: Optional[LocaleGlossary] = None
    lexical_tier: LexicalTier = field(default_factory=LexicalTier)
    name: str = "multilingual"

    def verify(
        self,
        claim: Claim,
        source: Source,
        *,
        threshold: float = 0.85,
    ) -> TierVerdict:
        if not claim.text or not source.text:
            return TierVerdict(
                name=self.name,
                verdict=Verdict.SKIPPED,
                threshold_used=threshold,
                detail="empty claim or source",
            )

        # Same-locale fast path: just lexical, no translation overhead.
        if (
            self.glossary is None
            or source.language == claim.metadata.get("language", source.language)
        ):
            return self.lexical_tier.verify(
                claim, source, threshold=threshold
            )

        if not self.glossary.supports(source.language):
            return self.lexical_tier.verify(
                claim, source, threshold=threshold
            )

        variants = _expand_text(claim.text, self.glossary)
        evidence: List[EvidencePointer] = []
        best_score = 0.0
        for variant in variants:
            sub_claim = Claim(
                text=variant,
                page=claim.page,
                citation_span=claim.citation_span,
                metadata=dict(claim.metadata),
            )
            sub_result = self.lexical_tier.verify(
                sub_claim, source, threshold=threshold
            )
            if sub_result.score > best_score:
                best_score = sub_result.score
            if sub_result.verdict == Verdict.GROUNDED:
                evidence.extend(sub_result.evidence)
                return TierVerdict(
                    name=self.name,
                    verdict=Verdict.GROUNDED,
                    score=sub_result.score,
                    threshold_used=threshold,
                    evidence=evidence,
                    detail=(
                        f"grounded after translation; "
                        f"variant len={len(variant)}"
                    ),
                )

        return TierVerdict(
            name=self.name,
            verdict=Verdict.UNGROUNDED,
            score=best_score,
            threshold_used=threshold,
            detail=(
                f"best translation variant score={best_score:.3f} "
                f"below {threshold:.3f}"
            ),
        )


__all__ = ["LocaleGlossary", "MultilingualVerifier"]
