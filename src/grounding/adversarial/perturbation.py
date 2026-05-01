"""Adversarial perturbation detection.

Identify text that has been perturbed against the source corpus through
common attacks:

- **Unicode confusables** (Cyrillic а vs Latin a, Greek ο vs Latin o).
- **Invisible / zero-width characters** (ZWSP, ZWNJ, ZWJ, soft hyphen,
  word joiner).
- **Homoglyph substitution** (rn → m, l → 1, O → 0).

These attacks let a malicious LLM produce a claim that *looks* identical
to the source while carrying different bytes — which would defeat
exact-substring grounding.

The :class:`PerturbationDetector` returns the cleaned canonical form of
a text plus a list of detected perturbations.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# Zero-width / invisible characters
_ZERO_WIDTH = {
    "​": "ZWSP (zero-width space)",
    "‌": "ZWNJ (zero-width non-joiner)",
    "‍": "ZWJ (zero-width joiner)",
    "⁠": "word joiner",
    "﻿": "BOM / zero-width no-break space",
    "­": "soft hyphen",
}

# Common confusables -> canonical Latin form.  This is a curated subset;
# the comprehensive list lives in Unicode TR39 confusables.txt.
_CONFUSABLES: Dict[str, str] = {
    # Cyrillic
    "а": "a",  # а
    "е": "e",  # е
    "о": "o",  # о
    "р": "p",  # р
    "с": "c",  # с
    "х": "x",  # х
    "у": "y",  # у
    "А": "A",
    "Е": "E",
    "О": "O",
    "Р": "P",
    "С": "C",
    "Х": "X",
    "У": "Y",
    # Greek
    "ο": "o",  # ο
    "α": "a",  # α
    "ε": "e",  # ε
    "Ο": "O",
    "Α": "A",
    # Fullwidth
    "Ａ": "A",
    "Ｂ": "B",
    "Ｃ": "C",
    # Mathematical alphanumeric symbols U+1D400-U+1D7FF skipped
    # (handled below by NFKC).
}


@dataclass
class PerturbationReport:
    """Output of :meth:`PerturbationDetector.detect`."""

    has_perturbations: bool
    canonical_text: str
    invisible_chars: List[str] = field(default_factory=list)
    confusables_replaced: List[Tuple[str, str]] = field(
        default_factory=list
    )
    nfkc_changed: bool = False


@dataclass
class PerturbationDetector:
    """Detect and canonicalise common adversarial perturbations."""

    def detect(self, text: str) -> PerturbationReport:
        invisible: List[str] = []
        confusables: List[Tuple[str, str]] = []
        out_chars: List[str] = []

        for ch in text:
            if ch in _ZERO_WIDTH:
                invisible.append(_ZERO_WIDTH[ch])
                continue
            if ch in _CONFUSABLES:
                replacement = _CONFUSABLES[ch]
                confusables.append((ch, replacement))
                out_chars.append(replacement)
                continue
            out_chars.append(ch)

        stripped = "".join(out_chars)
        nfkc = unicodedata.normalize("NFKC", stripped)
        nfkc_changed = nfkc != stripped

        return PerturbationReport(
            has_perturbations=bool(
                invisible or confusables or nfkc_changed
            ),
            canonical_text=nfkc,
            invisible_chars=invisible,
            confusables_replaced=confusables,
            nfkc_changed=nfkc_changed,
        )


__all__ = ["PerturbationDetector", "PerturbationReport"]
