"""Locale-aware numeric extractor.

Extracts numeric tokens from free-form text and normalises them to
floats, handling:

- US format: ``1,234.56`` (comma thousands, dot decimal)
- EU/IT format: ``1.234,56`` (dot thousands, comma decimal)
- Magnitude suffixes: ``K``, ``M``, ``B`` (case-insensitive),
  ``mln``, ``mld`` (Italian)
- Leading currency: ``EUR``, ``USD``, ``GBP``, ``CHF``, ``$``, ``€``,
  ``£``, ``¥`` (currency is captured as metadata, not part of the value)
- Trailing currency: ``8.4M €``, ``1234.56 USD``
- Percentages: ``4.5%`` (value 4.5, unit ``%``)
- Ratios: ``1.2x`` (value 1.2, unit ``x``)
- Years: ``2025`` (value 2025.0, unit ``year`` if 4-digit)
- Dates: ``31/12/2025`` (value parsed as ``year.month_day``)

Returns a list of :class:`ExtractedNumber` instances each carrying the
original token, its normalised value, the locale it was parsed under,
optional currency / unit metadata, and char offsets back into the
input text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------
# Regex patterns
# ----------------------------------------------------------------------

# Currency tokens (prefix or suffix).  Case-sensitive code patterns first;
# symbols matched as a separate alternation.
_CURRENCY_CODES = (
    "EUR", "USD", "GBP", "CHF", "JPY", "CNY", "CAD", "AUD",
)
_CURRENCY_SYMBOLS = "€$£¥"

# Multiplier suffix.
_MULT_PATTERN = r"(?P<mult>k|K|m|M|b|B|mln|MLN|mld|MLD)"

# Number body: either thousand-grouped (1,234.56 / 1.234,56) or plain
# (12345.67 / 12345 / 8.4).  The thousand-grouped alt requires AT LEAST
# ONE thousand separator group so it doesn't shadow the plain alt for
# values like "5000" (which would otherwise be matched as "500").
_NUM_BODY = (
    r"(?P<num>"
    r"\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?"   # thousand-grouped
    r"|"
    r"\d+(?:[.,]\d+)?"                       # plain
    r")"
)

# Percentage / ratio / year / date — handled as separate patterns.
_PERCENT_RE = re.compile(
    r"(?P<value>-?\d+(?:[.,]\d+)?)\s*%"
)
# Ratio: number followed by 'x' (case-insensitive) NOT followed by another
# alphanumeric. Excludes hex-like things.
_RATIO_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<value>-?\d+(?:[.,]\d+)?)\s*[xX](?![A-Za-z0-9])"
)
# Year: 4-digit integer in the modern range
_YEAR_RE = re.compile(r"(?<!\d)(?P<value>(?:19|20|21)\d{2})(?!\d)")
# Date: dd/mm/yyyy, dd-mm-yyyy, yyyy-mm-dd
_DATE_RE = re.compile(
    r"(?<!\d)"
    r"(?:"
    r"(?P<dd>\d{1,2})[/\-](?P<mm>\d{1,2})[/\-](?P<yyyy>\d{4})"  # dd/mm/yyyy
    r"|"
    r"(?P<yyyy2>\d{4})-(?P<mm2>\d{1,2})-(?P<dd2>\d{1,2})"        # yyyy-mm-dd
    r")"
    r"(?!\d)"
)

# Generic monetary number: optional currency before/after, optional multiplier.
_MONEY_RE = re.compile(
    rf"(?P<pre>(?:{'|'.join(_CURRENCY_CODES)})|[{_CURRENCY_SYMBOLS}])?\s*"
    rf"{_NUM_BODY}\s*"
    rf"(?:{_MULT_PATTERN})?\s*"
    rf"(?P<post>(?:{'|'.join(_CURRENCY_CODES)})|[{_CURRENCY_SYMBOLS}])?",
)

_MULT_MAP: Dict[str, float] = {
    "k": 1e3, "K": 1e3,
    "m": 1e6, "M": 1e6,
    "b": 1e9, "B": 1e9,
    "mln": 1e6, "MLN": 1e6,
    "mld": 1e9, "MLD": 1e9,
}


# ----------------------------------------------------------------------
# Output types
# ----------------------------------------------------------------------


@dataclass
class ExtractedNumber:
    """A normalised numeric token extracted from text."""

    raw: str
    value: float
    char_start: int
    char_end: int
    unit: str = ""              # "%", "x", "year", "date", "" for monetary/plain
    currency: str = ""          # "EUR", "USD", "$", "" otherwise
    multiplier: str = ""        # "M", "B", "mln", ...
    locale: str = "auto"        # "us", "eu", "auto"
    extras: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Normalisation helpers
# ----------------------------------------------------------------------


def _normalise_number_body(raw: str) -> Optional[float]:
    """Parse a numeric body to float, auto-detecting US vs EU separator."""
    raw = raw.strip()
    if not raw:
        return None
    has_dot = "." in raw
    has_comma = "," in raw

    if has_dot and has_comma:
        # The rightmost separator is the decimal.
        if raw.rfind(".") > raw.rfind(","):
            # US: comma thousands, dot decimal.
            raw = raw.replace(",", "")
        else:
            # EU: dot thousands, comma decimal.
            raw = raw.replace(".", "").replace(",", ".")
    elif has_dot:
        parts = raw.split(".")
        # Heuristic: dot used as thousands when the right part has exactly
        # 3 digits AND the left part has 1-3 digits; ambiguous in
        # isolation, but matches the common EU pattern (e.g. 8.400.000).
        if len(parts) > 2 or (
            len(parts) == 2
            and len(parts[1]) == 3
            and len(parts[0]) <= 3
        ):
            raw = "".join(parts)
        # Otherwise treat as decimal — leave as is.
    elif has_comma:
        parts = raw.split(",")
        if len(parts) > 2 or (
            len(parts) == 2
            and len(parts[1]) == 3
            and len(parts[0]) <= 3
        ):
            raw = "".join(parts)
        else:
            raw = raw.replace(",", ".")

    try:
        return float(raw)
    except ValueError:
        return None


def _resolve_currency(pre: Optional[str], post: Optional[str]) -> str:
    for tok in (pre, post):
        if tok:
            return tok.strip()
    return ""


# ----------------------------------------------------------------------
# Public extractor
# ----------------------------------------------------------------------


@dataclass
class NumberExtractor:
    """Extract :class:`ExtractedNumber` instances from free text.

    Order of detection per text:
    1. Dates (``dd/mm/yyyy`` / ``yyyy-mm-dd``)
    2. Years (4-digit modern years not embedded in a date)
    3. Percentages (``4.5%``)
    4. Ratios (``1.2x``)
    5. Money / plain numbers (with optional currency and multiplier)

    Higher-priority patterns "claim" their character ranges so a 4-digit
    year never gets re-interpreted as a money value.
    """

    min_money_value: float = 0.0
    """Lower bound on the absolute monetary value to retain."""

    name: str = "number_extractor"

    def extract(self, text: str) -> List[ExtractedNumber]:
        out: List[ExtractedNumber] = []
        if not text:
            return out
        claimed: List[tuple[int, int]] = []

        def _claim(start: int, end: int) -> bool:
            for s, e in claimed:
                if not (end <= s or start >= e):
                    return False
            claimed.append((start, end))
            return True

        # 1. Dates
        for m in _DATE_RE.finditer(text):
            if not _claim(m.start(), m.end()):
                continue
            if m.group("yyyy"):
                yyyy = int(m.group("yyyy"))
                mm = int(m.group("mm"))
                dd = int(m.group("dd"))
            else:
                yyyy = int(m.group("yyyy2"))
                mm = int(m.group("mm2"))
                dd = int(m.group("dd2"))
            value = float(yyyy) + (mm / 100.0) + (dd / 10000.0)
            out.append(
                ExtractedNumber(
                    raw=m.group(0),
                    value=value,
                    char_start=m.start(),
                    char_end=m.end(),
                    unit="date",
                    extras={"year": yyyy, "month": mm, "day": dd},
                )
            )

        # 2. Years
        for m in _YEAR_RE.finditer(text):
            if not _claim(m.start(), m.end()):
                continue
            value = float(m.group("value"))
            out.append(
                ExtractedNumber(
                    raw=m.group(0),
                    value=value,
                    char_start=m.start(),
                    char_end=m.end(),
                    unit="year",
                )
            )

        # 3. Percentages
        for m in _PERCENT_RE.finditer(text):
            if not _claim(m.start(), m.end()):
                continue
            v = _normalise_number_body(m.group("value"))
            if v is None:
                continue
            out.append(
                ExtractedNumber(
                    raw=m.group(0),
                    value=v,
                    char_start=m.start(),
                    char_end=m.end(),
                    unit="%",
                )
            )

        # 4. Ratios
        for m in _RATIO_RE.finditer(text):
            if not _claim(m.start(), m.end()):
                continue
            v = _normalise_number_body(m.group("value"))
            if v is None:
                continue
            out.append(
                ExtractedNumber(
                    raw=m.group(0),
                    value=v,
                    char_start=m.start(),
                    char_end=m.end(),
                    unit="x",
                )
            )

        # 5. Money / plain numbers
        for m in _MONEY_RE.finditer(text):
            if not _claim(m.start(), m.end()):
                continue
            num_raw = m.group("num")
            if not num_raw:
                continue
            base = _normalise_number_body(num_raw)
            if base is None:
                continue
            mult_token = m.group("mult") or ""
            mult_factor = _MULT_MAP.get(mult_token, 1.0)
            value = base * mult_factor
            if abs(value) < self.min_money_value:
                continue
            currency = _resolve_currency(m.group("pre"), m.group("post"))
            out.append(
                ExtractedNumber(
                    raw=m.group(0).strip(),
                    value=value,
                    char_start=m.start(),
                    char_end=m.end(),
                    unit="",
                    currency=currency,
                    multiplier=mult_token,
                )
            )

        out.sort(key=lambda n: n.char_start)
        return out


def numbers_match(
    a: float, b: float, *, tolerance: float = 0.05
) -> bool:
    """Return True if two values are within ``tolerance`` relative error.

    Tolerance is relative to ``max(|a|, |b|, 1e-9)`` so very small values
    don't silently pass.
    """
    ref = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / ref <= tolerance


__all__ = [
    "ExtractedNumber",
    "NumberExtractor",
    "numbers_match",
]
