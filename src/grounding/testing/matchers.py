"""grounding.testing.matchers — grounding-specific assert_* helpers.

Each helper raises ``AssertionError`` on failure with a human-readable
message, matching the numpy.testing.assert_* convention.
"""

from __future__ import annotations

from typing import Any


def assert_version(module: Any, expected: str) -> None:
    """Assert ``module.__version__ == expected``.

    Used by parity tests to guard against accidental version skew.
    """
    actual = getattr(module, "__version__", None)
    if actual != expected:
        raise AssertionError(
            f"version mismatch on {module!r}: expected {expected!r}, "
            f"got {actual!r}"
        )


def assert_grounded(result: Any) -> None:
    """Assert a GroundingResult-like object has verdict == 'GROUNDED'.

    Accepts both dataclass instances (post-P1) and plain dicts (current).
    """
    verdict = (
        getattr(result, "verdict", None)
        if not isinstance(result, dict)
        else result.get("verdict")
    )
    v = str(verdict).upper() if verdict is not None else ""
    if v != "GROUNDED":
        raise AssertionError(
            f"expected GROUNDED verdict, got {verdict!r}"
        )


def assert_ungrounded(result: Any) -> None:
    """Assert a GroundingResult-like object has verdict == 'UNGROUNDED'."""
    verdict = (
        getattr(result, "verdict", None)
        if not isinstance(result, dict)
        else result.get("verdict")
    )
    v = str(verdict).upper() if verdict is not None else ""
    if v != "UNGROUNDED":
        raise AssertionError(
            f"expected UNGROUNDED verdict, got {verdict!r}"
        )


__all__ = ["assert_version", "assert_grounded", "assert_ungrounded"]
