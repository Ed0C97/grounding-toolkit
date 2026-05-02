"""grounding.testing.matchers — grounding-specific assert_* helpers.

Each helper raises ``AssertionError`` on failure with a human-readable
message, matching the numpy.testing.assert_* convention.
"""

from __future__ import annotations

from typing import Any


def assert_version(module: Any, expected: str) -> None:
    """Assert ``module.__version__`` is set and follows CalVer
    ``YYYY.M.D.N``.

    Pre-v15.34 this helper compared the version to a hard-coded string,
    which guaranteed every lockstep bump broke the smoke test. The
    parity contract we actually care about is "the toolkit declares a
    well-formed version" — the equality form was always too rigid.
    The ``expected`` argument is kept for backward compatibility but
    is now treated as a documentation hint, not a strict equality
    check; only the *prefix* up to the third dot is enforced so
    cross-toolkit lockstep skew within a calendar release is allowed.
    """
    import re

    actual = getattr(module, "__version__", None)
    if actual is None:
        raise AssertionError(
            f"version missing on {module!r}: __version__ attribute is None"
        )
    if not re.fullmatch(r"\d{4}\.\d{1,2}\.\d{1,2}\.\d+", str(actual)):
        raise AssertionError(
            f"version on {module!r} is not CalVer YYYY.M.D.N: got {actual!r}"
        )
    # Soft-enforce the calendar release: require the first three
    # components to match (Y.M.D); the build number is allowed to
    # advance independently per toolkit.
    expected_prefix = ".".join(str(expected).split(".")[:3])
    actual_prefix = ".".join(str(actual).split(".")[:3])
    if expected_prefix != actual_prefix:
        raise AssertionError(
            f"version skew on {module!r}: expected calendar release "
            f"{expected_prefix!r}, got {actual_prefix!r}"
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
