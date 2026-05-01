"""Example 01 — Phase 0 smoke check.

Run::

    python examples/01_smoke.py

Verifies the toolkit is installed and the public surface is reachable.
The substantive examples (lexical cascade, preventive grounding,
multimodal, explainability) arrive in later phases.
"""

from __future__ import annotations


def main() -> None:
    import grounding
    from grounding.testing.smoke import run

    print(f"grounding-toolkit version: {grounding.__version__}")
    ok = run()
    print(f"smoke check: {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
