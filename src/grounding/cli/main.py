"""grounding CLI entry point.

Usage::

    grounding version
    grounding smoke
    grounding manifest

Subcommands are added phase-by-phase (verify, eval, calibrate, ...).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional


def _cmd_version(_: argparse.Namespace) -> int:
    import grounding

    print(grounding.__version__)
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from grounding.testing.smoke import run

    ok = run()
    print("OK" if ok else "FAIL", file=sys.stderr)
    return 0 if ok else 1


def _cmd_manifest(_: argparse.Namespace) -> int:
    from grounding.provisioning import descriptor

    print(json.dumps(descriptor(), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grounding",
        description=(
            "grounding-toolkit CLI: groundedness & hallucination "
            "detection for LLM outputs."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_version = sub.add_parser("version", help="Print toolkit version")
    p_version.set_defaults(func=_cmd_version)

    p_smoke = sub.add_parser("smoke", help="Run the 10-second smoke test")
    p_smoke.set_defaults(func=_cmd_smoke)

    p_manifest = sub.add_parser(
        "manifest", help="Print the provisioning manifest as JSON"
    )
    p_manifest.set_defaults(func=_cmd_manifest)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
