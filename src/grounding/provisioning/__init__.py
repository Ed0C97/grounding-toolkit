"""grounding.provisioning — Sentinel TestRunnerService bridge.

Exposes:

- ``descriptor() -> dict`` : the manifest as a Python dict.
- ``runner(group: str)``   : execute a manifest group, returning junit-xml.

Sentinel discovers this module via the ``sentinel_provisioning`` entry
point declared in pyproject.toml.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_HERE = Path(__file__).resolve().parent
_MANIFEST_PATH = _HERE / "manifest.yaml"


def descriptor() -> Dict[str, Any]:
    """Return the manifest contents as a dict.

    Shape::

        {
            "toolkit": "grounding-toolkit",
            "version": "...",
            "groups": [{"id": "...", "name": "...", "path": "...", ...}, ...]
        }
    """
    with _MANIFEST_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise RuntimeError(
            f"grounding manifest must be a mapping, got {type(data)!r}"
        )
    if "toolkit" not in data or "groups" not in data:
        raise RuntimeError(
            "grounding manifest missing required keys (toolkit, groups)"
        )
    return data


def runner(group: str, *, junit_path: Optional[Path] = None) -> str:
    """Execute the named group, returning junit-xml as a string.

    For ``runner: <module>:<callable>`` groups, calls the callable
    directly; the return value is wrapped in a minimal junit-xml document
    indicating success/failure.

    For ``path: <relpath>`` groups, runs ``pytest <relpath>`` against the
    grounding-toolkit checkout (resolved from this file's location).
    """
    desc = descriptor()
    groups = {g["id"]: g for g in desc.get("groups", [])}
    if group not in groups:
        raise KeyError(f"unknown group {group!r}; valid: {sorted(groups)}")
    spec = groups[group]

    if "runner" in spec:
        return _run_callable(spec["runner"])
    if "path" in spec:
        return _run_pytest(spec["path"], junit_path=junit_path)
    raise RuntimeError(f"group {group!r} has neither 'runner' nor 'path'")


def _run_callable(target: str) -> str:
    module_name, _, attr = target.partition(":")
    if not module_name or not attr:
        raise RuntimeError(
            f"invalid runner target {target!r}; expected 'module:callable'"
        )
    mod = importlib.import_module(module_name)
    fn = getattr(mod, attr)
    try:
        ok = bool(fn())
    except Exception as exc:
        return _junit_doc(group=target, ok=False, message=str(exc))
    return _junit_doc(
        group=target, ok=ok, message="" if ok else "callable returned False"
    )


def _run_pytest(rel_path: str, *, junit_path: Optional[Path]) -> str:
    repo_root = _HERE.parents[2]
    target = (repo_root / rel_path).resolve()
    if not target.exists():
        raise FileNotFoundError(f"pytest target {target} does not exist")
    junit = junit_path or (
        repo_root
        / ".pytest_cache"
        / f"junit-{rel_path.replace('/', '_')}.xml"
    )
    junit.parent.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        sys.executable,
        "-m",
        "pytest",
        str(target),
        f"--junitxml={junit}",
        "-q",
    ]
    proc = subprocess.run(
        cmd, cwd=str(repo_root), capture_output=True, text=True
    )
    if junit.exists():
        with junit.open("r", encoding="utf-8") as fh:
            return fh.read()
    return _junit_doc(
        group=rel_path,
        ok=proc.returncode == 0,
        message=(
            f"pytest exit={proc.returncode}; "
            f"stderr={proc.stderr[-400:]!r}"
        ),
    )


def _junit_doc(*, group: str, ok: bool, message: str) -> str:
    failures = "0" if ok else "1"
    body = "" if ok else f"<failure message={message!r}/>"
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<testsuites tests="1" failures="{failures}">'
        f'<testsuite name="grounding.provisioning" tests="1" '
        f'failures="{failures}">'
        f'<testcase classname="grounding.provisioning" name="{group}">'
        f"{body}"
        "</testcase></testsuite></testsuites>"
    )


__all__ = ["descriptor", "runner"]
