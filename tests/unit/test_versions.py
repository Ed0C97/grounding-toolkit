"""Version coherence tests: pyproject + manifest + __init__ must agree."""

from __future__ import annotations

from pathlib import Path

import yaml

import grounding


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_pyproject_version() -> str:
    text = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("version") and "=" in line:
            v = line.split("=", 1)[1].strip().strip('"').strip("'")
            return v
    raise RuntimeError("could not locate version in pyproject.toml")


def _read_manifest_version() -> str:
    p = (
        _REPO_ROOT
        / "src"
        / "grounding"
        / "provisioning"
        / "manifest.yaml"
    )
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return str(data["version"])


def test_pyproject_matches_init() -> None:
    assert _read_pyproject_version() == grounding.__version__


def test_manifest_matches_init() -> None:
    assert _read_manifest_version() == grounding.__version__


def test_calver_format() -> None:
    """Version follows YYYY.M.D.N CalVer."""
    parts = grounding.__version__.split(".")
    assert len(parts) == 4
    year = int(parts[0])
    month = int(parts[1])
    day = int(parts[2])
    patch = int(parts[3])
    assert year >= 2026
    assert 1 <= month <= 12
    assert 1 <= day <= 31
    assert patch >= 0
