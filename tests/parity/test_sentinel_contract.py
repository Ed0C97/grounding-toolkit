"""Cross-toolkit parity: sentinel contract mirror.

Sentinel's grounding adapter (in ``sentinel/adapters/grounding.py``)
expects a stable surface from ``grounding-toolkit``. The tests in this
file mirror Sentinel's expectations so a regression in the toolkit
breaks here BEFORE it reaches Sentinel CI.

Phase 0: only verifies the import surface. Real parity arrives with
P16-P17 (Sentinel migration + adapter).
"""

from __future__ import annotations

import importlib


_REQUIRED_SUBMODULES = [
    "grounding",
    "grounding.core",
    "grounding.tiers",
    "grounding.citations",
    "grounding.multimodal",
    "grounding.numerical",
    "grounding.temporal",
    "grounding.definitional",
    "grounding.crossdoc",
    "grounding.language",
    "grounding.explainability",
    "grounding.confidence",
    "grounding.audit",
    "grounding.adversarial",
    "grounding.calibration",
    "grounding.eval",
    "grounding.constitutional",
    "grounding.consensus",
    "grounding.tracking",
    "grounding.spatial",
    "grounding.answer",
    "grounding.testing",
    "grounding.provisioning",
    "grounding.cli",
]


def test_required_submodules_importable() -> None:
    """Every domain submodule scaffolded in P0 must import cleanly."""
    for name in _REQUIRED_SUBMODULES:
        mod = importlib.import_module(name)
        assert mod is not None, f"{name} import returned None"
