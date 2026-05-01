"""grounding.adversarial — adversarial perturbation detection + robustness."""

from __future__ import annotations

from grounding.adversarial.perturbation import (
    PerturbationDetector,
    PerturbationReport,
)
from grounding.adversarial.robustness import (
    RobustnessChecker,
    RobustnessResult,
)

__all__ = [
    "PerturbationDetector",
    "PerturbationReport",
    "RobustnessChecker",
    "RobustnessResult",
]
