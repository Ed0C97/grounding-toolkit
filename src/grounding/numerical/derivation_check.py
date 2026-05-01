"""Generic derivation verifier.

The consumer supplies a formula expression (string) and a dict of
grounded primitive values; the verifier recomputes the expression and
compares the result to the claimed derived value within tolerance.

Domain-agnostic: the toolkit ships no specific formulas (no DSCR, no
LTV, no ICR).  Consumers register their own formulas via
:class:`DerivationFormula` and pass them to :class:`DerivationVerifier`.

Safety: expressions are parsed with :mod:`ast` and walked through a
whitelisted evaluator that supports only ``Number``, ``Name``,
``BinOp(+ - * / // % **)``, and ``UnaryOp(+ -)``.  Function calls,
attribute access, comprehensions, etc. are rejected.

Example::

    f = DerivationFormula(name="DSCR", expression="cf / ds")
    check = DerivationCheck(
        formula=f,
        primitives={"cf": 12.0, "ds": 10.0},
        claimed_value=1.2,
    )
    result = DerivationVerifier().verify(check)
    assert result.ok and abs(result.computed - 1.2) < 0.01
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, Optional

from grounding.numerical.number_extraction import numbers_match


@dataclass(frozen=True)
class DerivationFormula:
    """A named formula expression.

    ``expression`` must be parseable by Python's :mod:`ast` module under
    ``mode="eval"`` and must reference only identifiers that the consumer
    will supply as primitives.
    """

    name: str
    expression: str
    description: str = ""


@dataclass
class DerivationCheck:
    """A single derivation verification input.

    ``primitives`` are the grounded inputs (already verified through
    other tiers); ``claimed_value`` is the LLM-asserted derived value.
    """

    formula: DerivationFormula
    primitives: Dict[str, float]
    claimed_value: float


@dataclass
class DerivationResult:
    """Output of :meth:`DerivationVerifier.verify`."""

    formula_name: str
    expression: str
    primitives: Dict[str, float]
    claimed: float
    computed: Optional[float]
    ok: bool
    error: str = ""

    @property
    def relative_error(self) -> Optional[float]:
        if self.computed is None:
            return None
        ref = max(abs(self.claimed), abs(self.computed), 1e-9)
        return abs(self.claimed - self.computed) / ref


@dataclass
class DerivationVerifier:
    """Recompute a formula from primitives and compare to claimed value."""

    tolerance: float = 0.05
    name: str = "derivation"

    def verify(self, check: DerivationCheck) -> DerivationResult:
        try:
            computed = self._evaluate(
                check.formula.expression, check.primitives
            )
        except (
            SyntaxError,
            NameError,
            ZeroDivisionError,
            ValueError,
            TypeError,
        ) as exc:
            return DerivationResult(
                formula_name=check.formula.name,
                expression=check.formula.expression,
                primitives=dict(check.primitives),
                claimed=check.claimed_value,
                computed=None,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        ok = numbers_match(
            computed, check.claimed_value, tolerance=self.tolerance
        )
        return DerivationResult(
            formula_name=check.formula.name,
            expression=check.formula.expression,
            primitives=dict(check.primitives),
            claimed=check.claimed_value,
            computed=computed,
            ok=ok,
        )

    @staticmethod
    def _evaluate(expr: str, primitives: Dict[str, float]) -> float:
        if not expr or not expr.strip():
            raise SyntaxError("empty expression")
        tree = ast.parse(expr, mode="eval")
        return _eval_node(tree.body, primitives)


# --------------------------------------------------------------------
# Whitelisted AST evaluator
# --------------------------------------------------------------------


def _eval_node(node: ast.AST, primitives: Dict[str, float]) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise SyntaxError("boolean literals not allowed")
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise SyntaxError(
            f"unsupported constant type: {type(node.value).__name__}"
        )
    if isinstance(node, ast.Name):
        if node.id not in primitives:
            raise NameError(f"unknown identifier: {node.id}")
        v = primitives[node.id]
        if not isinstance(v, (int, float)):
            raise TypeError(
                f"primitive {node.id!r} is not numeric: {type(v).__name__}"
            )
        return float(v)
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, primitives)
        right = _eval_node(node.right, primitives)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
        if isinstance(op, ast.FloorDiv):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return float(left // right)
        if isinstance(op, ast.Mod):
            if right == 0:
                raise ZeroDivisionError("modulo by zero")
            return left % right
        if isinstance(op, ast.Pow):
            return float(left**right)
        raise SyntaxError(
            f"unsupported binary operator: {type(op).__name__}"
        )
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, primitives)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise SyntaxError(
            f"unsupported unary operator: {type(node.op).__name__}"
        )
    raise SyntaxError(
        f"unsupported AST node: {type(node).__name__}"
    )


__all__ = [
    "DerivationFormula",
    "DerivationCheck",
    "DerivationResult",
    "DerivationVerifier",
]
