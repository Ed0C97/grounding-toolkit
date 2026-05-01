"""Structured emission templates for preventive citations.

Provides:

- :class:`GroundedClaim` / :class:`GroundedFindings` Pydantic models
  for downstream LLM-output validation.
- :func:`grounded_claim_system_prompt` instructional template.
- :class:`GroundedClaimSignature` DSPy Signature (only when ``dspy`` is
  installed; otherwise the symbol is set to ``None``).

The consumer wraps their LLM call with these so every emitted claim
arrives already annotated with a span pointing at the source location
the LLM believes supports the statement.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GroundedClaimSpan(BaseModel):
    """Pydantic shape of an LLM-emitted citation span."""

    page: int = Field(..., ge=1, description="1-indexed page number")
    char_start: int = Field(..., ge=0)
    char_end: int = Field(..., ge=0)


class GroundedClaim(BaseModel):
    """A claim with its preventive citation."""

    text: str
    citation_span: GroundedClaimSpan


class GroundedFindings(BaseModel):
    """A list of grounded claims (e.g. DD findings)."""

    findings: List[GroundedClaim]


_EN_PROMPT = (
    "For every claim you produce, you MUST attach a citation_span "
    "pointing at the source passage that supports the claim. The "
    "citation_span is a JSON object with fields: page (1-indexed), "
    "char_start (0-indexed inclusive), char_end (0-indexed exclusive). "
    "The text between char_start and char_end on that page must "
    "contain the evidence for the claim. If you cannot identify a "
    "supporting span, do NOT produce the claim."
)
_IT_PROMPT = (
    "Per ogni affermazione che produci, DEVI allegare un "
    "citation_span che punti al passaggio sorgente che la supporta. "
    "Il citation_span e' un oggetto JSON con i campi: page "
    "(1-indexed), char_start (0-indexed incluso), char_end (0-indexed "
    "escluso). Il testo tra char_start e char_end in quella pagina "
    "deve contenere l'evidenza per l'affermazione. Se non riesci a "
    "identificare uno span di supporto, NON produrre l'affermazione."
)


def grounded_claim_system_prompt(*, language: str = "en") -> str:
    """Return the system-prompt fragment for preventive citation."""
    if language.lower().startswith("it"):
        return _IT_PROMPT
    return _EN_PROMPT


# Optional DSPy signature.  Only available when the ``dspy`` extra is
# installed; otherwise the symbol is exported as ``None`` so consumers
# can do ``if GroundedClaimSignature is None: ...``.
GroundedClaimSignature: Optional[type] = None
try:
    import dspy  # type: ignore[import-not-found]

    class _GroundedClaimSignature(dspy.Signature):  # type: ignore[misc]
        """Produce a claim with a verifying citation_span.

        The model output ``citation_span`` must be a JSON string
        matching :class:`GroundedClaimSpan`.
        """

        source: str = dspy.InputField(  # type: ignore[attr-defined]
            desc="Source document text"
        )
        question: str = dspy.InputField(  # type: ignore[attr-defined]
            desc="Question or task"
        )
        claim: str = dspy.OutputField(  # type: ignore[attr-defined]
            desc="The claim text"
        )
        citation_span: str = dspy.OutputField(  # type: ignore[attr-defined]
            desc=(
                'JSON: {"page": int, "char_start": int, "char_end": int}'
            )
        )

    GroundedClaimSignature = _GroundedClaimSignature
except ImportError:
    pass


__all__ = [
    "GroundedClaimSpan",
    "GroundedClaim",
    "GroundedFindings",
    "GroundedClaimSignature",
    "grounded_claim_system_prompt",
]
