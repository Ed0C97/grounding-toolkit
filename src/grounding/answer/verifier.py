"""RAG-answer groundedness verifier.

Migrated from Sentinel's ``sentinel.agents.chat_verifier.ChatVerifier``
(P16 hard cutover).  Provider-agnostic: the LLM judge call is supplied
by the consumer via a ``chat_fn`` callable injected at construction.

The verifier:

1. Renders the retrieved chunks + answer into a structured prompt.
2. Asks the LLM judge to return ``{"grounded": bool,
   "ungrounded_claims": [...]}``.
3. Falls back to ``{"grounded": True, "ungrounded_claims": []}`` on any
   judge failure (consumer can add stricter handling around the
   verifier).
4. Optionally enriches the verdict with a deterministic feedback
   bundle (grounding / relevance / harmfulness scores) via
   :mod:`grounding.eval.rag_feedback`.

The judge's contract: an async callable receiving keyword arguments
``system_prompt``, ``user_message`` (and optional ``temperature`` and
``model``) and returning the parsed JSON dict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol

from grounding.eval.rag_feedback import evaluate_bundle

logger = logging.getLogger(__name__)


_DEFAULT_SYSTEM_PROMPT = (
    "You are a groundedness verifier. Check whether the ANSWER is fully "
    "supported by the DOCUMENT EXCERPTS. Identify any claims that are "
    "NOT present in the excerpts.\n\n"
    'Return JSON: {"grounded": true/false, "ungrounded_claims": ["claim1", ...]}\n'
    "- grounded=true: all claims in the answer are supported by the excerpts.\n"
    "- grounded=false: some claims lack support. List them in ungrounded_claims."
)


class ChatJsonFn(Protocol):
    """Async callable that takes a system + user prompt and returns
    parsed JSON.  Mirrors the signature of Sentinel's
    ``sentinel.agents.cohere_client.chat_json`` so the migration is
    drop-in for the existing call site."""

    async def __call__(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]: ...


@dataclass
class AnswerVerifier:
    """RAG-answer groundedness verifier.

    Args:
        chat_fn: async callable conforming to :class:`ChatJsonFn`.
        model: optional model identifier passed to ``chat_fn``.
        system_prompt: override the default verifier prompt.
        attach_feedback_bundle: when True (default), enrich the
            verdict with deterministic grounding / relevance /
            harmfulness scores under ``feedback_scores``.
        max_chunks: cap the number of chunks rendered into the prompt.
        max_answer_chars: cap the answer length rendered into the prompt.
        max_chunk_chars: cap each chunk text length rendered into the prompt.
    """

    chat_fn: Optional[ChatJsonFn] = None
    model: Optional[str] = None
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    attach_feedback_bundle: bool = True
    max_chunks: int = 8
    max_answer_chars: int = 3000
    max_chunk_chars: int = 500

    async def verify(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
        query: str,
    ) -> Dict[str, Any]:
        chunks_text = "\n".join(
            f"[Chunk {i + 1}]: "
            f"{str(c.get('text', ''))[: self.max_chunk_chars]}"
            for i, c in enumerate(chunks[: self.max_chunks])
        )
        user_msg = (
            f"QUERY: {query}\n\n"
            f"ANSWER TO VERIFY:\n{answer[: self.max_answer_chars]}\n\n"
            f"DOCUMENT EXCERPTS:\n{chunks_text}"
        )

        verdict: Dict[str, Any] = {
            "grounded": True,
            "ungrounded_claims": [],
        }
        if self.chat_fn is not None:
            try:
                resp = await self.chat_fn(
                    system_prompt=self.system_prompt,
                    user_message=user_msg,
                    model=self.model,
                    temperature=0.0,
                )
                if isinstance(resp, dict):
                    verdict = {
                        "grounded": bool(resp.get("grounded", True)),
                        "ungrounded_claims": list(
                            resp.get("ungrounded_claims", []) or []
                        ),
                    }
            except Exception as exc:
                logger.warning(
                    "[answer-verifier] chat_fn raised: %s", exc
                )

        if self.attach_feedback_bundle:
            try:
                sources = [
                    str(c.get("text", ""))
                    for c in chunks[: self.max_chunks]
                ]
                scores = evaluate_bundle(
                    query=query, answer=answer, sources=sources
                )
                verdict["feedback_scores"] = {
                    name: score.to_dict()
                    for name, score in scores.items()
                }
            except Exception as exc:
                logger.debug(
                    "[answer-verifier] feedback bundle skipped: %s", exc
                )

        return verdict


__all__ = ["AnswerVerifier", "ChatJsonFn"]
