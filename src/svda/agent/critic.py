"""Reflection / critic (Ch. 4) - the piece the whole project turns on.

The critic re-reads the draft against the retrieved chunks and decides whether
each claim is actually supported. The pipeline then regenerates, declines, or
escalates to human review. This is the mechanism behind "I can't support this"
rather than a confidently wrong answer.
"""

from __future__ import annotations

from typing import Any

from ..config import get_settings
from ..llm import LLMClient, LLMRequest, Task, Tier
from ..reliability import Degradation, parse_json_lenient
from ..schemas import ClaimCheck, CriticAction, CriticVerdict, Draft
from ..tracing import Tracer
from . import prompts


class Critic:
    def __init__(self, llm: LLMClient, tracer: Tracer) -> None:
        self.llm = llm
        self.tracer = tracer
        self.settings = get_settings()

    async def check(
        self,
        query: str,
        draft: Draft,
        contexts: list[dict[str, Any]],
        *,
        degradations: list[Degradation] | None = None,
    ) -> CriticVerdict:
        with self.tracer.span("critic", sentences=len(draft.sentences)) as span:
            if not draft.sentences:
                verdict = CriticVerdict(
                    faithfulness=0.0,
                    citation_coverage=0.0,
                    action=CriticAction.DECLINE,
                    reason="draft contained no claims",
                )
                span.attributes["action"] = verdict.action.value
                return verdict

            chunk_to_block = {ctx["chunk_id"]: i for i, ctx in enumerate(contexts, start=1)}
            payload_sentences = [
                {
                    "text": s.text,
                    "citations": [chunk_to_block[cid] for cid in s.citation_ids if cid in chunk_to_block],
                }
                for s in draft.sentences
            ]

            request = LLMRequest(
                task=Task.CRITIQUE,
                prompt=prompts.critique_prompt(query, contexts, payload_sentences),
                system=prompts.SYSTEM_BASE,
                payload={"query": query, "contexts": contexts, "sentences": payload_sentences},
                tier=Tier.STRONG,  # the critic is the last line of defence
                max_tokens=2048,
            )

            try:
                response = await self.llm.complete(request)
                data = parse_json_lenient(response.text)
                verdict = self._assemble(data, draft, contexts, checked_by="llm")
            except Exception as exc:  # noqa: BLE001
                if degradations is not None:
                    degradations.append(Degradation("critic", f"{type(exc).__name__}: {exc}"))
                # Fail closed: an unverifiable answer goes to human review, it
                # is never silently shipped as if it had been checked.
                verdict = CriticVerdict(
                    faithfulness=0.0,
                    citation_coverage=_citation_coverage(draft),
                    action=CriticAction.HUMAN_REVIEW,
                    reason=f"critic unavailable ({type(exc).__name__}); cannot certify the answer",
                    checked_by="fallback",
                )

            span.attributes["action"] = verdict.action.value
            span.attributes["faithfulness"] = round(verdict.faithfulness, 3)
            span.attributes["citation_coverage"] = round(verdict.citation_coverage, 3)
            return verdict

    def _assemble(
        self,
        data: dict[str, Any],
        draft: Draft,
        contexts: list[dict[str, Any]],
        *,
        checked_by: str,
    ) -> CriticVerdict:
        claims: list[ClaimCheck] = []
        for raw in data.get("claims", []):
            evidence = raw.get("evidence")
            chunk_id = None
            if isinstance(evidence, int) and 1 <= evidence <= len(contexts):
                chunk_id = contexts[evidence - 1]["chunk_id"]
            claims.append(
                ClaimCheck(
                    claim=str(raw.get("claim", ""))[:400],
                    supported=bool(raw.get("supported", False)),
                    evidence_chunk_id=chunk_id,
                    reason=str(raw.get("reason", ""))[:200],
                )
            )

        faithfulness = (
            sum(1 for c in claims if c.supported) / len(claims) if claims else 0.0
        )
        coverage = _citation_coverage(draft)

        # The model proposes a verdict; the thresholds decide. Keeping the
        # decision numeric is what makes the ablation table reproducible.
        raw_verdict = str(data.get("verdict", "")).strip().lower()
        if faithfulness >= self.settings.faithfulness_accept_threshold and raw_verdict != "decline":
            action = CriticAction.ACCEPT
        elif faithfulness >= self.settings.human_review_threshold:
            action = CriticAction.REGENERATE
        else:
            action = CriticAction.DECLINE

        return CriticVerdict(
            faithfulness=faithfulness,
            citation_coverage=coverage,
            claims=claims,
            action=action,
            reason=str(data.get("reason", ""))[:200],
            checked_by=checked_by,
        )


def _citation_coverage(draft: Draft) -> float:
    from .guardrails import _needs_citation

    factual = [s for s in draft.sentences if _needs_citation(s.text)]
    if not factual:
        return 1.0 if draft.sentences else 0.0
    return sum(1 for s in factual if s.has_citation) / len(factual)
