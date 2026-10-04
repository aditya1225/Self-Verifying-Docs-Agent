"""The agent graph.

    query
      -> input guardrails (Ch. 18)
      -> router (Ch. 2) -----------------> chitchat / out-of-scope short circuits
      -> planner (Ch. 6, multi-hop only)
      -> parallel hybrid retrieval + RRF + re-rank (Ch. 3)
      -> generator with citations (Ch. 14)
      -> critic / reflection (Ch. 4) ----> regenerate | decline | human review
      -> output guardrails (Ch. 18)
      -> answer

`Profile` turns individual stages off, which is what produces the ablation
table: the same corpus, golden set and metrics, four configurations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..config import get_settings
from ..index.store import CorpusIndex
from ..llm import LLMClient, Tier, build_llm
from ..reliability import Degradation
from ..retrieval.hybrid import hybrid_retrieve
from ..schemas import (
    AgentAnswer,
    AnswerStatus,
    Citation,
    CriticAction,
    CriticVerdict,
    Draft,
    GuardrailStage,
    GuardrailVerdict,
    Route,
    RouteDecision,
    Usage,
)
from ..tracing import Tracer
from .critic import Critic
from .generator import Generator, build_contexts
from .guardrails import check_output, scan_input
from .memory import ConversationMemory
from .planner import Planner
from .review_queue import ReviewQueue
from .router import Router

CHITCHAT_REPLY = (
    "I answer questions about *Designing Data-Intensive Applications* and cite the "
    "chapter and page each claim comes from. Ask me about replication, sharding, "
    "transactions, consensus, batch or stream processing."
)

OUT_OF_SCOPE_REPLY = (
    "I can't support this from the indexed corpus. This agent only answers from "
    "*Designing Data-Intensive Applications*, and that question falls outside it."
)

UNSUPPORTED_REPLY = (
    "I can't support this from the indexed corpus. I found related passages, but "
    "they don't actually state an answer to this question, so I'd rather decline "
    "than guess."
)


@dataclass(frozen=True)
class Profile:
    """One point on the ablation curve."""

    name: str
    use_bm25: bool = True
    use_vector: bool = True
    use_rerank: bool = True
    use_router: bool = True
    use_planner: bool = True
    use_critic: bool = True
    use_output_guardrail: bool = True
    use_tiering: bool = True  # Ch. 16: route decides the model tier
    description: str = ""


PROFILES: dict[str, Profile] = {
    "naive": Profile(
        name="naive",
        use_bm25=False,
        use_rerank=False,
        use_router=False,
        use_planner=False,
        use_critic=False,
        use_output_guardrail=False,
        use_tiering=False,
        description="Dense retrieval -> stuff context -> generate. The baseline.",
    ),
    "hybrid": Profile(
        name="hybrid",
        use_router=False,
        use_planner=False,
        use_critic=False,
        use_output_guardrail=False,
        use_tiering=False,
        description="+ BM25, reciprocal rank fusion and cross-encoder re-ranking.",
    ),
    "agentic": Profile(
        name="agentic",
        use_critic=False,
        use_output_guardrail=False,
        description="+ routing and multi-hop planning with parallel sub-question retrieval.",
    ),
    "full": Profile(
        name="full",
        description="+ reflection: faithfulness critic, regeneration, and refusal.",
    ),
}

DEFAULT_PROFILE = "full"


class Agent:
    def __init__(
        self,
        index: CorpusIndex,
        *,
        llm: LLMClient | None = None,
        profile: str | Profile = DEFAULT_PROFILE,
        memory: ConversationMemory | None = None,
        review_queue: ReviewQueue | None = None,
    ) -> None:
        self.index = index
        self.settings = get_settings()
        self.profile = profile if isinstance(profile, Profile) else PROFILES[profile]
        self.memory = memory
        self.review_queue = review_queue
        self._degradations: list[Degradation] = []
        self.llm = llm or build_llm(self._degradations)

    # ------------------------------------------------------------------
    async def answer(self, query: str, *, trace_id: str | None = None) -> AgentAnswer:
        started = time.perf_counter()
        tracer = Tracer(trace_id=trace_id)
        degradations: list[Degradation] = []
        self._degradations = degradations
        usage = Usage()
        profile = self.profile

        def finish(
            status: AnswerStatus,
            answer_text: str,
            **kwargs,
        ) -> AgentAnswer:
            result = AgentAnswer(
                query=query,
                status=status,
                answer=answer_text,
                usage=usage,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                stage_latency_ms=tracer.stage_latencies(),
                trace_id=tracer.trace_id,
                profile=profile.name,
                warnings=[str(d) for d in degradations],
                **kwargs,
            )
            tracer.event("answer", status=status.value, latency_ms=round(result.latency_ms, 1))
            tracer.flush()
            if self.memory is not None and status is AnswerStatus.ANSWERED:
                self.memory.add(query, answer_text)
            if (
                self.review_queue is not None
                and status is AnswerStatus.HUMAN_REVIEW
            ):
                self.review_queue.enqueue(result, reason=result.critic.reason if result.critic else "")
            return result

        with tracer.span("pipeline", profile=profile.name):
            # -- 1. input guardrails ------------------------------------
            with tracer.span("guardrails.input"):
                input_verdict = scan_input(query)
            if not input_verdict.passed:
                tracer.event("blocked", rules=[v.rule for v in input_verdict.blocking])
                return finish(
                    AnswerStatus.BLOCKED,
                    "I can't process that request: it looks like an attempt to override my "
                    "instructions rather than a question about the book.",
                    input_guardrail=input_verdict,
                )

            # -- 2. memory (follow-up resolution) ------------------------
            retrieval_query = query
            if self.memory is not None:
                retrieval_query, rewritten = self.memory.contextualize(query)
                if rewritten:
                    tracer.event("memory.rewrite", rewritten_query=retrieval_query)

            # -- 3. routing ----------------------------------------------
            router = Router(self.llm, self.index, tracer)
            if profile.use_router:
                route = await router.route(retrieval_query, degradations=degradations)
                usage = usage.merge(_last_usage(self.llm))
            else:
                route = RouteDecision(
                    route=Route.SIMPLE,
                    confidence=1.0,
                    reason="routing disabled by profile",
                    model_tier="cheap",
                    decided_by="disabled",
                )

            if route.route is Route.CHITCHAT:
                return finish(AnswerStatus.ANSWERED, CHITCHAT_REPLY, route=route,
                              input_guardrail=input_verdict)
            if route.route is Route.OUT_OF_SCOPE:
                return finish(AnswerStatus.DECLINED, OUT_OF_SCOPE_REPLY, route=route,
                              input_guardrail=input_verdict)

            # -- 4. planning ---------------------------------------------
            if profile.use_planner and route.route is Route.MULTI_HOP:
                plan = await Planner(self.llm, tracer).plan(retrieval_query, degradations=degradations)
            else:
                plan = Planner.passthrough(retrieval_query)

            # -- 5. retrieval ---------------------------------------------
            retrieval = await hybrid_retrieve(
                self.index,
                [retrieval_query] + [s for s in plan.sub_questions if s != retrieval_query],
                tracer=tracer,
                use_bm25=profile.use_bm25,
                use_vector=profile.use_vector,
                use_rerank=profile.use_rerank,
                degradations=degradations,
            )
            if not retrieval.chunks:
                return finish(
                    AnswerStatus.DECLINED,
                    OUT_OF_SCOPE_REPLY,
                    route=route,
                    plan=plan,
                    input_guardrail=input_verdict,
                    retrieval_degraded=retrieval.degraded,
                )

            contexts = build_contexts(retrieval, self.settings.final_context_chunks)

            # -- 6. generate (+ 7. reflect, with one repair attempt) ------
            tier = (
                Tier(route.model_tier)
                if profile.use_tiering and route.model_tier in {t.value for t in Tier}
                else Tier.CHEAP
            )
            generator = Generator(self.llm, tracer)
            critic = Critic(self.llm, tracer)

            draft: Draft
            citations: list[Citation]
            verdict: CriticVerdict | None = None
            extra_instruction = ""

            for attempt in range(1, self.settings.max_regeneration_attempts + 2):
                draft, citations = await generator.generate(
                    retrieval_query,
                    contexts,
                    tier=tier,
                    attempt=attempt,
                    extra_instruction=extra_instruction,
                    degradations=degradations,
                )
                usage = usage.merge(_last_usage(self.llm))

                if draft.declined:
                    return finish(
                        AnswerStatus.DECLINED,
                        UNSUPPORTED_REPLY,
                        route=route,
                        plan=plan,
                        input_guardrail=input_verdict,
                        retrieved_chunk_ids=retrieval.chunk_ids,
                        retrieval_degraded=retrieval.degraded,
                    )

                if not profile.use_critic:
                    verdict = None
                    break

                verdict = await critic.check(
                    retrieval_query, draft, contexts, degradations=degradations
                )
                usage = usage.merge(_last_usage(self.llm))

                if verdict.action is CriticAction.ACCEPT:
                    break
                if verdict.action is CriticAction.REGENERATE and attempt <= self.settings.max_regeneration_attempts:
                    unsupported = [c.claim for c in verdict.claims if not c.supported][:3]
                    extra_instruction = (
                        "The previous draft contained claims the cited passages do not "
                        f"support: {unsupported}. Drop them. Keep only what the context "
                        "states outright; if that leaves nothing, decline."
                    )
                    tracer.event("reflection.regenerate", unsupported=len(unsupported))
                    continue
                break

            # -- 8. act on the critic's verdict ---------------------------
            if verdict is not None:
                if verdict.action is CriticAction.DECLINE:
                    tracer.event("reflection.decline", faithfulness=round(verdict.faithfulness, 3))
                    return finish(
                        AnswerStatus.DECLINED,
                        UNSUPPORTED_REPLY,
                        route=route,
                        plan=plan,
                        critic=verdict,
                        input_guardrail=input_verdict,
                        retrieved_chunk_ids=retrieval.chunk_ids,
                        retrieval_degraded=retrieval.degraded,
                    )
                if verdict.action is CriticAction.HUMAN_REVIEW:
                    return finish(
                        AnswerStatus.HUMAN_REVIEW,
                        UNSUPPORTED_REPLY,
                        route=route,
                        plan=plan,
                        critic=verdict,
                        citations=citations,
                        input_guardrail=input_verdict,
                        retrieved_chunk_ids=retrieval.chunk_ids,
                        retrieval_degraded=retrieval.degraded,
                    )
                if verdict.action is CriticAction.REGENERATE:
                    # Out of attempts and still unsupported - escalate rather
                    # than ship it.
                    return finish(
                        AnswerStatus.HUMAN_REVIEW,
                        UNSUPPORTED_REPLY,
                        route=route,
                        plan=plan,
                        critic=verdict,
                        citations=citations,
                        input_guardrail=input_verdict,
                        retrieved_chunk_ids=retrieval.chunk_ids,
                        retrieval_degraded=retrieval.degraded,
                    )

            # -- 9. output guardrails --------------------------------------
            output_verdict = GuardrailVerdict(stage=GuardrailStage.OUTPUT, passed=True)
            if profile.use_output_guardrail:
                with tracer.span("guardrails.output"):
                    output_verdict = check_output(draft, citations)
                if not output_verdict.passed:
                    tracer.event("blocked.output", rules=[v.rule for v in output_verdict.blocking])
                    return finish(
                        AnswerStatus.HUMAN_REVIEW,
                        UNSUPPORTED_REPLY,
                        route=route,
                        plan=plan,
                        critic=verdict,
                        citations=citations,
                        input_guardrail=input_verdict,
                        output_guardrail=output_verdict,
                        retrieved_chunk_ids=retrieval.chunk_ids,
                        retrieval_degraded=retrieval.degraded,
                    )

            return finish(
                AnswerStatus.ANSWERED,
                draft.answer,
                route=route,
                plan=plan,
                critic=verdict,
                citations=citations,
                input_guardrail=input_verdict,
                output_guardrail=output_verdict,
                retrieved_chunk_ids=retrieval.chunk_ids,
                retrieval_degraded=retrieval.degraded,
            )


def _last_usage(llm: LLMClient) -> Usage:
    """Usage is accumulated on the client by `UsageTrackingLLM`; plain clients
    report nothing and the totals simply stay at zero."""
    take = getattr(llm, "take_usage", None)
    return take() if callable(take) else Usage()
